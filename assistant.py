#!/usr/bin/env python3
"""
Persönlicher Notion- & Sprachnotiz-Assistent (100% Lokal via Ollama & Whisper).
Belastet keine Antigravity-Kontingente.

Funktionen:
1. Nimmt Sprachnachrichten (.m4a, .mp3, .ogg, .wav) und Textnotizen (.txt, .md)
   aus `inbox_drop/`, aus einem optionalen Telegram-Bot oder direkt aus der
   Notion-Datenbank "AI Inbox" (Status = "Neu") entgegen.
2. Transkribiert Audiodateien lokal mit `faster-whisper` (falls installiert).
3. Wertet den Text lokal über Ollama (`qwen2.5-coder:7b`, `devstral:24b` o.ä.) aus:
   - Kurze Zusammenfassung
   - Konkrete To-Dos (inkl. Priorität, Fälligkeitsdatum, nächster Schritt)
   - Proaktive Vorschläge ("Dinge, die man tun kann")
4. Synchronisiert Ergebnisse mit Notion ("AI Inbox" + verknüpfte "Tasks")
   und speichert sie lokal in `local_state.json`.
5. Erzeugt ein tägliches Briefing inkl. Windows-Desktop-Erinnerung.
"""

from __future__ import annotations

import argparse
import datetime as dt
import email
from email.header import decode_header
import imaplib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from urllib import error, request

import chat_guard
import jev_client
import llm_core
from core import agent_tools
from core.storage import StateStore, atomic_write_json, safe_read_json
from integrations import calendar_sync, telegram_ui
from integrations.notion_client import NotionClient

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
ENV_PATH = BASE_DIR / ".env"

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac"}
TEXT_EXTENSIONS = {".txt", ".md"}

log = logging.getLogger("assistant")
_LOGGING_INITIALIZED = False
_STORE_CACHE: dict[str, StateStore] = {}


def setup_logging(log_file: Path | None = None) -> None:
    """Konfiguriert rotierendes Datei-Logging (assistant.log) + Konsolenausgabe."""
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return
    target = log_file or (BASE_DIR / "assistant.log")
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        stream_h = logging.StreamHandler()
        stream_h.setFormatter(fmt)
        root.addHandler(stream_h)
        try:
            rot_h = RotatingFileHandler(target, maxBytes=2_000_000, backupCount=4, encoding="utf-8")
            rot_h.setFormatter(fmt)
            root.addHandler(rot_h)
        except Exception:
            pass
    _LOGGING_INITIALIZED = True


def get_store(db_path: Path | None = None) -> StateStore:
    """Gibt die thread-sichere SQLite-StateStore-Instanz zurück."""
    target = db_path or Path(os.environ.get("ASSISTANT_DB_PATH") or (BASE_DIR / "assistant.db"))
    key = str(target.resolve())
    if key not in _STORE_CACHE:
        st = StateStore(target)
        try:
            st.migrate_from_legacy_json(BASE_DIR)
        except Exception:
            pass
        _STORE_CACHE[key] = st
    return _STORE_CACHE[key]


def load_dotenv_file(env_path: Path) -> None:
    """Liest eine einfache .env-Datei ohne externe Abhängigkeiten ein."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config() -> dict[str, Any]:
    load_dotenv_file(ENV_PATH)
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_local_state(state_path: Path) -> dict[str, Any]:
    data = safe_read_json(state_path, {"items": [], "tasks": [], "telegram_offset": 0})
    if not isinstance(data, dict):
        data = {}
    data.setdefault("items", [])
    data.setdefault("tasks", [])
    data.setdefault("telegram_offset", 0)
    return data


def save_local_state(state_path: Path, state: dict[str, Any]) -> None:
    atomic_write_json(state_path, state, indent=2)


def get_date_reference_block() -> str:
    """Erzeugt eine exakte deutsche Kalender-Referenztabelle zur fehlerfreien Datums-Arithmetik."""
    today = dt.date.today()
    today_iso = today.isoformat()
    weekday_names = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    weekday = weekday_names[today.weekday()]

    days_map: dict[str, str] = {}
    for i in range(1, 8):
        d = today + dt.timedelta(days=i)
        days_map[weekday_names[d.weekday()]] = d.isoformat()

    four_weeks = (today + dt.timedelta(weeks=4)).isoformat()
    return f"""Heutiges Datum: {weekday}, {today_iso}
Kalender-Referenztabelle für relative Datumsangaben:
- Heute: {today_iso} ({weekday})
- Morgen: {(today + dt.timedelta(days=1)).isoformat()} ({weekday_names[(today.weekday()+1)%7]})
- Übermorgen: {(today + dt.timedelta(days=2)).isoformat()} ({weekday_names[(today.weekday()+2)%7]})
- Nächster Montag: {days_map['Montag']}
- Nächster Dienstag: {days_map['Dienstag']}
- Nächster Mittwoch: {days_map['Mittwoch']}
- Nächster Donnerstag: {days_map['Donnerstag']}
- Nächster Freitag: {days_map['Freitag']}
- Nächster Samstag: {days_map['Samstag']}
- Nächster Sonntag: {days_map['Sonntag']}
- In 4 Wochen: {four_weeks}

WICHTIGE REGELN:
1. Berechne relative Zeitangaben EXAKT anhand dieser Referenztabelle.
2. Ein anstehender Termin darf NIEMALS in der Vergangenheit liegen (also nicht vor {today_iso}).
3. Wenn der Nutzer 'Sonntag' sagt und heute Sonntag ist, meint er heute ({today_iso}) oder nächsten Sonntag ({days_map['Sonntag']}).
"""


def transcribe_audio_local(audio_path: Path, whisper_cfg: dict[str, Any]) -> str:
    """Transkribiert eine lokale Audiodatei mit faster-whisper (rein lokal, ohne HF-Netzwerk-Abhängigkeit)."""
    try:
        import av  # type: ignore
        # Kompatibilitäts-Fix für neuere PyAV-Versionen, die metadata_errors entfernt haben
        orig_av_open = av.open
        av.open = lambda *a, **kw: orig_av_open(*a, **{k: v for k, v in kw.items() if k != "metadata_errors"})

        from faster_whisper import WhisperModel  # type: ignore

        model_size = whisper_cfg.get("model_size", "small")
        language = whisper_cfg.get("language", "de")
        device = whisper_cfg.get("device", "auto")
        print(f"[Whisper] Transkribiere {audio_path.name} (Modell: {model_size})...")

        # Versuche zuerst 100% offline aus dem lokalen Cache (verhindert HF-Netzwerkfehler)
        try:
            model = WhisperModel(model_size, device=device, compute_type="int8", local_files_only=True)
        except Exception:
            model = WhisperModel(model_size, device=device, compute_type="int8", local_files_only=False)

        segments, _ = model.transcribe(str(audio_path), language=language)
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        return transcript or f"[Leere Sprachnotiz: {audio_path.name}]"
    except ImportError:
        return (
            f"[Hinweis: Für lokale Audio-Transkription von '{audio_path.name}' "
            "bitte `pip install faster-whisper` ausführen.]"
        )
    except Exception as exc:
        print(f"[Whisper Fehler] {exc}")
        return f"[Fehler bei lokaler Whisper-Transkription: {exc}]"


def get_llm_choice(cfg: dict[str, Any]) -> tuple[str, str]:
    """Ermittelt den aktiven LLM-Provider ('gemini' oder 'ollama') und das Modell."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    llm_cfg = cfg.get("llm", {})
    provider = llm_cfg.get("provider", "gemini" if gemini_key else "ollama")
    if provider == "gemini" and not gemini_key:
        print("[LLM Provider] GEMINI_API_KEY nicht in .env gefunden -> automatischer Fallback auf lokales Ollama.")
        provider = "ollama"

    if provider == "gemini":
        model = llm_cfg.get("model", "gemini-3.8-flash")
    else:
        model = cfg.get("ollama", {}).get("model", "devstral:24b")
    return provider, model


def transcribe_audio(audio_path: Path, cfg: dict[str, Any]) -> str:
    """
    Transkribiert Sprachdateien:
    - Primär über Gemini 3.8 Flash (hochpräzise, fehlerfreie deutsche Grammatik & Interpunktion)
    - Fallback: Lokales faster-whisper (offline)
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    trans_cfg = cfg.get("transcription", {})
    provider = trans_cfg.get("provider", "gemini" if gemini_key else "whisper")

    if provider == "gemini" and gemini_key:
        try:
            print(f"[Gemini Audio] Transkribiere {audio_path.name} mit Gemini Flash...")
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)
            ext = audio_path.suffix.lower()
            mime_map = {
                ".ogg": "audio/ogg",
                ".oga": "audio/ogg",
                ".opus": "audio/ogg",
                ".mp3": "audio/mp3",
                ".wav": "audio/wav",
                ".m4a": "audio/m4a",
                ".aac": "audio/aac",
                ".flac": "audio/flac",
            }
            mime_type = mime_map.get(ext, "audio/ogg")
            audio_bytes = audio_path.read_bytes()
            prompt = (
                "Transkribiere diese deutsche Sprachnachricht wortgetreu, vollständig und präzise. "
                "Achte auf korrekte deutsche Rechtschreibung, Grammatik, Kommasetzung und Eigennamen. "
                "Gib ausschließlich den reinen transkribierten Text zurück, ohne Kommentare oder Anführungszeichen."
            )
            response = client.models.generate_content(
                model=cfg.get("llm", {}).get("model", "gemini-3.8-flash"),
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    prompt,
                ],
            )
            text = (response.text or "").strip()
            if text:
                return text
        except Exception as exc:
            print(f"[Gemini Audio Hinweis] {exc} -> Nutze lokales Whisper-Fallback.")

    return transcribe_audio_local(audio_path, cfg.get("whisper", {}))


def call_llm_json(
    prompt: str,
    cfg: dict[str, Any],
    system_instruction: str | None = None,
    model_override: str | None = None,
    task: str = "extract",
) -> dict[str, Any]:
    """Ruft Gemini (Standard) oder lokales Ollama mit strukturierter JSON-Rückgabe und optimiertem Thinking auf."""
    provider, default_model = get_llm_choice(cfg)

    # Modell und Thinking-Level bestimmen
    tier_model, tier_thinking = llm_core.resolve_task_model(task, cfg)
    model_name = model_override or tier_model or default_model

    if provider == "gemini":
        try:
            from google import genai

            gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
            client = genai.Client(api_key=gemini_key)
            system = system_instruction or (
                "Du bist der persönliche deutsche KI-Assistent von Stefan Kollmeier. "
                "Analysiere präzise und antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt."
            )
            resp = llm_core.execute_gemini_call(
                prompt=prompt,
                client=client,
                task=task,
                model=model_name,
                thinking_level=tier_thinking,
                system_instruction=system,
                response_mime_type="application/json",
            )
            raw = resp.text or "{}"
            return json.loads(raw)
        except Exception as exc:
            print(f"[Gemini JSON Fehler] {exc} -> Fallback auf Ollama.")

    return llm_core.call_ollama_json_safe(
        prompt=prompt,
        system=system_instruction,
        ollama_cfg=cfg.get("ollama", {}),
        model_override=model_override,
    )


def call_llm_text(
    prompt: str,
    cfg: dict[str, Any],
    system_instruction: str | None = None,
    model_override: str | None = None,
    task: str = "chat_simple",
) -> str:
    """Ruft Gemini oder Ollama für Textantworten auf."""
    provider, default_model = get_llm_choice(cfg)
    tier_model, tier_thinking = llm_core.resolve_task_model(task, cfg)
    model_name = model_override or tier_model or default_model

    if provider == "gemini":
        try:
            from google import genai

            gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
            client = genai.Client(api_key=gemini_key)
            resp = llm_core.execute_gemini_call(
                prompt=prompt,
                client=client,
                task=task,
                model=model_name,
                thinking_level=tier_thinking,
                system_instruction=system_instruction,
            )
            return (resp.text or "").strip()
        except Exception as exc:
            print(f"[Gemini Text Fehler] {exc} -> Fallback auf Ollama.")

    # Ollama Fallback
    ollama_cfg = cfg.get("ollama", {})
    res = llm_core.call_ollama_json_safe(
        prompt=prompt + "\n\nGib deine Antwort als JSON mit genau dem Feld 'answer': {\"answer\": \"...\"}",
        system=system_instruction,
        ollama_cfg=ollama_cfg,
        model_override=model_override,
    )
    return str(res.get("answer") or res.get("text") or "")


def call_ollama_json(
    prompt: str,
    ollama_cfg: dict[str, Any],
    model_override: str | None = None,
) -> dict[str, Any]:
    """Ruft das lokale Ollama-Modell auf und gibt strukturiertes JSON zurück."""
    base_url = os.environ.get("OLLAMA_BASE_URL", ollama_cfg.get("base_url", "http://localhost:11434"))
    models_to_try = [model_override] if model_override else [
        os.environ.get("OLLAMA_MODEL", ollama_cfg.get("model", "qwen2.5-coder:7b")),
        *ollama_cfg.get("fallback_models", []),
    ]

    last_err: Exception | None = None
    for model_name in [m for m in models_to_try if m]:
        payload = {
            "model": model_name,
            "format": "json",
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Du bist der persönliche deutsche KI-Assistent von Stefan Kollmeier. "
                        "Analysiere Sprachnotizen und Kurznotizen präzise. "
                        "Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        req = request.Request(
            f"{base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body.get("message", {}).get("content", "{}")
                return json.loads(content)
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(f"Lokaler Ollama-Aufruf fehlgeschlagen: {last_err}")


def analyze_note(
    title: str,
    text: str,
    cfg: dict[str, Any],
    model_override: str | None = None,
) -> dict[str, Any]:
    ref_block = get_date_reference_block()

    prompt = f"""{ref_block}

Analysiere die folgende Notiz / Sprachnachricht von Stefan Kollmeier und extrahiere strukturiert:
1. "title": Einen kurzen, prägnanten Titel (max. 60 Zeichen), falls der bisherige Titel generisch ist.
2. "summary": Eine kurze Zusammenfassung (1-2 Sätze auf fehlerfreiem Deutsch).
3. "area": Passender Lebensbereich – genau einer von: "Hochschule/Lehre", "Life on Pause", "Gesundheit/Forschung", "Familie/Privat", "Finanzen/Orga".
4. "todos": Eine Liste konkreter Aufgaben (To-Dos), die erledigt werden müssen (für die Notion-Aufgabenliste).
   Jedes To-Do hat:
   - "title": Kurze Aufgabenbeschreibung (präzise auf Deutsch)
   - "priority": "Hoch", "Mittel" oder "Niedrig"
   - "due_date": Konkretes Fälligkeitsdatum im Format "YYYY-MM-DD" (berechne relative Angaben wie "morgen", "Mittwoch", "nächste Woche", "Sonntag" EXAKT anhand der obigen Kalender-Referenztabelle) oder null, wenn kein Datum genannt ist.
   - "next_step": Konkreter erster Arbeitsschritt (1 Satz).
   - "estimated_minutes": Realistische geschätzte Dauer in Minuten (Ganzzahl, z. B. 15, 30, 60, 90).
5. "calendar_events": Eine Liste echter Kalendertermine mit festem Zeitpunkt/Zeitblock (z. B. Besprechung um 14 Uhr, Vorlesung, Arzttermin, Fokusblock 19-20 Uhr, Geburtstag).
   WICHTIGE ABGRENZUNG ZWISCHEN TODOS UND CALENDAR_EVENTS (KEINE DUPLIKATE!):
   - Eine allgemeine Aufgabe oder Besorgung an einem Tag ohne feste Uhrzeit (z. B. "Erinnere mich daran, am Mittwoch eine Überweisung zu holen" oder "Nächste Woche ETF-Anlage mit Frau machen") gehört NUR in "todos" (mit "due_date"), NIEMALS zusätzlich in "calendar_events"!
   - Ein Termin mit Uhrzeit oder ein echtes ganztägiges Ereignis gehört in "calendar_events", und dann NICHT nochmal als identisches To-Do in "todos"!
   - Erzeuge für dieselbe Sache NIEMALS gleichzeitig ein To-Do und einen Kalendereintrag!
   Jeder Kalendereintrag hat:
   - "title": Kurzer Titel des Termins
   - "date": Datum im Format "YYYY-MM-DD" (anhand der obigen Referenztabelle berechnet; NIEMALS in der Vergangenheit!)
   - "start_time": Startuhrzeit im Format "HH:MM" (24h, z. B. "11:00") oder null, falls ganztägig
   - "end_time": Enduhrzeit im Format "HH:MM" (Standard: 1 Stunde nach start_time, z. B. "12:00") oder null
   - "location": Ort / Raum (oder "")
   - "description": Kurzer Kontext zum Termin
6. "suggestions": 1 bis 2 hilfreiche, konkret umsetzbare Impulse / Ideen zur Notiz.
   Jeder Vorschlag hat:
   - "title": Kurzer Titel des Vorschlags
   - "priority": "Mittel" oder "Niedrig"
   - "next_step": Warum das sinnvoll ist und wie man konkret startet.

Bisheriger Titel: {title}
Notiz-Inhalt:
\"\"\"
{text}
\"\"\"

Gib genau dieses JSON-Schema zurück:
{{
  "title": "...",
  "summary": "...",
  "area": "Hochschule/Lehre | Life on Pause | Gesundheit/Forschung | Familie/Privat | Finanzen/Orga",
  "todos": [
    {{"title": "...", "priority": "Hoch|Mittel|Niedrig", "due_date": "YYYY-MM-DD oder null", "next_step": "...", "estimated_minutes": 30}}
  ],
  "calendar_events": [
    {{"title": "...", "date": "YYYY-MM-DD", "start_time": "HH:MM oder null", "end_time": "HH:MM oder null", "location": "...", "description": "..."}}
  ],
  "suggestions": [
    {{"title": "...", "priority": "Mittel|Niedrig", "next_step": "..."}}
  ]
}}"""

    result = call_llm_json(prompt, cfg, model_override=model_override)
    todos = result.get("todos") or []
    cal_events = result.get("calendar_events") or []
    note_area = result.get("area") or agent_tools.detect_area(f"{title} {text}")

    # Sicherheits-Deduplizierung: Falls das LLM denselben Punkt trotzdem in todos UND calendar_events packt:
    filtered_cal: list[dict[str, Any]] = []
    filtered_todos = list(todos)

    for ev in cal_events:
        ev_title_words = {w for w in (ev.get("title") or "").lower().split() if len(w) >= 4}
        ev_date = ev.get("date")
        ev_start = _normalize_hhmm(ev.get("start_time"))
        overlapping_todo = None
        for td in filtered_todos:
            td_words = {w for w in (td.get("title") or "").lower().split() if len(w) >= 4}
            if (ev_title_words and (ev_title_words & td_words)) or (ev_date and td.get("due_date") == ev_date and len(filtered_todos) == 1 and len(cal_events) == 1):
                overlapping_todo = td
                break
        if overlapping_todo is not None:
            if not ev_start:
                if not overlapping_todo.get("due_date") and ev_date:
                    overlapping_todo["due_date"] = ev_date
                continue
            else:
                filtered_todos = [t for t in filtered_todos if t is not overlapping_todo]
        filtered_cal.append(ev)

    for td in filtered_todos:
        td.setdefault("area", note_area)
        try:
            td["estimated_minutes"] = max(10, int(td.get("estimated_minutes") or 30))
        except Exception:
            td["estimated_minutes"] = 30

    return {
        "title": result.get("title") or title,
        "summary": result.get("summary") or "",
        "area": note_area,
        "todos": filtered_todos,
        "calendar_events": filtered_cal,
        "suggestions": result.get("suggestions") or [],
    }


def analyze_note_with_ollama(
    title: str,
    text: str,
    ollama_cfg: dict[str, Any],
    model_override: str | None = None,
) -> dict[str, Any]:
    """Rückwärtskompatibler Wrapper."""
    cfg = {"llm": {"provider": "ollama"}, "ollama": ollama_cfg}
    return analyze_note(title, text, cfg, model_override=model_override)


GCAL_CACHE_PATH = calendar_sync.GCAL_CACHE_PATH


def _normalize_hhmm(value: Any) -> str | None:
    return calendar_sync._normalize_hhmm(value)


def export_ics_event(ev: dict[str, Any]) -> Path:
    return calendar_sync.export_ics_event(ev)


def get_gcalendar_bin() -> str | None:
    return calendar_sync.get_gcalendar_bin()


def parse_ics_events(ics_text: str, start_date: dt.date, end_date: dt.date) -> list[dict[str, Any]]:
    return calendar_sync.parse_ics_events(ics_text, start_date, end_date)


def create_google_calendar_event(ev: dict[str, Any]) -> str | None:
    return calendar_sync.create_google_calendar_event(ev)


def handle_extracted_calendar_events(
    events: list[dict[str, Any]],
    source_title: str,
    state: dict[str, Any],
    notion: NotionClient | None,
    tasks_db_id: str,
    inbox_page_id: str | None = None,
) -> list[dict[str, Any]]:
    if "calendar_events" not in state:
        state["calendar_events"] = []

    created_records: list[dict[str, Any]] = []
    for ev in events:
        title = ev.get("title") or "Termin"
        date_str = ev.get("date")
        if not date_str or str(date_str).lower() == "null":
            continue

        start_t = _normalize_hhmm(ev.get("start_time"))
        end_t = _normalize_hhmm(ev.get("end_time"))
        time_info = f" ({start_t}–{end_t} Uhr)" if start_t and end_t else (f" ({start_t} Uhr)" if start_t else " (Ganztägig)")
        desc = ev.get("description") or ""
        loc = ev.get("location") or ""
        step_text = f"Termin am {date_str}{time_info}" + (f" | Ort: {loc}" if loc else "") + (f" | {desc}" if desc else "")

        ics_path = export_ics_event(ev)
        gcal_link = create_google_calendar_event(ev)
        if gcal_link:
            print(f"[Google Kalender] Termin eingetragen: '{title}' am {date_str}{time_info} -> {gcal_link}")
        else:
            print(f"[Kalender] Termin vorgemerkt & .ics erstellt ({ics_path.name}): '{title}' am {date_str}{time_info}")
            if notion:
                notion.create_task_page(
                    tasks_db_id=tasks_db_id,
                    title=f"{title}{time_info}",
                    kind="Erinnerung",
                    priority="Hoch",
                    due_date=date_str,
                    next_step=step_text,
                    inbox_page_id=inbox_page_id,
                    area=agent_tools.detect_area(f"{title} {desc}"),
                )

        rec = {
            "title": title,
            "date": date_str,
            "start_time": start_t,
            "end_time": end_t,
            "location": loc,
            "description": desc,
            "source_title": source_title,
            "ics_file": str(ics_path.relative_to(BASE_DIR)),
            "gcal_link": gcal_link,
            "gcal_event_id": ev.get("gcal_event_id"),
            "synced_to_gcal": bool(gcal_link),
        }
        state["calendar_events"].append(rec)
        created_records.append(rec)

    return created_records


def sync_pending_calendar_events(state: dict[str, Any]) -> int:
    """Synchronisiert noch offene zukünftige Kalendereinträge aus local_state.json mit Google Kalender."""
    today_iso = dt.date.today().isoformat()
    count = 0
    for ev in state.get("calendar_events", []):
        if ev.get("synced_to_gcal"):
            continue
        if (ev.get("date") or "") < today_iso:
            continue
        link = create_google_calendar_event(ev)
        if link:
            ev["gcal_link"] = link
            ev["synced_to_gcal"] = True
            if ev.get("gcal_event_id"):
                ev["gcal_event_id"] = ev["gcal_event_id"]
            count += 1
            print(f"[Google Kalender] Nachträglich synchronisiert: '{ev.get('title')}' ({ev.get('date')}) -> {link}")
    return count


def fetch_google_calendar_events(days_ahead: int = 14) -> list[dict[str, Any]]:
    return calendar_sync.fetch_google_calendar_events(days_ahead=days_ahead)


def find_matching_google_calendar_event(
    query_title: str,
    query_date: str | None = None,
    days_ahead: int = 30,
) -> dict[str, Any] | None:
    return calendar_sync.find_matching_google_calendar_event(query_title, query_date=query_date, days_ahead=days_ahead)


def update_google_calendar_event(
    event_id: str,
    new_date: str | None = None,
    new_start_time: str | None = None,
    new_end_time: str | None = None,
    new_title: str | None = None,
) -> bool:
    return calendar_sync.update_google_calendar_event(
        event_id=event_id,
        new_date=new_date,
        new_start_time=new_start_time,
        new_end_time=new_end_time,
        new_title=new_title,
    )


def delete_google_calendar_event(event_id: str) -> bool:
    return calendar_sync.delete_google_calendar_event(event_id)



def format_suggestions_text(suggestions: list[dict[str, Any]]) -> str:
    lines = []
    for s in suggestions:
        title = s.get("title", "Vorschlag")
        step = s.get("next_step", "")
        lines.append(f"• {title}: {step}" if step else f"• {title}")
    return "\n".join(lines)


SENSITIVE_EMAIL_PATTERNS = [
    r"passwort",
    r"password",
    r"reset",
    r"zurücksetzen",
    r"bestätigungscode",
    r"verification\s*code",
    r"security\s*code",
    r"auth\s*code",
    r"einmal-?passwort",
    r"one-?time\s*password",
    r"\b2fa\b",
    r"\bmfa\b",
    r"\botp\b",
    r"\btan\b",
    r"\biban\b",
    r"kreditkarte",
    r"credit\s*card",
    r"cvv\b",
    r"cvc\b",
    r"banking",
]


def decode_email_header(header_value: str) -> str:
    """Dekodiert E-Mail-Header wie Betreff oder Absender sicher."""
    if not header_value:
        return ""
    try:
        decoded_fragments = decode_header(header_value)
        result = []
        for fragment, enc in decoded_fragments:
            if isinstance(fragment, bytes):
                result.append(fragment.decode(enc or "utf-8", errors="replace"))
            else:
                result.append(str(fragment))
        return "".join(result).strip()
    except Exception:
        return str(header_value).strip()


def get_email_body_text(msg: Any) -> str:
    """Extrahiert den reinen Text-Inhalt einer E-Mail (ohne Anhänge)."""
    body = ""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition", ""))
                if content_type == "text/plain" and "attachment" not in disposition:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        body += payload.decode(charset, errors="replace") + "\n"
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                body = payload.decode(charset, errors="replace")
    except Exception as exc:
        print(f"[E-Mail Parser Hinweis] {exc}")
    return body.strip()


def is_sensitive_email(subject: str, body: str, sender: str) -> bool:
    """Harter Sicherheitsfilter: Erkennt 2FA-Codes, Passwörter und Bankdaten."""
    full_text = f"{subject}\n{body}\n{sender}".lower()
    for pattern in SENSITIVE_EMAIL_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            return True
    return False


def fetch_labeled_emails(cfg: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Sicherer Lesezugriff auf markierte E-Mails (Variante A: Opt-in Label).
    Liest ausschließlich E-Mails mit dem Label 'Assistent' (oder mit Stern).
    """
    mail_cfg = cfg.get("email", {})
    if not mail_cfg.get("enabled", False):
        return []

    user = os.environ.get("EMAIL_USER", "").strip() or os.environ.get("GMAIL_USER", "").strip()
    password = os.environ.get("EMAIL_PASSWORD", "").strip() or os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    if not user or not password:
        return []

    server = mail_cfg.get("server", "imap.gmail.com")
    port = int(mail_cfg.get("port", 993))
    label = mail_cfg.get("label", "Assistent")
    only_starred = mail_cfg.get("only_starred", False)
    max_emails = int(mail_cfg.get("max_emails", 5))

    processed_ids = set(state.get("processed_email_ids", []))
    emails_out: list[dict[str, Any]] = []

    try:
        mail = imaplib.IMAP4_SSL(server, port, timeout=15)
        mail.login(user, password)

        mailbox_name = '"[Gmail]/Starred"' if only_starred else f'"{label}"'
        status, _ = mail.select(mailbox_name, readonly=True)
        if status != "OK":
            status, _ = mail.select("INBOX", readonly=True)
            if status != "OK":
                mail.logout()
                return []

        search_criteria = "ALL"
        status, messages = mail.search(None, search_criteria)
        if status != "OK" or not messages[0]:
            mail.logout()
            return []

        msg_ids = messages[0].split()
        for msg_id in reversed(msg_ids[-max_emails:]):
            str_id = msg_id.decode()
            if str_id in processed_ids:
                continue

            status, data = mail.fetch(msg_id, "(RFC822)")
            if status != "OK" or not data or not data[0]:
                continue

            raw_email = data[0][1]
            if not isinstance(raw_email, bytes):
                continue

            msg = email.message_from_bytes(raw_email)
            subject = decode_email_header(msg.get("Subject", "(Kein Betreff)"))
            sender = decode_email_header(msg.get("From", "(Unbekannt)"))
            date_str = msg.get("Date", "")
            body = get_email_body_text(msg)

            # 1. Sicherheitsprüfung vor jeder KI-Analyse
            if is_sensitive_email(subject, body, sender):
                print(f"[Sicherheits-Gate] E-Mail '{subject}' von '{sender}' verworfen (enthält sensible Daten/2FA).")
                processed_ids.add(str_id)
                continue

            emails_out.append({
                "id": str_id,
                "sender": sender,
                "subject": subject,
                "date": date_str,
                "body": body[:4000],
            })
            processed_ids.add(str_id)

        mail.logout()
        state["processed_email_ids"] = list(processed_ids)[-200:]
    except Exception as exc:
        print(f"[E-Mail Abruf Hinweis] {exc}")

    return emails_out


def process_emails(
    cfg: dict[str, Any],
    state: dict[str, Any],
    notion: NotionClient | None,
    model_override: str | None = None,
) -> int:
    """Verarbeitet neu markierte E-Mails sicher nach Notion AI Inbox & Tasks."""
    new_emails = fetch_labeled_emails(cfg, state)
    if not new_emails:
        return 0

    count = 0
    for em in new_emails:
        subject = em.get("subject", "E-Mail")
        sender = em.get("sender", "")
        body = em.get("body", "")
        raw_text = f"E-Mail von: {sender}\nBetreff: {subject}\n\n{body}"

        print(f"[E-Mail Assistent] Werte markierte Mail aus: '{subject}'...")
        analysis = analyze_note(
            title=f"Mail: {subject[:45]}",
            text=raw_text,
            cfg=cfg,
            model_override=model_override,
        )
        suggestions_str = format_suggestions_text(analysis["suggestions"])

        inbox_page_id: str | None = None
        note_area = analysis.get("area") or agent_tools.detect_area(raw_text)
        if notion:
            inbox_page_id = notion.create_inbox_page(
                inbox_db_id=cfg["notion"]["inbox_database_id"],
                title=f"Mail: {analysis['title']}",
                source=f"E-Mail ({sender})",
                raw_text=raw_text,
                summary=analysis["summary"],
                suggestions_text=suggestions_str,
                area=note_area,
            )
            for todo in analysis["todos"]:
                notion.create_task_page(
                    tasks_db_id=cfg["notion"]["tasks_database_id"],
                    title=todo.get("title", "Aufgabe"),
                    kind="To-Do",
                    priority=todo.get("priority", "Mittel"),
                    due_date=todo.get("due_date"),
                    next_step=todo.get("next_step", ""),
                    inbox_page_id=inbox_page_id,
                    area=todo.get("area") or note_area,
                    estimated_minutes=todo.get("estimated_minutes"),
                )

        state["items"].append({
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "file": f"email_{em['id']}",
            "source": f"E-Mail ({sender})",
            "raw_text": raw_text,
            "analysis": analysis,
            "notion_inbox_page_id": inbox_page_id,
            "synced_to_notion": bool(inbox_page_id),
        })
        for todo in analysis["todos"]:
            state["tasks"].append({
                "kind": "To-Do",
                "title": todo.get("title", ""),
                "priority": todo.get("priority", "Mittel"),
                "due_date": todo.get("due_date"),
                "next_step": todo.get("next_step", ""),
                "source_title": f"Mail: {analysis['title']}",
                "status": "Nicht begonnen",
                "area": todo.get("area") or note_area,
                "estimated_minutes": int(todo.get("estimated_minutes") or 30),
            })

        handle_extracted_calendar_events(
            events=analysis.get("calendar_events", []),
            source_title=f"Mail: {analysis['title']}",
            state=state,
            notion=notion,
            tasks_db_id=cfg["notion"]["tasks_database_id"],
            inbox_page_id=inbox_page_id,
        )
        count += 1
        print(f"[OK] E-Mail '{subject}' verarbeitet ({len(analysis['todos'])} To-Dos).")

    return count


def process_local_drop_folder(
    cfg: dict[str, Any],
    state: dict[str, Any],
    notion: NotionClient | None,
    model_override: str | None = None,
) -> list[dict[str, Any]]:
    drop_dir = BASE_DIR / cfg["paths"]["inbox_drop_dir"]
    processed_dir = BASE_DIR / cfg["paths"]["processed_dir"]
    drop_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed_items: list[dict[str, Any]] = []
    files = sorted(
        f for f in drop_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (AUDIO_EXTENSIONS | TEXT_EXTENSIONS)
    )

    for file_path in files:
        is_audio = file_path.suffix.lower() in AUDIO_EXTENSIONS
        source = "Sprachnotiz (Audio)" if is_audio else "Lokaler Ordner"
        if is_audio:
            raw_text = transcribe_audio(file_path, cfg)
        else:
            raw_text = file_path.read_text(encoding="utf-8").strip()

        if not raw_text:
            continue

        print(f"[LLM] Werte Notiz '{file_path.name}' aus...")
        analysis = analyze_note(
            title=file_path.stem.replace("_", " "),
            text=raw_text,
            cfg=cfg,
            model_override=model_override,
        )
        suggestions_str = format_suggestions_text(analysis["suggestions"])

        inbox_page_id: str | None = None
        note_area = analysis.get("area") or agent_tools.detect_area(raw_text)
        if notion:
            inbox_page_id = notion.create_inbox_page(
                inbox_db_id=cfg["notion"]["inbox_database_id"],
                title=analysis["title"],
                source=source,
                raw_text=raw_text,
                summary=analysis["summary"],
                suggestions_text=suggestions_str,
                area=note_area,
            )
            for todo in analysis["todos"]:
                notion.create_task_page(
                    tasks_db_id=cfg["notion"]["tasks_database_id"],
                    title=todo.get("title", "Aufgabe"),
                    kind="To-Do",
                    priority=todo.get("priority", "Mittel"),
                    due_date=todo.get("due_date"),
                    next_step=todo.get("next_step", ""),
                    inbox_page_id=inbox_page_id,
                    area=todo.get("area") or note_area,
                    estimated_minutes=todo.get("estimated_minutes"),
                )
            # Hinweis: KI-Vorschläge werden NICHT als Tasks angelegt, um die Aufgabenliste nicht zu überladen.

        record = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "file": file_path.name,
            "source": source,
            "raw_text": raw_text,
            "analysis": analysis,
            "notion_inbox_page_id": inbox_page_id,
            "synced_to_notion": bool(inbox_page_id),
        }
        state["items"].append(record)
        for todo in analysis["todos"]:
            state["tasks"].append({
                "kind": "To-Do",
                "title": todo.get("title", ""),
                "priority": todo.get("priority", "Mittel"),
                "due_date": todo.get("due_date"),
                "next_step": todo.get("next_step", ""),
                "source_title": analysis["title"],
                "status": "Nicht begonnen",
                "area": todo.get("area") or note_area,
                "estimated_minutes": int(todo.get("estimated_minutes") or 30),
            })

        handle_extracted_calendar_events(
            events=analysis.get("calendar_events", []),
            source_title=analysis["title"],
            state=state,
            notion=notion,
            tasks_db_id=cfg["notion"]["tasks_database_id"],
            inbox_page_id=inbox_page_id,
        )

        target = processed_dir / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_path.name}"
        shutil.move(str(file_path), str(target))
        processed_items.append(record)
        print(
            f"[OK] '{analysis['title']}' ausgewertet ("
            f"{len(analysis['todos'])} To-Dos, "
            f"{len(analysis.get('calendar_events', []))} Kalendereinträge, "
            f"{len(analysis['suggestions'])} Vorschläge)."
        )

    return processed_items


def is_question_message(text: str) -> bool:
    """Erkennt, ob eine Nachricht eine Frage/Abfrage an den Assistenten ist statt einer neuen Notiz."""
    t = text.strip().lower()
    if t.startswith(("/ask", "/frage", "frage:")):
        return True
    question_starters = (
        "was ", "welche", "wann ", "wie ", "wo ", "wer ", "warum ", "wieso ",
        "habe ich", "hab ich", "steht heute", "steht morgen", "gibt es",
        "zeig mir", "zeig ", "liste ", "kannst du mir sagen", "erinnere mich an meine",
    )
    if t.endswith("?") and len(t.split()) <= 25:
        return True
    return any(t.startswith(q) for q in question_starters)


def find_matching_task(query: str, tasks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Findet eine passende Aufgabe in der Aufgabenliste anhand von Titel oder Schlüsselwörtern."""
    if not query:
        return None
    clean_q = query.strip().lower()
    for t in tasks:
        tt = (t.get("title") or "").strip().lower()
        if clean_q in tt or tt in clean_q:
            return t
    q_words = {w for w in clean_q.split() if len(w) >= 3}
    best_t = None
    best_overlap = 0
    for t in tasks:
        t_words = {w for w in (t.get("title") or "").lower().split() if len(w) >= 3}
        ov = len(q_words & t_words)
        if ov > best_overlap:
            best_overlap = ov
            best_t = t
    return best_t


def parse_task_indices(text: str, max_idx: int) -> list[int]:
    """Extrahiert 1-basierte Aufgaben-Indizes aus Strings wie '1, 3, 5', '1-3', '1 und 4'."""
    if not text or max_idx <= 0:
        return []
    indices: set[int] = set()

    # 1. Bereiche: z. B. 1-3 oder 1 bis 3
    for match in re.finditer(r"\b(\d+)\s*(?:-|bis)\s*(\d+)\b", text, flags=re.IGNORECASE):
        start, end = int(match.group(1)), int(match.group(2))
        if start > end:
            start, end = end, start
        for i in range(start, end + 1):
            if 1 <= i <= max_idx:
                indices.add(i)

    # 2. Einzelne Nummern: z. B. 1, 3, 4
    cleaned = re.sub(r"\b\d+\s*(?:-|bis)\s*\d+\b", " ", text, flags=re.IGNORECASE)
    for num_str in re.findall(r"\b\d+\b", cleaned):
        val = int(num_str)
        if 1 <= val <= max_idx:
            indices.add(val)

    return sorted(indices)


def resolve_tasks_to_complete(
    text: str, open_todos: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Löst Benutzereingaben (mehrere Nummern, Bereiche, Namen oder Freitext) in Aufgaben auf."""
    if not text or not open_todos:
        return []

    # 1. Nummern oder Bereiche (z. B. '1, 3', '1-3', '1 und 4')
    indices = parse_task_indices(text, len(open_todos))
    if indices:
        return [open_todos[i - 1] for i in indices]

    # 2. Bereinigung für Textsuche
    clean = re.sub(
        r"^(/erledigt|/done|erledigt:?|abgehakt:?|fertig:?)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    clean = re.sub(
        r"\b(habe|hab|ich habe|bitte|als erledigt markieren|erledigt|abgehakt|fertig|schon)\b",
        "",
        clean,
        flags=re.IGNORECASE,
    ).strip()

    matched: list[dict[str, Any]] = []
    matched_ids: set[str] = set()

    # Nach Komma oder 'und' aufteilen
    sub_queries = [p.strip() for p in re.split(r",|\bund\b", clean, flags=re.IGNORECASE) if p.strip()]
    if not sub_queries:
        sub_queries = [clean] if clean else []

    for q in sub_queries:
        if len(q) < 2:
            continue
        t = find_matching_task(q, open_todos)
        if t:
            tid = str(t.get("notion_page_id") or t.get("title"))
            if tid not in matched_ids:
                matched.append(t)
                matched_ids.add(tid)

    if not matched and clean:
        t = find_matching_task(clean, open_todos)
        if t:
            matched.append(t)

    return matched


def answer_question(
    question: str,
    state: dict[str, Any],
    cfg: dict[str, Any],
    notion: NotionClient | None = None,
    model_override: str | None = None,
) -> str:
    """
    Umfassender persönlicher Dialog-Assistent:
    Beantwortet Fragen, formuliert E-Mails/Nachrichten vor, gibt Ratschläge zur Tagesplanung
    und nutzt Live-Kontext aus Google Kalender, Notion Tasks und Notizen.
    """
    today_iso = dt.date.today().isoformat()
    weekday_names = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    weekday = weekday_names[dt.date.today().weekday()]

    open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
    suggestions = [t for t in state.get("tasks", []) if t.get("kind") == "KI-Vorschlag / Idee" and t.get("status") != "Erledigt"]

    # Echte Google-Kalendertermine für die nächsten 14 Tage live abrufen
    gcal_events = fetch_google_calendar_events(days_ahead=14)
    seen_keys = {(ev.get("title"), ev.get("date"), ev.get("start_time")) for ev in gcal_events}
    all_cal = list(gcal_events)
    for ev in state.get("calendar_events", []):
        if (ev.get("date") or "") < today_iso:
            continue
        key = (ev.get("title"), ev.get("date"), ev.get("start_time"))
        if key not in seen_keys:
            all_cal.append(ev)
    all_cal.sort(key=lambda x: (x.get("date") or "", x.get("start_time") or "00:00"))

    recent_notes = [
        {"title": it.get("analysis", {}).get("title"), "summary": it.get("analysis", {}).get("summary"), "date": it.get("timestamp")}
        for it in state.get("items", [])[-10:]
    ]

    # Falls die Frage nach Notizen oder Notion-Inhalten fragt, suche kurz in Notion
    notion_knowledge = []
    if notion and any(w in question.lower() for w in ("notion", "notiz", "wissen", "life on pause", "vorlesung", "dokument")):
        try:
            notion_knowledge = notion.search_workspace(question, limit=4)
        except Exception:
            pass

    # Schlanke Payloads deterministisch filtern (spart >60% Kontext-Tokens)
    slimmed_todos = [chat_guard.slim_task(t) for t in open_todos[:15]]
    slimmed_cal = [chat_guard.slim_calendar_event(ev) for ev in all_cal[:20]]
    slimmed_notes = [chat_guard.slim_recent_note(it.get("analysis", {})) for it in state.get("items", [])[-6:]]
    slimmed_search = [chat_guard.slim_search_hit(h) for h in notion_knowledge[:4]]

    cal_json = chat_guard.truncate_context_json(slimmed_cal, max_chars=1800)
    todos_json = chat_guard.truncate_context_json(slimmed_todos, max_chars=1800)
    notes_json = chat_guard.truncate_context_json(slimmed_notes, max_chars=1200)
    search_json = chat_guard.truncate_context_json(slimmed_search, max_chars=800)

    # Modell-Tiering: Nur bei expliziter Analyse auf gemini-3.8-flash eskalieren
    is_complex = any(k in question.lower() for k in ("/analyse", "analysiere", "komplexe planung", "strategie", "mehrstufig"))
    task_tier = "chat_complex" if is_complex else "chat_simple"

    ref_block = get_date_reference_block()
    prompt = f"""{ref_block}

Du bist der persönliche KI-Assistent & Co-Pilot von Stefan Kollmeier (Hochschuldozent, Videoproduzent 'Life on Pause', Familienmensch).
Stefan spricht oder schreibt dir über Telegram folgende Nachricht:
"{question}"

Hier ist sein relevanter Kontext (Live aus Google Kalender, Notion & lokalem Speicher):
- Echte Kalendertermine (nächste Tage): {cal_json}
- Offene To-Dos (aus Notion Tasks): {todos_json}
- Zuletzt erfasste Notizen: {notes_json}
- Relevante Notion-Seiten: {search_json}

Rolle & Verhalten:
1. Antworte immer auf fehlerfreiem Deutsch, professionell, präzise, freundlich und auf den Punkt für einen Telegram-Chat.
2. Wenn Stefan nach Terminen, Plänen oder freier Zeit fragt, nutze die echten Google-Kalendertermine und berechne freie Zeitfenster.
3. Wenn er bittet, eine E-Mail, Nachricht oder einen Text vorzuformulieren, schreibe direkt einen druckreifen Entwurf.
4. Wenn er nach Aufgaben fragt, liste die relevanten To-Dos mit Priorität und Fälligkeit auf.
5. Verwende klare Formatierung (Aufzählungspunkte, Zeilenumbrüche).

Gib ein JSON-Objekt mit genau einem Feld "answer" (String) zurück:
{{"answer": "..."}}"""

    try:
        res = call_llm_json(prompt, cfg, model_override=model_override, task=task_tier)
        return str(res.get("answer") or "Ich konnte dazu leider keine passende Antwort generieren.")
    except Exception as exc:
        return f"Fehler bei der Antwortgenerierung: {exc}"


def answer_question_with_ollama(
    question: str,
    state: dict[str, Any],
    ollama_cfg: dict[str, Any],
    notion: NotionClient | None = None,
    model_override: str | None = None,
) -> str:
    """Rückwärtskompatibler Wrapper."""
    cfg = {"llm": {"provider": "ollama"}, "ollama": ollama_cfg}
    return answer_question(question, state, cfg, notion=notion, model_override=model_override)


def classify_intent(
    text: str,
    cfg: dict[str, Any],
    model_override: str | None = None,
) -> dict[str, Any]:
    """
    Klassifiziert die Absicht (Intent) einer Sprach- oder Textnachricht umfassend.
    """
    ref_block = get_date_reference_block()
    prompt = f"""{ref_block}

Stefan spricht oder schreibt folgende Nachricht an seinen persönlichen Assistenten:
\"\"\"{text}\"\"\"

Klassifiziere die genaue Absicht (Intent):
1. "FEEDBACK_OR_CORRECTION": Wenn Stefan mitteilt, dass ein Termin oder eine Aufgabe falsch eingetragen wurde, korrigiert werden muss oder sich über ein Missverständnis beschwert (z. B. "Die Materialien und Kuchentermine sind falsch eingetragen. Ich habe gesagt Sonntag!", "Das stimmt nicht", "Das war falsch").
2. "RESCHEDULE_EVENT": Einen BESTEHENDEN Termin im Kalender ändern, verschieben oder Uhrzeit/Datum anpassen (z. B. "Ändere den Termin Elterngespräch auf 15 Uhr", "Verschiebe Treffen morgen um 1h").
3. "DELETE_EVENT": Einen BESTEHENDEN Kalendertermin absagen / löschen (z. B. "Lösche das Treffen...", "Termin absagen").
4. "CREATE_EVENT": Explizit einen NEUEN Termin oder Blocker im Google Kalender eintragen (z. B. "Trage morgen um 14 Uhr Zahnarzt ein").
5. "QUERY_CALENDAR": Nach Kalenderterminen, Tagesplan oder freier Zeit fragen (z. B. "Was steht heute im Kalender?", "Habe ich heute Nachmittag 2 Stunden frei?", "Wann ist mein nächster Termin?").
6. "UPDATE_TASK": Eine BESTEHENDE Aufgabe in Notion anpassen (z. B. "Verschiebe die Aufgabe Video schneiden auf nächsten Freitag", "Setze Priorität von X auf Hoch").
7. "COMPLETE_TASK": Eine Aufgabe / To-Do als erledigt melden (z. B. "Ich habe das Video fertig geschnitten", "Habe Aufgabe X erledigt").
8. "DELETE_TASK": Eine Aufgabe in Notion löschen / streichen / archivieren (z. B. "Lösche die Aufgabe Testtask", "Streiche Aufgabe X").
9. "QUERY_TASKS": Nach Aufgaben fragen / filtern (z. B. "Welche To-Dos sind heute fällig?", "Welche Aufgaben mit hoher Prio gibt es?", "Was steht für Life on Pause an?").
10. "SEARCH_KNOWLEDGE": Nach Notizen, Dokumenten oder Wissen im Notion-Workspace suchen (z. B. "Suche in Notion nach...", "Was hatte ich mir zu X notiert?").
11. "GENERAL_CHAT": Eine allgemeine Frage, Hilfe beim Formulieren von Texten/E-Mails, Brainstorming, Rat oder lockeres Gespräch (z. B. "Formuliere mir eine E-Mail...", "Wie strukturiere ich...").
12. "NOTE_OR_TASK": Eine neue Notiz erfassen (Gedanken, neue Aufgaben für Notion Inbox & Tasks).

WICHTIG:
1. Wenn Stefan sich über einen falschen Eintrag beschwert oder einen Termin korrigieren möchte, wähle IMMER "FEEDBACK_OR_CORRECTION" oder "RESCHEDULE_EVENT". Erstelle daraus NIEMALS ein neues To-Do!
2. Berechne relative Zeit- und Datumsangaben EXAKT anhand der obigen Kalender-Referenztabelle als YYYY-MM-DD.

Gib ein JSON-Objekt zurück mit genau dieser Struktur:
{{
  "intent": "FEEDBACK_OR_CORRECTION | RESCHEDULE_EVENT | DELETE_EVENT | CREATE_EVENT | QUERY_CALENDAR | UPDATE_TASK | COMPLETE_TASK | DELETE_TASK | QUERY_TASKS | SEARCH_KNOWLEDGE | GENERAL_CHAT | NOTE_OR_TASK",
  "event_title": "Name des Termins oder null",
  "new_date": "YYYY-MM-DD (berechnet ab Referenztabelle) oder null",
  "new_start_time": "HH:MM oder null",
  "new_end_time": "HH:MM oder null",
  "task_title": "Name der Aufgabe oder null",
  "new_priority": "Hoch | Mittel | Niedrig oder null",
  "query": "Suchbegriff oder Frage oder null",
  "correction_details": "Kurze Zusammenfassung was korrigiert werden soll oder null"
}}"""

    # Optionaler Jev-System-One-Vorfilter für schnelle Intent-Klassifikation
    # Streng geschützt: Nur wenn JEV_ALLOW_CHAT_ROUTING explizit aktiv ist und PII maskiert wurde.
    if jev_client.is_jev_chat_allowed():
        jclient = jev_client.JevClient()
        if jclient.is_configured:
            try:
                # Maskiere persönliche Kontaktdaten und begrenze auf kurze Metadaten
                safe_text = jev_client.redact_pii_for_external_router(text, max_chars=180)
                options = {
                    "QUERY_CALENDAR": "Fragen zum Kalender, Tagesplan, Terminen heute/morgen, freie Zeit",
                    "QUERY_TASKS": "Fragen zu offenen Aufgaben, To-Dos, fällige Tasks, Prioritäten",
                    "SEARCH_KNOWLEDGE": "Suchen in Notion, Notizen, Dokumenten oder Wissensspeicher",
                    "GENERAL_CHAT": "Allgemeine Fragen, E-Mail/Text vorformulieren, Brainstorming, Rat",
                    "OTHER_ACTION": "Termin neu eintragen, ändern, absagen, Task updaten, erledigen oder Notiz",
                }
                choice, conf, _ = jclient.evaluate_choice(
                    state=safe_text,
                    instructions="Welche Intention hat die kurze Benutzer-Nachricht an den Assistenten?",
                    options=options,
                    question_id="chat_intent",
                )
                if conf >= 0.70 and choice in ("QUERY_CALENDAR", "QUERY_TASKS", "SEARCH_KNOWLEDGE", "GENERAL_CHAT"):
                    print(f"[Jev Router] Slotfreier Intent '{choice}' mit Confidence {conf:.2f} deterministisch erkannt.")
                    return {"intent": choice, "query": text}
            except Exception as jexc:
                print(f"[Jev Router Hinweis] {jexc} -> Fallback auf Gemini Flash-Lite.")

    fast_model = model_override or cfg.get("llm", {}).get("fast_model", "gemini-3.5-flash-lite")
    try:
        return call_llm_json(prompt, cfg, model_override=fast_model, task="intent")
    except Exception as exc:
        print(f"[Intent Router Fallback] {exc}")
        if is_question_message(text):
            return {"intent": "GENERAL_CHAT", "query": text}
        return {"intent": "NOTE_OR_TASK"}


def _execute_undo(
    store: StateStore,
    chat_id: str,
    state: dict[str, Any],
    notion: NotionClient | None,
) -> str:
    """Macht die letzte protokollierte Aktion (Task erledigen/anlegen, Termin löschen/anlegen) rückgängig."""
    entry = store.pop_last_undo("main", chat_id)
    if not entry:
        return "ℹ️ Keine kürzliche Aktion zum Rückgängigmachen vorhanden."

    action = entry.get("action_type", "")
    payload = entry.get("payload") or {}

    if action == "reopen_tasks":
        titles = payload.get("titles") or []
        page_ids = payload.get("page_ids") or []
        for pid in page_ids:
            if notion and pid:
                try:
                    notion.reopen_task(pid)
                except Exception as exc:
                    log.warning("Undo reopen_task Fehler (%s): %s", pid, exc)
        for t in state.get("tasks", []):
            if t.get("notion_page_id") in page_ids or t.get("title") in titles:
                t["status"] = "Nicht begonnen"
        label = ", ".join(telegram_ui.esc(x) for x in titles[:3]) or f"{len(page_ids)} Aufgabe(n)"
        return f"↩️ <b>Wieder geöffnet:</b> {label}"

    if action == "archive_task":
        pid = payload.get("page_id")
        title = payload.get("title") or "Aufgabe"
        if notion and pid:
            try:
                notion.archive_task(pid)
            except Exception as exc:
                log.warning("Undo archive_task Fehler (%s): %s", pid, exc)
        state["tasks"] = [
            t for t in state.get("tasks", [])
            if t.get("notion_page_id") != pid and t.get("title") != title
        ]
        return f"↩️ <b>Aufgabe wieder entfernt:</b> {telegram_ui.esc(title)}"

    if action == "reopen_task":
        pid = payload.get("page_id")
        title = payload.get("title") or "Aufgabe"
        if notion and pid:
            try:
                notion.reopen_task(pid)
            except Exception as exc:
                log.warning("Undo reopen_task Fehler (%s): %s", pid, exc)
        return f"↩️ <b>Gelöschte Aufgabe wiederhergestellt:</b> {telegram_ui.esc(title)}"

    if action == "delete_event":
        eid = payload.get("event_id")
        title = payload.get("title") or "Termin"
        ok = bool(eid and delete_google_calendar_event(eid))
        state["calendar_events"] = [
            e for e in state.get("calendar_events", [])
            if e.get("gcal_event_id") != eid and e.get("title") != title
        ]
        if ok or not eid:
            return f"↩️ <b>Termin wieder entfernt:</b> {telegram_ui.esc(title)}"
        return f"⚠️ Konnte Termin '{telegram_ui.esc(title)}' in Google Kalender nicht entfernen."

    if action == "restore_event":
        ev = dict(payload)
        link = create_google_calendar_event(ev)
        if link:
            state.setdefault("calendar_events", []).append({
                "title": ev.get("title", "Termin"),
                "date": ev.get("date"),
                "start_time": ev.get("start_time"),
                "end_time": ev.get("end_time"),
                "location": ev.get("location", ""),
                "description": ev.get("description", ""),
                "gcal_link": link,
                "gcal_event_id": ev.get("gcal_event_id"),
                "synced_to_gcal": True,
            })
            return f"↩️ <b>Termin wiederhergestellt:</b> {telegram_ui.esc(ev.get('title', 'Termin'))}"
        return f"⚠️ Konnte Termin '{telegram_ui.esc(ev.get('title', 'Termin'))}' nicht wiederherstellen."

    return "ℹ️ Aktionstyp kann nicht rückgängig gemacht werden."


def _build_slots_reply(
    target_date: dt.date,
    cfg: dict[str, Any],
    state: dict[str, Any],
    store: StateStore,
) -> tuple[str, dict[str, Any] | None]:
    """Erstellt die /slots Übersicht (inkl. anonymer Arbeitskalender-Blocker + Time-Block Buttons)."""
    import uuid

    date_iso = target_date.isoformat()
    today_iso = dt.date.today().isoformat()
    days_delta = max(1, (target_date - dt.date.today()).days + 1)

    gcal_events = fetch_google_calendar_events(days_ahead=days_delta)
    work_busy = calendar_sync.fetch_work_busy_slots(days_ahead=days_delta, store=store)

    seen_keys = {(ev.get("title"), ev.get("date"), ev.get("start_time")) for ev in gcal_events}
    private_events = [ev for ev in gcal_events if ev.get("date") == date_iso]
    for ev in state.get("calendar_events", []):
        if ev.get("date") != date_iso or (ev.get("date") or "") < today_iso:
            continue
        key = (ev.get("title"), ev.get("date"), ev.get("start_time"))
        if key not in seen_keys:
            private_events.append(ev)
            seen_keys.add(key)
    private_events.sort(key=lambda x: x.get("start_time") or "00:00")

    day_work_busy = [w for w in work_busy if w.get("date") == date_iso]

    work_hours = cfg.get("work_hours", {})
    work_start = os.environ.get("WORK_HOURS_START") or work_hours.get("start", "08:00")
    work_end = os.environ.get("WORK_HOURS_END") or work_hours.get("end", "20:00")

    combined = private_events + day_work_busy
    free_slots = calendar_sync.compute_free_slots(
        combined,
        target_date=date_iso,
        work_start=work_start,
        work_end=work_end,
        min_slot_minutes=30,
    )
    open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
    block_suggestions = calendar_sync.suggest_time_blocks(free_slots, open_todos, max_suggestions=3)

    weekday_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    wd = weekday_names[target_date.weekday()]
    lines = [f"⏱️ <b>Zeitfenster & Fokus-Slots ({wd}, {date_iso})</b>\n"]

    if private_events:
        lines.append("📅 <b>Private Termine:</b>")
        for ev in private_events:
            t_lbl = (
                f"{ev['start_time']}–{ev.get('end_time') or ''}"
                if ev.get("start_time")
                else "Ganztägig"
            )
            lines.append(f"  • {telegram_ui.esc(t_lbl)} · {telegram_ui.esc(ev.get('title', 'Termin'))}")
        lines.append("")

    if day_work_busy:
        lines.append(f"🔒 <b>Arbeitskalender (nur Frei/Belegt, {len(day_work_busy)} Blöcke):</b>")
        for wb in day_work_busy:
            w_lbl = (
                f"{wb['start_time']}–{wb.get('end_time') or ''}"
                if wb.get("start_time")
                else "Ganztägig"
            )
            lines.append(f"  • {telegram_ui.esc(w_lbl)} · Arbeit: Belegt")
        lines.append("")

    if free_slots:
        lines.append("⏳ <b>Freie Zeitfenster:</b>")
        for sl in free_slots:
            lines.append(
                f"  • <b>{sl['start_time']}–{sl['end_time']}</b> ({sl['duration_minutes']} Min.)"
            )
    else:
        lines.append("⏳ <b>Freie Zeitfenster:</b> Keine freien Blöcke ≥ 30 Min. gefunden.")

    kb_rows: list[list[tuple[str, str]]] = []
    if block_suggestions:
        lines.append("\n🎯 <b>Vorgeschlagene Time-Blocks (Top-To-Dos):</b>")
        for sug in block_suggestions:
            lines.append(
                f"  • {sug['start_time']}–{sug['end_time']}: "
                f"<b>{telegram_ui.esc(sug['task_title'])}</b> [{telegram_ui.esc(sug['priority'])}]"
            )
            tok = uuid.uuid4().hex[:10]
            store.put_callback_payload(tok, "time_block", sug)
            lbl = f"⏱️ Blocken: {sug['task_title'][:22]} ({sug['start_time']})"
            kb_rows.append([(lbl, f"block:{tok}")])

    markup = telegram_ui.build_inline_keyboard(kb_rows) if kb_rows else None
    return "\n".join(lines), markup


def _handle_telegram_callback_query(
    cq: dict[str, Any],
    bot_token: str,
    api_base: str,
    cfg: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    notion: NotionClient | None,
    store: StateStore,
) -> bool:
    """Verarbeitet Inline-Button-Klicks (`done:`, `defer:`, `block:`, `undo:`)."""
    cq_id = cq.get("id", "")
    data = str(cq.get("data") or "")
    msg = cq.get("message") or {}
    chat_id = str((msg.get("chat") or {}).get("id") or (cq.get("from") or {}).get("id") or "")

    if not telegram_ui.is_authorized_chat(chat_id, env_keys=("TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USERS")):
        telegram_ui.answer_callback_query(bot_token, cq_id, "Keine Berechtigung.", show_alert=True)
        return False

    open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]

    if data.startswith("done:"):
        try:
            idx = int(data.split(":", 1)[1])
        except ValueError:
            idx = 0
        if 1 <= idx <= len(open_todos):
            t = open_todos[idx - 1]
            t["status"] = "Erledigt"
            n_id = t.get("notion_page_id")
            if notion and n_id:
                try:
                    notion.mark_task_done(n_id)
                except Exception as exc:
                    log.warning("Callback mark_task_done Fehler: %s", exc)
            store.push_undo(
                "main",
                chat_id,
                "reopen_tasks",
                {"page_ids": [n_id] if n_id else [], "titles": [t.get("title", "")]},
            )
            save_local_state(state_path, state)
            telegram_ui.answer_callback_query(bot_token, cq_id, "✅ Als erledigt markiert!")
            markup = telegram_ui.build_inline_keyboard([[("↩️ Rückgängig", "undo:last")]])
            _send_telegram_message(
                api_base,
                chat_id,
                f"✅ <b>Erledigt:</b> {telegram_ui.esc(t.get('title', 'Aufgabe'))}",
                reply_markup=markup,
            )
            return True
        telegram_ui.answer_callback_query(bot_token, cq_id, "Aufgabe nicht mehr in offener Liste.")
        return False

    if data.startswith("defer:"):
        try:
            idx = int(data.split(":", 1)[1])
        except ValueError:
            idx = 0
        if 1 <= idx <= len(open_todos):
            t = open_todos[idx - 1]
            tomorrow = (dt.date.today() + dt.timedelta(days=1)).isoformat()
            t["due_date"] = tomorrow
            n_id = t.get("notion_page_id")
            if notion and n_id:
                try:
                    notion.update_task(page_id=n_id, due_date=tomorrow)
                except Exception as exc:
                    log.warning("Callback defer update_task Fehler: %s", exc)
            defers = store.increment_task_defer(n_id or t.get("title", ""))
            save_local_state(state_path, state)
            telegram_ui.answer_callback_query(bot_token, cq_id, f"📅 Auf {tomorrow} verschoben")
            hint = (
                f"\n💡 <i>Hinweis: Schon {defers}× verschoben – evtl. in kleineren 15-Min-Schritt teilen oder streichen?</i>"
                if defers >= 3
                else ""
            )
            _send_telegram_message(
                api_base,
                chat_id,
                f"📅 <b>Auf morgen ({tomorrow}) verschoben:</b> {telegram_ui.esc(t.get('title', 'Aufgabe'))}{hint}",
            )
            return True
        telegram_ui.answer_callback_query(bot_token, cq_id, "Aufgabe nicht gefunden.")
        return False

    if data.startswith("block:"):
        tok = data.split(":", 1)[1]
        entry = store.get_callback_payload(tok)
        if not entry:
            telegram_ui.answer_callback_query(bot_token, cq_id, "Slot-Vorschlag abgelaufen.")
            return False
        _, payload = entry
        ev_dict = {
            "title": f"Fokus: {payload.get('task_title', 'Aufgabe')}",
            "date": payload.get("date"),
            "start_time": payload.get("start_time"),
            "end_time": payload.get("end_time"),
            "location": "",
            "description": "Smart Time-Block via Telegram",
        }
        created = handle_extracted_calendar_events(
            events=[ev_dict],
            source_title="Time-Block",
            state=state,
            notion=notion,
            tasks_db_id=cfg["notion"]["tasks_database_id"],
        )
        if created and created[0].get("gcal_event_id"):
            store.push_undo(
                "main",
                chat_id,
                "delete_event",
                {"event_id": created[0]["gcal_event_id"], "title": created[0]["title"]},
            )
        save_local_state(state_path, state)
        telegram_ui.answer_callback_query(bot_token, cq_id, "⏱️ Zeitblock eingetragen!")
        markup = telegram_ui.build_inline_keyboard([[("↩️ Rückgängig", "undo:last")]])
        _send_telegram_message(
            api_base,
            chat_id,
            f"⏱️ <b>Fokus-Slot im Kalender geblockt:</b>\n"
            f"• {telegram_ui.esc(ev_dict['title'])}\n"
            f"• {ev_dict['date']} {ev_dict['start_time']}–{ev_dict['end_time']} Uhr",
            reply_markup=markup,
        )
        return True

    if data == "undo:last":
        res_msg = _execute_undo(store, chat_id, state, notion)
        save_local_state(state_path, state)
        telegram_ui.answer_callback_query(bot_token, cq_id, "Rückgängig ausgeführt")
        _send_telegram_message(api_base, chat_id, res_msg)
        return True

    telegram_ui.answer_callback_query(bot_token, cq_id, "Unbekannte Aktion.")
    return False


def _persist_analysis_and_reply(
    api_base: str,
    chat_id: str,
    source: str,
    update_id: int,
    raw_text: str,
    analysis: dict[str, Any],
    cfg: dict[str, Any],
    state: dict[str, Any],
    notion: NotionClient | None,
) -> None:
    """Speichert eine ausgewertete Notiz (Text, Voice oder Foto/PDF) in Notion & lokalem Zustand und sendet HTML-Antwort."""
    suggestions_str = format_suggestions_text(analysis.get("suggestions") or [])
    note_area = analysis.get("area") or agent_tools.detect_area(raw_text)

    inbox_page_id: str | None = None
    if notion:
        inbox_page_id = notion.create_inbox_page(
            inbox_db_id=cfg["notion"]["inbox_database_id"],
            title=analysis["title"],
            source=source,
            raw_text=raw_text,
            summary=analysis["summary"],
            suggestions_text=suggestions_str,
            area=note_area,
        )
        for todo in analysis.get("todos") or []:
            notion.create_task_page(
                tasks_db_id=cfg["notion"]["tasks_database_id"],
                title=todo.get("title", "Aufgabe"),
                kind="To-Do",
                priority=todo.get("priority", "Mittel"),
                due_date=todo.get("due_date"),
                next_step=todo.get("next_step", ""),
                inbox_page_id=inbox_page_id,
                area=todo.get("area") or note_area,
                estimated_minutes=todo.get("estimated_minutes"),
            )

    record = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "file": f"telegram_{update_id}",
        "source": source,
        "raw_text": raw_text,
        "analysis": analysis,
        "notion_inbox_page_id": inbox_page_id,
        "synced_to_notion": bool(inbox_page_id),
    }
    state["items"].append(record)
    for todo in analysis.get("todos") or []:
        state["tasks"].append({
            "kind": "To-Do",
            "title": todo.get("title", ""),
            "priority": todo.get("priority", "Mittel"),
            "due_date": todo.get("due_date"),
            "next_step": todo.get("next_step", ""),
            "source_title": analysis["title"],
            "status": "Nicht begonnen",
            "area": todo.get("area") or note_area,
            "estimated_minutes": int(todo.get("estimated_minutes") or 30),
        })

    created_cal = handle_extracted_calendar_events(
        events=analysis.get("calendar_events", []),
        source_title=analysis["title"],
        state=state,
        notion=notion,
        tasks_db_id=cfg["notion"]["tasks_database_id"],
        inbox_page_id=inbox_page_id,
    )

    todo_lines = (
        "\n".join(
            f"• {telegram_ui.esc(t.get('title'))} ({telegram_ui.esc(t.get('priority', 'Mittel'))})"
            for t in (analysis.get("todos") or [])
        )
        or "Keine direkten To-Dos"
    )
    cal_lines = (
        "\n".join(
            f"• {telegram_ui.esc(c.get('title'))} ({c.get('date')} {c.get('start_time') or 'Ganztägig'})"
            for c in created_cal
        )
        or "Keine Termine"
    )
    sug_lines = (
        "\n".join(f"• {telegram_ui.esc(s.get('title'))}" for s in (analysis.get("suggestions") or []))
        or "Keine"
    )
    reply = (
        f"✅ <b>Ausgewertet:</b> {telegram_ui.esc(analysis['title'])} "
        f"<i>[{telegram_ui.esc(note_area)}]</i>\n\n"
        f"📝 {telegram_ui.esc(analysis['summary'])}\n\n"
        f"📌 <b>To-Dos:</b>\n{todo_lines}\n\n"
        f"📅 <b>Kalendereinträge:</b>\n{cal_lines}\n\n"
        f"💡 <b>Vorschläge:</b>\n{sug_lines}"
    )
    _send_telegram_message(api_base, chat_id, reply)


def process_telegram_updates(
    cfg: dict[str, Any],
    state: dict[str, Any],
    notion: NotionClient | None,
    model_override: str | None = None,
    poll_timeout: int = 5,
) -> int:
    """
    Bidirektionale Telegram-Steuerung (2 Richtungen):
    - Inline-Buttons (`callback_query`: erledigen, auf morgen, Slot blocken, Undo)
    - Sprachnotizen, Fotos (Terminzettel/Rechnungen) & Dokumente (PDF/TXT/MD)
    - Gemini Multi-Tool Function Calling mit StepBudget & Multi-Turn-Gedächtnis
    - Schnellbefehle: /morgen, /abend, /todos, /kalender, /slots, /review, /undo, /suche, /status
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token:
        return 0

    offset = int(state.get("telegram_offset", 0))
    api_base = f"https://api.telegram.org/bot{bot_token}"

    try:
        req = request.Request(f"{api_base}/getUpdates?offset={offset}&timeout={poll_timeout}")
        with request.urlopen(req, timeout=poll_timeout + 10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"[Telegram] Fehler beim Abrufen von Updates: {exc}")
        return 0

    if not data.get("ok"):
        return 0

    processed_dir = BASE_DIR / cfg["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    state_path = BASE_DIR / cfg["paths"]["local_state_file"]
    store = get_store()
    count = 0

    for update in data.get("result", []):
        update_id = update["update_id"]
        state["telegram_offset"] = max(int(state.get("telegram_offset", 0)), update_id + 1)
        # WICHTIG: Offset sofort persistent speichern, damit niemals eine Nachricht mehrfach beantwortet wird!
        save_local_state(state_path, state)

        # 0. Inline-Button Callback-Queries behandeln
        cq = update.get("callback_query")
        if cq:
            _handle_telegram_callback_query(cq, bot_token, api_base, cfg, state, state_path, notion, store)
            count += 1
            continue

        msg = update.get("message") or update.get("edited_message")
        if not msg:
            continue

        chat_id = str(msg.get("chat", {}).get("id", ""))
        if not telegram_ui.is_authorized_chat(chat_id, env_keys=("TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USERS")):
            print(f"[Telegram] Ignoriere Nachricht von unautorisierter Chat-ID {chat_id}.")
            continue

        # Chat-ID erst NACH Autorisierung für Morgen-/Abend-Briefings merken & sichern
        if chat_id:
            state["last_telegram_chat_id"] = str(chat_id)
            try:
                save_local_state(state_path, state)
            except Exception:
                pass

        raw_text = ""
        source = "Telegram"

        # 1A. Sprachnachricht (Voice) oder Audiodatei
        voice_or_audio = msg.get("voice") or msg.get("audio")
        if voice_or_audio:
            file_id = voice_or_audio["file_id"]
            local_audio = telegram_ui.download_telegram_file(bot_token, file_id, processed_dir, prefix=f"tg_{update_id}")
            if not local_audio:
                _send_telegram_message(api_base, chat_id, "⚠️ Sprachnachricht konnte nicht heruntergeladen werden.")
                continue
            raw_text = transcribe_audio(local_audio, cfg)
            source = "Telegram"
        # 1B. Foto (Terminzettel, Rechnung, Screenshot)
        elif msg.get("photo"):
            photo_id = msg["photo"][-1]["file_id"]
            caption = (msg.get("caption") or "").strip()
            local_img = telegram_ui.download_telegram_file(bot_token, photo_id, processed_dir, prefix=f"tg_img_{update_id}")
            if not local_img:
                _send_telegram_message(api_base, chat_id, "⚠️ Bild konnte nicht heruntergeladen werden.")
                continue
            ext = local_img.suffix.lower()
            mime = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")
            mm_analysis = agent_tools.analyze_multimodal_file(local_img, mime, caption, cfg, get_date_reference_block())
            _persist_analysis_and_reply(
                api_base, chat_id, "Telegram (Foto)", update_id,
                mm_analysis.get("raw_text") or caption or local_img.name,
                mm_analysis, cfg, state, notion,
            )
            save_local_state(state_path, state)
            count += 1
            continue
        # 1C. Dokument (PDF, Bild oder TXT/MD)
        elif msg.get("document"):
            doc = msg["document"]
            file_id = doc["file_id"]
            caption = (msg.get("caption") or "").strip()
            fname = str(doc.get("file_name") or "")
            doc_mime = str(doc.get("mime_type") or "")
            local_doc = telegram_ui.download_telegram_file(bot_token, file_id, processed_dir, prefix=f"tg_doc_{update_id}")
            if not local_doc:
                _send_telegram_message(api_base, chat_id, "⚠️ Dokument konnte nicht heruntergeladen werden.")
                continue
            ext = Path(fname).suffix.lower() or local_doc.suffix.lower()
            if ext in (".txt", ".md"):
                file_body = local_doc.read_text(encoding="utf-8", errors="replace").strip()
                raw_text = f"{caption}\n\n{file_body}".strip() if caption else file_body
                source = "Telegram (Textdatei)"
            elif ext == ".pdf" or "pdf" in doc_mime.lower() or ext in (".png", ".jpg", ".jpeg", ".webp"):
                mime = "application/pdf" if (ext == ".pdf" or "pdf" in doc_mime.lower()) else (
                    "image/png" if ext == ".png" else "image/jpeg"
                )
                mm_analysis = agent_tools.analyze_multimodal_file(local_doc, mime, caption, cfg, get_date_reference_block())
                _persist_analysis_and_reply(
                    api_base, chat_id, "Telegram (Dokument)", update_id,
                    mm_analysis.get("raw_text") or caption or fname or local_doc.name,
                    mm_analysis, cfg, state, notion,
                )
                save_local_state(state_path, state)
                count += 1
                continue
            else:
                raw_text = caption
        else:
            raw_text = (msg.get("text") or msg.get("caption") or "").strip()

        if not raw_text:
            continue

        cmd_lower = raw_text.lower().strip()

        # 2. Direkte Bot-Kommandos (Schnellzugriff ohne LLM-Umweg)
        if cmd_lower in ("status", "dienst", "info", "zeitplan", "schedule", "läufe", "laeufe") or cmd_lower.startswith(("/status", "/dienst", "/info", "/zeitplan", "/schedule", "status ", "zeitplan ")):
            try:
                sched_state = _load_scheduler_state()
                now_berlin = _scheduler_now("Europe/Berlin")
                time_str = now_berlin.strftime("%H:%M Uhr (%d.%m.%Y)")
                today_str = now_berlin.date().isoformat()
                now_t = now_berlin.time().replace(tzinfo=None)
                sched_cfg = {**DEFAULT_SCHEDULE, **(cfg.get("schedule") or {})}
                job_lines = []
                job_names = {
                    "rss_morning": "📰 RSS-Monitor Morgen",
                    "briefing_morning": "☀️ Morgen-Briefing",
                    "rss_evening": "📰 RSS-Monitor Abend",
                    "briefing_evening": "🌙 Abend-Briefing",
                    "weekly_review": "🗓️ Wochen-Review (So)",
                    "weekly_synthesis": "🧠 RSS-Synthese (So)",
                }
                for j in sched_cfg.get("jobs", []):
                    jid = j.get("id", "")
                    jtime = j.get("time", "")
                    j_name = job_names.get(jid, f"📌 {jid}")
                    last_run = sched_state.get(jid)
                    already_ran_today = (last_run == today_str)

                    try:
                        start_t = _parse_hhmm(jtime)
                        until_t = _parse_hhmm(j.get("catchup_until", "23:59"))
                        if already_ran_today:
                            next_run_str = f"Morgen, {jtime} Uhr"
                        elif now_t < start_t:
                            next_run_str = f"Heute, {jtime} Uhr"
                        elif start_t <= now_t <= until_t:
                            next_run_str = f"Heute fällig (Zeitfenster aktiv bis {j.get('catchup_until', '23:59')} Uhr)"
                        else:
                            next_run_str = f"Morgen, {jtime} Uhr (heute verpasst nach {j.get('catchup_until', '23:59')} Uhr)"
                    except Exception:
                        next_run_str = f"Geplant um {jtime} Uhr"

                    if already_ran_today:
                        last_label = f"Heute gelaufen ({jtime} Uhr)"
                    elif last_run:
                        last_label = f"{last_run}"
                    else:
                        last_label = "noch nicht gelaufen"

                    job_lines.append(
                        f"• <b>{telegram_ui.esc(j_name)}</b> ({jtime} Uhr)\n"
                        f"   ↳ Zuletzt: <code>{telegram_ui.esc(last_label)}</code> | Nächster Lauf: <b>{telegram_ui.esc(next_run_str)}</b>"
                    )

                cal_events = fetch_google_calendar_events(days_ahead=7)
                work_busy = calendar_sync.fetch_work_busy_slots(days_ahead=7, store=store)
                cal_cnt = len(cal_events)
                todo_cnt = len([t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"])

                if cal_cnt > 0:
                    cal_details = []
                    for ev in cal_events[:4]:
                        t_str = f" um {ev['start_time']} Uhr" if ev.get("start_time") else " (Ganztägig)"
                        loc_str = f" [{telegram_ui.esc(ev['location'])}]" if ev.get("location") else ""
                        cal_details.append(f"   ↳ <b>{ev.get('date')}</b>{t_str}: {telegram_ui.esc(ev.get('title'))}{loc_str}")
                    cal_section = f"📅 <b>Google Kalender:</b> {cal_cnt} Termine in den nächsten 7 Tagen:\n" + "\n".join(cal_details)
                else:
                    has_ics = bool(os.environ.get("GOOGLE_CALENDAR_ICS_URL", "").strip())
                    has_gcal = bool(get_gcalendar_bin())
                    has_oauth = bool(os.environ.get("GOOGLE_CALENDAR_REFRESH_TOKEN", "").strip())
                    if has_ics or has_gcal or has_oauth:
                        cal_section = "📅 <b>Google Kalender:</b> 0 Termine in den nächsten 7 Tagen gefunden (Quelle aktiv)."
                    else:
                        cal_section = (
                            "📅 <b>Google Kalender:</b> Keine Termine gefunden (Quelle fehlt)\n"
                            "   ⚠️ <i>Tipp: Trage GOOGLE_CALENDAR_ICS_URL oder Google OAuth2 in .env ein.</i>"
                        )

                work_line = (
                    f"\n🔒 <b>Arbeitskalender (nur Frei/Belegt):</b> {len(work_busy)} anonyme Belegt-Blöcke (7 Tage)"
                    if work_busy
                    else ""
                )

                import platform as _plat
                _host = _plat.node() or "unknown"
                _commit = ""
                try:
                    _cproc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True, timeout=3)
                    if _cproc.returncode == 0:
                        _commit = _cproc.stdout.strip()
                except Exception:
                    pass
                _env_label = f"<code>{sys.platform}</code> @ <code>{telegram_ui.esc(_host)}</code> (PID <code>{os.getpid()}</code>" + (f", Commit <code>{telegram_ui.esc(_commit)}</code>" if _commit else "") + ")"

                status_text = (
                    f"🤖 <b>Assistent & Scheduler Status</b>\n\n"
                    f"🖥️ <b>Host:</b> {_env_label}\n"
                    f"🕒 <b>Serverzeit (Berlin):</b> {time_str}\n"
                    f"💬 <b>Deine Chat-ID:</b> <code>{telegram_ui.esc(chat_id)}</code> (autorisiert)\n"
                    f"✅ <b>Notion Tasks:</b> {todo_cnt} offene Aufgaben\n"
                    f"{cal_section}{work_line}\n\n"
                    f"⏰ <b>Automatischer Zeitplan (nächste Läufe):</b>\n" + "\n".join(job_lines) + "\n\n"
                    f"Tipp: <code>/morgen</code>, <code>/slots</code>, <code>/todos</code> oder <code>/review</code> nutzen."
                )
                _send_telegram_message(api_base, chat_id, status_text)
            except Exception as s_err:
                _send_telegram_message(api_base, chat_id, f"🤖 <b>Assistent Status</b>\n\n⚠️ Fehler: {telegram_ui.esc(s_err)}")
            count += 1
            continue

        if cmd_lower in ("/restart", "/neustart", "restart", "neustart", "/reload"):
            _send_telegram_message(api_base, chat_id, "⚡ Assistent wird jetzt neu gestartet...")
            count += 1
            trigger_restart()
            continue

        if cmd_lower in ("start", "help", "hilfe") or cmd_lower.startswith(("/start", "/help", "/hilfe")):
            _send_telegram_message(
                api_base,
                chat_id,
                "👋 <b>Hallo Stefan! Dein persönlicher KI-Assistent ist bereit.</b>\n\n"
                "📅 <b>Kalender & Freie Slots:</b>\n"
                "• <code>/kalender</code> – Terminvorschau (7 Tage)\n"
                "• <code>/slots [heute|morgen|YYYY-MM-DD]</code> – Freie Zeitfenster (inkl. anonymer Arbeits-Blocker) &amp; Smart Time-Blocking\n"
                "• Natürliche Sprache: <i>„Trage morgen 14 Uhr Zahnarzt ein“</i> oder <i>„Verschiebe ...“</i>\n\n"
                "📌 <b>Aufgaben &amp; Review (Notion Tasks):</b>\n"
                "• <code>/todos</code> – Offene Aufgaben mit Schnell-Buttons (✅ Erledigt / 📅 Morgen)\n"
                "• <code>/erledigt 1, 3</code> – Aufgaben abhaken\n"
                "• <code>/undo</code> – Letzte Aktion rückgängig machen\n"
                "• <code>/review</code> – Wochen-Review &amp; Engpass-Analyse\n\n"
                "🎙️ <b>Multimodalität &amp; Wissen:</b>\n"
                "• Sprachnotiz, Foto (Terminzettel/Rechnung) oder PDF einfach in den Chat senden\n"
                "• <code>/suche &lt;Begriff&gt;</code> – Notion, Notizen &amp; RSS-Wissensspeicher (BM25) durchsuchen\n"
                "• <code>/morgen</code> / <code>/abend</code> / <code>/ideen</code> / <code>/mails</code> / <code>/themen</code> / <code>/status</code> / <code>/update</code>",
            )
            count += 1
            continue

        if cmd_lower.startswith("/slots"):
            parts = raw_text.split(maxsplit=1)
            arg = parts[1].strip().lower() if len(parts) > 1 else "heute"
            target_d = dt.date.today()
            if arg in ("morgen", "tomorrow"):
                target_d = dt.date.today() + dt.timedelta(days=1)
            elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", arg):
                try:
                    target_d = dt.date.fromisoformat(arg)
                except Exception:
                    pass
            slots_html, slots_kb = _build_slots_reply(target_d, cfg, state, store)
            _send_telegram_message(api_base, chat_id, slots_html, reply_markup=slots_kb)
            count += 1
            continue

        if cmd_lower in ("/undo", "undo", "/rueckgaengig", "/rückgängig"):
            res_msg = _execute_undo(store, chat_id, state, notion)
            save_local_state(state_path, state)
            _send_telegram_message(api_base, chat_id, res_msg)
            count += 1
            continue

        if cmd_lower.startswith("/review"):
            _send_telegram_message(api_base, chat_id, "🗓️ Erstelle Wochen-Review (Notion + Kalender + RSS)...")
            review_html, _ = agent_tools.generate_weekly_review(cfg, state, notion, store)
            _send_telegram_message(api_base, chat_id, review_html)
            count += 1
            continue

        if cmd_lower in ("briefing", "morgen") or cmd_lower.startswith(("/briefing", "/morgen", "briefing ", "morgen ")):
            briefing_msg = generate_briefing(cfg, state, notify=False, mode="morning")
            _send_telegram_message(api_base, chat_id, briefing_msg)
            count += 1
            continue

        if cmd_lower in ("abend", "nacht", "checkliste") or cmd_lower.startswith(("/abend", "abend ")):
            briefing_msg = generate_briefing(cfg, state, notify=False, mode="evening")
            _send_telegram_message(api_base, chat_id, briefing_msg)
            count += 1
            continue

        if cmd_lower.startswith("/kalender"):
            upcoming = fetch_google_calendar_events(days_ahead=7)
            seen_keys = {(ev.get("title"), ev.get("date"), ev.get("start_time")) for ev in upcoming}
            today_iso = dt.date.today().isoformat()
            end_iso = (dt.date.today() + dt.timedelta(days=7)).isoformat()
            for ev in state.get("calendar_events", []):
                d_str = ev.get("date") or ""
                key = (ev.get("title"), d_str, ev.get("start_time"))
                if today_iso <= d_str <= end_iso and key not in seen_keys:
                    upcoming.append(ev)
                    seen_keys.add(key)
            upcoming.sort(key=lambda x: (x.get("date") or "", x.get("start_time") or "00:00"))
            work_busy = calendar_sync.fetch_work_busy_slots(days_ahead=7, store=store)

            if not upcoming and not work_busy:
                _send_telegram_message(api_base, chat_id, "📅 In den nächsten 7 Tagen stehen keine Termine im Kalender an.")
            else:
                weekday_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
                lines = ["📅 <b>Deine Termine der nächsten 7 Tage (Google Kalender):</b>"]
                for e in upcoming:
                    d_str = e.get("date") or ""
                    try:
                        wd = weekday_names[dt.date.fromisoformat(d_str).weekday()]
                        d_label = f"{wd}, {d_str}"
                    except Exception:
                        d_label = d_str
                    t_str = f" {e['start_time']}–{e['end_time']} Uhr" if e.get("start_time") and e.get("end_time") else (f" um {e['start_time']} Uhr" if e.get("start_time") else " (Ganztägig)")
                    loc_str = f" 📍 {telegram_ui.esc(e['location'])}" if e.get("location") else ""
                    lines.append(f"• <b>{telegram_ui.esc(d_label)}</b>{t_str}: {telegram_ui.esc(e.get('title'))}{loc_str}")
                if work_busy:
                    lines.append(f"\n🔒 <i>Zusätzlich {len(work_busy)} anonyme Arbeitskalender-Blocker in <code>/slots</code> berücksichtigt.</i>")
                _send_telegram_message(api_base, chat_id, "\n".join(lines))
            count += 1
            continue

        if cmd_lower.startswith("/todos"):
            open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
            if not open_todos:
                _send_telegram_message(api_base, chat_id, "✅ Du hast aktuell keine offenen To-Dos!")
            else:
                lines = ["📌 <b>Deine offenen To-Dos (Notion):</b>"]
                for i, t in enumerate(open_todos, 1):
                    due = f" [Fällig: {t['due_date']}]" if t.get("due_date") else ""
                    area_lbl = f" [{telegram_ui.esc(t['area'])}]" if t.get("area") else ""
                    dur_lbl = f" ~{t['estimated_minutes']}m" if t.get("estimated_minutes") else ""
                    lines.append(
                        f"{i}. [{telegram_ui.esc(t.get('priority', 'Mittel'))}]{area_lbl}{due}{dur_lbl} "
                        f"<b>{telegram_ui.esc(t.get('title'))}</b>"
                    )
                lines.append("\nZum Abhaken: <code>/erledigt 1, 3</code>, <code>/erledigt 1-3</code> oder Schnell-Buttons:")
                kb_rows: list[list[tuple[str, str]]] = []
                for idx in range(1, min(3, len(open_todos)) + 1):
                    kb_rows.append([
                        (f"✅ #{idx} erledigen", f"done:{idx}"),
                        (f"📅 #{idx} auf morgen", f"defer:{idx}"),
                    ])
                markup = telegram_ui.build_inline_keyboard(kb_rows) if kb_rows else None
                _send_telegram_message(api_base, chat_id, "\n".join(lines), reply_markup=markup)
            count += 1
            continue

        if cmd_lower.startswith("/ideen"):
            recent_sugs: list[dict[str, Any]] = []
            for it in reversed(state.get("items", [])):
                for s in it.get("analysis", {}).get("suggestions", []):
                    if s.get("title") and not any(x.get("title") == s.get("title") for x in recent_sugs):
                        recent_sugs.append(s)
                    if len(recent_sugs) >= 5:
                        break
                if len(recent_sugs) >= 5:
                    break
            if not recent_sugs:
                _send_telegram_message(api_base, chat_id, "💡 Aktuell keine offenen KI-Vorschläge.")
            else:
                lines = ["💡 <b>Proaktive Vorschläge aus deinen letzten Notizen:</b>"]
                for i, s in enumerate(recent_sugs, 1):
                    lines.append(f"{i}. <b>{telegram_ui.esc(s.get('title'))}</b>\n   ↳ {telegram_ui.esc(s.get('next_step', ''))}")
                _send_telegram_message(api_base, chat_id, "\n".join(lines))
            count += 1
            continue

        if cmd_lower.startswith(("/mails", "/email", "/emails")):
            _send_telegram_message(api_base, chat_id, "📬 Prüfe markierte E-Mails (Label: Assistent)...")
            mail_count = process_emails(cfg, state, notion, model_override=model_override)
            if mail_count > 0:
                _send_telegram_message(api_base, chat_id, f"✅ {mail_count} neue E-Mail(s) ausgewertet und in Notion AI Inbox &amp; Tasks übertragen.")
            else:
                _send_telegram_message(api_base, chat_id, "ℹ️ Keine neuen ungelesenen E-Mails mit dem Label 'Assistent' gefunden.")
            count += 1
            continue

        if cmd_lower.startswith(("/erledigt", "/done")):
            parts = raw_text.split(maxsplit=1)
            arg = parts[1].strip() if len(parts) > 1 else ""
            open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
            if not open_todos:
                _send_telegram_message(api_base, chat_id, "✅ Du hast aktuell keine offenen To-Dos!")
                count += 1
                continue

            if not arg:
                _send_telegram_message(
                    api_base,
                    chat_id,
                    "ℹ️ Bitte Nummer(n) oder Name(n) angeben, z. B.:\n"
                    "• <code>/erledigt 1, 3</code> (mehrere auf einmal)\n"
                    "• <code>/erledigt 1-3</code> (Bereich von 1 bis 3)\n"
                    "• <code>/erledigt Video schneiden</code> (nach Name/Stichwort)\n\n"
                    "📋 Liste deiner Aufgaben anzeigen: <code>/todos</code>",
                )
                count += 1
                continue

            target_tasks = resolve_tasks_to_complete(arg, open_todos)
            if target_tasks:
                page_ids: list[str] = []
                titles_done: list[str] = []
                for t in target_tasks:
                    t["status"] = "Erledigt"
                    titles_done.append(t.get("title", ""))
                    n_page_id = t.get("notion_page_id")
                    if n_page_id:
                        page_ids.append(n_page_id)
                    if notion and n_page_id:
                        try:
                            notion.mark_task_done(n_page_id)
                        except Exception as _e:
                            print(f"[Notion Task Error] {_e}")
                store.push_undo("main", chat_id, "reopen_tasks", {"page_ids": page_ids, "titles": titles_done})
                save_local_state(state_path, state)

                if len(target_tasks) == 1:
                    resp = f"✅ Als erledigt markiert:\n• <b>{telegram_ui.esc(target_tasks[0].get('title'))}</b>"
                else:
                    titles = "\n".join(f"• <b>{telegram_ui.esc(t.get('title'))}</b>" for t in target_tasks)
                    resp = f"✅ <b>{len(target_tasks)} Aufgaben als erledigt markiert:</b>\n{titles}"

                remaining = [t for t in open_todos if t not in target_tasks]
                if remaining:
                    resp += f"\n\n📋 Noch <i>{len(remaining)} offene Aufgabe(n)</i> (Liste mit <code>/todos</code>)."
                else:
                    resp += "\n\n🎉 <b>Alle offenen To-Dos erledigt!</b>"

                markup = telegram_ui.build_inline_keyboard([[("↩️ Rückgängig", "undo:last")]])
                _send_telegram_message(api_base, chat_id, resp, reply_markup=markup)
                count += 1
                continue
            else:
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"⚠️ Konnte keine passende offene Aufgabe zu „{telegram_ui.esc(arg)}“ finden.\n"
                    f"Nutze <code>/todos</code> für die aktuelle Nummernliste.",
                )
                count += 1
                continue

        if cmd_lower.startswith(("/suche", "/search")):
            parts = raw_text.split(maxsplit=1)
            search_query = parts[1].strip() if len(parts) > 1 else ""
            if not search_query:
                _send_telegram_message(api_base, chat_id, "Bitte Suchbegriff angeben, z. B. <code>/suche Life on Pause</code>.")
                count += 1
                continue
            notion_results = notion.search_workspace(search_query, limit=5) if notion else []
            rss_hits = store.search_knowledge_bm25(search_query, limit=5)
            matching_notes = [
                it for it in state.get("items", [])
                if search_query.lower() in (it.get("raw_text") or "").lower() or search_query.lower() in (it.get("analysis", {}).get("title") or "").lower()
            ]
            lines = [f"🔍 <b>Suchergebnisse für „{telegram_ui.esc(search_query)}“:</b>\n"]
            if notion_results:
                lines.append("📄 <b>Notion-Seiten:</b>")
                for r in notion_results[:4]:
                    url = r.get("url", "")
                    t_lbl = telegram_ui.esc(r.get("title", "Seite"))
                    lines.append(f'• <a href="{telegram_ui.esc(url)}">{t_lbl}</a>' if url else f"• {t_lbl}")
                lines.append("")
            if rss_hits:
                lines.append("📡 <b>RSS-Wissensspeicher (BM25):</b>")
                for rh in rss_hits[:4]:
                    r_url = rh.get("link", "")
                    r_lbl = telegram_ui.esc(rh.get("title", "Artikel"))
                    lines.append(
                        f'• [{telegram_ui.esc(rh.get("topic_id", ""))}] <a href="{telegram_ui.esc(r_url)}">{r_lbl}</a>'
                        if r_url
                        else f"• [{telegram_ui.esc(rh.get('topic_id', ''))}] {r_lbl}"
                    )
                lines.append("")
            if matching_notes:
                lines.append("🎙️ <b>Notizen-Verlauf:</b>")
                for n in matching_notes[:3]:
                    lines.append(f"• <b>{telegram_ui.esc(n.get('analysis', {}).get('title'))}</b>: {telegram_ui.esc(n.get('analysis', {}).get('summary'))}")
            if not notion_results and not rss_hits and not matching_notes:
                lines.append("Keine Treffer in Notion, RSS-Wissensspeicher oder Notizen gefunden.")
            _send_telegram_message(api_base, chat_id, "\n".join(lines))
            count += 1
            continue

        if cmd_lower in ("/update", "/upgrade", "update", "/pull"):
            _send_telegram_message(api_base, chat_id, "⏳ Prüfe GitHub auf Updates...")
            needs_restart, update_msg = perform_self_update()
            _send_telegram_message(api_base, chat_id, update_msg)
            count += 1
            if needs_restart:
                trigger_restart()
            continue

        if cmd_lower in ("/run", "/sync", "run", "sync") or cmd_lower.startswith(("/run ", "/sync ")):
            _send_telegram_message(api_base, chat_id, "📡 RSS-Monitor gestartet. Neue Funde werden gleich gemeldet...")
            def _async_rss_run():
                try:
                    import rss_monitor
                    import argparse as _ap
                    ns = _ap.Namespace(topic=None, dry_run=False, no_notify=False, no_notion=False, verbose=False)
                    rss_monitor.run(ns)
                except Exception as _e:
                    print(f"[RSS Telegram Run Error] {_e}")
            threading.Thread(target=_async_rss_run, daemon=True).start()
            count += 1
            continue

        if cmd_lower.startswith(("/themen", "/rss", "/topic")):
            parts = raw_text.split(maxsplit=2)
            subcmd = parts[1].lower() if len(parts) > 1 else "list"
            arg = parts[2].strip() if len(parts) > 2 else ""

            try:
                import rss_monitor

                if subcmd in ("list", "show", "liste"):
                    feeds_cfg = rss_monitor.load_feeds_config()
                    topics = feeds_cfg.get("topics", [])
                    lines = [f"📡 <b>Deine RSS-Themengebiete ({len(topics)}):</b>\n"]
                    for t in topics:
                        st = "🟢" if t.get("enabled", True) else "⚪ (inaktiv)"
                        lines.append(f"{st} {t.get('emoji', '📌')} <b>{telegram_ui.esc(t['name'])}</b> (<code>{telegram_ui.esc(t['id'])}</code>)")
                        lines.append(f"   <i>{len(t.get('sources', []))} Quellen</i>")
                    lines.append("\n⚡ <b>Befehle:</b>")
                    lines.append("• <code>/themen an &lt;id&gt;</code> – Thema aktivieren")
                    lines.append("• <code>/themen aus &lt;id&gt;</code> – Thema pausieren")
                    lines.append("• <code>/themen neu &lt;Name&gt;</code> – Neues Thema per KI generieren")
                    lines.append("• <code>/themen run</code> – Jetzt sofort Feeds abfragen")
                    _send_telegram_message(api_base, chat_id, "\n".join(lines))
                    count += 1
                    continue

                elif subcmd in ("an", "on", "enable", "aktivieren"):
                    if not arg:
                        _send_telegram_message(api_base, chat_id, "Bitte Themen-ID angeben, z. B. <code>/themen an parkinson</code>.")
                        count += 1
                        continue
                    ok = rss_monitor.cmd_topic_set_enabled(arg, True)
                    if ok:
                        _send_telegram_message(api_base, chat_id, f"🟢 Thema <code>{telegram_ui.esc(arg)}</code> wurde aktiviert.")
                    else:
                        _send_telegram_message(api_base, chat_id, f"❌ Thema <code>{telegram_ui.esc(arg)}</code> nicht gefunden.")
                    count += 1
                    continue

                elif subcmd in ("aus", "off", "disable", "deaktivieren"):
                    if not arg:
                        _send_telegram_message(api_base, chat_id, "Bitte Themen-ID angeben, z. B. <code>/themen aus parkinson</code>.")
                        count += 1
                        continue
                    ok = rss_monitor.cmd_topic_set_enabled(arg, False)
                    if ok:
                        _send_telegram_message(api_base, chat_id, f"⚪ Thema <code>{telegram_ui.esc(arg)}</code> wurde deaktiviert.")
                    else:
                        _send_telegram_message(api_base, chat_id, f"❌ Thema <code>{telegram_ui.esc(arg)}</code> nicht gefunden.")
                    count += 1
                    continue

                elif subcmd in ("neu", "add", "anlegen"):
                    new_name = arg or (parts[1] if len(parts) > 1 and parts[1] not in ("neu", "add") else "")
                    if not new_name:
                        _send_telegram_message(api_base, chat_id, "Bitte Namen des neuen Themas angeben, z. B. <code>/themen neu Multiple Sklerose</code>.")
                        count += 1
                        continue
                    _send_telegram_message(api_base, chat_id, f"⏳ Generiere Feeds &amp; Kriterien für „{telegram_ui.esc(new_name)}“ per KI...")
                    import argparse as _ap
                    ns = _ap.Namespace(name=new_name, id=None, emoji=None, focus=None, rss=None, auto=True)
                    ok = rss_monitor.cmd_topic_add(ns)
                    if ok:
                        _send_telegram_message(api_base, chat_id, f"✅ Thema <b>{telegram_ui.esc(new_name)}</b> wurde mit KI-generierten Quellen erfolgreich angelegt und aktiviert!")
                    else:
                        _send_telegram_message(api_base, chat_id, f"❌ Konnte Thema <b>{telegram_ui.esc(new_name)}</b> nicht anlegen.")
                    count += 1
                    continue

                elif subcmd in ("run", "sync", "start"):
                    _send_telegram_message(api_base, chat_id, "📡 RSS-Monitor gestartet. Neue Funde werden gleich gemeldet...")
                    def _async_rss():
                        try:
                            import argparse as _ap
                            ns = _ap.Namespace(topic=None, dry_run=False, no_notify=False, no_notion=False, verbose=False)
                            rss_monitor.run(ns)
                        except Exception as _e:
                            print(f"[RSS Telegram Run Error] {_e}")
                    threading.Thread(target=_async_rss, daemon=True).start()
                    count += 1
                    continue

                else:
                    _send_telegram_message(api_base, chat_id, "Unbekannter Befehl. Nutze <code>/themen</code> für die Übersicht.")
                    count += 1
                    continue
            except Exception as exc:
                _send_telegram_message(api_base, chat_id, f"Fehler bei /themen: {telegram_ui.esc(exc)}")
                count += 1
                continue

        # 3. Harte Loop-Bremse & Token-Budget für den Chat-Turn
        chat_cfg = cfg.get("chat", {})
        budget = chat_guard.StepBudget(
            max_iterations=int(chat_cfg.get("max_iterations", 3)),
            max_estimated_tokens=int(chat_cfg.get("max_input_tokens", 12000)),
        )

        try:
            budget.step(extra_text_len=len(raw_text))
        except chat_guard.ChatBudgetExceeded as bexc:
            print(f"[Telegram Chat-Guard] {bexc}")
            _send_telegram_message(api_base, chat_id, f"⚠️ {telegram_ui.esc(bexc)}")
            count += 1
            continue

        # 3B. Multi-Tool Function Calling Agent (Gemini mit Multi-Turn-Memory & StepBudget)
        if chat_cfg.get("enable_tool_agent", True):
            tool_res = agent_tools.run_tool_agent(
                user_text=raw_text,
                chat_id=chat_id,
                cfg=cfg,
                state=state,
                notion=notion,
                store=store,
                date_ref_block=get_date_reference_block(),
                find_matching_task_fn=find_matching_task,
                resolve_tasks_fn=resolve_tasks_to_complete,
                find_gcal_event_fn=find_matching_google_calendar_event,
                handle_cal_events_fn=handle_extracted_calendar_events,
                save_state_fn=lambda: save_local_state(state_path, state),
            )
            if tool_res is not None:
                reply_html, has_undo = tool_res
                markup = telegram_ui.build_inline_keyboard([[("↩️ Rückgängig", "undo:last")]]) if has_undo else None
                _send_telegram_message(api_base, chat_id, reply_html, reply_markup=markup)
                save_local_state(state_path, state)
                count += 1
                continue

        # 4. Klassisches Intent-Routing als Fallback (z. B. Ollama-Betrieb)
        print(f"[Telegram Router] Analysiere Nachricht: '{raw_text[:60]}...'")
        intent_info = classify_intent(raw_text, cfg, model_override=model_override)
        intent = intent_info.get("intent", "NOTE_OR_TASK")
        print(f"[Telegram Router] Erkannter Intent: {intent}")

        # 0) FEEDBACK_OR_CORRECTION
        if intent == "FEEDBACK_OR_CORRECTION":
            details = intent_info.get("correction_details") or raw_text
            print(f"[Telegram Feedback] Behandle Korrektur: '{details}'...")
            prompt_fix = f"""Stefan korrigiert einen kürzlichen Fehler oder ein Missverständnis des Assistenten:
"{raw_text}"

Aktueller Kalender & To-Dos:
- Zuletzt erfasste Kalendertermine: {json.dumps(state.get("calendar_events", []), ensure_ascii=False)}
- Offene To-Dos: {json.dumps([t.get('title') for t in state.get('tasks', [])], ensure_ascii=False)}

Antworte Stefan direkt, freundlich und hilfsbereit auf Deutsch. Bestätige ihm die Korrektur oder erkläre kurz, wie es nun angepasst ist.
WICHTIG: Erstelle daraus KEINE neue Aufgabe in Notion Tasks."""
            reply = call_llm_text(prompt_fix, cfg, model_override=model_override)
            _send_telegram_message(api_base, chat_id, f"🫡 {telegram_ui.esc(reply)}")
            count += 1
            continue

        # A) RESCHEDULE_EVENT
        if intent == "RESCHEDULE_EVENT":
            ev_title = intent_info.get("event_title") or ""
            new_date = intent_info.get("new_date")
            new_start = _normalize_hhmm(intent_info.get("new_start_time"))
            new_end = _normalize_hhmm(intent_info.get("new_end_time"))

            matched = find_matching_google_calendar_event(ev_title, new_date)
            if not matched and ev_title:
                matched = find_matching_google_calendar_event(ev_title, None)

            if matched and matched.get("id"):
                target_date = new_date or matched.get("date") or dt.date.today().isoformat()
                ok = update_google_calendar_event(
                    event_id=matched["id"],
                    new_date=target_date,
                    new_start_time=new_start,
                    new_end_time=new_end,
                )
                if ok:
                    for ev in state.get("calendar_events", []):
                        if ev.get("title") == matched.get("title"):
                            if target_date:
                                ev["date"] = target_date
                            if new_start:
                                ev["start_time"] = new_start
                            if new_end:
                                ev["end_time"] = new_end
                    time_str = f" um {new_start} Uhr" if new_start else ""
                    _send_telegram_message(
                        api_base,
                        chat_id,
                        f"📅 <b>Termin erfolgreich verschoben:</b>\n"
                        f"• {telegram_ui.esc(matched['title'])}\n"
                        f"• Neues Datum/Zeit: {target_date}{time_str}\n\n"
                        f"✅ Direkt im Google Kalender aktualisiert.",
                    )
                    count += 1
                    continue
                else:
                    _send_telegram_message(
                        api_base,
                        chat_id,
                        f"⚠️ Termin '{telegram_ui.esc(matched['title'])}' gefunden, aber Google Kalender konnte nicht aktualisiert werden.",
                    )
                    count += 1
                    continue
            else:
                upcoming = fetch_google_calendar_events(days_ahead=7)
                upcoming_lines = "\n".join(f"• {telegram_ui.esc(e['title'])} ({e['date']} {e.get('start_time') or ''})" for e in upcoming[:5])
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"⚠️ Ich konnte keinen passenden Termin zu '{telegram_ui.esc(ev_title)}' im Kalender finden.\n\n"
                    f"Deine nächsten Termine:\n{upcoming_lines or 'Keine anstehenden Termine gefunden.'}",
                )
                count += 1
                continue

        # B) DELETE_EVENT
        if intent == "DELETE_EVENT":
            ev_title = intent_info.get("event_title") or ""
            target_date = intent_info.get("new_date")
            matched = find_matching_google_calendar_event(ev_title, target_date)
            if matched and matched.get("id"):
                ok = delete_google_calendar_event(matched["id"])
                if ok:
                    store.push_undo("main", chat_id, "restore_event", matched)
                    state["calendar_events"] = [
                        e for e in state.get("calendar_events", [])
                        if e.get("title") != matched.get("title")
                    ]
                    markup = telegram_ui.build_inline_keyboard([[("↩️ Rückgängig", "undo:last")]])
                    _send_telegram_message(
                        api_base,
                        chat_id,
                        f"🗑️ Termin <b>{telegram_ui.esc(matched['title'])}</b> ({matched.get('date')}) wurde aus deinem Google Kalender gelöscht.",
                        reply_markup=markup,
                    )
                    count += 1
                    continue
            _send_telegram_message(
                api_base,
                chat_id,
                f"⚠️ Konnte keinen Termin zu '{telegram_ui.esc(ev_title)}' im Google Kalender zum Löschen finden.",
            )
            count += 1
            continue

        # C) CREATE_EVENT
        if intent == "CREATE_EVENT":
            multi_words = ("außerdem", "und auch", "und am", "sowie", "zudem", "und außerdem", "und um", "danach", "sowie am", "und eine")
            has_multi = any(w in raw_text.lower() for w in multi_words) or len(re.findall(r"\d{1,2}(?::\d{2})?\s*uhr", raw_text.lower())) > 1
            if has_multi:
                intent = "NOTE_OR_TASK"
            else:
                ev_title = intent_info.get("event_title") or "Termin"
                ev_date = intent_info.get("new_date") or dt.date.today().isoformat()
                ev_start = intent_info.get("new_start_time")
                ev_end = intent_info.get("new_end_time")
                ev_dict = {
                    "title": ev_title,
                    "date": ev_date,
                    "start_time": ev_start,
                    "end_time": ev_end,
                    "location": "",
                    "description": "Erstellt via Telegram Sprach-/Textnachricht",
                }
                created_cal = handle_extracted_calendar_events(
                    events=[ev_dict],
                    source_title=ev_title,
                    state=state,
                    notion=notion,
                    tasks_db_id=cfg["notion"]["tasks_database_id"],
                )
                has_undo = False
                if created_cal and created_cal[0].get("gcal_event_id"):
                    store.push_undo(
                        "main",
                        chat_id,
                        "delete_event",
                        {"event_id": created_cal[0]["gcal_event_id"], "title": ev_title},
                    )
                    has_undo = True
                time_str = f" ({ev_start}–{ev_end} Uhr)" if ev_start and ev_end else (f" ({ev_start} Uhr)" if ev_start else " (Ganztägig)")
                markup = telegram_ui.build_inline_keyboard([[("↩️ Rückgängig", "undo:last")]]) if has_undo else None
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"📅 <b>Neuer Termin eingetragen:</b>\n• <b>{telegram_ui.esc(ev_title)}</b> am {ev_date}{time_str}\n\nIn Google Kalender &amp; Notion Erinnerungen gesichert.",
                    reply_markup=markup,
                )
                count += 1
                continue

        # D) QUERY_CALENDAR
        if intent == "QUERY_CALENDAR":
            print(f"[Telegram Kalender-Abfrage] Beantworte: '{raw_text}'...")
            answer = answer_question(raw_text, state, cfg, notion=notion, model_override=model_override)
            _send_telegram_message(api_base, chat_id, f"📅 {telegram_ui.esc(answer)}")
            count += 1
            continue

        # E) UPDATE_TASK
        if intent == "UPDATE_TASK":
            task_q = intent_info.get("task_title") or raw_text
            new_date = intent_info.get("new_date")
            new_prio = intent_info.get("new_priority")
            open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
            matched_task = find_matching_task(task_q, open_todos)

            if matched_task:
                target_due = new_date or matched_task.get("due_date")
                target_prio = new_prio or matched_task.get("priority")
                if notion and matched_task.get("notion_page_id"):
                    try:
                        notion.update_task(
                            page_id=matched_task["notion_page_id"],
                            due_date=new_date,
                            priority=new_prio,
                        )
                    except Exception as exc:
                        print(f"[Notion Update Task Fehler] {exc}")
                if new_date:
                    matched_task["due_date"] = new_date
                if new_prio:
                    matched_task["priority"] = new_prio

                due_text = f"Fällig am {target_due}" if target_due else "Kein Fälligkeitsdatum"
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"📌 <b>Aufgabe aktualisiert:</b>\n"
                    f"• <b>{telegram_ui.esc(matched_task.get('title'))}</b>\n"
                    f"• {due_text}\n"
                    f"• Priorität: {telegram_ui.esc(target_prio)}\n\n"
                    f"✅ In Notion &amp; lokal aktualisiert.",
                )
                count += 1
                continue
            else:
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"⚠️ Konnte keine offene Aufgabe zu '{telegram_ui.esc(task_q)}' finden. Liste deine Aufgaben mit <code>/todos</code> auf.",
                )
                count += 1
                continue

        # F) DELETE_TASK
        if intent == "DELETE_TASK":
            task_q = intent_info.get("task_title") or raw_text
            open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
            matched_task = find_matching_task(task_q, open_todos)
            if matched_task:
                n_id = matched_task.get("notion_page_id")
                if notion and n_id:
                    try:
                        notion.archive_task(n_id)
                    except Exception:
                        pass
                store.push_undo("main", chat_id, "reopen_task", {"page_id": n_id, "title": matched_task.get("title", "")})
                state["tasks"] = [t for t in state.get("tasks", []) if t != matched_task]
                markup = telegram_ui.build_inline_keyboard([[("↩️ Rückgängig", "undo:last")]])
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"🗑️ Aufgabe <b>{telegram_ui.esc(matched_task.get('title'))}</b> wurde aus Notion archiviert/gelöscht.",
                    reply_markup=markup,
                )
                count += 1
                continue
            else:
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"⚠️ Konnte keine offene Aufgabe zu '{telegram_ui.esc(task_q)}' zum Löschen finden.",
                )
                count += 1
                continue

        # G) COMPLETE_TASK
        if intent == "COMPLETE_TASK":
            task_title_q = intent_info.get("task_title") or raw_text
            open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]

            target_tasks = resolve_tasks_to_complete(raw_text, open_todos)
            if not target_tasks and task_title_q != raw_text:
                target_tasks = resolve_tasks_to_complete(task_title_q, open_todos)

            if target_tasks:
                page_ids = []
                titles_done = []
                for t in target_tasks:
                    t["status"] = "Erledigt"
                    titles_done.append(t.get("title", ""))
                    n_id = t.get("notion_page_id")
                    if n_id:
                        page_ids.append(n_id)
                    if notion and n_id:
                        try:
                            notion.mark_task_done(n_id)
                        except Exception as _e:
                            print(f"[Notion Task Error] {_e}")
                store.push_undo("main", chat_id, "reopen_tasks", {"page_ids": page_ids, "titles": titles_done})
                save_local_state(state_path, state)

                if len(target_tasks) == 1:
                    resp = f"✅ In Notion &amp; lokal als erledigt markiert:\n• <b>{telegram_ui.esc(target_tasks[0].get('title'))}</b>"
                else:
                    titles = "\n".join(f"• <b>{telegram_ui.esc(t.get('title'))}</b>" for t in target_tasks)
                    resp = f"✅ In Notion &amp; lokal als erledigt markiert ({len(target_tasks)} Aufgaben):\n{titles}"

                remaining = [t for t in open_todos if t not in target_tasks]
                if remaining:
                    resp += f"\n\n📋 Noch <i>{len(remaining)} offene Aufgabe(n)</i> (Übersicht mit <code>/todos</code>)."
                else:
                    resp += "\n\n🎉 <b>Alle offenen To-Dos erledigt!</b>"

                markup = telegram_ui.build_inline_keyboard([[("↩️ Rückgängig", "undo:last")]])
                _send_telegram_message(api_base, chat_id, resp, reply_markup=markup)
                count += 1
                continue
            else:
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"⚠️ Konnte keine passende offene Aufgabe zu '{telegram_ui.esc(task_title_q)}' finden (Übersicht mit <code>/todos</code>).",
                )
                count += 1
                continue

        # H) QUERY_TASKS
        if intent == "QUERY_TASKS":
            print(f"[Telegram Aufgaben-Abfrage] Beantworte: '{raw_text}'...")
            answer = answer_question(raw_text, state, cfg, notion=notion, model_override=model_override)
            _send_telegram_message(api_base, chat_id, f"📌 {telegram_ui.esc(answer)}")
            count += 1
            continue

        # I) SEARCH_KNOWLEDGE
        if intent == "SEARCH_KNOWLEDGE":
            search_q = intent_info.get("query") or raw_text
            notion_results = notion.search_workspace(search_q, limit=5) if notion else []
            rss_hits = store.search_knowledge_bm25(search_q, limit=4)
            matching_notes = [
                it for it in state.get("items", [])
                if search_q.lower() in (it.get("raw_text") or "").lower() or search_q.lower() in (it.get("analysis", {}).get("title") or "").lower()
            ]
            lines = [f"🔍 <b>Suchergebnisse für „{telegram_ui.esc(search_q)}“:</b>\n"]
            if notion_results:
                lines.append("📄 <b>Notion-Seiten:</b>")
                for r in notion_results[:4]:
                    url = r.get("url", "")
                    t_lbl = telegram_ui.esc(r.get("title", "Seite"))
                    lines.append(f'• <a href="{telegram_ui.esc(url)}">{t_lbl}</a>' if url else f"• {t_lbl}")
                lines.append("")
            if rss_hits:
                lines.append("📡 <b>RSS-Wissensspeicher:</b>")
                for rh in rss_hits[:4]:
                    lines.append(f"• [{telegram_ui.esc(rh.get('topic_id', ''))}] {telegram_ui.esc(rh.get('title', ''))}")
                lines.append("")
            if matching_notes:
                lines.append("🎙️ <b>Notizen-Verlauf:</b>")
                for n in matching_notes[:3]:
                    lines.append(f"• <b>{telegram_ui.esc(n.get('analysis', {}).get('title'))}</b>: {telegram_ui.esc(n.get('analysis', {}).get('summary'))}")
            if not notion_results and not rss_hits and not matching_notes:
                lines.append("Keine Treffer in Notion oder Notizen gefunden.")
            _send_telegram_message(api_base, chat_id, "\n".join(lines))
            count += 1
            continue

        # J) GENERAL_CHAT
        if intent == "GENERAL_CHAT" or is_question_message(raw_text):
            q_clean = intent_info.get("query") or raw_text
            if cmd_lower.startswith(("/ask", "/frage")) and " " in raw_text:
                q_clean = raw_text.split(maxsplit=1)[1]
            print(f"[Telegram Dialog] Beantworte: '{q_clean}'...")
            answer = answer_question(q_clean, state, cfg, notion=notion, model_override=model_override)
            store.append_chat_turn("main", chat_id, "user", raw_text)
            store.append_chat_turn("main", chat_id, "assistant", answer)
            _send_telegram_message(api_base, chat_id, f"🤖 {telegram_ui.esc(answer)}")
            count += 1
            continue

        # K) NOTE_OR_TASK (Default)
        print(f"[Telegram Notiz] Werte Nachricht aus: '{raw_text[:60]}...'")
        analysis = analyze_note(
            title="Telegram-Notiz",
            text=raw_text,
            cfg=cfg,
            model_override=model_override,
        )
        _persist_analysis_and_reply(
            api_base, chat_id, source, update_id, raw_text, analysis, cfg, state, notion
        )
        count += 1

    return count


def _send_telegram_message(
    api_base: str,
    chat_id: str,
    text: str,
    parse_mode: str = "HTML",
    reply_markup: dict[str, Any] | None = None,
) -> bool:
    """Sendet Nachricht an Telegram über das gehärtete UI-Transportmodul (inkl. HTML->Plaintext Fallback)."""
    if not api_base or not chat_id or not text:
        return False
    token = api_base.rsplit("/bot", 1)[-1] if "/bot" in api_base else api_base
    return telegram_ui.send_message(
        token=token,
        chat_id=chat_id,
        text=text,
        parse_mode=parse_mode,
        reply_markup=reply_markup,
    )


def process_notion_inbox(
    cfg: dict[str, Any],
    state: dict[str, Any],
    notion: NotionClient,
    model_override: str | None = None,
) -> int:
    inbox_db_id = cfg["notion"]["inbox_database_id"]
    tasks_db_id = cfg["notion"]["tasks_database_id"]
    new_pages = notion.query_new_inbox_items(inbox_db_id)
    count = 0

    for page in new_pages:
        page_id = page["id"]
        props = page.get("properties", {})
        title_arr = props.get("Name", {}).get("title", [])
        title = "".join(t.get("plain_text", "") for t in title_arr).strip() or "Sprachnotiz"
        raw_arr = props.get("Transkript", {}).get("rich_text", [])
        raw_text = "".join(t.get("plain_text", "") for t in raw_arr).strip()
        if not raw_text:
            raw_text = notion.get_page_blocks_text(page_id)
        if not raw_text:
            raw_text = title

        print(f"[Notion Inbox] Werte '{title}' aus...")
        analysis = analyze_note(title, raw_text, cfg, model_override=model_override)
        suggestions_str = format_suggestions_text(analysis["suggestions"])
        note_area = analysis.get("area") or agent_tools.detect_area(raw_text)

        notion.update_inbox_page(
            page_id=page_id,
            raw_text=raw_text,
            summary=analysis["summary"],
            suggestions_text=suggestions_str,
        )
        for todo in analysis["todos"]:
            notion.create_task_page(
                tasks_db_id=tasks_db_id,
                title=todo.get("title", "Aufgabe"),
                kind="To-Do",
                priority=todo.get("priority", "Mittel"),
                due_date=todo.get("due_date"),
                next_step=todo.get("next_step", ""),
                inbox_page_id=page_id,
                area=todo.get("area") or note_area,
                estimated_minutes=todo.get("estimated_minutes"),
            )
        # Hinweis: KI-Vorschläge werden NICHT als Tasks angelegt, um die Aufgabenliste nicht zu überladen.
        handle_extracted_calendar_events(
            events=analysis.get("calendar_events", []),
            source_title=analysis["title"],
            state=state,
            notion=notion,
            tasks_db_id=tasks_db_id,
            inbox_page_id=page_id,
        )
        count += 1
        print(f"[OK] Notion-Eintrag '{title}' ausgewertet.")

    return count


def show_windows_notification(title: str, message: str) -> None:
    """Zeigt optional eine Windows-Desktop-Benachrichtigung an."""
    if sys.platform != "win32":
        return
    try:
        from win11toast import toast  # type: ignore

        toast(title, message)
    except Exception:
        pass


def _resolve_primary_chat_id(state: dict[str, Any]) -> str:
    """Ermittelt die Telegram-Ziel-Chat-ID aus ENV oder lokalem Cache."""
    chat_id = (
        os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        or os.environ.get("TELEGRAM_RSS_CHAT_ID", "").strip()
        or str(state.get("last_telegram_chat_id", "")).strip()
    )
    if not chat_id:
        rss_state_file = BASE_DIR / "rss_state.json"
        if rss_state_file.exists():
            r_st = safe_read_json(rss_state_file, {})
            if isinstance(r_st, dict):
                chat_id = str(r_st.get("last_rss_chat_id", "")).strip()
    if not chat_id:
        allowed = os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip()
        if allowed and (allowed.isdigit() or (allowed.startswith("-") and allowed[1:].isdigit())):
            chat_id = allowed
    return chat_id


def generate_briefing(
    cfg: dict[str, Any],
    state: dict[str, Any],
    notify: bool = False,
    mode: str = "morning",
    send_telegram: bool = False,
    raise_on_telegram_failure: bool = False,
) -> str:
    today_iso = dt.date.today().isoformat()
    open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
    suggestions = [t for t in state.get("tasks", []) if t.get("kind") == "KI-Vorschlag / Idee" and t.get("status") != "Erledigt"]
    today_items = [it for it in state.get("items", []) if str(it.get("timestamp", "")).startswith(today_iso)]

    # Live-Termine aus Google Kalender abrufen
    live_gcal = fetch_google_calendar_events(days_ahead=3 if mode == "evening" else 2)
    seen_keys = {(ev.get("title"), ev.get("date"), ev.get("start_time")) for ev in live_gcal}
    all_cal = list(live_gcal)
    for ev in state.get("calendar_events", []):
        if (ev.get("date") or "") < today_iso:
            continue
        key = (ev.get("title"), ev.get("date"), ev.get("start_time"))
        if key not in seen_keys:
            all_cal.append(ev)
    all_cal.sort(key=lambda x: (x.get("date") or "", x.get("start_time") or "00:00"))

    today_cal = [ev for ev in all_cal if ev.get("date") == today_iso]
    upcoming_cal = [ev for ev in all_cal if (ev.get("date") or "") > today_iso]

    # STRIKT ANONYME Arbeitskalender-Belegtzeiten (nur Free/Busy-Intervalle für Slot-Berechnung)
    store = get_store()
    work_busy: list[dict[str, Any]] = []
    try:
        work_busy = calendar_sync.fetch_work_busy_slots(days_ahead=3 if mode == "evening" else 2, store=store)
    except Exception:
        work_busy = []
    today_work_busy = [wb for wb in work_busy if wb.get("date") == today_iso]

    target_slot_date = (dt.date.today() + dt.timedelta(days=1)).isoformat() if mode == "evening" else today_iso
    work_hours = cfg.get("work_hours", {})
    work_start = os.environ.get("WORK_HOURS_START") or work_hours.get("start", "08:00")
    work_end = os.environ.get("WORK_HOURS_END") or work_hours.get("end", "20:00")
    free_slots = calendar_sync.compute_free_slots(
        all_cal + work_busy,
        target_date=target_slot_date,
        work_start=work_start,
        work_end=work_end,
        min_slot_minutes=30,
    )
    focus_blocks = calendar_sync.suggest_time_blocks(free_slots, open_todos, max_suggestions=2)

    if mode == "evening":
        header = f"🌙 <b>Abend-Checkliste & Rückblick ({today_iso})</b>"
    else:
        header = f"☀️ <b>Guten Morgen Stefan! Dein Tages-Briefing ({today_iso})</b>"

    lines = [header, ""]

    if mode == "evening":
        lines.append(f"🎙️ <b>Heute neu erfasste Notizen ({len(today_items)}):</b>")
        if today_items:
            for idx, it in enumerate(today_items, 1):
                a = it.get("analysis", {})
                lines.append(f"  {idx}. {telegram_ui.esc(a.get('title', 'Notiz'))} – {telegram_ui.esc(a.get('summary', ''))}")
        else:
            lines.append("  Heute wurden keine neuen Sprach-/Textnotizen erfasst.")
        lines.append("")

    # Heutige private Termine
    if today_cal:
        lines.append(f"📅 <b>Heutige Termine ({len(today_cal)}):</b>")
        for idx, ev in enumerate(today_cal, 1):
            t_str = f" {ev['start_time']}–{ev.get('end_time') or ''} Uhr" if ev.get("start_time") else " (Ganztägig)"
            loc_str = f" [Ort: {telegram_ui.esc(ev['location'])}]" if ev.get("location") else ""
            lines.append(f"  {idx}. [{t_str.strip()}]{loc_str} {telegram_ui.esc(ev.get('title'))}")
    else:
        lines.append("📅 <b>Heutige Termine:</b> Keine Termine für heute im Google Kalender eingetragen.")

    if today_work_busy:
        lines.append(f"🔒 <i>Arbeitskalender: {len(today_work_busy)} anonyme Belegt-Blöcke (nur Frei/Belegt) berücksichtigt</i>")

    if free_slots:
        slot_lbl = "morgen" if mode == "evening" else "heute"
        lines.append("")
        lines.append(f"⏳ <b>Freie Zeitfenster ({slot_lbl}):</b>")
        for sl in free_slots[:4]:
            lines.append(f"  • {sl['start_time']}–{sl['end_time']} Uhr ({sl['duration_minutes']} Min.)")

    if focus_blocks:
        lines.append("🎯 <b>Fokus-Empfehlung:</b>")
        for fb in focus_blocks:
            lines.append(
                f"  • {fb['start_time']}–{fb['end_time']}: <b>{telegram_ui.esc(fb['task_title'])}</b>"
            )

    # Vorschau auf nächste Tage
    if upcoming_cal:
        lines.append("")
        lines.append("🗓️ <b>Vorschau auf anstehende Termine:</b>")
        for idx, ev in enumerate(upcoming_cal[:4], 1):
            t_str = f" ({ev['start_time']} Uhr)" if ev.get("start_time") else ""
            lines.append(f"  • {ev.get('date')}{t_str}: {telegram_ui.esc(ev.get('title'))}")

    lines.append("")
    lines.append(f"📌 <b>Offene To-Dos ({len(open_todos)}):</b>")
    if open_todos:
        for idx, t in enumerate(open_todos, 1):
            due = f" [Fällig: {t['due_date']}]" if t.get("due_date") else ""
            area_lbl = f" [{telegram_ui.esc(t['area'])}]" if t.get("area") else ""
            lines.append(f"  {idx}. [{telegram_ui.esc(t.get('priority', 'Mittel'))}]{area_lbl}{due} {telegram_ui.esc(t.get('title'))}")
            if t.get("next_step"):
                lines.append(f"     ↳ Nächster Schritt: {telegram_ui.esc(t['next_step'])}")
    else:
        lines.append("  Keine offenen To-Dos im System.")

    # KI-Impulse aus den letzten Notizen (max. 3, ohne die Aufgabenliste zu belasten)
    recent_suggestions: list[dict[str, Any]] = []
    for it in reversed(state.get("items", [])):
        for s in it.get("analysis", {}).get("suggestions", []):
            if s.get("title") and not any(x.get("title") == s.get("title") for x in recent_suggestions):
                recent_suggestions.append(s)
            if len(recent_suggestions) >= 3:
                break
        if len(recent_suggestions) >= 3:
            break

    if recent_suggestions:
        lines.append("")
        lines.append(f"💡 <b>KI-Impulse aus deinen letzten Notizen ({len(recent_suggestions)}):</b>")
        for idx, s in enumerate(recent_suggestions, 1):
            step = f" → {telegram_ui.esc(s['next_step'])}" if s.get("next_step") else ""
            lines.append(f"  • <b>{telegram_ui.esc(s.get('title'))}</b>{step}")

    if mode == "evening" and open_todos:
        lines.append("\nTipp: Nutze die Buttons unten, <code>/erledigt 1, 3</code> oder <code>/slots morgen</code>.")

    text = "\n".join(lines)
    if notify and (open_todos or suggestions or all_cal):
        if today_cal:
            top_focus = today_cal[0].get("title", "Termin")
        elif open_todos:
            top_focus = open_todos[0].get("title", "Aufgabe")
        elif suggestions:
            top_focus = suggestions[0].get("title", "Vorschlag")
        elif all_cal:
            top_focus = all_cal[0].get("title", "Termin")
        else:
            top_focus = "Fokus"
        show_windows_notification(
            telegram_ui.strip_html_tags(header),
            f"{len(today_cal)} Termine heute, {len(open_todos)} To-Dos. Fokus: {top_focus}",
        )

    if send_telegram:
        bot_token = (
            os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
            or os.environ.get("TELEGRAM_RSSBOT_TOKEN", "").strip()
        )
        chat_id = _resolve_primary_chat_id(state)

        # Fallback für Server ohne gesetztes TELEGRAM_CHAT_ID: Letzte Chat-ID direkt aus Telegram getUpdates abfragen
        if not chat_id and bot_token:
            try:
                gu_req = request.Request(f"https://api.telegram.org/bot{bot_token}/getUpdates?limit=10")
                with request.urlopen(gu_req, timeout=10) as gu_resp:
                    gu_data = json.loads(gu_resp.read().decode("utf-8"))
                    for u in reversed(gu_data.get("result", [])):
                        m = u.get("message") or u.get("edited_message")
                        if m and m.get("chat", {}).get("id"):
                            cand_id = str(m["chat"]["id"])
                            if telegram_ui.is_authorized_chat(cand_id):
                                chat_id = cand_id
                                state["last_telegram_chat_id"] = chat_id
                                try:
                                    save_local_state(BASE_DIR / cfg["paths"]["local_state_file"], state)
                                except Exception:
                                    pass
                                break
            except Exception:
                pass

        if bot_token and chat_id:
            api_base = f"https://api.telegram.org/bot{bot_token}"
            kb_rows: list[list[tuple[str, str]]] = []
            for idx in range(1, min(2, len(open_todos)) + 1):
                kb_rows.append([
                    (f"✅ #{idx} erledigen", f"done:{idx}"),
                    (f"📅 #{idx} auf morgen", f"defer:{idx}"),
                ])
            reply_markup = telegram_ui.build_inline_keyboard(kb_rows) if kb_rows else None
            ok = _send_telegram_message(api_base, chat_id, text, reply_markup=reply_markup)
            if ok:
                print(f"[Telegram] {mode.capitalize()}-Briefing an Chat-ID {chat_id} gesendet.")
            else:
                err_msg = f"{mode.capitalize()}-Briefing an Chat-ID {chat_id} konnte nicht übertragen werden."
                print(f"[Telegram Fehler] {err_msg}")
                if raise_on_telegram_failure:
                    raise RuntimeError(err_msg)
        else:
            missing_info = f"Bot-Token ({bool(bot_token)}) oder Chat-ID ({bool(chat_id)}) fehlt."
            print(f"[Telegram] Hinweis: {missing_info} Briefing nicht per Telegram gesendet.")
            if raise_on_telegram_failure:
                raise RuntimeError(missing_info)

    return text


def perform_self_update() -> tuple[bool, str]:
    """Führt ein git pull aus und aktualisiert ggf. Abhängigkeiten.
    Gibt (needs_restart, message) zurück.
    """
    import subprocess
    try:
        pull_proc = subprocess.run(
            ["git", "pull", "--autostash"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if pull_proc.returncode != 0:
            subprocess.run(["git", "stash"], cwd=BASE_DIR, capture_output=True, timeout=15)
            pull_proc = subprocess.run(
                ["git", "pull"],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=60,
            )

        if pull_proc.returncode != 0:
            err = pull_proc.stderr.strip() or pull_proc.stdout.strip()
            return False, f"❌ <code>git pull</code> fehlgeschlagen:\n<pre>{telegram_ui.esc(err[:800])}</pre>"

        pull_out = pull_proc.stdout.strip()
        already_up_to_date = "Already up to date" in pull_out or "Bereits aktuell" in pull_out

        log_proc = subprocess.run(
            ["git", "log", "-1", "--format=%h - %s (%cr)"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit_info = log_proc.stdout.strip() if log_proc.returncode == 0 else ""

        # Falls heute das Morgen- oder Abend-Briefing noch nicht kam oder verpasst wurde,
        # setzen wir den Status dafür in scheduler_state.json zurück, damit der Neustart es nachholt.
        try:
            if SCHEDULER_STATE_PATH.exists():
                st_data = safe_read_json(SCHEDULER_STATE_PATH, {})
                if isinstance(st_data, dict):
                    st_data.pop("briefing_morning", None)
                    st_data.pop("briefing_evening", None)
                    atomic_write_json(SCHEDULER_STATE_PATH, st_data, indent=2)
        except Exception:
            pass

        if already_up_to_date:
            msg = (
                f"✅ <b>Das System ist bereits auf dem neuesten Stand!</b>\n\n"
                f"📌 <b>Aktueller Commit:</b>\n<code>{telegram_ui.esc(commit_info)}</code>\n\n"
                f"⚡ Starte Dienst neu, um alle Module frisch zu laden..."
            )
            return True, msg

        pip_note = ""
        if "requirements.txt" in pull_out:
            pip_proc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
                cwd=BASE_DIR,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if pip_proc.returncode == 0:
                pip_note = "\n📦 Neue Python-Pakete wurden installiert."
            else:
                pip_note = f"\n⚠️ Warnung bei pip install: {telegram_ui.esc(pip_proc.stderr[:120])}"

        msg = (
            f"🔄 <b>Update erfolgreich heruntergeladen!</b>\n\n"
            f"📌 <b>Neuer Commit:</b>\n<code>{telegram_ui.esc(commit_info)}</code>\n"
            f"{pip_note}\n\n"
            f"⚡ Starte Dienst jetzt neu..."
        )
        return True, msg
    except Exception as exc:
        return False, f"❌ Fehler beim Aktualisieren: {telegram_ui.esc(exc)}"


def trigger_restart() -> None:
    """Startet den Prozess sauber neu.
    Funktioniert unter systemd (Restart=always), Windows Task Scheduler und direkt via subprocess / os.execv.
    """
    def _restart():
        time.sleep(1.5)
        pid_file = BASE_DIR / "daemon.pid"
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass
        script_name = Path(sys.argv[0]).name if sys.argv and Path(sys.argv[0]).name.endswith(".py") else "assistant.py"
        script = str(BASE_DIR / script_name)
        cmd_args = [sys.executable, script] + [a for a in sys.argv[1:] if a != script]
        if script_name == "assistant.py" and "daemon" not in cmd_args and "run" not in cmd_args and "briefing" not in cmd_args:
            cmd_args.append("daemon")
        try:
            print(f"[Restart] Führe Neustart aus: {' '.join(cmd_args)}")
            if os.environ.get("INVOCATION_ID") or os.environ.get("JOURNAL_STREAM"):
                print("[Restart] Systemd-Dienst erkannt -> beende Prozess für sauberen systemd-Neustart.")
                os._exit(0)
            if sys.platform == "win32":
                subprocess.Popen(cmd_args, cwd=BASE_DIR, close_fds=True)
                os._exit(0)
            else:
                os.execv(sys.executable, cmd_args)
        except Exception as exc:
            print(f"[Restart Fehler] Neustart fehlgeschlagen: {exc} -> Beende Prozess für externen Restart.")
            os._exit(0)

    threading.Thread(target=_restart, daemon=True).start()


SCHEDULER_STATE_PATH = BASE_DIR / "scheduler_state.json"
SCHEDULER_LOG_PATH = BASE_DIR / "scheduler.log"
SCHEDULER_RETRY_COOLDOWN: dict[str, float] = {}

# Standard-Zeitplan (überschreibbar via config.json -> "schedule").
# "catchup_until": Bis zu dieser Uhrzeit wird ein verpasster Lauf (z. B. nach Neustart) noch nachgeholt.
# "day_of_week": Optional 0=Mo .. 6=So (z. B. 6 für Sonntags-Review & RSS-Synthese)
DEFAULT_SCHEDULE: dict[str, Any] = {
    "enabled": True,
    "timezone": "Europe/Berlin",
    "jobs": [
        {"id": "rss_morning", "time": "07:30", "catchup_until": "18:00", "command": ["rss_monitor.py", "run"]},
        {"id": "briefing_morning", "time": "08:00", "catchup_until": "17:00",
         "command": ["assistant.py", "briefing", "--mode", "morning", "--telegram"]},
        {"id": "rss_evening", "time": "18:30", "catchup_until": "23:59", "command": ["rss_monitor.py", "run"]},
        {"id": "briefing_evening", "time": "19:30", "catchup_until": "23:59",
         "command": ["assistant.py", "briefing", "--mode", "evening", "--telegram"]},
        {"id": "weekly_review", "day_of_week": 6, "time": "18:00", "catchup_until": "23:59",
         "command": ["assistant.py", "review"]},
        {"id": "weekly_synthesis", "day_of_week": 6, "time": "18:15", "catchup_until": "23:59",
         "command": ["rss_bot.py"]},
    ],
}


def _scheduler_now(tz_name: str) -> dt.datetime:
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo(tz_name))
    except Exception:
        return dt.datetime.now()


def _parse_hhmm(value: str) -> dt.time:
    hh, mm = str(value).strip().split(":", 1)
    return dt.time(int(hh), int(mm))


def _load_scheduler_state() -> dict[str, Any]:
    data = safe_read_json(SCHEDULER_STATE_PATH, {})
    return data if isinstance(data, dict) else {}


def _save_scheduler_state(sched_state: dict[str, Any]) -> None:
    try:
        atomic_write_json(SCHEDULER_STATE_PATH, sched_state, indent=2)
    except Exception as exc:
        print(f"[Scheduler] Konnte Zustand nicht speichern: {exc}")


def run_scheduled_jobs(
    cfg: dict[str, Any],
    running: dict[str, Any],
    state: dict[str, Any] | None = None,
    notion: NotionClient | None = None,
) -> None:
    """Startet fällige Zeitplan-Jobs (RSS-Monitor, Briefings, Sonntags-Review/Synthese) im Hintergrund.

    Ersetzt den Windows Task Scheduler, damit der Daemon auch unter Linux/GCP alles selbst erledigt.
    `running` hält aktive Threads & Prozesse (job_id -> Thread/Popen), damit nichts doppelt startet.
    """
    schedule = {**DEFAULT_SCHEDULE, **(cfg.get("schedule") or {})}
    if not schedule.get("enabled", True):
        return

    # Beendete Threads & Prozesse aufräumen
    for job_id, item in list(running.items()):
        if isinstance(item, threading.Thread):
            if not item.is_alive():
                running.pop(job_id, None)
        else:
            rc = item.poll()
            if rc is not None:
                print(f"[Scheduler] Job '{job_id}' beendet (Exit-Code {rc}).")
                running.pop(job_id, None)

    now = _scheduler_now(schedule.get("timezone", "Europe/Berlin"))
    today = now.date().isoformat()
    sched_state = _load_scheduler_state()

    for job in schedule.get("jobs", []):
        job_id = job.get("id")
        if not job_id or job.get("enabled", True) is False or job_id in running:
            continue
        if "day_of_week" in job and now.weekday() != int(job["day_of_week"]):
            continue
        if SCHEDULER_RETRY_COOLDOWN.get(job_id, 0) > time.time():
            continue
        if sched_state.get(job_id) == today:
            continue
        try:
            start_t = _parse_hhmm(job["time"])
            until_t = _parse_hhmm(job.get("catchup_until", "23:59"))
        except Exception:
            print(f"[Scheduler] Ungültige Zeitangabe für Job '{job_id}' – übersprungen.")
            continue
        if not (start_t <= now.time().replace(tzinfo=None) <= until_t):
            continue

        # A) Briefing-Jobs direkt im In-Memory-Thread des Daemons ausführen
        if "briefing" in job_id and state is not None:
            mode = "evening" if "evening" in job_id else "morning"

            def _briefing_worker(jid=job_id, m=mode, d=today):
                try:
                    print(f"[Scheduler] Starte Briefing-Job '{jid}' (Modus: {m})...")
                    if notion:
                        try:
                            state["tasks"] = notion.query_open_tasks(cfg["notion"]["tasks_database_id"])
                        except Exception as ne:
                            print(f"[Scheduler Briefing] Notion-Refresh Hinweis: {ne}")
                    text = generate_briefing(cfg, state, notify=False, mode=m, send_telegram=True, raise_on_telegram_failure=True)
                    try:
                        with open(SCHEDULER_LOG_PATH, "a", encoding="utf-8") as f:
                            f.write(f"\n===== {_scheduler_now(schedule.get('timezone', 'Europe/Berlin')).isoformat(timespec='seconds')} Briefing '{jid}' gesendet =====\n")
                            f.write(text[:300] + "...\n")
                    except Exception:
                        pass
                    st = _load_scheduler_state()
                    st[jid] = d
                    _save_scheduler_state(st)
                    SCHEDULER_RETRY_COOLDOWN.pop(jid, None)
                    print(f"[Scheduler] Briefing-Job '{jid}' erfolgreich abgeschlossen.")
                except Exception as exc:
                    SCHEDULER_RETRY_COOLDOWN[jid] = time.time() + 300
                    print(f"[Scheduler Fehler] Briefing-Job '{jid}' fehlgeschlagen (erneuter Versuch in 5 Min.): {exc}")
                    traceback.print_exc()

            t = threading.Thread(target=_briefing_worker, daemon=True)
            t.start()
            running[job_id] = t
            continue

        # B) RSS-Monitor-Jobs im Thread ausführen
        if "rss" in job_id:
            def _rss_worker(jid=job_id, d=today):
                try:
                    print(f"[Scheduler] Starte RSS-Job '{jid}'...")
                    import rss_monitor
                    import argparse as _ap
                    ns = _ap.Namespace(topic=None, dry_run=False, no_notify=False, no_notion=False, verbose=False)
                    rss_monitor.run(ns)
                    st = _load_scheduler_state()
                    st[jid] = d
                    _save_scheduler_state(st)
                    SCHEDULER_RETRY_COOLDOWN.pop(jid, None)
                    print(f"[Scheduler] RSS-Job '{jid}' erfolgreich abgeschlossen.")
                except Exception as exc:
                    SCHEDULER_RETRY_COOLDOWN[jid] = time.time() + 300
                    print(f"[Scheduler Fehler] RSS-Job '{jid}' fehlgeschlagen (erneuter Versuch in 5 Min.): {exc}")
                    traceback.print_exc()

            t = threading.Thread(target=_rss_worker, daemon=True)
            t.start()
            running[job_id] = t
            continue

        # C) Sonntags-Wochenreview
        if job_id == "weekly_review" and state is not None:
            def _review_worker(jid=job_id, d=today):
                try:
                    print("[Scheduler] Starte Weekly-Review...")
                    if notion:
                        try:
                            state["tasks"] = notion.query_open_tasks(cfg["notion"]["tasks_database_id"])
                        except Exception:
                            pass
                    review_html, _ = agent_tools.generate_weekly_review(cfg, state, notion, get_store())
                    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
                    chat_id = _resolve_primary_chat_id(state)
                    if bot_token and chat_id:
                        telegram_ui.send_message(bot_token, chat_id, review_html)
                    st = _load_scheduler_state()
                    st[jid] = d
                    _save_scheduler_state(st)
                    SCHEDULER_RETRY_COOLDOWN.pop(jid, None)
                except Exception as exc:
                    SCHEDULER_RETRY_COOLDOWN[jid] = time.time() + 300
                    print(f"[Scheduler Fehler] Weekly-Review fehlgeschlagen: {exc}")

            t = threading.Thread(target=_review_worker, daemon=True)
            t.start()
            running[job_id] = t
            continue

        # D) Sonntags-RSS-Synthese
        if job_id == "weekly_synthesis":
            def _synth_worker(jid=job_id, d=today):
                try:
                    print("[Scheduler] Starte wöchentliche RSS-Synthese...")
                    import rss_bot
                    rss_token = os.environ.get("TELEGRAM_RSSBOT_TOKEN", "").strip()
                    rss_chat_id = os.environ.get("TELEGRAM_RSS_CHAT_ID", "").strip() or _resolve_primary_chat_id(state or {})
                    if rss_token and rss_chat_id:
                        synth_html = rss_bot.handle_synthese(rss_chat_id)
                        rss_bot.send_message(rss_token, rss_chat_id, synth_html)
                    st = _load_scheduler_state()
                    st[jid] = d
                    _save_scheduler_state(st)
                    SCHEDULER_RETRY_COOLDOWN.pop(jid, None)
                except Exception as exc:
                    SCHEDULER_RETRY_COOLDOWN[jid] = time.time() + 300
                    print(f"[Scheduler Fehler] RSS-Synthese fehlgeschlagen: {exc}")

            t = threading.Thread(target=_synth_worker, daemon=True)
            t.start()
            running[job_id] = t
            continue

        # E) Generischer Fallback für sonstige externe Befehle als Subprozess
        cmd = [sys.executable] + [str(BASE_DIR / job["command"][0])] + list(job["command"][1:])
        try:
            log_fh = open(SCHEDULER_LOG_PATH, "a", encoding="utf-8")
            log_fh.write(f"\n===== {now.isoformat(timespec='seconds')} Start '{job_id}': {' '.join(cmd)} =====\n")
            log_fh.flush()
            proc = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=log_fh, stderr=subprocess.STDOUT)
            log_fh.close()
            running[job_id] = proc
            print(f"[Scheduler] Job '{job_id}' gestartet (PID {proc.pid}).")
            sched_state[job_id] = today
            _save_scheduler_state(sched_state)
        except Exception as exc:
            print(f"[Scheduler] Job '{job_id}' konnte nicht gestartet werden: {exc}")


def run_daemon(
    cfg: dict[str, Any],
    state_path: Path,
    notion: NotionClient | None,
    model_override: str | None = None,
) -> None:
    """
    Echtzeit-Hintergrunddienst (Daemon):
    - Long-Polling auf Telegram (Antwort in 1–2 Sekunden)
    - Pre-Meeting-Briefings (40 Min vor privaten Terminen)
    - Zeitplan (Morgen-/Abend-Briefing, RSS-Monitor, Sonntags-Review & Synthese)
    - Regelmäßige Sync-Schleife (Drop-Ordner, Mails, Notion Inbox)
    """
    setup_logging()

    pid_file = BASE_DIR / "daemon.pid"
    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text(encoding="utf-8").strip())
            if old_pid != os.getpid():
                is_running = False
                if sys.platform == "win32":
                    proc = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {old_pid}", "/NH"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    is_running = str(old_pid) in proc.stdout and ("python" in proc.stdout.lower())
                else:
                    try:
                        os.kill(old_pid, 0)
                        cmdline_path = Path(f"/proc/{old_pid}/cmdline")
                        if cmdline_path.exists():
                            cmdline = cmdline_path.read_text(encoding="utf-8", errors="replace")
                            is_running = "assistant.py" in cmdline or "python" in cmdline
                        else:
                            is_running = True
                    except OSError:
                        is_running = False
                if is_running:
                    print(f"[Daemon] Läuft bereits unter PID {old_pid}. Beende Duplikat.")
                    return
        except Exception:
            pass
    try:
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        pass

    print(f"[Daemon] Echtzeit-Assistent gestartet (PID {os.getpid()}). Warte auf Telegram-Nachrichten...")

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if bot_token:
        telegram_ui.register_bot_commands(bot_token, telegram_ui.MAIN_BOT_COMMANDS)

    # Dedizierten RSS- & Wissensspeicher-Bot starten, falls konfiguriert
    if os.environ.get("TELEGRAM_RSSBOT_TOKEN", "").strip():
        try:
            import rss_bot
            rss_bot.start_in_background()
            print("[Daemon] Dedizierter RSS-Bot erfolgreich im Hintergrund gestartet.")
        except Exception as _e:
            print(f"[Daemon] Hinweis: Konnte RSS-Bot nicht starten: {_e}")

    last_full_sync = 0.0
    scheduled_procs: dict[str, Any] = {}
    store = get_store()

    try:
        while True:
            state = load_local_state(state_path)
            now = time.time()

            # Zeitplan: RSS 07:30/18:30, Briefing 08:00/19:30, So Review/Synthese
            try:
                run_scheduled_jobs(cfg, scheduled_procs, state=state, notion=notion)
            except Exception as exc:
                print(f"[Scheduler Hinweis] {exc}")

            # Alle 5 Minuten: Offene Tasks, Drop-Ordner, Mails, Notion Inbox & Pre-Meeting-Pings prüfen
            if now - last_full_sync >= 300:
                last_full_sync = now
                if notion:
                    try:
                        state["tasks"] = notion.query_open_tasks(cfg["notion"]["tasks_database_id"])
                    except Exception:
                        pass
                try:
                    process_local_drop_folder(cfg, state, notion, model_override=model_override)
                    process_emails(cfg, state, notion, model_override=model_override)
                    if notion:
                        process_notion_inbox(cfg, state, notion, model_override=model_override)
                    sync_pending_calendar_events(state)
                    save_local_state(state_path, state)
                except Exception as exc:
                    print(f"[Daemon Sync Hinweis] {exc}")

                # Pre-Meeting-Briefings (z. B. 40 Min vor privatem Termin)
                try:
                    lead_min = int(cfg.get("pre_meeting_lead_minutes", 40))
                    pings = agent_tools.check_upcoming_meeting_pings(cfg, state, notion, store, lead_minutes=lead_min)
                    chat_id = _resolve_primary_chat_id(state)
                    if pings and bot_token and chat_id:
                        for p_html in pings:
                            telegram_ui.send_message(bot_token, chat_id, p_html)
                except Exception as exc:
                    print(f"[Pre-Meeting Ping Hinweis] {exc}")

            # Echtzeit-Telegram-Abfrage (Long-Polling bis zu 20s)
            try:
                tg_count = process_telegram_updates(
                    cfg,
                    state,
                    notion,
                    model_override=model_override,
                    poll_timeout=20,
                )
                if tg_count > 0:
                    if notion:
                        try:
                            state["tasks"] = notion.query_open_tasks(cfg["notion"]["tasks_database_id"])
                        except Exception:
                            pass
                    save_local_state(state_path, state)
            except Exception as exc:
                print(f"[Daemon Telegram Hinweis] {exc}")
                time.sleep(5)
    finally:
        try:
            if pid_file.exists() and pid_file.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pid_file.unlink()
        except Exception:
            pass


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    setup_logging()
    parser = argparse.ArgumentParser(description="Persönlicher Notion-, Kalender- & Sprachnotiz-Assistent")
    parser.add_argument(
        "command",
        choices=["run", "daemon", "briefing", "review", "add", "sync-calendar"],
        nargs="?",
        default="run",
        help="run = Einmaliger Sync | daemon = Echtzeit-Hintergrunddienst | briefing = Morgen-/Abend-Briefing | review = Wochen-Review | add = schnelle Textnotiz | sync-calendar = Termine synchronisieren",
    )
    parser.add_argument("--text", help="Direkter Notiztext für das Kommando 'add'")
    parser.add_argument("--title", default="Schnellnotiz", help="Titel für 'add'")
    parser.add_argument("--model", help="Optionales lokales Ollama-Modell (z. B. qwen2.5-coder:7b)")
    parser.add_argument("--notify", action="store_true", help="Windows-Benachrichtigung anzeigen")
    parser.add_argument("--telegram", action="store_true", help="Briefing aktiv per Telegram aufs Handy senden")
    parser.add_argument(
        "--mode",
        choices=["morning", "evening"],
        default="morning",
        help="morning = Morgen-Briefing (Standard), evening = Abend-Checkliste & Rückblick",
    )
    args = parser.parse_args()

    cfg = load_config()
    state_path = BASE_DIR / cfg["paths"]["local_state_file"]
    state = load_local_state(state_path)

    notion_token = os.environ.get("NOTION_TOKEN", "").strip()
    notion = NotionClient(notion_token) if notion_token else None

    if args.command == "daemon":
        run_daemon(cfg, state_path, notion, model_override=args.model)
        return

    if notion:
        try:
            state["tasks"] = notion.query_open_tasks(cfg["notion"]["tasks_database_id"])
        except Exception as exc:
            print(f"[Notion Sync] Hinweis beim Laden offener Tasks: {exc}")

    if args.command == "sync-calendar":
        synced = sync_pending_calendar_events(state)
        save_local_state(state_path, state)
        print(f"[Fertig] {synced} Kalendereinträge mit Google Kalender synchronisiert.")
        return

    if args.command == "review":
        review_html, _ = agent_tools.generate_weekly_review(cfg, state, notion, get_store())
        print(telegram_ui.strip_html_tags(review_html))
        if args.telegram:
            bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
            chat_id = _resolve_primary_chat_id(state)
            if bot_token and chat_id:
                telegram_ui.send_message(bot_token, chat_id, review_html)
        return

    if args.command == "add":
        if not args.text:
            print("Bitte --text \"Deine Notiz...\" angeben.")
            sys.exit(1)
        drop_dir = BASE_DIR / cfg["paths"]["inbox_drop_dir"]
        drop_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.title.replace(' ', '_')}.txt"
        (drop_dir / fname).write_text(args.text, encoding="utf-8")
        print(f"[Eingang] Notiz gespeichert unter inbox_drop/{fname}. Starte Auswertung...")

    if args.command in ("run", "add"):
        local_results = process_local_drop_folder(cfg, state, notion, model_override=args.model)
        tg_count = process_telegram_updates(cfg, state, notion, model_override=args.model)
        mail_count = process_emails(cfg, state, notion, model_override=args.model)
        notion_count = 0
        if notion:
            notion_count = process_notion_inbox(cfg, state, notion, model_override=args.model)
        if notion and (local_results or tg_count or mail_count or notion_count):
            try:
                state["tasks"] = notion.query_open_tasks(cfg["notion"]["tasks_database_id"])
            except Exception:
                pass
        sync_pending_calendar_events(state)
        save_local_state(state_path, state)
        print(
            f"\n[Fertig] {len(local_results)} lokale Datei(en), {tg_count} Telegram-Interaktion(en), "
            f"{mail_count} E-Mail(s) und {notion_count} Notion-Inbox-Einträge verarbeitet."
        )
        print("\n" + generate_briefing(cfg, state, notify=args.notify, mode=args.mode, send_telegram=args.telegram))
    elif args.command == "briefing":
        print(generate_briefing(cfg, state, notify=args.notify, mode=args.mode, send_telegram=args.telegram))


if __name__ == "__main__":
    main()
