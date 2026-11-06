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
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
ENV_PATH = BASE_DIR / ".env"

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac"}
TEXT_EXTENSIONS = {".txt", ".md"}


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
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"items": [], "tasks": [], "telegram_offset": 0}


def save_local_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
) -> dict[str, Any]:
    """Ruft Gemini (Standard) oder lokales Ollama mit strukturierter JSON-Rückgabe auf."""
    provider, default_model = get_llm_choice(cfg)
    model_name = model_override or default_model

    if provider == "gemini":
        try:
            from google import genai
            from google.genai import types

            gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
            client = genai.Client(api_key=gemini_key)
            config = types.GenerateContentConfig(
                system_instruction=system_instruction or (
                    "Du bist der persönliche deutsche KI-Assistent von Stefan Kollmeier. "
                    "Analysiere präzise und antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt."
                ),
                response_mime_type="application/json",
                temperature=0.1,
            )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            raw = response.text or "{}"
            return json.loads(raw)
        except Exception as exc:
            print(f"[Gemini JSON Fehler] {exc} -> Fallback auf Ollama.")

    return call_ollama_json(prompt, cfg.get("ollama", {}), model_override=model_override)


def call_llm_text(
    prompt: str,
    cfg: dict[str, Any],
    system_instruction: str | None = None,
    model_override: str | None = None,
) -> str:
    """Ruft Gemini oder Ollama für Textantworten auf."""
    provider, default_model = get_llm_choice(cfg)
    model_name = model_override or default_model

    if provider == "gemini":
        try:
            from google import genai
            from google.genai import types

            gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
            client = genai.Client(api_key=gemini_key)
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            ) if system_instruction else types.GenerateContentConfig(temperature=0.3)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            return (response.text or "").strip()
        except Exception as exc:
            print(f"[Gemini Text Fehler] {exc} -> Fallback auf Ollama.")

    # Ollama Fallback
    ollama_cfg = cfg.get("ollama", {})
    res = call_ollama_json(
        prompt + "\n\nGib deine Antwort als JSON mit genau dem Feld 'answer': {\"answer\": \"...\"}",
        ollama_cfg,
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
3. "todos": Eine Liste konkreter Aufgaben (To-Dos), die erledigt werden müssen (für die Notion-Aufgabenliste).
   Jedes To-Do hat:
   - "title": Kurze Aufgabenbeschreibung (präzise auf Deutsch)
   - "priority": "Hoch", "Mittel" oder "Niedrig"
   - "due_date": Konkretes Fälligkeitsdatum im Format "YYYY-MM-DD" (berechne relative Angaben wie "morgen", "Mittwoch", "nächste Woche", "Sonntag" EXAKT anhand der obigen Kalender-Referenztabelle) oder null, wenn kein Datum genannt ist.
   - "next_step": Konkreter erster Arbeitsschritt (1 Satz).
4. "calendar_events": Eine Liste echter Kalendertermine mit festem Zeitpunkt/Zeitblock (z. B. Besprechung um 14 Uhr, Vorlesung, Arzttermin, Fokusblock 19-20 Uhr, Geburtstag).
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
5. "suggestions": 1 bis 2 hilfreiche, konkret umsetzbare Impulse / Ideen zur Notiz.
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
  "todos": [
    {{"title": "...", "priority": "Hoch|Mittel|Niedrig", "due_date": "YYYY-MM-DD oder null", "next_step": "..."}}
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

    # Sicherheits-Deduplizierung: Falls das LLM denselben Punkt trotzdem in todos UND calendar_events packt:
    # - Hat der Kalendereintrag KEINE Uhrzeit (start_time is None) und existiert ein ähnliches To-Do am selben Tag,
    #   behalten wir nur das To-Do (mit due_date).
    # - Hat der Kalendereintrag eine feste Uhrzeit (start_time gesetzt), behalten wir den Kalendereintrag.
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
                # Ganztägige "Erinnerung" ist bereits als To-Do mit Fälligkeitsdatum erfasst -> kein doppelter Kalendereintrag
                if not overlapping_todo.get("due_date") and ev_date:
                    overlapping_todo["due_date"] = ev_date
                continue
            else:
                # Echter Uhrzeit-Termin -> aus To-Dos entfernen, damit er nur im Kalender steht
                filtered_todos = [t for t in filtered_todos if t is not overlapping_todo]
        filtered_cal.append(ev)

    return {
        "title": result.get("title") or title,
        "summary": result.get("summary") or "",
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


class NotionClient:
    """Leichtgewichtiger Notion-API-Client für den lokalen Hintergrundbetrieb."""

    def __init__(self, token: str) -> None:
        self.token = token
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=self.headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Notion API Fehler ({exc.code}): {details}") from exc

    def query_new_inbox_items(self, inbox_db_id: str) -> list[dict[str, Any]]:
        payload = {
            "filter": {
                "property": "Status",
                "status": {"equals": "Neu"},
            }
        }
        res = self._request("POST", f"/databases/{inbox_db_id}/query", payload)
        return res.get("results", [])

    def get_page_blocks_text(self, page_id: str) -> str:
        res = self._request("GET", f"/blocks/{page_id}/children?page_size=50")
        lines: list[str] = []
        for block in res.get("results", []):
            btype = block.get("type", "")
            bdata = block.get(btype, {})
            rich = bdata.get("rich_text", [])
            text = "".join(rt.get("plain_text", "") for rt in rich).strip()
            if text:
                lines.append(text)
        return "\n".join(lines).strip()

    def create_inbox_page(
        self,
        inbox_db_id: str,
        title: str,
        source: str,
        raw_text: str,
        summary: str,
        suggestions_text: str,
    ) -> str:
        payload: dict[str, Any] = {
            "parent": {"database_id": inbox_db_id},
            "icon": {"type": "emoji", "emoji": "🎙️"},
            "properties": {
                "Name": {"title": [{"text": {"content": title[:100]}}]},
                "Status": {"status": {"name": "Ausgewertet"}},
                "Quelle": {"select": {"name": source}},
                "Transkript": {"rich_text": [{"text": {"content": raw_text[:1900]}}]},
                "Zusammenfassung": {"rich_text": [{"text": {"content": summary[:1900]}}]},
                "Vorschlaege": {"rich_text": [{"text": {"content": suggestions_text[:1900]}}]},
            },
        }
        res = self._request("POST", "/pages", payload)
        return res["id"]

    def update_inbox_page(
        self,
        page_id: str,
        raw_text: str,
        summary: str,
        suggestions_text: str,
    ) -> None:
        payload: dict[str, Any] = {
            "properties": {
                "Status": {"status": {"name": "Ausgewertet"}},
                "Transkript": {"rich_text": [{"text": {"content": raw_text[:1900]}}]},
                "Zusammenfassung": {"rich_text": [{"text": {"content": summary[:1900]}}]},
                "Vorschlaege": {"rich_text": [{"text": {"content": suggestions_text[:1900]}}]},
            }
        }
        self._request("PATCH", f"/pages/{page_id}", payload)

    def create_task_page(
        self,
        tasks_db_id: str,
        title: str,
        kind: str,
        priority: str,
        due_date: str | None,
        next_step: str,
        inbox_page_id: str | None = None,
    ) -> str:
        props: dict[str, Any] = {
            "Aufgabenbezeichnung": {"title": [{"text": {"content": title[:100]}}]},
            "Status": {"status": {"name": "Nicht begonnen"}},
            "Art": {"select": {"name": kind}},
            "Prioritaet": {"select": {"name": priority if priority in ("Hoch", "Mittel", "Niedrig") else "Mittel"}},
            "Schritt": {"rich_text": [{"text": {"content": next_step[:1900]}}]},
        }
        if due_date and len(due_date) == 10 and due_date != "null":
            props["Fällig"] = {"date": {"start": due_date}}
        if inbox_page_id:
            props["Notiz"] = {"relation": [{"id": inbox_page_id}]}

        icon_emoji = "✅" if kind == "To-Do" else ("📅" if kind == "Erinnerung" else "💡")
        payload: dict[str, Any] = {
            "parent": {"database_id": tasks_db_id},
            "icon": {"type": "emoji", "emoji": icon_emoji},
            "properties": props,
        }
        res = self._request("POST", "/pages", payload)
        return res["id"]

    def query_open_tasks(self, tasks_db_id: str) -> list[dict[str, Any]]:
        payload = {
            "filter": {
                "and": [
                    {"property": "Status", "status": {"does_not_equal": "Erledigt"}},
                    {"property": "Status", "status": {"does_not_equal": "Archiviert"}},
                ]
            }
        }
        res = self._request("POST", f"/databases/{tasks_db_id}/query", payload)
        tasks_out: list[dict[str, Any]] = []
        for page in res.get("results", []):
            p = page.get("properties", {})
            title_arr = p.get("Aufgabenbezeichnung", {}).get("title", [])
            title = "".join(t.get("plain_text", "") for t in title_arr).strip()
            if not title:
                continue
            status = (p.get("Status", {}).get("status") or {}).get("name", "Nicht begonnen")
            kind = (p.get("Art", {}).get("select") or {}).get("name", "To-Do")
            prio = (p.get("Prioritaet", {}).get("select") or {}).get("name", "Mittel")
            due = (p.get("Fällig", {}).get("date") or {}).get("start")
            step_arr = p.get("Schritt", {}).get("rich_text", [])
            step = "".join(t.get("plain_text", "") for t in step_arr).strip()
            tasks_out.append({
                "notion_page_id": page["id"],
                "kind": kind,
                "title": title,
                "priority": prio,
                "due_date": due,
                "next_step": step,
                "status": status,
            })
        return tasks_out

    def mark_task_done(self, page_id: str) -> None:
        self._request("PATCH", f"/pages/{page_id}", {"properties": {"Status": {"status": {"name": "Erledigt"}}}})

    def update_task(
        self,
        page_id: str,
        due_date: str | None = None,
        priority: str | None = None,
        title: str | None = None,
    ) -> None:
        props: dict[str, Any] = {}
        if due_date and len(due_date) == 10 and due_date != "null":
            props["Fällig"] = {"date": {"start": due_date}}
        if priority and priority in ("Hoch", "Mittel", "Niedrig"):
            props["Prioritaet"] = {"select": {"name": priority}}
        if title:
            props["Aufgabenbezeichnung"] = {"title": [{"text": {"content": title[:100]}}]}
        if props:
            self._request("PATCH", f"/pages/{page_id}", {"properties": props})

    def archive_task(self, page_id: str) -> None:
        self._request("PATCH", f"/pages/{page_id}", {
            "archived": True,
            "properties": {"Status": {"status": {"name": "Archiviert"}}},
        })

    def search_workspace(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        payload = {
            "query": query,
            "page_size": limit,
            "sort": {"direction": "descending", "timestamp": "last_edited_time"},
        }
        res = self._request("POST", "/search", payload)
        out: list[dict[str, Any]] = []
        for obj in res.get("results", []):
            obj_type = obj.get("object", "page")
            props = obj.get("properties", {})
            title = ""
            for pval in props.values():
                if pval.get("type") == "title":
                    title = "".join(t.get("plain_text", "") for t in pval.get("title", [])).strip()
                    break
            if not title:
                title_arr = obj.get("title", [])
                if isinstance(title_arr, list):
                    title = "".join(t.get("plain_text", "") for t in title_arr).strip()
            if not title:
                title = obj.get("url", "Unbenannt")
            out.append({
                "id": obj.get("id"),
                "type": obj_type,
                "title": title,
                "url": obj.get("url", ""),
            })
        return out


def _normalize_hhmm(value: Any) -> str | None:
    if not value or str(value).lower() == "null":
        return None
    s = str(value).strip()
    if len(s) >= 5 and s[2] == ":":
        return s[:5]
    return None


def export_ics_event(ev: dict[str, Any]) -> Path:
    """Schreibt zusätzlich eine standardkonforme .ics-Kalenderdatei in calendar_outbox/."""
    outbox = BASE_DIR / "calendar_outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    title = ev.get("title") or "Termin"
    date_str = ev.get("date") or dt.date.today().isoformat()
    start_t = _normalize_hhmm(ev.get("start_time"))
    end_t = _normalize_hhmm(ev.get("end_time"))
    location = ev.get("location") or ""
    description = ev.get("description") or ""

    date_compact = date_str.replace("-", "")
    if start_t:
        st_compact = start_t.replace(":", "") + "00"
        if end_t:
            et_compact = end_t.replace(":", "") + "00"
        else:
            hour = (int(start_t[:2]) + 1) % 24
            et_compact = f"{hour:02d}{start_t[3:5]}00"
        dtstart_line = f"DTSTART:{date_compact}T{st_compact}"
        dtend_line = f"DTEND:{date_compact}T{et_compact}"
    else:
        try:
            next_day = (dt.date.fromisoformat(date_str) + dt.timedelta(days=1)).isoformat().replace("-", "")
        except ValueError:
            next_day = date_compact
        dtstart_line = f"DTSTART;VALUE=DATE:{date_compact}"
        dtend_line = f"DTEND;VALUE=DATE:{next_day}"

    uid = f"{date_compact}-{abs(hash(title + date_str))}@notion-local-assistant"
    ics_content = "\r\n".join([
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Stefan Kollmeier//Notion Local Assistant//DE",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"SUMMARY:{title}",
        dtstart_line,
        dtend_line,
        f"LOCATION:{location}",
        f"DESCRIPTION:{description}",
        "END:VEVENT",
        "END:VCALENDAR",
        "",
    ])
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title)[:40]
    ics_path = outbox / f"{date_str}_{safe_name}.ics"
    ics_path.write_text(ics_content, encoding="utf-8")
    return ics_path


GCAL_CACHE_PATH = BASE_DIR / "gcal_cache.json"


def get_gcalendar_bin() -> str | None:
    """Findet die gcalendar CLI Executable bzw. Batch-Datei."""
    bin_name = shutil.which("gcalendar") or shutil.which("gcalendar.cmd")
    if bin_name:
        return bin_name
    wrapper = Path(os.environ.get("USERPROFILE", "")) / ".gemini/config/plugins/gcalendar/skills/gcalendar/scripts/gcalendar.cmd"
    if wrapper.exists():
        return str(wrapper)
    return None


def parse_ics_events(ics_text: str, start_date: dt.date, end_date: dt.date) -> list[dict[str, Any]]:
    """Parst Termine aus einem Standard-iCal/ICS-Feed (z. B. Google Calendar Secret iCal URL)."""
    events: list[dict[str, Any]] = []
    # Zeilenfortsetzungen (RFC 5545 unfolding) auflösen
    unfolded = re.sub(r"\r?\n[ \t]", "", ics_text)
    in_event = False
    cur: dict[str, str] = {}

    for raw_line in unfolded.splitlines():
        line = raw_line.strip()
        if line == "BEGIN:VEVENT":
            in_event = True
            cur = {}
        elif line == "END:VEVENT" and in_event:
            in_event = False
            dtstart = cur.get("DTSTART", "")
            dtend = cur.get("DTEND", "")
            summary = cur.get("SUMMARY", "Termin").replace("\\,", ",").replace("\\;", ";").replace("\\n", " ")
            loc = cur.get("LOCATION", "").replace("\\,", ",").replace("\\n", " ").strip()
            uid = cur.get("UID", "")

            if len(dtstart) >= 8 and dtstart[:8].isdigit():
                d_str = f"{dtstart[:4]}-{dtstart[4:6]}-{dtstart[6:8]}"
                try:
                    ev_date = dt.date.fromisoformat(d_str)
                except ValueError:
                    continue
                if start_date <= ev_date <= end_date:
                    st_time = None
                    en_time = None
                    if "T" in dtstart and len(dtstart) >= 13:
                        t_part = dtstart.split("T", 1)[1]
                        st_time = f"{t_part[:2]}:{t_part[2:4]}"
                    if "T" in dtend and len(dtend) >= 13:
                        t_part = dtend.split("T", 1)[1]
                        en_time = f"{t_part[:2]}:{t_part[2:4]}"
                    events.append({
                        "id": uid,
                        "title": summary,
                        "date": d_str,
                        "start_time": st_time,
                        "end_time": en_time,
                        "location": loc,
                        "link": "",
                        "source": "Google Kalender (ICS)",
                        "synced_to_gcal": True,
                    })
        elif in_event and ":" in line:
            k_part, val = line.split(":", 1)
            key = k_part.split(";", 1)[0].upper()
            cur[key] = val.strip()

    events.sort(key=lambda x: (x.get("date") or "", x.get("start_time") or "00:00"))
    return events


def create_google_calendar_event(ev: dict[str, Any]) -> str | None:
    """
    Erstellt einen Termin direkt im Google Kalender:
    1. Über die `gcalendar` CLI (wenn verfügbar / in Antigravity).
    2. Oder über ein Google Calendar OAuth Access Token (`GOOGLE_CALENDAR_ACCESS_TOKEN` in .env).
    Gibt den Google-Calendar-Link zurück, falls erfolgreich, sonst None.
    """
    title = ev.get("title") or "Termin"
    date_str = ev.get("date")
    if not date_str or str(date_str).lower() == "null" or len(str(date_str)) != 10:
        return None

    start_t = _normalize_hhmm(ev.get("start_time"))
    end_t = _normalize_hhmm(ev.get("end_time"))
    location = (ev.get("location") or "").strip()
    description = (ev.get("description") or "").strip()

    # Lokaler Offset (Europe/Berlin)
    local_offset = dt.datetime.now().astimezone().strftime("%z")
    tz_offset = f"{local_offset[:3]}:{local_offset[3:]}" if len(local_offset) == 5 else "+01:00"

    if start_t:
        if not end_t:
            end_hour = min(int(start_t[:2]) + 1, 23)
            end_t = f"{end_hour:02d}:{start_t[3:5]}"
        start_rfc = f"{date_str}T{start_t}:00{tz_offset}"
        end_rfc = f"{date_str}T{end_t}:00{tz_offset}"
        all_day = False
    else:
        try:
            next_day = (dt.date.fromisoformat(date_str) + dt.timedelta(days=1)).isoformat()
        except ValueError:
            return None
        start_rfc = date_str
        end_rfc = next_day
        all_day = True

    # Weg 1: gcalendar CLI (falls im PATH)
    gcal_bin = get_gcalendar_bin()
    if gcal_bin:
        cmd = [
            gcal_bin,
            "mutate",
            "create",
            "--summary",
            title,
            "--start",
            start_rfc,
            "--end",
            end_rfc,
            "--send-updates",
            "none",
        ]
        if all_day:
            cmd.append("--all-day")
        if description:
            cmd.extend(["--description", description])
        if location:
            cmd.extend(["--location", location])
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20, check=False)
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    if "Link:" in line or "https://www.google.com/calendar" in line:
                        return line.split("Link:", 1)[-1].strip()
                return "created-via-gcalendar"
        except Exception:
            pass

    # Weg 2: Direkter Google Calendar REST API Aufruf (falls GOOGLE_CALENDAR_ACCESS_TOKEN gesetzt ist)
    gcal_token = os.environ.get("GOOGLE_CALENDAR_ACCESS_TOKEN", "").strip()
    if gcal_token:
        cal_id = os.environ.get("GOOGLE_CALENDAR_ID", "primary").strip()
        body: dict[str, Any] = {
            "summary": title,
            "description": description,
            "location": location,
            "start": {"date": start_rfc} if all_day else {"dateTime": start_rfc},
            "end": {"date": end_rfc} if all_day else {"dateTime": end_rfc},
        }
        req = request.Request(
            f"https://www.googleapis.com/calendar/v3/calendars/{cal_id}/events?sendUpdates=none",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {gcal_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=20) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res.get("htmlLink") or "created-via-rest"
        except Exception as exc:
            print(f"[Google Calendar API] Hinweis: {exc}")

    return None


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

        # 1. Im Google Kalender eintragen + .ics Datei erzeugen
        ics_path = export_ics_event(ev)
        gcal_link = create_google_calendar_event(ev)
        if gcal_link:
            print(f"[Google Kalender] Termin eingetragen: '{title}' am {date_str}{time_info} -> {gcal_link}")
        else:
            print(f"[Kalender] Termin vorgemerkt & .ics erstellt ({ics_path.name}): '{title}' am {date_str}{time_info}")
            # Nur falls der Google Kalender gerade nicht direkt erreichbar war, als Erinnerung in Notion sichern
            if notion:
                notion.create_task_page(
                    tasks_db_id=tasks_db_id,
                    title=f"{title}{time_info}",
                    kind="Erinnerung",
                    priority="Hoch",
                    due_date=date_str,
                    next_step=step_text,
                    inbox_page_id=inbox_page_id,
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
            count += 1
            print(f"[Google Kalender] Nachträglich synchronisiert: '{ev.get('title')}' ({ev.get('date')}) -> {link}")
    return count


def fetch_google_calendar_events(days_ahead: int = 14) -> list[dict[str, Any]]:
    """
    Holt die tatsächlichen Termine aus dem verbundenen Google Kalender:
    1. Über die `gcalendar` CLI (speichert Ergebnisse zudem automatisch in `gcal_cache.json`).
    2. Über eine geheime iCal-URL (`GOOGLE_CALENDAR_ICS_URL` in .env oder `config.json`), falls gesetzt.
    3. Fallback auf den lokalen Kalender-Cache (`gcal_cache.json`), falls der Hintergrund-Task
       temporär keinen OAuth-Token hat.
    """
    events_out: list[dict[str, Any]] = []
    today = dt.date.today()
    end_date = today + dt.timedelta(days=days_ahead)
    start_iso = f"{today.isoformat()}T00:00:00Z"
    end_iso = f"{end_date.isoformat()}T23:59:59Z"

    # 1. Versuch: gcalendar CLI
    gcal_bin = get_gcalendar_bin()
    if gcal_bin:
        # Wir rufen immer 30 Tage ab, damit der Cache für den Hintergrundbetrieb reichhaltig gefüllt bleibt
        cache_end_iso = f"{(today + dt.timedelta(days=max(days_ahead, 30))).isoformat()}T23:59:59Z"
        cmd = [gcal_bin, "readonly", "events", "--start", start_iso, "--end", cache_end_iso, "--max", "100", "--json"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, check=False)
            if proc.returncode == 0 and proc.stdout.strip():
                raw_events = json.loads(proc.stdout)
                all_fetched: list[dict[str, Any]] = []
                for ev in raw_events:
                    summary = ev.get("summary") or "Termin"
                    start_obj = ev.get("start", {})
                    start_dt = start_obj.get("dateTime") or start_obj.get("date") or ""
                    end_obj = ev.get("end", {})
                    end_dt = end_obj.get("dateTime") or end_obj.get("date") or ""

                    date_part = start_dt[:10] if len(start_dt) >= 10 else ""
                    start_time = start_dt[11:16] if "T" in start_dt else None
                    end_time = end_dt[11:16] if "T" in end_dt else None
                    loc = (ev.get("location") or "").strip()
                    item = {
                        "id": ev.get("id"),
                        "title": summary,
                        "date": date_part,
                        "start_time": start_time,
                        "end_time": end_time,
                        "location": loc,
                        "link": ev.get("htmlLink", ""),
                        "source": "Google Kalender",
                        "synced_to_gcal": True,
                    }
                    all_fetched.append(item)
                    if today.isoformat() <= date_part <= end_date.isoformat():
                        events_out.append(item)

                # Cache aktualisieren für Hintergrundläufe
                try:
                    GCAL_CACHE_PATH.write_text(
                        json.dumps(
                            {
                                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
                                "events": all_fetched,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
                events_out.sort(key=lambda x: (x.get("date") or "", x.get("start_time") or "00:00"))
                return events_out
        except Exception as exc:
            print(f"[Google Kalender CLI Hinweis] {exc}")

    # 2. Versuch: Google Calendar Secret iCal URL (falls konfiguriert)
    ics_url = os.environ.get("GOOGLE_CALENDAR_ICS_URL", "").strip()
    if ics_url:
        try:
            req = request.Request(ics_url, headers={"User-Agent": "NotionLocalAssistant/1.0"})
            with request.urlopen(req, timeout=15) as resp:
                ics_text = resp.read().decode("utf-8", errors="replace")
            events_out = parse_ics_events(ics_text, today, end_date)
            if events_out:
                return events_out
        except Exception as exc:
            print(f"[Google Kalender ICS Hinweis] {exc}")

    # 3. Fallback: Lokaler Kalender-Cache (gcal_cache.json)
    if GCAL_CACHE_PATH.exists():
        try:
            cache_data = json.loads(GCAL_CACHE_PATH.read_text(encoding="utf-8"))
            for ev in cache_data.get("events", []):
                d_part = ev.get("date") or ""
                if today.isoformat() <= d_part <= end_date.isoformat():
                    events_out.append(ev)
        except Exception:
            pass

    events_out.sort(key=lambda x: (x.get("date") or "", x.get("start_time") or "00:00"))
    return events_out


def find_matching_google_calendar_event(
    query_title: str,
    query_date: str | None = None,
    days_ahead: int = 30,
) -> dict[str, Any] | None:
    """Sucht einen Termin im Google Kalender nach Titel und optional Datum."""
    events = fetch_google_calendar_events(days_ahead=days_ahead)
    if not events:
        return None

    clean_q = (query_title or "").strip().lower()
    if not clean_q:
        return None

    # 1. Datum & exakter/Teil-Titel
    date_filtered = [ev for ev in events if ev.get("date") == query_date] if query_date else events
    for ev in date_filtered:
        t = (ev.get("title") or "").strip().lower()
        if clean_q in t or t in clean_q:
            return ev

    # 2. Ohne Datumsteilsuche, falls 1 keinen Treffer ergab
    if query_date and date_filtered != events:
        for ev in events:
            t = (ev.get("title") or "").strip().lower()
            if clean_q in t or t in clean_q:
                return ev

    # 3. Wort-Überlappung (z. B. "Elterngespräch" vs "Elterngespräch Jasper")
    q_words = {w for w in clean_q.replace("-", " ").split() if len(w) >= 3}
    best_ev = None
    best_overlap = 0
    for ev in events:
        ev_words = {w for w in (ev.get("title") or "").lower().replace("-", " ").split() if len(w) >= 3}
        overlap = len(q_words & ev_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_ev = ev

    return best_ev


def update_google_calendar_event(
    event_id: str,
    new_date: str | None = None,
    new_start_time: str | None = None,
    new_end_time: str | None = None,
    new_title: str | None = None,
) -> bool:
    """Aktualisiert einen bestehenden Google-Kalendertermin über gcalendar mutate update."""
    gcal_bin = get_gcalendar_bin()
    if not gcal_bin or not event_id:
        return False

    cmd = [gcal_bin, "mutate", "update", event_id]

    local_offset = dt.datetime.now().astimezone().strftime("%z")
    tz_offset = f"{local_offset[:3]}:{local_offset[3:]}" if len(local_offset) == 5 else "+01:00"

    if new_date and new_start_time:
        start_t = _normalize_hhmm(new_start_time)
        end_t = _normalize_hhmm(new_end_time)
        if not end_t and start_t:
            end_hour = min(int(start_t[:2]) + 1, 23)
            end_t = f"{end_hour:02d}:{start_t[3:5]}"
        start_rfc = f"{new_date}T{start_t}:00{tz_offset}"
        end_rfc = f"{new_date}T{end_t}:00{tz_offset}"
        cmd.extend(["--start", start_rfc, "--end", end_rfc])
    elif new_date and not new_start_time:
        try:
            next_day = (dt.date.fromisoformat(new_date) + dt.timedelta(days=1)).isoformat()
        except ValueError:
            next_day = new_date
        cmd.extend(["--start", f"{new_date}T00:00:00{tz_offset}", "--end", f"{next_day}T00:00:00{tz_offset}"])

    if new_title:
        cmd.extend(["--summary", new_title])

    cmd.extend(["--send-updates", "none"])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
        return proc.returncode == 0
    except Exception as exc:
        print(f"[Google Kalender Update Fehler] {exc}")
        return False


def delete_google_calendar_event(event_id: str) -> bool:
    """Löscht einen Google-Kalendertermin über gcalendar mutate delete."""
    gcal_bin = get_gcalendar_bin()
    if not gcal_bin or not event_id:
        return False

    cmd = [gcal_bin, "mutate", "delete", event_id, "--send-updates", "none"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
        return proc.returncode == 0
    except Exception as exc:
        print(f"[Google Kalender Delete Fehler] {exc}")
        return False


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
        if notion:
            inbox_page_id = notion.create_inbox_page(
                inbox_db_id=cfg["notion"]["inbox_database_id"],
                title=f"Mail: {analysis['title']}",
                source=f"E-Mail ({sender})",
                raw_text=raw_text,
                summary=analysis["summary"],
                suggestions_text=suggestions_str,
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
        if notion:
            inbox_page_id = notion.create_inbox_page(
                inbox_db_id=cfg["notion"]["inbox_database_id"],
                title=analysis["title"],
                source=source,
                raw_text=raw_text,
                summary=analysis["summary"],
                suggestions_text=suggestions_str,
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

    ref_block = get_date_reference_block()
    prompt = f"""{ref_block}

Du bist der persönliche KI-Assistent & Co-Pilot von Stefan Kollmeier (Hochschuldozent, Videoproduzent 'Life on Pause', Familienmensch).
Stefan spricht oder schreibt dir über Telegram folgende Nachricht:
"{question}"

Hier ist sein aktueller Kontext (Live aus Google Kalender, Notion & lokalem Speicher):
- Echte Kalendertermine (Google Kalender für die nächsten 14 Tage): {json.dumps(all_cal, ensure_ascii=False)}
- Offene To-Dos (aus Notion Tasks): {json.dumps(open_todos, ensure_ascii=False)}
- Zuletzt erfasste Notizen: {json.dumps(recent_notes, ensure_ascii=False)}
- Relevante Notion-Seiten: {json.dumps(notion_knowledge, ensure_ascii=False)}

Rolle & Verhalten:
1. Antworte immer auf fehlerfreiem Deutsch, professionell, präzise, freundlich und auf den Punkt für einen Telegram-Chat.
2. Wenn Stefan nach Terminen, Plänen oder freier Zeit fragt, nutze die echten Google-Kalendertermine und berechne freie Zeitfenster.
3. Wenn er bittet, eine E-Mail, Nachricht oder einen Text vorzuformulieren, schreibe direkt einen druckreifen Entwurf.
4. Wenn er nach Aufgaben fragt, liste die relevanten To-Dos mit Priorität und Fälligkeit auf.
5. Verwende klare Formatierung (Aufzählungspunkte, Zeilenumbrüche).

Gib ein JSON-Objekt mit genau einem Feld "answer" (String) zurück:
{{"answer": "..."}}"""

    try:
        res = call_llm_json(prompt, cfg, model_override=model_override)
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

    try:
        return call_llm_json(prompt, cfg, model_override=model_override)
    except Exception as exc:
        print(f"[Intent Router Fallback] {exc}")
        if is_question_message(text):
            return {"intent": "GENERAL_CHAT", "query": text}
        return {"intent": "NOTE_OR_TASK"}


def process_telegram_updates(
    cfg: dict[str, Any],
    state: dict[str, Any],
    notion: NotionClient | None,
    model_override: str | None = None,
    poll_timeout: int = 5,
) -> int:
    """
    Bidirektionale Telegram-Steuerung (2 Richtungen):
    - Nimmt neue Sprach- und Textnotizen auf -> Whisper + Ollama -> Notion & Google Kalender
    - Beantwortet Fragen an den Assistenten (per Sprache oder Text)
    - Unterstützt Schnellbefehle: /briefing, /morgen, /abend, /todos, /ideen, /erledigt <Nr>
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not bot_token:
        return 0

    allowed_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    offset = int(state.get("telegram_offset", 0))
    api_base = f"https://api.telegram.org/bot{bot_token}"
    file_base = f"https://api.telegram.org/file/bot{bot_token}"

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
    count = 0

    for update in data.get("result", []):
        update_id = update["update_id"]
        state["telegram_offset"] = max(int(state.get("telegram_offset", 0)), update_id + 1)
        # WICHTIG: Offset sofort persistent speichern, damit niemals eine Nachricht mehrfach beantwortet wird!
        save_local_state(state_path, state)
        count += 1

        msg = update.get("message") or update.get("edited_message")
        if not msg:
            continue

        chat_id = str(msg.get("chat", {}).get("id", ""))
        if allowed_chat_id and chat_id != allowed_chat_id:
            print(f"[Telegram] Ignoriere Nachricht von unbekannter Chat-ID {chat_id}.")
            continue

        # Chat-ID automatisch für automatische Morgen-/Abend-Briefings merken
        if chat_id:
            state["last_telegram_chat_id"] = chat_id

        raw_text = ""
        source = "Telegram"

        # 1. Sprachnachricht (Voice) oder Audiodatei
        voice_or_audio = msg.get("voice") or msg.get("audio")
        if voice_or_audio:
            file_id = voice_or_audio["file_id"]
            try:
                f_req = request.Request(f"{api_base}/getFile?file_id={file_id}")
                with request.urlopen(f_req, timeout=15) as f_resp:
                    f_info = json.loads(f_resp.read().decode("utf-8"))
                file_path_remote = f_info["result"]["file_path"]
                ext = Path(file_path_remote).suffix or ".ogg"
                local_audio = processed_dir / f"tg_{update_id}{ext}"
                with request.urlopen(f"{file_base}/{file_path_remote}", timeout=30) as dl_resp:
                    local_audio.write_bytes(dl_resp.read())
                raw_text = transcribe_audio(local_audio, cfg)
                source = "Telegram"
            except Exception as exc:
                print(f"[Telegram] Fehler beim Laden der Sprachnachricht: {exc}")
                continue
        else:
            raw_text = (msg.get("text") or msg.get("caption") or "").strip()

        if not raw_text:
            continue

        cmd_lower = raw_text.lower().strip()

        # 2. Direkte Bot-Kommandos (Schnellzugriff)
        if cmd_lower.startswith(("/start", "/help", "/hilfe")):
            _send_telegram_message(
                api_base,
                chat_id,
                f"👋 **Hallo Stefan! Dein persönlicher KI-Assistent ist bereit.**\n\n"
                "Ich bin dein 100% lokaler Co-Pilot auf deinem Rechner (Ollama & Whisper) und helfe dir bei:\n\n"
                "📅 **Kalender (Google Kalender):**\n"
                "• Termine anlegen: *„Morgen 14 Uhr Zahnarzt eintragen“*\n"
                "• Termine ändern: *„Verschiebe den Termin X auf 16 Uhr“*\n"
                "• Termine absagen: *„Lösche den Termin Y“*\n"
                "• Freie Zeit abfragen: *„Habe ich heute Nachmittag 2 Stunden frei?“*\n\n"
                "📌 **Aufgaben & Projekte (Notion Tasks):**\n"
                "• Aufgaben anlegen: *„Neues To-Do: Video schneiden bis 3.11. Prio Hoch“*\n"
                "• Aufgaben anpassen: *„Verschiebe die Aufgabe Vorlesungsunterlagen auf Montag“*\n"
                "• Aufgaben abhaken: *„Habe das Video fertig geschnitten“* oder `/erledigt <Nr>`\n"
                "• Aufgaben löschen: *„Lösche die Aufgabe Testtask“*\n"
                "• Aufgaben filtern: *„Welche To-Dos sind heute fällig?“*\n\n"
                "🔍 **Wissen & Notizen (Notion AI Inbox & Docs):**\n"
                "• Sprach- oder Textnotiz senden -> wird automatisch ausgewertet & strukturiert in Notion abgelegt\n"
                "• Suche: *„Suche in Notion nach [Begriff]“* oder `/suche [Begriff]`\n\n"
                "🤖 **Persönlicher Dialog & Ratgeber:**\n"
                "• E-Mails/Nachrichten formulieren: *„Formuliere eine E-Mail an...“*\n"
                "• Brainstorming, Fragen & Tagesplanung\n\n"
                "⚡ **Schnellbefehle:**\n"
                "• `/morgen` oder `/briefing` – Dein Tages-Briefing\n"
                "• `/abend` – Deine Abend-Checkliste\n"
                "• `/kalender` – Terminvorschau (7 Tage)\n"
                "• `/todos` – Offene Aufgabenliste\n"
                "• `/ideen` – Proaktive Vorschläge\n"
                "• `/erledigt <Nr>` – Aufgabe direkt per Nummer abhaken\n"
                "• `/suche <Text>` – Notion durchsuchen\n"
                "• `/themen` – RSS-Themengebiete verwalten (/themen an|aus|neu|run)",
            )
            continue

        if cmd_lower.startswith(("/briefing", "/morgen")):
            briefing_msg = generate_briefing(cfg, state, notify=False, mode="morning")
            _send_telegram_message(api_base, chat_id, briefing_msg)
            continue

        if cmd_lower.startswith("/abend"):
            briefing_msg = generate_briefing(cfg, state, notify=False, mode="evening")
            _send_telegram_message(api_base, chat_id, briefing_msg)
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

            if not upcoming:
                _send_telegram_message(api_base, chat_id, "📅 In den nächsten 7 Tagen stehen keine Termine im Google Kalender an.")
            else:
                weekday_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
                lines = ["📅 **Deine Termine der nächsten 7 Tage (Google Kalender):**"]
                for e in upcoming:
                    d_str = e.get("date") or ""
                    try:
                        wd = weekday_names[dt.date.fromisoformat(d_str).weekday()]
                        d_label = f"{wd}, {d_str}"
                    except Exception:
                        d_label = d_str
                    t_str = f" {e['start_time']}–{e['end_time']} Uhr" if e.get("start_time") and e.get("end_time") else (f" um {e['start_time']} Uhr" if e.get("start_time") else " (Ganztägig)")
                    loc_str = f" 📍 {e['location']}" if e.get("location") else ""
                    lines.append(f"• **{d_label}**{t_str}: {e.get('title')}{loc_str}")
                _send_telegram_message(api_base, chat_id, "\n".join(lines))
            continue

        if cmd_lower.startswith("/todos"):
            open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
            if not open_todos:
                _send_telegram_message(api_base, chat_id, "✅ Du hast aktuell keine offenen To-Dos!")
            else:
                lines = ["📌 **Deine offenen To-Dos (Notion):**"]
                for i, t in enumerate(open_todos, 1):
                    due = f" [Fällig: {t['due_date']}]" if t.get("due_date") else ""
                    lines.append(f"{i}. [{t.get('priority', 'Mittel')}]{due} **{t.get('title')}**")
                lines.append("\nZum Abhaken einfach `/erledigt <Nummer>` senden oder per Sprache mitteilen!")
                _send_telegram_message(api_base, chat_id, "\n".join(lines))
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
                lines = ["💡 **Proaktive Vorschläge aus deinen letzten Notizen:**"]
                for i, s in enumerate(recent_sugs, 1):
                    lines.append(f"{i}. **{s.get('title')}**\n   ↳ {s.get('next_step', '')}")
                _send_telegram_message(api_base, chat_id, "\n".join(lines))
            continue

        if cmd_lower.startswith(("/mails", "/email", "/emails")):
            _send_telegram_message(api_base, chat_id, "📬 Prüfe markierte E-Mails (Label: Assistent)...")
            mail_count = process_emails(cfg, state, notion, model_override=model_override)
            if mail_count > 0:
                _send_telegram_message(api_base, chat_id, f"✅ {mail_count} neue E-Mail(s) ausgewertet und in Notion AI Inbox & Tasks übertragen.")
            else:
                _send_telegram_message(api_base, chat_id, "ℹ️ Keine neuen ungelesenen E-Mails mit dem Label 'Assistent' gefunden.")
            continue

        if cmd_lower.startswith(("/erledigt", "/done")):
            parts = raw_text.split(maxsplit=1)
            open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
            if len(parts) > 1 and parts[1].strip().isdigit():
                idx = int(parts[1].strip()) - 1
                if 0 <= idx < len(open_todos):
                    target_task = open_todos[idx]
                    target_task["status"] = "Erledigt"
                    n_page_id = target_task.get("notion_page_id")
                    if notion and n_page_id:
                        try:
                            notion.mark_task_done(n_page_id)
                        except Exception:
                            pass
                    _send_telegram_message(api_base, chat_id, f"✅ Als erledigt markiert:\n• **{target_task.get('title')}**")
                    count += 1
                    continue
            _send_telegram_message(api_base, chat_id, "Bitte Nummer angeben, z. B. `/erledigt 1` (Liste mit `/todos`).")
            continue

        if cmd_lower.startswith(("/suche", "/search")):
            parts = raw_text.split(maxsplit=1)
            search_query = parts[1].strip() if len(parts) > 1 else ""
            if not search_query:
                _send_telegram_message(api_base, chat_id, "Bitte Suchbegriff angeben, z. B. `/suche Life on Pause`.")
                continue
            notion_results = notion.search_workspace(search_query, limit=5) if notion else []
            matching_notes = [
                it for it in state.get("items", [])
                if search_query.lower() in (it.get("raw_text") or "").lower() or search_query.lower() in (it.get("analysis", {}).get("title") or "").lower()
            ]
            lines = [f"🔍 **Suchergebnisse für „{search_query}“:**\n"]
            if notion_results:
                lines.append("📄 **Notion-Seiten:**")
                for r in notion_results[:4]:
                    lines.append(f"• [{r['title']}]({r['url']})")
                lines.append("")
            if matching_notes:
                lines.append("🎙️ **Notizen-Verlauf:**")
                for n in matching_notes[:3]:
                    lines.append(f"• **{n.get('analysis', {}).get('title')}**: {n.get('analysis', {}).get('summary')}")
            if not notion_results and not matching_notes:
                lines.append("Keine Treffer in Notion oder Notizen gefunden.")
            _send_telegram_message(api_base, chat_id, "\n".join(lines))
            continue

        if cmd_lower in ("/run", "/sync", "run", "sync") or cmd_lower.startswith(("/run ", "/sync ")):
            _send_telegram_message(api_base, chat_id, "📡 RSS-Monitor gestartet. Neue Funde werden gleich gemeldet...")
            import threading
            def _async_rss():
                try:
                    import rss_monitor
                    import argparse as _ap
                    ns = _ap.Namespace(topic=None, dry_run=False, no_notify=False, no_notion=False, verbose=False)
                    rss_monitor.run(ns)
                except Exception as _e:
                    print(f"[RSS Telegram Run Error] {_e}")
            threading.Thread(target=_async_rss, daemon=True).start()
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
                    lines = [f"📡 **Deine RSS-Themengebiete ({len(topics)}):**\n"]
                    for t in topics:
                        st = "🟢" if t.get("enabled", True) else "⚪ (inaktiv)"
                        lines.append(f"{st} {t.get('emoji', '📌')} **{t['name']}** (`{t['id']}`)")
                        lines.append(f"   _{len(t.get('sources', []))} Quellen_")
                    lines.append("\n⚡ **Befehle:**")
                    lines.append("• `/themen an <id>` – Thema aktivieren")
                    lines.append("• `/themen aus <id>` – Thema pausieren")
                    lines.append("• `/themen neu <Name>` – Neues Thema per KI generieren")
                    lines.append("• `/themen run` – Jetzt sofort Feeds abfragen")
                    _send_telegram_message(api_base, chat_id, "\n".join(lines))
                    continue

                elif subcmd in ("an", "on", "enable", "aktivieren"):
                    if not arg:
                        _send_telegram_message(api_base, chat_id, "Bitte Themen-ID angeben, z. B. `/themen an parkinson`.")
                        continue
                    ok = rss_monitor.cmd_topic_set_enabled(arg, True)
                    if ok:
                        _send_telegram_message(api_base, chat_id, f"🟢 Thema `{arg}` wurde aktiviert.")
                    else:
                        _send_telegram_message(api_base, chat_id, f"❌ Thema `{arg}` nicht gefunden.")
                    continue

                elif subcmd in ("aus", "off", "disable", "deaktivieren"):
                    if not arg:
                        _send_telegram_message(api_base, chat_id, "Bitte Themen-ID angeben, z. B. `/themen aus parkinson`.")
                        continue
                    ok = rss_monitor.cmd_topic_set_enabled(arg, False)
                    if ok:
                        _send_telegram_message(api_base, chat_id, f"⚪ Thema `{arg}` wurde deaktiviert.")
                    else:
                        _send_telegram_message(api_base, chat_id, f"❌ Thema `{arg}` nicht gefunden.")
                    continue

                elif subcmd in ("neu", "add", "anlegen"):
                    new_name = arg or (parts[1] if len(parts) > 1 and parts[1] not in ("neu", "add") else "")
                    if not new_name:
                        _send_telegram_message(api_base, chat_id, "Bitte Namen des neuen Themas angeben, z. B. `/themen neu Multiple Sklerose`.")
                        continue
                    _send_telegram_message(api_base, chat_id, f"⏳ Generiere Feeds & Kriterien für „{new_name}“ per KI...")
                    import argparse as _ap
                    ns = _ap.Namespace(name=new_name, id=None, emoji=None, focus=None, rss=None, auto=True)
                    ok = rss_monitor.cmd_topic_add(ns)
                    if ok:
                        _send_telegram_message(api_base, chat_id, f"✅ Thema **{new_name}** wurde mit KI-generierten Quellen erfolgreich angelegt und aktiviert!")
                    else:
                        _send_telegram_message(api_base, chat_id, f"❌ Konnte Thema **{new_name}** nicht anlegen.")
                    continue

                elif subcmd in ("run", "sync", "start"):
                    _send_telegram_message(api_base, chat_id, "📡 RSS-Monitor gestartet. Neue Funde werden gleich gemeldet...")
                    import threading
                    def _async_rss():
                        try:
                            import argparse as _ap
                            ns = _ap.Namespace(topic=None, dry_run=False, no_notify=False, no_notion=False, verbose=False)
                            rss_monitor.run(ns)
                        except Exception as _e:
                            print(f"[RSS Telegram Run Error] {_e}")
                    threading.Thread(target=_async_rss, daemon=True).start()
                    continue

                else:
                    _send_telegram_message(api_base, chat_id, "Unbekannter Befehl. Nutze `/themen` für die Übersicht.")
                    continue
            except Exception as exc:
                _send_telegram_message(api_base, chat_id, f"Fehler bei /themen: {exc}")
                continue

        # 3. Umfassendes Intent-Routing via KI (Gemini Flash / lokales Modell)
        print(f"[Telegram Router] Analysiere Nachricht: '{raw_text[:60]}...'")
        intent_info = classify_intent(raw_text, cfg, model_override=model_override)
        intent = intent_info.get("intent", "NOTE_OR_TASK")
        print(f"[Telegram Router] Erkannter Intent: {intent}")

        # 0) FEEDBACK_OR_CORRECTION: Stefan korrigiert ein Missverständnis oder falschen Eintrag
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
            _send_telegram_message(api_base, chat_id, f"🫡 {reply}")
            count += 1
            continue

        # A) RESCHEDULE_EVENT: Bestehenden Termin im Google Kalender ändern
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
                        f"📅 **Termin erfolgreich verschoben:**\n"
                        f"• {matched['title']}\n"
                        f"• Neues Datum/Zeit: {target_date}{time_str}\n\n"
                        f"✅ Direkt im Google Kalender aktualisiert.",
                    )
                    count += 1
                    continue
                else:
                    _send_telegram_message(
                        api_base,
                        chat_id,
                        f"⚠️ Termin '{matched['title']}' gefunden, aber Google Kalender konnte nicht aktualisiert werden.",
                    )
                    count += 1
                    continue
            else:
                upcoming = fetch_google_calendar_events(days_ahead=7)
                upcoming_lines = "\n".join(f"• {e['title']} ({e['date']} {e.get('start_time') or ''})" for e in upcoming[:5])
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"⚠️ Ich konnte keinen passenden Termin zu '{ev_title}' im Kalender finden.\n\n"
                    f"Deine nächsten Termine:\n{upcoming_lines or 'Keine anstehenden Termine gefunden.'}",
                )
                count += 1
                continue

        # B) DELETE_EVENT: Termin im Google Kalender löschen / absagen
        if intent == "DELETE_EVENT":
            ev_title = intent_info.get("event_title") or ""
            target_date = intent_info.get("new_date")
            matched = find_matching_google_calendar_event(ev_title, target_date)
            if matched and matched.get("id"):
                ok = delete_google_calendar_event(matched["id"])
                if ok:
                    state["calendar_events"] = [
                        e for e in state.get("calendar_events", [])
                        if e.get("title") != matched.get("title")
                    ]
                    _send_telegram_message(
                        api_base,
                        chat_id,
                        f"🗑️ Termin **{matched['title']}** ({matched.get('date')}) wurde aus deinem Google Kalender gelöscht.",
                    )
                    count += 1
                    continue
            _send_telegram_message(
                api_base,
                chat_id,
                f"⚠️ Konnte keinen Termin zu '{ev_title}' im Google Kalender zum Löschen finden.",
            )
            count += 1
            continue

        # C) CREATE_EVENT: Expliziten neuen Kalendertermin anlegen
        if intent == "CREATE_EVENT":
            # Falls die Nachricht mehrere Termine, To-Dos oder Aufzählungen enthält, als umfassende Notiz verarbeiten!
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
                time_str = f" ({ev_start}–{ev_end} Uhr)" if ev_start and ev_end else (f" ({ev_start} Uhr)" if ev_start else " (Ganztägig)")
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"📅 **Neuer Termin eingetragen:**\n• **{ev_title}** am {ev_date}{time_str}\n\nIn Google Kalender & Notion Erinnerungen gesichert.",
                )
                count += 1
                continue

        # D) QUERY_CALENDAR: Frage zu Terminen, Zeitplan oder freier Zeit
        if intent == "QUERY_CALENDAR":
            print(f"[Telegram Kalender-Abfrage] Beantworte: '{raw_text}'...")
            answer = answer_question(raw_text, state, cfg, notion=notion, model_override=model_override)
            _send_telegram_message(api_base, chat_id, f"📅 {answer}")
            count += 1
            continue

        # E) UPDATE_TASK: Bestehende Aufgabe in Notion verschieben / Priorität ändern
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
                    f"📌 **Aufgabe aktualisiert:**\n"
                    f"• **{matched_task.get('title')}**\n"
                    f"• {due_text}\n"
                    f"• Priorität: {target_prio}\n\n"
                    f"✅ In Notion & lokal aktualisiert.",
                )
                count += 1
                continue
            else:
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"⚠️ Konnte keine offene Aufgabe zu '{task_q}' finden. Liste deine Aufgaben mit `/todos` auf.",
                )
                count += 1
                continue

        # F) DELETE_TASK: Aufgabe in Notion löschen / archivieren
        if intent == "DELETE_TASK":
            task_q = intent_info.get("task_title") or raw_text
            open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
            matched_task = find_matching_task(task_q, open_todos)
            if matched_task:
                if notion and matched_task.get("notion_page_id"):
                    try:
                        notion.archive_task(matched_task["notion_page_id"])
                    except Exception:
                        pass
                state["tasks"] = [t for t in state.get("tasks", []) if t != matched_task]
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"🗑️ Aufgabe **{matched_task.get('title')}** wurde aus Notion archiviert/gelöscht.",
                )
                count += 1
                continue
            else:
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"⚠️ Konnte keine offene Aufgabe zu '{task_q}' zum Löschen finden.",
                )
                count += 1
                continue

        # G) COMPLETE_TASK: Aufgabe / To-Do als erledigt markieren
        if intent == "COMPLETE_TASK":
            task_title_q = intent_info.get("task_title") or raw_text
            open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
            best_t = find_matching_task(task_title_q, open_todos)

            if best_t:
                best_t["status"] = "Erledigt"
                n_id = best_t.get("notion_page_id")
                if notion and n_id:
                    try:
                        notion.mark_task_done(n_id)
                    except Exception:
                        pass
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"✅ In Notion & lokal als erledigt markiert:\n• **{best_t.get('title')}**",
                )
                count += 1
                continue
            else:
                _send_telegram_message(
                    api_base,
                    chat_id,
                    f"⚠️ Konnte keine passende offene Aufgabe zu '{task_title_q}' finden (Übersicht mit `/todos`).",
                )
                count += 1
                continue

        # H) QUERY_TASKS: Nach Aufgaben filtern / fragen
        if intent == "QUERY_TASKS":
            print(f"[Telegram Aufgaben-Abfrage] Beantworte: '{raw_text}'...")
            answer = answer_question(raw_text, state, cfg, notion=notion, model_override=model_override)
            _send_telegram_message(api_base, chat_id, f"📌 {answer}")
            count += 1
            continue

        # I) SEARCH_KNOWLEDGE: Wissen in Notion durchsuchen
        if intent == "SEARCH_KNOWLEDGE":
            search_q = intent_info.get("query") or raw_text
            notion_results = notion.search_workspace(search_q, limit=5) if notion else []
            matching_notes = [
                it for it in state.get("items", [])
                if search_q.lower() in (it.get("raw_text") or "").lower() or search_q.lower() in (it.get("analysis", {}).get("title") or "").lower()
            ]
            lines = [f"🔍 **Suchergebnisse für „{search_q}“:**\n"]
            if notion_results:
                lines.append("📄 **Notion-Seiten:**")
                for r in notion_results[:4]:
                    lines.append(f"• [{r['title']}]({r['url']})")
                lines.append("")
            if matching_notes:
                lines.append("🎙️ **Notizen-Verlauf:**")
                for n in matching_notes[:3]:
                    lines.append(f"• **{n.get('analysis', {}).get('title')}**: {n.get('analysis', {}).get('summary')}")
            if not notion_results and not matching_notes:
                lines.append("Keine Treffer in Notion oder Notizen gefunden.")
            _send_telegram_message(api_base, chat_id, "\n".join(lines))
            count += 1
            continue

        # J) GENERAL_CHAT: Freie Frage, Formulierungshilfe, Brainstorming, Coaching
        if intent == "GENERAL_CHAT" or is_question_message(raw_text):
            q_clean = intent_info.get("query") or raw_text
            if cmd_lower.startswith(("/ask", "/frage")) and " " in raw_text:
                q_clean = raw_text.split(maxsplit=1)[1]
            print(f"[Telegram Dialog] Beantworte: '{q_clean}'...")
            answer = answer_question(q_clean, state, cfg, notion=notion, model_override=model_override)
            _send_telegram_message(api_base, chat_id, f"🤖 {answer}")
            count += 1
            continue

        # K) NOTE_OR_TASK (Default): Sprachnotiz / Textnotiz erfassen -> Notion AI Inbox & Tasks & Kalender
        print(f"[Telegram Notiz] Werte Nachricht aus: '{raw_text[:60]}...'")
        analysis = analyze_note(
            title="Telegram-Notiz",
            text=raw_text,
            cfg=cfg,
            model_override=model_override,
        )
        suggestions_str = format_suggestions_text(analysis["suggestions"])

        inbox_page_id: str | None = None
        if notion:
            inbox_page_id = notion.create_inbox_page(
                inbox_db_id=cfg["notion"]["inbox_database_id"],
                title=analysis["title"],
                source=source,
                raw_text=raw_text,
                summary=analysis["summary"],
                suggestions_text=suggestions_str,
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
                )
            # Hinweis: KI-Vorschläge werden NICHT als Tasks angelegt, um die Aufgabenliste nicht zu überladen.

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
        for todo in analysis["todos"]:
            state["tasks"].append({
                "kind": "To-Do",
                "title": todo.get("title", ""),
                "priority": todo.get("priority", "Mittel"),
                "due_date": todo.get("due_date"),
                "next_step": todo.get("next_step", ""),
                "source_title": analysis["title"],
                "status": "Nicht begonnen",
            })

        created_cal = handle_extracted_calendar_events(
            events=analysis.get("calendar_events", []),
            source_title=analysis["title"],
            state=state,
            notion=notion,
            tasks_db_id=cfg["notion"]["tasks_database_id"],
            inbox_page_id=inbox_page_id,
        )

        # Kurze Bestätigung direkt in Telegram zurücksenden
        todo_lines = "\n".join(f"• {t.get('title')} ({t.get('priority', 'Mittel')})" for t in analysis["todos"]) or "Keine direkten To-Dos"
        cal_lines = (
            "\n".join(
                f"• {c.get('title')} ({c.get('date')} {c.get('start_time') or 'Ganztägig'})"
                for c in created_cal
            )
            or "Keine Termine"
        )
        sug_lines = "\n".join(f"• {s.get('title')}" for s in analysis["suggestions"]) or "Keine"
        reply = (
            f"✅ Ausgewertet: {analysis['title']}\n\n"
            f"📝 {analysis['summary']}\n\n"
            f"📌 To-Dos:\n{todo_lines}\n\n"
            f"📅 Kalendereinträge:\n{cal_lines}\n\n"
            f"💡 Vorschläge:\n{sug_lines}"
        )
        _send_telegram_message(api_base, chat_id, reply)
        count += 1

    return count


def _send_telegram_message(api_base: str, chat_id: str, text: str) -> None:
    try:
        payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
        req = request.Request(
            f"{api_base}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        request.urlopen(req, timeout=10).read()
    except Exception:
        pass


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


def generate_briefing(
    cfg: dict[str, Any],
    state: dict[str, Any],
    notify: bool = False,
    mode: str = "morning",
    send_telegram: bool = False,
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

    if mode == "evening":
        header = f"🌙 Abend-Checkliste & Rückblick ({today_iso})"
    else:
        header = f"☀️ Guten Morgen Stefan! Dein Tages-Briefing ({today_iso})"

    lines = [header, ""]

    if mode == "evening":
        lines.append(f"🎙️ Heute neu erfasste Notizen ({len(today_items)}):")
        if today_items:
            for idx, it in enumerate(today_items, 1):
                a = it.get("analysis", {})
                lines.append(f"  {idx}. {a.get('title', 'Notiz')} – {a.get('summary', '')}")
        else:
            lines.append("  Heute wurden keine neuen Sprach-/Textnotizen erfasst.")
        lines.append("")

    # Heutige Termine
    if today_cal:
        lines.append(f"📅 Heutige Termine ({len(today_cal)}):")
        for idx, ev in enumerate(today_cal, 1):
            t_str = f" {ev['start_time']}–{ev.get('end_time') or ''} Uhr" if ev.get("start_time") else " (Ganztägig)"
            loc_str = f" [Ort: {ev['location']}]" if ev.get("location") else ""
            lines.append(f"  {idx}. [{t_str.strip()}]{loc_str} {ev.get('title')}")
    else:
        lines.append("📅 Heutige Termine: Keine Termine für heute im Google Kalender eingetragen.")

    # Vorschau auf nächste Tage
    if upcoming_cal:
        lines.append("")
        lines.append("🗓️ Vorschau auf anstehende Termine:")
        for idx, ev in enumerate(upcoming_cal[:4], 1):
            t_str = f" ({ev['start_time']} Uhr)" if ev.get("start_time") else ""
            lines.append(f"  • {ev.get('date')}{t_str}: {ev.get('title')}")

    lines.append("")
    lines.append(f"📌 Offene To-Dos ({len(open_todos)}):")
    if open_todos:
        for idx, t in enumerate(open_todos, 1):
            due = f" [Fällig: {t['due_date']}]" if t.get("due_date") else ""
            lines.append(f"  {idx}. [{t.get('priority', 'Mittel')}]{due} {t.get('title')}")
            if t.get("next_step"):
                lines.append(f"     -> Nächster Schritt: {t['next_step']}")
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
        lines.append(f"💡 KI-Impulse aus deinen letzten Notizen ({len(recent_suggestions)}):")
        for idx, s in enumerate(recent_suggestions, 1):
            step = f" -> {s['next_step']}" if s.get("next_step") else ""
            lines.append(f"  • **{s.get('title')}**{step}")

    if mode == "evening" and open_todos:
        lines.append("\nTipp: Antworte direkt hier im Chat mit `/erledigt <Nummer>`, um erledigte Aufgaben abzuhaken!")

    text = "\n".join(lines)
    if notify and (open_todos or suggestions or all_cal):
        top_focus = (today_cal[0]["title"] if today_cal else (open_todos[0]["title"] if open_todos else suggestions[0]["title"]))
        show_windows_notification(
            header,
            f"{len(today_cal)} Termine heute, {len(open_todos)} To-Dos. Fokus: {top_focus}",
        )

    if send_telegram:
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip() or str(state.get("last_telegram_chat_id", "")).strip()
        if bot_token and chat_id:
            api_base = f"https://api.telegram.org/bot{bot_token}"
            _send_telegram_message(api_base, chat_id, text)
            print(f"[Telegram] {mode.capitalize()}-Briefing an Chat-ID {chat_id} gesendet.")
        else:
            print("[Telegram] Hinweis: Für Telegram-Push bitte TELEGRAM_BOT_TOKEN in .env setzen und dem Bot einmal /start schreiben.")

    return text


def run_daemon(
    cfg: dict[str, Any],
    state_path: Path,
    notion: NotionClient | None,
    model_override: str | None = None,
) -> None:
    """
    Echtzeit-Hintergrunddienst (Daemon):
    - Long-Polling auf Telegram (Antwort in 1–2 Sekunden statt 15 Minuten Wartezeit)
    - Prüft regelmäßig lokale Dateien, Notion Inbox und markierte E-Mails
    """
    import time

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
    
    # Dedizierten RSS- & Wissensspeicher-Bot starten, falls konfiguriert
    if os.environ.get("TELEGRAM_RSSBOT_TOKEN", "").strip():
        try:
            import rss_bot
            rss_bot.start_in_background()
            print("[Daemon] Dedizierter RSS-Bot (@RSS8472Bot) erfolgreich im Hintergrund gestartet.")
        except Exception as _e:
            print(f"[Daemon] Hinweis: Konnte RSS-Bot nicht starten: {_e}")

    last_full_sync = 0.0

    try:
        while True:
            state = load_local_state(state_path)
            now = time.time()

            # Alle 5 Minuten: Offene Tasks aus Notion aktualisieren, Drop-Ordner, Mails & Notion Inbox prüfen
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
    parser = argparse.ArgumentParser(description="Lokaler Notion- & Sprachnotiz-Assistent (Ollama)")
    parser.add_argument(
        "command",
        choices=["run", "daemon", "briefing", "add", "sync-calendar"],
        nargs="?",
        default="run",
        help="run = Einmaliger Sync | daemon = Echtzeit-Hintergrunddienst | briefing = Morgen-/Abend-Briefing | add = schnelle Textnotiz | sync-calendar = Termine synchronisieren",
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
