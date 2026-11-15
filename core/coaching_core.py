"""
Gemeinsamer Kern für Coach-Bot und Elterncoaching-Bot:
- LPP- & Familien-Profilverwaltung (lokal in data/*.json + SQLite StateStore)
- LPP-Importer (.zip, .docx, .pdf, .txt/.md)
- Zielverwaltung (/ziele) + gecachte Notion-Kontext-Abfrage (About Stefan, Memories, Ziele, Tasks)
- Kalender-Belastungs- & Free-Slot-Analyse (inkl. Schutz für Regenerations-/Fokusblöcke)
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from core.storage import StateStore, atomic_write_json, safe_read_json
from integrations import calendar_sync

log = logging.getLogger("core.coaching_core")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
COACH_PROFILE_PATH = DATA_DIR / "coach_profile.json"
COACH_EXAMPLE_PATH = DATA_DIR / "coach_profile.example.json"
PARENT_PROFILE_PATH = DATA_DIR / "parent_profile.json"
PARENT_EXAMPLE_PATH = DATA_DIR / "parent_profile.example.json"


# --------------------------------------------------------------------------------------
# Profil-Verwaltung (Lokal .gitignore + SQLite-Sync)
# --------------------------------------------------------------------------------------

def _default_coach_profile() -> dict[str, Any]:
    if COACH_EXAMPLE_PATH.exists():
        data = safe_read_json(COACH_EXAMPLE_PATH, {})
        if isinstance(data, dict) and data:
            return data
    return {
        "name": "Stefan Kollmeier",
        "roles": ["Hochschuldozent & Teamleiter", "Videoproduzent ('Life on Pause')", "Familienmensch"],
        "lpp_summary": {
            "motives": ["Sicherheitsmotiv", "Kreativitätsmotiv"],
            "core_strengths": ["Empathie", "Selbständiges Denken", "Reflexionsvermögen", "Konfliktkompetenz", "Innovationskompetenz"],
            "development_areas": [],
            "ist_soll_gaps": [],
        },
        "coaching_preferences": {
            "tone": "Wertschätzend, präzise, bildhaft aber mit klarem nächstem Handlungsschritt.",
            "max_impulse_words": 230,
        },
        "custom_notes": [],
    }


def _default_parent_profile() -> dict[str, Any]:
    if PARENT_EXAMPLE_PATH.exists():
        data = safe_read_json(PARENT_EXAMPLE_PATH, {})
        if isinstance(data, dict) and data:
            return data
    return {
        "family_context": {
            "children_summary": "Alltag mit Stress-, Konflikt- und ADHS-Fokus (Details via /profil ergänzen).",
            "focus_areas": [
                "Umgang mit Stress & elterliche Co-Regulation",
                "Konflikte ruhig und klar begleiten",
                "ADHS im Familienalltag (Übergänge, Reizüberflutung, Struktur)",
            ],
            "working_strategies": [],
            "known_triggers": [],
        },
        "coaching_preferences": {
            "tone": "Warm, entlastend, konkret, neurodiversitäts-sensibel, sofort umsetzbar.",
            "bridge_to_parent_personality": "Hohe Empathie nutzen, eigene Anspannung früh regulieren, klare ruhige Grenzen statt Harmoniefalle.",
        },
        "custom_notes": [],
    }


def load_coach_profile(store: StateStore | None = None) -> dict[str, Any]:
    if COACH_PROFILE_PATH.exists():
        data = safe_read_json(COACH_PROFILE_PATH, None)
        if isinstance(data, dict) and data:
            if store is not None:
                try:
                    if not store.get_kv("coach_profile"):
                        store.set_kv("coach_profile", data)
                except Exception:
                    pass
            return data
    if store is not None:
        kv_data = store.get_kv("coach_profile")
        if isinstance(kv_data, dict) and kv_data:
            return kv_data
    return _default_coach_profile()


def save_coach_profile(profile: dict[str, Any], store: StateStore | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(COACH_PROFILE_PATH, profile, indent=2)
    if store is not None:
        store.set_kv("coach_profile", profile)


def append_coach_note(note: str, store: StateStore | None = None) -> dict[str, Any]:
    prof = load_coach_profile(store)
    notes = prof.setdefault("custom_notes", [])
    notes.append({"at": dt.datetime.now().isoformat(timespec="seconds"), "text": note.strip()[:800]})
    prof["custom_notes"] = notes[-25:]
    save_coach_profile(prof, store)
    return prof


def load_parent_profile(store: StateStore | None = None) -> dict[str, Any]:
    if PARENT_PROFILE_PATH.exists():
        data = safe_read_json(PARENT_PROFILE_PATH, None)
        if isinstance(data, dict) and data:
            if store is not None:
                try:
                    if not store.get_kv("parent_profile"):
                        store.set_kv("parent_profile", data)
                except Exception:
                    pass
            return data
    if store is not None:
        kv_data = store.get_kv("parent_profile")
        if isinstance(kv_data, dict) and kv_data:
            return kv_data
    return _default_parent_profile()


def save_parent_profile(profile: dict[str, Any], store: StateStore | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_json(PARENT_PROFILE_PATH, profile, indent=2)
    if store is not None:
        store.set_kv("parent_profile", profile)


def append_parent_note(note: str, store: StateStore | None = None) -> dict[str, Any]:
    prof = load_parent_profile(store)
    notes = prof.setdefault("custom_notes", [])
    notes.append({"at": dt.datetime.now().isoformat(timespec="seconds"), "text": note.strip()[:800]})
    prof["custom_notes"] = notes[-30:]
    save_parent_profile(prof, store)
    return prof


def format_coach_profile_context(profile: dict[str, Any]) -> str:
    lpp = profile.get("lpp_summary", {})
    lines = [
        f"Name: {profile.get('name', 'Stefan Kollmeier')}",
        "Rollen: " + ", ".join(profile.get("roles", [])),
    ]
    if lpp.get("motives"):
        lines.append("Zentrale Motive: " + " | ".join(lpp["motives"]))
    if lpp.get("core_strengths"):
        lines.append("Kernstärken: " + ", ".join(lpp["core_strengths"]))
    if lpp.get("big_five_highlights"):
        lines.append("Big-Five-Muster: " + " ; ".join(lpp["big_five_highlights"]))
    if lpp.get("development_areas"):
        lines.append("LPP-Handlungsfelder:")
        for da in lpp["development_areas"]:
            lines.append(f"  - {da.get('title')}: Risiken={da.get('risks', '')} | Interventionen={', '.join(da.get('interventions', [])[:3])}")
    if lpp.get("ist_soll_gaps"):
        lines.append("Wichtigste IST-SOLL-Lücken: " + " ; ".join(lpp["ist_soll_gaps"]))
    prefs = profile.get("coaching_preferences", {})
    if prefs.get("tone"):
        lines.append(f"Bevorzugter Coaching-Ton: {prefs['tone']}")
    notes = profile.get("custom_notes") or []
    if notes:
        lines.append("Aktuelle eigene Profil-Ergänzungen:")
        for n in notes[-6:]:
            lines.append(f"  - ({n.get('at', '')[:10]}) {n.get('text', '')}")
    return "\n".join(lines)


def format_parent_profile_context(profile: dict[str, Any]) -> str:
    fc = profile.get("family_context", {})
    prefs = profile.get("coaching_preferences", {})
    lines = [
        f"Familienkontext: {fc.get('children_summary', '')}",
        "Schwerpunkte: " + ", ".join(fc.get("focus_areas", [])),
    ]
    if fc.get("working_strategies"):
        lines.append("Bewährte Strategien: " + " | ".join(fc["working_strategies"]))
    if fc.get("known_triggers"):
        lines.append("Typische Reibungs-/Trigger-Zeiten: " + " | ".join(fc["known_triggers"]))
    if prefs.get("bridge_to_parent_personality"):
        lines.append(f"Verbindung zu Stefans Persönlichkeitsprofil: {prefs['bridge_to_parent_personality']}")
    if prefs.get("tone"):
        lines.append(f"Gewünschte Tonalität: {prefs['tone']}")
    notes = profile.get("custom_notes") or []
    if notes:
        lines.append("Aktuelle Familien-Notizen & Updates:")
        for n in notes[-8:]:
            lines.append(f"  - ({n.get('at', '')[:10]}) {n.get('text', '')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# LPP-Importer (.zip, .docx, .pdf, .txt/.md)
# --------------------------------------------------------------------------------------

def extract_text_from_docx(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path) as zf:
        xml_data = zf.read("word/document.xml")
    root = ET.fromstring(xml_data)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paras: list[str] = []
    for p in root.findall(".//w:p", ns):
        txt = "".join(t.text for t in p.findall(".//w:t", ns) if t.text).strip()
        if txt:
            paras.append(txt)
    return "\n".join(paras)


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        proc = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=25,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    except Exception:
        return ""


def extract_text_from_lpp_path(input_path: Path) -> list[tuple[str, str]]:
    """Extrahiert (Dateiname, Text) aus einer Einzeldatei, einem Ordner oder einer ZIP-Datei."""
    extracted: list[tuple[str, str]] = []
    if not input_path.exists():
        return extracted

    if input_path.is_dir():
        for p in sorted(input_path.rglob("*")):
            if p.is_file() and p.suffix.lower() in (".docx", ".pdf", ".txt", ".md"):
                extracted.extend(extract_text_from_lpp_path(p))
        return extracted

    ext = input_path.suffix.lower()
    if ext == ".zip":
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(input_path) as zf:
                zf.extractall(tmpdir)
            return extract_text_from_lpp_path(Path(tmpdir))
    if ext == ".docx":
        txt = extract_text_from_docx(input_path)
        if txt:
            extracted.append((input_path.name, txt))
    elif ext == ".pdf":
        txt = extract_text_from_pdf(input_path)
        if txt:
            extracted.append((input_path.name, txt))
    elif ext in (".txt", ".md"):
        txt = input_path.read_text(encoding="utf-8", errors="replace").strip()
        if txt:
            extracted.append((input_path.name, txt))
    return extracted


def import_lpp_into_profile(
    input_path: Path,
    cfg: dict[str, Any],
    store: StateStore | None = None,
    call_llm_json_fn: Any = None,
) -> tuple[dict[str, Any], int]:
    """
    Liest LPP-Dokumente (.zip/.docx/.pdf/.txt) ein und verdichtet sie in coach_profile.json.
    Gibt (aktualisiertes_profil, anzahl_verarbeitete_dateien) zurück.
    """
    docs = extract_text_from_lpp_path(input_path)
    if not docs:
        return load_coach_profile(store), 0

    prof = load_coach_profile(store)
    sources = [name for name, _ in docs]
    combined = "\n\n".join(f"=== {name} ===\n{text[:4500]}" for name, text in docs[:8])

    if call_llm_json_fn is not None:
        prompt = f"""Verdichte die folgenden LPP-/Feedback-Unterlagen zu einem strukturierten Coaching-Profil auf Deutsch.
Antworte AUSSCHLIESSLICH mit einem JSON-Objekt mit diesen Feldern:
{{
  "motives": ["..."],
  "core_strengths": ["..."],
  "big_five_highlights": ["..."],
  "development_areas": [
    {{"id": "kurz_id", "title": "...", "risks": "...", "interventions": ["...", "..."]}}
  ],
  "ist_soll_gaps": ["..."]
}}

Unterlagen:
{combined[:18000]}"""
        try:
            parsed = call_llm_json_fn(prompt, cfg, task="coach_deep")
            if isinstance(parsed, dict) and (parsed.get("core_strengths") or parsed.get("development_areas")):
                lpp = prof.setdefault("lpp_summary", {})
                lpp["sources"] = sources
                lpp["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
                for k in ("motives", "core_strengths", "big_five_highlights", "development_areas", "ist_soll_gaps"):
                    if parsed.get(k):
                        lpp[k] = parsed[k]
                save_coach_profile(prof, store)
                return prof, len(docs)
        except Exception as exc:
            log.warning("LLM-LPP-Synthese fehlgeschlagen (%s) – nutze deterministischen Fallback.", exc)

    # Deterministischer Fallback: Quellen + IST/SOLL-Gaps extrahieren
    gaps: list[str] = []
    for _, txt in docs:
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        for i, l in enumerate(lines):
            if l.startswith("IST:") and i > 0 and i + 1 < len(lines):
                entry = f"{lines[i-1]}: {l} | {lines[i+1]}"
                if entry not in gaps:
                    gaps.append(entry)
    lpp = prof.setdefault("lpp_summary", {})
    lpp["sources"] = sources
    lpp["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    if gaps:
        lpp["ist_soll_gaps"] = gaps[:12]
    save_coach_profile(prof, store)
    return prof, len(docs)


# --------------------------------------------------------------------------------------
# Ziele-Verwaltung (SQLite + Notion-Kontext)
# --------------------------------------------------------------------------------------

def list_goals(store: StateStore, include_done: bool = False) -> list[dict[str, Any]]:
    goals = store.get_kv("coach_goals", []) or []
    if not isinstance(goals, list):
        return []
    if include_done:
        return goals
    return [g for g in goals if g.get("status") != "done"]


def add_goal(store: StateStore, title: str, area: str = "Allgemein", horizon: str = "Quartal") -> dict[str, Any]:
    goals = store.get_kv("coach_goals", []) or []
    if not isinstance(goals, list):
        goals = []
    next_id = max((int(g.get("id", 0)) for g in goals), default=0) + 1
    goal = {
        "id": next_id,
        "title": title.strip()[:180],
        "area": area,
        "horizon": horizon,
        "status": "active",
        "created_at": dt.date.today().isoformat(),
    }
    goals.append(goal)
    store.set_kv("coach_goals", goals)
    return goal


def complete_goal(store: StateStore, index_or_query: str) -> dict[str, Any] | None:
    goals = store.get_kv("coach_goals", []) or []
    active = [g for g in goals if g.get("status") != "done"]
    q = index_or_query.strip()
    target = None
    if q.isdigit():
        idx = int(q) - 1
        if 0 <= idx < len(active):
            target = active[idx]
    if target is None:
        ql = q.lower()
        for g in active:
            if ql in (g.get("title") or "").lower():
                target = g
                break
    if target is None:
        return None
    target["status"] = "done"
    target["completed_at"] = dt.date.today().isoformat()
    store.set_kv("coach_goals", goals)
    return target


def delete_goal(store: StateStore, index_or_query: str) -> dict[str, Any] | None:
    goals = store.get_kv("coach_goals", []) or []
    active = [g for g in goals if g.get("status") != "done"]
    q = index_or_query.strip()
    target = None
    if q.isdigit():
        idx = int(q) - 1
        if 0 <= idx < len(active):
            target = active[idx]
    if target is None:
        ql = q.lower()
        for g in active:
            if ql in (g.get("title") or "").lower():
                target = g
                break
    if target is None:
        return None
    goals = [g for g in goals if g is not target]
    store.set_kv("coach_goals", goals)
    return target


def fetch_notion_coaching_context(
    cfg: dict[str, Any],
    notion: Any,
    store: StateStore,
    force_refresh: bool = False,
    cache_ttl_seconds: int = 1800,
) -> dict[str, Any]:
    """
    Lädt gecachten Kontext aus Notion für den Coach-Bot:
    - About Stefan (about_page_id)
    - Optionale Ziele-Seite / -Datenbank (goals_page_id / goals_database_id oder Workspace-Suche 'Ziele')
    - Memories DB (letzte Einträge)
    - Offene Tasks (inkl. mehrfach verschobene Aufgaben)
    """
    cache_key = "coach_notion_ctx_cache"
    if not force_refresh:
        cached = store.get_kv(cache_key)
        if isinstance(cached, dict):
            ts = float(cached.get("fetched_ts", 0))
            if (dt.datetime.now().timestamp() - ts) < cache_ttl_seconds:
                return cached.get("data", {})

    ctx: dict[str, Any] = {
        "about_text": "",
        "goals_text": "",
        "memories": [],
        "open_tasks": [],
        "deferred_tasks": [],
    }
    if notion is None:
        return ctx

    notion_cfg = cfg.get("notion", {})
    about_id = (notion_cfg.get("about_page_id") or "").strip()
    if about_id:
        try:
            about = notion.get_page_summary(about_id, max_blocks=60)
            ctx["about_text"] = (about.get("text") or "")[:2200]
        except Exception as exc:
            log.debug("About-Seite konnte nicht geladen werden: %s", exc)

    goals_page_id = (notion_cfg.get("goals_page_id") or "").strip()
    goals_db_id = (notion_cfg.get("goals_database_id") or "").strip()
    goals_parts: list[str] = []
    if goals_page_id:
        try:
            gp = notion.get_page_summary(goals_page_id, max_blocks=60)
            if gp.get("text"):
                goals_parts.append(f"[{gp.get('title', 'Ziele')}]\n{gp['text'][:1800]}")
        except Exception as exc:
            log.debug("Goals-Seite konnte nicht geladen werden: %s", exc)
    if goals_db_id:
        try:
            g_items = notion.query_database_items(goals_db_id, page_size=15)
            for gi in g_items:
                goals_parts.append(f"• {gi.get('title')} ({gi.get('details', '')})")
        except Exception as exc:
            log.debug("Goals-DB konnte nicht geladen werden: %s", exc)
    if not goals_parts:
        try:
            hits = notion.search_workspace("Ziele", limit=3)
            for h in hits[:2]:
                if h.get("type") == "page" and h.get("id"):
                    gp = notion.get_page_summary(h["id"], max_blocks=35)
                    if gp.get("text"):
                        goals_parts.append(f"[{gp.get('title', 'Ziele')}]\n{gp['text'][:900]}")
        except Exception:
            pass
    ctx["goals_text"] = "\n".join(goals_parts)[:2500]

    mem_db_id = (notion_cfg.get("memories_database_id") or "").strip()
    if mem_db_id:
        try:
            mems = notion.query_database_items(mem_db_id, page_size=10)
            ctx["memories"] = [
                f"{m.get('title')}: {m.get('details', '')}"[:220]
                for m in mems[:8]
            ]
        except Exception as exc:
            log.debug("Memories-DB konnte nicht geladen werden: %s", exc)

    tasks_db_id = (notion_cfg.get("tasks_database_id") or "").strip()
    if tasks_db_id:
        try:
            tasks = notion.query_open_tasks(tasks_db_id)
            todos = [t for t in tasks if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
            ctx["open_tasks"] = [
                {
                    "title": t.get("title", ""),
                    "priority": t.get("priority", "Mittel"),
                    "due_date": t.get("due_date"),
                    "area": t.get("area", ""),
                    "defer_count": store.get_task_defer_count(t.get("notion_page_id") or t.get("title", "")),
                }
                for t in todos[:15]
            ]
            ctx["deferred_tasks"] = [
                t["title"] for t in ctx["open_tasks"] if t["defer_count"] >= 2
            ][:5]
        except Exception as exc:
            log.debug("Tasks-DB konnte nicht geladen werden: %s", exc)

    store.set_kv(cache_key, {"fetched_ts": dt.datetime.now().timestamp(), "data": ctx})
    return ctx


def fetch_day_load_context(cfg: dict[str, Any], store: StateStore) -> dict[str, Any]:
    """Berechnet heutige Terminlast & freie Zeitfenster (inkl. anonymer Arbeitskalender-Blocker)."""
    today_iso = dt.date.today().isoformat()
    try:
        priv = calendar_sync.fetch_google_calendar_events(days_ahead=2)
    except Exception:
        priv = []
    try:
        work = calendar_sync.fetch_work_busy_slots(days_ahead=2, store=store)
    except Exception:
        work = []

    today_priv = [e for e in priv if e.get("date") == today_iso]
    today_work = [w for w in work if w.get("date") == today_iso]

    work_hours = cfg.get("work_hours", {})
    work_start = os.environ.get("WORK_HOURS_START") or work_hours.get("start", "08:00")
    work_end = os.environ.get("WORK_HOURS_END") or work_hours.get("end", "20:00")
    free_slots = calendar_sync.compute_free_slots(
        priv + work,
        target_date=today_iso,
        work_start=work_start,
        work_end=work_end,
        min_slot_minutes=30,
    )
    return {
        "date": today_iso,
        "private_events_today": [
            f"{e.get('start_time') or 'Ganztägig'} {e.get('title', '')}".strip()
            for e in today_priv[:8]
        ],
        "work_busy_count": len(today_work),
        "free_slots": free_slots[:5],
    }


def book_coaching_slot(
    cfg: dict[str, Any],
    store: StateStore,
    title: str,
    duration_minutes: int = 30,
    description: str = "Geschützter Fokus-/Regenerationsblock via Coach-Bot",
) -> dict[str, Any] | None:
    """Findet heute (oder morgen) ein passendes freies Zeitfenster und trägt einen Block im Kalender ein."""
    now_local = dt.datetime.now()
    for day_offset in (0, 1):
        target_date = (dt.date.today() + dt.timedelta(days=day_offset)).isoformat()
        try:
            priv = calendar_sync.fetch_google_calendar_events(days_ahead=2)
        except Exception:
            priv = []
        try:
            work = calendar_sync.fetch_work_busy_slots(days_ahead=2, store=store)
        except Exception:
            work = []
        work_hours = cfg.get("work_hours", {})
        work_start = os.environ.get("WORK_HOURS_START") or work_hours.get("start", "08:00")
        work_end = os.environ.get("WORK_HOURS_END") or work_hours.get("end", "20:00")
        free = calendar_sync.compute_free_slots(
            priv + work,
            target_date=target_date,
            work_start=work_start,
            work_end=work_end,
            min_slot_minutes=duration_minutes,
        )
        for sl in free:
            st_str = sl["start_time"]
            if day_offset == 0:
                hh, mm = map(int, st_str.split(":"))
                slot_dt = now_local.replace(hour=hh, minute=mm, second=0, microsecond=0)
                if slot_dt < now_local + dt.timedelta(minutes=15):
                    # Slot innerhalb desselben Blocks nach hinten schieben falls noch genug Zeit bleibt
                    shifted = now_local + dt.timedelta(minutes=15)
                    shifted_min = shifted.hour * 60 + shifted.minute
                    eh, em = map(int, sl["end_time"].split(":"))
                    end_min = eh * 60 + em
                    if end_min - shifted_min < duration_minutes:
                        continue
                    st_str = f"{shifted_min // 60:02d}:{shifted_min % 60:02d}"
            sh, sm = map(int, st_str.split(":"))
            end_tot = sh * 60 + sm + duration_minutes
            end_str = f"{end_tot // 60:02d}:{end_tot % 60:02d}"
            ev = {
                "title": title[:80],
                "date": target_date,
                "start_time": st_str,
                "end_time": end_str,
                "location": "",
                "description": description,
            }
            link = calendar_sync.create_google_calendar_event(ev)
            if not link:
                calendar_sync.export_ics_event(ev)
            ev["gcal_link"] = link
            return ev
    return None
