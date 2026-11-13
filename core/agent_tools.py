"""
Agenten-Kern: Multi-Tool Function Calling (unter StepBudget), Multi-Turn-Memory,
Multimodalität (Fotos/PDFs), Pre-Meeting-Briefings, Undo & Weekly Review.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import chat_guard
import llm_core
from core.storage import StateStore
from integrations import calendar_sync, telegram_ui

log = logging.getLogger("core.agent_tools")

AREAS = [
    "Hochschule/Lehre",
    "Life on Pause",
    "Gesundheit/Forschung",
    "Familie/Privat",
    "Finanzen/Orga",
]


def detect_area(text: str) -> str:
    """Schnelle deterministische Bereichs-Heuristik als Basis/Fallback."""
    t = (text or "").lower()
    if any(k in t for k in ("vorlesung", "studier", "dozent", "klausur", "prüfung", "hochschule", "hs-rm", "seminar", "lehre", "kapitel")):
        return "Hochschule/Lehre"
    if any(k in t for k in ("life on pause", "youtube", "video", "schneiden", "dreh", "thumbnail", "skript", "kamera", "podcast")):
        return "Life on Pause"
    if any(k in t for k in ("parkinson", "long covid", "me/cfs", "arzt", "klinik", "studie", "therapie", "blut", "rezept", "medikament")):
        return "Gesundheit/Forschung"
    if any(k in t for k in ("steuer", "überweisung", "etf", "bank", "vertrag", "versicherung", "rechnung", "finanz")):
        return "Finanzen/Orga"
    return "Familie/Privat"


def _build_gemini_tool_declarations():
    """Definiert die Function-Calling-Tools für Gemini."""
    from google.genai import types

    decls = [
        types.FunctionDeclaration(
            name="create_calendar_event",
            description="Erstellt einen neuen Termin im Google Kalender (z. B. Zahnarzt, Treffen, Fokusblock).",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING", "description": "Kurzer Titel des Termins"},
                    "date": {"type": "STRING", "description": "Datum im Format YYYY-MM-DD"},
                    "start_time": {"type": "STRING", "description": "Startzeit HH:MM (24h) oder null für ganztägig"},
                    "end_time": {"type": "STRING", "description": "Endzeit HH:MM (24h) oder null"},
                    "location": {"type": "STRING", "description": "Optionaler Ort/Raum"},
                },
                "required": ["title", "date"],
            },
        ),
        types.FunctionDeclaration(
            name="reschedule_calendar_event",
            description="Verschiebt oder ändert einen bestehenden Termin im Google Kalender.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "event_title": {"type": "STRING", "description": "Name oder Stichwort des Termins"},
                    "new_date": {"type": "STRING", "description": "Neues Datum YYYY-MM-DD"},
                    "new_start_time": {"type": "STRING", "description": "Neue Startzeit HH:MM"},
                    "new_end_time": {"type": "STRING", "description": "Neue Endzeit HH:MM"},
                },
                "required": ["event_title"],
            },
        ),
        types.FunctionDeclaration(
            name="delete_calendar_event",
            description="Löscht einen bestehenden Termin aus dem Google Kalender.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "event_title": {"type": "STRING", "description": "Name oder Stichwort des Termins"},
                    "date": {"type": "STRING", "description": "Optionales Datum YYYY-MM-DD"},
                },
                "required": ["event_title"],
            },
        ),
        types.FunctionDeclaration(
            name="get_calendar_and_free_slots",
            description="Liest anstehende Termine und berechnet freie Zeitfenster (berücksichtigt auch anonyme Arbeits-Belegtzeiten).",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "date": {"type": "STRING", "description": "Zieldatum YYYY-MM-DD (Standard: heute)"},
                    "days_ahead": {"type": "INTEGER", "description": "Anzahl Tage Vorschau (1 bis 14)"},
                    "min_minutes": {"type": "INTEGER", "description": "Mindestdauer eines freien Slots in Minuten (Standard 30)"},
                },
            },
        ),
        types.FunctionDeclaration(
            name="create_task",
            description="Legt eine neue Aufgabe (To-Do) in Notion Tasks an.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING", "description": "Aufgabentitel"},
                    "priority": {"type": "STRING", "description": "Hoch, Mittel oder Niedrig"},
                    "due_date": {"type": "STRING", "description": "Fälligkeitsdatum YYYY-MM-DD oder null"},
                    "next_step": {"type": "STRING", "description": "Erster konkreter Schritt"},
                    "area": {"type": "STRING", "description": "Bereich: Hochschule/Lehre, Life on Pause, Gesundheit/Forschung, Familie/Privat oder Finanzen/Orga"},
                    "estimated_minutes": {"type": "INTEGER", "description": "Geschätzte Dauer in Minuten (z. B. 30, 60)"},
                },
                "required": ["title"],
            },
        ),
        types.FunctionDeclaration(
            name="complete_task",
            description="Markiert eine oder mehrere offene Aufgaben in Notion als erledigt.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "task_query": {"type": "STRING", "description": "Aufgabenname, Stichwort oder Nummern wie '1, 3'"},
                },
                "required": ["task_query"],
            },
        ),
        types.FunctionDeclaration(
            name="update_task",
            description="Ändert Fälligkeitsdatum oder Priorität einer bestehenden Notion-Aufgabe.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "task_query": {"type": "STRING", "description": "Name oder Stichwort der Aufgabe"},
                    "due_date": {"type": "STRING", "description": "Neues Datum YYYY-MM-DD"},
                    "priority": {"type": "STRING", "description": "Hoch, Mittel oder Niedrig"},
                },
                "required": ["task_query"],
            },
        ),
        types.FunctionDeclaration(
            name="delete_task",
            description="Archiviert/löscht eine bestehende Aufgabe in Notion.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "task_query": {"type": "STRING", "description": "Name oder Stichwort der Aufgabe"},
                },
                "required": ["task_query"],
            },
        ),
        types.FunctionDeclaration(
            name="list_open_tasks",
            description="Listet offene To-Dos aus Notion Tasks auf (optional gefiltert nach Bereich oder Priorität).",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "area": {"type": "STRING", "description": "Optionaler Bereichsfilter"},
                    "priority": {"type": "STRING", "description": "Optionaler Prioritätsfilter (Hoch, Mittel, Niedrig)"},
                },
            },
        ),
        types.FunctionDeclaration(
            name="search_knowledge",
            description="Durchsucht Notion-Seiten, frühere Notizen und den RSS-Wissensspeicher nach einem Begriff.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {"type": "STRING", "description": "Suchbegriff oder Thema"},
                },
                "required": ["query"],
            },
        ),
        types.FunctionDeclaration(
            name="save_inbox_note",
            description="Speichert eine allgemeine Notiz, Idee oder einen Gedanken in Notion AI Inbox.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING", "description": "Prägnanter Titel"},
                    "summary": {"type": "STRING", "description": "Kurze Zusammenfassung"},
                    "raw_text": {"type": "STRING", "description": "Originalinhalt der Notiz"},
                    "area": {"type": "STRING", "description": "Lebensbereich"},
                },
                "required": ["title", "summary", "raw_text"],
            },
        ),
    ]
    return [types.Tool(function_declarations=decls)]


def run_tool_agent(
    user_text: str,
    chat_id: str,
    cfg: dict[str, Any],
    state: dict[str, Any],
    notion: Any,
    store: StateStore,
    date_ref_block: str,
    find_matching_task_fn: Callable[[str, list[dict[str, Any]]], dict[str, Any] | None],
    resolve_tasks_fn: Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]],
    find_gcal_event_fn: Callable[[str, str | None], dict[str, Any] | None],
    handle_cal_events_fn: Callable[..., list[dict[str, Any]]],
    save_state_fn: Callable[[], None],
) -> tuple[str, bool] | None:
    """
    Führt Multi-Tool Function Calling mit Gemini unter hartem `StepBudget` aus.
    Gibt `(reply_html, has_undoable_action)` zurück, oder `None` für Fallback auf klassischen Router.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    provider = cfg.get("llm", {}).get("provider", "gemini" if gemini_key else "ollama")
    if provider != "gemini" or not gemini_key:
        return None

    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None

    is_complex = any(k in user_text.lower() for k in ("/analyse", "analysiere", "strategie", "komplex", "plane meine woche"))
    task_tier = "chat_complex" if is_complex else "chat_simple"
    model_name, thinking_level = llm_core.resolve_task_model(task_tier, cfg)

    chat_cfg = cfg.get("chat", {})
    budget = chat_guard.StepBudget(
        max_iterations=int(chat_cfg.get("max_iterations", 4)),
        max_estimated_tokens=int(chat_cfg.get("max_input_tokens", 14000)),
    )

    history = store.get_recent_chat_turns("main", chat_id, limit=6)
    history_lines = "\n".join(f"{h['role'].upper()}: {h['content'][:350]}" for h in history)

    system_instr = (
        f"{date_ref_block}\n\n"
        "Du bist der persönliche KI-Assistent von Stefan Kollmeier (Hochschuldozent, "
        "Videoproduzent 'Life on Pause', Familienmensch) auf Telegram.\n"
        "Du hast Zugriff auf Werkzeuge für Google Kalender, anonyme Arbeits-Frei/Belegt-Slots, "
        "Notion Tasks, Notion AI Inbox und den Wissensspeicher.\n\n"
        "REGELN:\n"
        "1. Nutze Werkzeuge aktiv, auch mehrere hintereinander in einem Turn, wenn Stefan mehrere Dinge in einer Nachricht nennt.\n"
        "2. Wenn sich Stefan über einen früheren Fehler beschwert ('Das war falsch...'), korrigiere direkt per Tool statt ein neues To-Do anzulegen.\n"
        "3. Wenn Stefan allgemeine Gedanken/Notizen schickt (ohne reine Frage), speichere per save_inbox_note und extrahiere ggf. konkrete To-Dos/Termine.\n"
        "4. Dupliziere NIEMALS dieselbe Sache als Termin UND als To-Do.\n"
        "5. Bei freien Slots berücksichtige immer auch die zurückgegebenen Arbeits-Belegtzeiten.\n"
        "6. Antworte kompakt, freundlich und auf fehlerfreiem Deutsch für Telegram (ohne Markdown-Codeblöcke um die ganze Antwort)."
    )

    contents: list[Any] = []
    if history_lines:
        contents.append(f"Bisheriger Kurzzeit-Chatverlauf:\n{history_lines}\n\nAktuelle Nachricht von Stefan:\n{user_text}")
    else:
        contents.append(user_text)

    tools = _build_gemini_tool_declarations()
    client = genai.Client(api_key=gemini_key)
    undo_pushed = False

    def _exec_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        nonlocal undo_pushed
        tasks_db_id = cfg["notion"]["tasks_database_id"]
        inbox_db_id = cfg["notion"]["inbox_database_id"]

        if name == "create_calendar_event":
            ev_dict = {
                "title": args.get("title") or "Termin",
                "date": args.get("date") or dt.date.today().isoformat(),
                "start_time": calendar_sync._normalize_hhmm(args.get("start_time")),
                "end_time": calendar_sync._normalize_hhmm(args.get("end_time")),
                "location": args.get("location") or "",
                "description": "Erstellt via Telegram Assistent",
            }
            created = handle_cal_events_fn(
                events=[ev_dict],
                source_title=ev_dict["title"],
                state=state,
                notion=notion,
                tasks_db_id=tasks_db_id,
            )
            save_state_fn()
            if created and created[0].get("gcal_event_id"):
                store.push_undo("main", chat_id, "delete_event", {
                    "event_id": created[0]["gcal_event_id"],
                    "title": ev_dict["title"],
                })
                undo_pushed = True
            return {"status": "ok", "created": created[0] if created else ev_dict}

        if name == "reschedule_calendar_event":
            ev_title = str(args.get("event_title") or "")
            new_date = args.get("new_date")
            new_start = calendar_sync._normalize_hhmm(args.get("new_start_time"))
            new_end = calendar_sync._normalize_hhmm(args.get("new_end_time"))
            matched = find_gcal_event_fn(ev_title, new_date) or find_gcal_event_fn(ev_title, None)
            if not matched or not matched.get("id"):
                return {"status": "not_found", "message": f"Kein Termin passend zu '{ev_title}' gefunden."}
            old_snapshot = {
                "event_id": matched["id"],
                "old_date": matched.get("date"),
                "old_start_time": matched.get("start_time"),
                "old_end_time": matched.get("end_time"),
                "title": matched.get("title"),
            }
            target_date = new_date or matched.get("date") or dt.date.today().isoformat()
            ok = calendar_sync.update_google_calendar_event(
                event_id=matched["id"],
                new_date=target_date,
                new_start_time=new_start,
                new_end_time=new_end,
            )
            if ok:
                store.push_undo("main", chat_id, "restore_event", old_snapshot)
                undo_pushed = True
                for ev in state.get("calendar_events", []):
                    if ev.get("title") == matched.get("title"):
                        ev["date"] = target_date
                        if new_start:
                            ev["start_time"] = new_start
                        if new_end:
                            ev["end_time"] = new_end
                save_state_fn()
                return {"status": "ok", "title": matched["title"], "new_date": target_date, "new_start_time": new_start}
            return {"status": "error", "message": "Google Kalender Update fehlgeschlagen."}

        if name == "delete_calendar_event":
            ev_title = str(args.get("event_title") or "")
            matched = find_gcal_event_fn(ev_title, args.get("date"))
            if not matched or not matched.get("id"):
                return {"status": "not_found", "message": f"Kein Termin '{ev_title}' gefunden."}
            ok = calendar_sync.delete_google_calendar_event(matched["id"])
            if ok:
                state["calendar_events"] = [e for e in state.get("calendar_events", []) if e.get("title") != matched.get("title")]
                save_state_fn()
                return {"status": "ok", "deleted_title": matched.get("title"), "date": matched.get("date")}
            return {"status": "error", "message": "Löschen im Google Kalender fehlgeschlagen."}

        if name == "get_calendar_and_free_slots":
            target_d = str(args.get("date") or dt.date.today().isoformat())
            days_ahead = max(1, min(int(args.get("days_ahead") or 3), 14))
            min_min = max(15, int(args.get("min_minutes") or 30))
            priv = calendar_sync.fetch_google_calendar_events(days_ahead=days_ahead)
            work = calendar_sync.fetch_work_busy_slots(days_ahead=days_ahead, store=store)
            combined = priv + work
            combined.sort(key=lambda x: (x.get("date") or "", x.get("start_time") or "00:00"))
            work_start = os.environ.get("WORK_HOURS_START", "08:00")
            work_end = os.environ.get("WORK_HOURS_END", "20:00")
            free = calendar_sync.compute_free_slots(
                combined,
                target_date=target_d,
                work_start=work_start,
                work_end=work_end,
                min_slot_minutes=min_min,
            )
            return {
                "events": [chat_guard.slim_calendar_event(e) for e in combined[:20]],
                "free_slots_on_target_date": free,
                "work_calendar_considered": len(work) > 0,
            }

        if name == "create_task":
            title = str(args.get("title") or "Aufgabe").strip()
            prio = args.get("priority") if args.get("priority") in ("Hoch", "Mittel", "Niedrig") else "Mittel"
            due = args.get("due_date")
            step = str(args.get("next_step") or "")
            area = args.get("area") or detect_area(f"{title} {step}")
            est = int(args.get("estimated_minutes") or 30)
            n_id = None
            if notion:
                n_id = notion.create_task_page(
                    tasks_db_id=tasks_db_id,
                    title=title,
                    kind="To-Do",
                    priority=prio,
                    due_date=due,
                    next_step=step,
                    area=area,
                    estimated_minutes=est,
                )
                store.push_undo("main", chat_id, "archive_task", {"notion_page_id": n_id, "title": title})
                undo_pushed = True
            state.setdefault("tasks", []).append({
                "notion_page_id": n_id,
                "kind": "To-Do",
                "title": title,
                "priority": prio,
                "due_date": due,
                "next_step": step,
                "status": "Nicht begonnen",
                "area": area,
                "estimated_minutes": est,
            })
            save_state_fn()
            return {"status": "ok", "title": title, "priority": prio, "due_date": due, "area": area}

        if name == "complete_task":
            q = str(args.get("task_query") or "")
            open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
            targets = resolve_tasks_fn(q, open_todos)
            if not targets:
                return {"status": "not_found", "message": f"Keine offene Aufgabe passend zu '{q}'."}
            done_titles = []
            reopen_ids = []
            for t in targets:
                t["status"] = "Erledigt"
                done_titles.append(t.get("title"))
                if notion and t.get("notion_page_id"):
                    notion.mark_task_done(t["notion_page_id"])
                    reopen_ids.append(t["notion_page_id"])
            if reopen_ids:
                store.push_undo("main", chat_id, "reopen_tasks", {"page_ids": reopen_ids, "titles": done_titles})
                undo_pushed = True
            save_state_fn()
            return {"status": "ok", "completed": done_titles}

        if name == "update_task":
            q = str(args.get("task_query") or "")
            open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
            matched = find_matching_task_fn(q, open_todos)
            if not matched:
                return {"status": "not_found", "message": f"Keine offene Aufgabe zu '{q}'."}
            due = args.get("due_date")
            prio = args.get("priority")
            if notion and matched.get("notion_page_id"):
                notion.update_task(matched["notion_page_id"], due_date=due, priority=prio)
            if due:
                if matched.get("due_date") and matched.get("due_date") != due:
                    store.increment_task_defer(matched.get("notion_page_id") or matched.get("title", ""))
                matched["due_date"] = due
            if prio in ("Hoch", "Mittel", "Niedrig"):
                matched["priority"] = prio
            save_state_fn()
            return {"status": "ok", "title": matched.get("title"), "due_date": matched.get("due_date"), "priority": matched.get("priority")}

        if name == "delete_task":
            q = str(args.get("task_query") or "")
            open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
            matched = find_matching_task_fn(q, open_todos)
            if not matched:
                return {"status": "not_found"}
            if notion and matched.get("notion_page_id"):
                notion.archive_task(matched["notion_page_id"])
                store.push_undo("main", chat_id, "reopen_tasks", {"page_ids": [matched["notion_page_id"]], "titles": [matched.get("title")]})
                undo_pushed = True
            state["tasks"] = [t for t in state.get("tasks", []) if t != matched]
            save_state_fn()
            return {"status": "ok", "deleted": matched.get("title")}

        if name == "list_open_tasks":
            area_f = (args.get("area") or "").lower().strip()
            prio_f = (args.get("priority") or "").lower().strip()
            todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
            if area_f:
                todos = [t for t in todos if area_f in (t.get("area") or "").lower() or area_f in (t.get("title") or "").lower()]
            if prio_f:
                todos = [t for t in todos if (t.get("priority") or "").lower() == prio_f]
            return {"tasks": [chat_guard.slim_task(t) for t in todos[:20]]}

        if name == "search_knowledge":
            q = str(args.get("query") or "")
            notion_hits = notion.search_workspace(q, limit=5) if notion else []
            local_bm25 = store.search_knowledge_bm25(q, limit=5)
            note_hits = [
                chat_guard.slim_recent_note(it.get("analysis", {}))
                for it in state.get("items", [])
                if q.lower() in (it.get("raw_text") or "").lower() or q.lower() in (it.get("analysis", {}).get("title") or "").lower()
            ][:4]
            return {
                "notion_pages": [{"title": h["title"], "url": h.get("url", "")} for h in notion_hits],
                "rss_knowledge": [{"title": r["title"], "summary": (r.get("summary") or "")[:200], "link": r.get("link", "")} for r in local_bm25],
                "recent_notes": note_hits,
            }

        if name == "save_inbox_note":
            title = str(args.get("title") or "Telegram-Notiz")
            summary = str(args.get("summary") or "")
            raw_text = str(args.get("raw_text") or user_text)
            area = args.get("area") or detect_area(raw_text)
            page_id = None
            if notion:
                page_id = notion.create_inbox_page(
                    inbox_db_id=inbox_db_id,
                    title=title,
                    source="Telegram",
                    raw_text=raw_text,
                    summary=summary,
                    suggestions_text="",
                    area=area,
                )
            state.setdefault("items", []).append({
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "file": f"telegram_tool_{int(dt.datetime.now().timestamp())}",
                "source": "Telegram",
                "raw_text": raw_text,
                "analysis": {"title": title, "summary": summary, "area": area, "todos": [], "calendar_events": [], "suggestions": []},
                "notion_inbox_page_id": page_id,
                "synced_to_notion": bool(page_id),
            })
            save_state_fn()
            return {"status": "ok", "title": title, "area": area, "notion_page_id": page_id}

        return {"status": "unknown_tool", "name": name}

    # Tool-Calling-Loop mit StepBudget
    try:
        while budget.can_proceed():
            budget.step(extra_text_len=len(user_text))
            config_kwargs: dict[str, Any] = {
                "system_instruction": system_instr,
                "tools": tools,
                "thinking_config": types.ThinkingConfig(thinking_level=thinking_level),
            }
            resp = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            llm_core.log_gemini_usage(task_tier, model_name, thinking_level, resp)

            fn_calls = getattr(resp, "function_calls", None) or []
            if not fn_calls:
                text_out = (resp.text or "").strip()
                if text_out:
                    store.append_chat_turn("main", chat_id, "user", user_text)
                    store.append_chat_turn("main", chat_id, "assistant", text_out)
                    return telegram_ui.esc(text_out), undo_pushed
                return None

            # Model-Turn anhängen und Tool-Antworten einspeisen
            if getattr(resp, "candidates", None):
                contents.append(resp.candidates[0].content)

            fn_parts = []
            for fc in fn_calls:
                fname = fc.name
                fargs = dict(fc.args or {})
                log.info("[ToolAgent] Rufe Tool '%s' mit %s auf", fname, fargs)
                try:
                    tool_res = _exec_tool(fname, fargs)
                except Exception as t_exc:
                    log.warning("[ToolAgent] Tool '%s' Fehler: %s", fname, t_exc)
                    tool_res = {"status": "error", "message": str(t_exc)}
                fn_parts.append(types.Part.from_function_response(name=fname, response=tool_res))

            contents.append(types.Content(role="user", parts=fn_parts))

        return "⚠️ Anfrage-Budget erreicht. Bitte formuliere die nächste Aktion etwas kürzer.", undo_pushed
    except chat_guard.ChatBudgetExceeded as bexc:
        return f"⚠️ {telegram_ui.esc(str(bexc))}", undo_pushed
    except Exception as exc:
        log.warning("[ToolAgent] Fehler (%s) -> Fallback auf Standard-Router.", exc)
        return None


# --------------------------------------------------------------------------------------
# Multimodalität: Fotos & PDF/Dokumente via Gemini
# --------------------------------------------------------------------------------------

def analyze_multimodal_file(
    file_path: Path,
    mime_type: str,
    caption: str,
    cfg: dict[str, Any],
    date_ref_block: str,
) -> dict[str, Any]:
    """Analysiert ein Foto (Terminzettel, Rechnung, Screenshot) oder Dokument (PDF/TXT) via Gemini."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    if not gemini_key:
        return {
            "title": file_path.stem,
            "summary": f"Datei empfangen ({file_path.name}). Für Bild-/PDF-Analyse bitte GEMINI_API_KEY setzen.",
            "area": "Familie/Privat",
            "todos": [],
            "calendar_events": [],
            "suggestions": [],
        }

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gemini_key)
    model_name, thinking_level = llm_core.resolve_task_model("extract", cfg)
    raw_bytes = file_path.read_bytes()

    prompt = f"""{date_ref_block}

Stefan schickt diese Datei/dieses Foto über Telegram.
Begleittext von Stefan: "{caption or '(keiner)'}"

Analysiere den Inhalt (Text, Termine, Aufgaben, Beträge, Fristen) und gib AUSSCHLIESSLICH dieses JSON-Objekt zurück:
{{
  "title": "kurzer prägnanter Titel (max. 60 Zeichen)",
  "summary": "2-3 Sätze Zusammenfassung der wichtigsten Fakten auf Deutsch",
  "area": "Hochschule/Lehre | Life on Pause | Gesundheit/Forschung | Familie/Privat | Finanzen/Orga",
  "raw_text": "relevanter extrahierter Text/Kernaussagen",
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

    resp = llm_core.execute_gemini_call(
        prompt=[types.Part.from_bytes(data=raw_bytes, mime_type=mime_type), prompt],
        client=client,
        task="extract",
        model=model_name,
        thinking_level=thinking_level,
        response_mime_type="application/json",
    )
    data = json.loads(resp.text or "{}")
    return {
        "title": data.get("title") or file_path.stem,
        "summary": data.get("summary") or "",
        "area": data.get("area") or detect_area(data.get("summary", "")),
        "raw_text": data.get("raw_text") or data.get("summary") or caption or file_path.name,
        "todos": data.get("todos") or [],
        "calendar_events": data.get("calendar_events") or [],
        "suggestions": data.get("suggestions") or [],
    }


# --------------------------------------------------------------------------------------
# Pre-Meeting-Briefing-Check (30-45 Min vor Termin)
# --------------------------------------------------------------------------------------

def check_upcoming_meeting_pings(
    cfg: dict[str, Any],
    state: dict[str, Any],
    notion: Any,
    store: StateStore,
    lead_minutes: int = 40,
) -> list[str]:
    """
    Prüft, ob ein privater Google-Kalendertermin in den nächsten `lead_minutes` Minuten beginnt,
    sucht passenden Notion-/Notiz-Kontext und gibt fertige HTML-Ping-Nachrichten zurück.
    """
    tz_berlin = calendar_sync._berlin_tz()
    now = dt.datetime.now(tz_berlin) if tz_berlin else dt.datetime.now()
    today_iso = now.date().isoformat()
    events = calendar_sync.fetch_google_calendar_events(days_ahead=1)
    open_todos = [t for t in state.get("tasks", []) if t.get("status") != "Erledigt"]
    pings: list[str] = []

    for ev in events:
        if ev.get("date") != today_iso or not ev.get("start_time"):
            continue
        try:
            hh, mm = map(int, ev["start_time"].split(":"))
            ev_dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except Exception:
            continue
        diff_min = (ev_dt - now).total_seconds() / 60.0
        if not (5 <= diff_min <= lead_minutes):
            continue

        ev_key = f"{today_iso}|{ev['start_time']}|{ev.get('title', '')}"
        if store.was_meeting_pinged(ev_key):
            continue
        store.mark_meeting_pinged(ev_key)

        title = ev.get("title", "Termin")
        loc = ev.get("location") or ""
        words = [w.lower() for w in title.split() if len(w) >= 4]
        rel_tasks = [
            t.get("title")
            for t in open_todos
            if any(w in (t.get("title") or "").lower() for w in words)
        ][:3]

        notion_hits = []
        if notion and words:
            try:
                notion_hits = notion.search_workspace(words[0], limit=3)
            except Exception:
                pass

        lines = [
            f"⏰ <b>Gleich in {int(round(diff_min))} Min:</b> {telegram_ui.esc(title)}",
            f"🕒 {telegram_ui.esc(ev['start_time'])} Uhr" + (f" · 📍 {telegram_ui.esc(loc)}" if loc else ""),
        ]
        if rel_tasks:
            lines.append("\n📌 <b>Passende offene Aufgaben:</b>")
            for rt_title in rel_tasks:
                lines.append(f"• {telegram_ui.esc(rt_title)}")
        if notion_hits:
            lines.append("\n📄 <b>Relevante Notion-Seiten:</b>")
            for nh in notion_hits[:2]:
                url = nh.get("url") or ""
                t_lbl = telegram_ui.esc(nh.get("title", "Seite"))
                lines.append(f'• <a href="{telegram_ui.esc(url)}">{t_lbl}</a>' if url else f"• {t_lbl}")
        pings.append("\n".join(lines))

    return pings


# --------------------------------------------------------------------------------------
# Weekly Review & RSS Synthese
# --------------------------------------------------------------------------------------

def generate_weekly_review(
    cfg: dict[str, Any],
    state: dict[str, Any],
    notion: Any,
    store: StateStore,
) -> tuple[str, str | None]:
    """Erstellt ein strukturiertes Wochen-Review (Telegram HTML + Notion Documents Seite)."""
    today = dt.date.today()
    week_ago = (today - dt.timedelta(days=7)).isoformat()

    open_todos = [t for t in state.get("tasks", []) if t.get("kind") == "To-Do" and t.get("status") != "Erledigt"]
    deferred_tasks = [
        t for t in open_todos
        if store.get_task_defer_count(t.get("notion_page_id") or t.get("title", "")) >= 2
    ]

    completed: list[dict[str, Any]] = []
    if notion:
        try:
            all_done = notion.query_recently_completed_tasks(cfg["notion"]["tasks_database_id"], limit=25)
            completed = [c for c in all_done if (c.get("last_edited_time") or "")[:10] >= week_ago]
        except Exception:
            pass

    recent_notes = [
        it for it in state.get("items", [])
        if str(it.get("timestamp", ""))[:10] >= week_ago
    ]
    rss_items = store.get_recent_knowledge(days=7, limit=15)
    upcoming_cal = calendar_sync.fetch_google_calendar_events(days_ahead=7)

    html_lines = [
        f"🗓️ <b>Wochen-Review & Vorschau ({week_ago} bis {today.isoformat()})</b>\n",
        f"✅ <b>Erledigt diese Woche:</b> {len(completed)} Aufgaben",
    ]
    for c in completed[:6]:
        html_lines.append(f"  • {telegram_ui.esc(c.get('title', ''))}")

    html_lines.append(f"\n📌 <b>Noch offen:</b> {len(open_todos)} To-Dos")
    if deferred_tasks:
        html_lines.append("⚠️ <b>Mehrfach verschoben (aufteilen oder streichen?):</b>")
        for dt_t in deferred_tasks[:4]:
            html_lines.append(f"  • {telegram_ui.esc(dt_t.get('title', ''))}")

    html_lines.append(f"\n🎙️ <b>Erfasste Notizen (7 Tage):</b> {len(recent_notes)}")
    html_lines.append(f"📡 <b>Neue RSS-Funde (7 Tage):</b> {len(rss_items)}")
    for r in rss_items[:4]:
        html_lines.append(f"  • [{telegram_ui.esc(r.get('topic_id', ''))}] {telegram_ui.esc(r.get('title', ''))}")

    html_lines.append(f"\n📅 <b>Kommende 7 Tage:</b> {len(upcoming_cal)} private Termine")
    for ev in upcoming_cal[:5]:
        t_lbl = f" {ev['start_time']}" if ev.get("start_time") else ""
        html_lines.append(f"  • {ev.get('date')}{t_lbl}: {telegram_ui.esc(ev.get('title', ''))}")

    doc_url: str | None = None
    if notion:
        md_lines = [
            f"## Wochen-Review {today:%d.%m.%Y}",
            f"- Erledigte Aufgaben (7 Tage): {len(completed)}",
            *[f"- ✅ {c.get('title', '')}" for c in completed[:15]],
            "## Offene Aufgaben & Engpässe",
            *[f"- [{t.get('priority', 'Mittel')}] {t.get('title', '')}" for t in open_todos[:15]],
            "## RSS-Highlights der Woche",
            *[f"- [{r.get('topic_id', '')}] {r.get('title', '')} ({r.get('link', '')})" for r in rss_items[:10]],
        ]
        try:
            doc_url = notion.create_document_page(
                cfg["notion"]["documents_database_id"],
                title=f"Weekly Review {today:%d.%m.%Y}",
                category="Review",
                markdown_lines=md_lines,
                emoji="🗓️",
            )
            html_lines.append(f'\n📄 <a href="{telegram_ui.esc(doc_url)}">Weekly Review in Notion öffnen</a>')
        except Exception as exc:
            log.warning("Weekly Review Notion-Dokument fehlgeschlagen: %s", exc)

    return "\n".join(html_lines), doc_url
