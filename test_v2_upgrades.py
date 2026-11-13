"""
Tests für v2-Stabilisierung, Interaktivität, Datenschutz-Arbeitskalender & Agent-Funktionen.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import sys
from pathlib import Path
from typing import Any
from urllib import error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import assistant
import rss_bot
from core import agent_tools
from core.storage import StateStore, atomic_write_json, safe_read_json
from integrations import calendar_sync, telegram_ui
from integrations.notion_client import NotionClient


class _DummyHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_DummyHTTPResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


# --------------------------------------------------------------------------------------
# 1. core.storage: Atomic JSON, SQLite StateStore, Undo, Callbacks, BM25, Circuit Breaker
# --------------------------------------------------------------------------------------

def test_storage_atomic_json_and_corrupt_recovery(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"ok": True, "items": [1, 2]})
    assert safe_read_json(target, {}) == {"ok": True, "items": [1, 2]}

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{kaputt", encoding="utf-8")
    assert safe_read_json(corrupt, {"fallback": 1}) == {"fallback": 1}


def test_storage_undo_callbacks_bm25_and_circuit_breaker(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "assistant_test.db")

    # Undo-Stack (LIFO)
    store.push_undo("main", "111", "reopen_tasks", {"page_ids": ["p1"], "titles": ["Task 1"]})
    store.push_undo("main", "111", "delete_event", {"event_id": "e1", "title": "Meet"})
    last = store.pop_last_undo("main", "111")
    assert last is not None
    assert last["action_type"] == "delete_event"
    assert last["payload"] == {"event_id": "e1", "title": "Meet"}
    prev = store.pop_last_undo("main", "111")
    assert prev is not None
    assert prev["action_type"] == "reopen_tasks"
    assert prev["payload"] == {"page_ids": ["p1"], "titles": ["Task 1"]}
    assert store.pop_last_undo("main", "111") is None

    # Callback-Payloads
    store.put_callback_payload("tok123", "time_block", {"task_title": "Fokus", "date": "2026-11-20"})
    assert store.get_callback_payload("tok123") == ("time_block", {"task_title": "Fokus", "date": "2026-11-20"})
    assert store.get_callback_payload("unknown") is None

    # BM25 Wissenssuche
    store.add_knowledge_record({
        "topic_id": "parkinson",
        "key": "k1",
        "title": "Neue Biomarker bei Parkinson erkannt",
        "summary": "Alpha-Synuclein Studie aus Boston.",
        "link": "https://example.org/1",
        "relevance": "Hoch",
    })
    store.add_knowledge_record({
        "topic_id": "deals",
        "key": "k2",
        "title": "Adobe Creative Cloud Rabatt für Dozenten",
        "summary": "50 Prozent Nachlass auf Jahresabo.",
        "link": "https://example.org/2",
        "relevance": "Mittel",
    })
    hits = store.search_knowledge_bm25("Biomarker Parkinson Synuclein", limit=3)
    assert hits and hits[0]["item_key"] == "k1"

    # RSS Feed Circuit Breaker
    assert not store.is_feed_tripped("feed_a")
    assert not store.record_feed_failure("feed_a", "parkinson", "Feed A", "timeout", trip_threshold=3)
    assert not store.record_feed_failure("feed_a", "parkinson", "Feed A", "timeout", trip_threshold=3)
    tripped = store.record_feed_failure("feed_a", "parkinson", "Feed A", "timeout", trip_threshold=3)
    assert tripped is True
    assert store.is_feed_tripped("feed_a") is True
    store.record_feed_success("feed_a", "parkinson", "Feed A")
    assert store.is_feed_tripped("feed_a") is False


def test_storage_work_busy_slots_never_persists_titles(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "work_privacy.db")
    stored = store.replace_work_busy_slots(
        [
            {
                "title": "GEHEIMER VORSTANDS-CALL MIT GEHÄLTERN",
                "date": "2026-11-20",
                "start_time": "10:00",
                "end_time": "11:30",
                "location": "Raum 404",
            }
        ],
        source="file",
    )
    assert stored == 1
    out = store.get_work_busy_slots("2026-11-20", "2026-11-20")
    assert len(out) == 1
    assert out[0]["title"] == "Arbeit: Belegt"
    assert out[0]["location"] == ""
    assert out[0]["is_work_busy_only"] is True
    assert "VORSTAND" not in json.dumps(out)


# --------------------------------------------------------------------------------------
# 2. calendar_sync: ICS-Parsing (TZID/RRULE/EXDATE/CANCELLED), Privacy Gate & Slots
# --------------------------------------------------------------------------------------

def test_parse_ics_events_recurrence_exdate_and_cancelled() -> None:
    future_start = dt.date.today() + dt.timedelta(days=2)
    exdate = future_start + dt.timedelta(days=1)
    end_range = future_start + dt.timedelta(days=3)

    ics = "\r\n".join([
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Test//DE",
        "BEGIN:VEVENT",
        "UID:ser-1",
        "SUMMARY:Daily Standup",
        f"DTSTART;TZID=Europe/Berlin:{future_start.strftime('%Y%m%d')}T090000",
        f"DTEND;TZID=Europe/Berlin:{future_start.strftime('%Y%m%d')}T093000",
        "RRULE:FREQ=DAILY;COUNT=3",
        f"EXDATE;TZID=Europe/Berlin:{exdate.strftime('%Y%m%d')}T090000",
        "END:VEVENT",
        "BEGIN:VEVENT",
        "UID:can-1",
        "SUMMARY:Abgesagter Termin",
        f"DTSTART;TZID=Europe/Berlin:{future_start.strftime('%Y%m%d')}T120000",
        f"DTEND;TZID=Europe/Berlin:{future_start.strftime('%Y%m%d')}T130000",
        "STATUS:CANCELLED",
        "END:VEVENT",
        "END:VCALENDAR",
    ])

    events = calendar_sync.parse_ics_events(ics, future_start, end_range)
    dates = [ev["date"] for ev in events if ev["title"] == "Daily Standup"]
    assert future_start.isoformat() in dates
    assert exdate.isoformat() not in dates
    assert (future_start + dt.timedelta(days=2)).isoformat() in dates
    assert all(ev["title"] != "Abgesagter Termin" for ev in events)


def test_parse_ics_work_anonymize_for_privacy() -> None:
    target = dt.date.today() + dt.timedelta(days=2)
    d_comp = target.strftime("%Y%m%d")
    ics = "\r\n".join([
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "BEGIN:VEVENT",
        "UID:work-secret-1",
        "SUMMARY:Vertrauliche Personalakte Dr. Müller",
        "LOCATION:chef@firma.de / Raum 9",
        f"DTSTART;TZID=Europe/Berlin:{d_comp}T100000",
        f"DTEND;TZID=Europe/Berlin:{d_comp}T110000",
        "TRANSP:OPAQUE",
        "END:VEVENT",
        "BEGIN:VEVENT",
        "UID:work-free-1",
        "SUMMARY:Hinweis Frei",
        f"DTSTART;TZID=Europe/Berlin:{d_comp}T140000",
        f"DTEND;TZID=Europe/Berlin:{d_comp}T150000",
        "TRANSP:TRANSPARENT",
        "END:VEVENT",
        "END:VCALENDAR",
    ])
    slots = calendar_sync.parse_ics_events(ics, target, target, source_label="Work", anonymize_for_privacy=True)
    assert len(slots) == 1
    assert slots[0]["title"] == "Arbeit: Belegt"
    assert slots[0]["start_time"] == "10:00"
    assert slots[0]["end_time"] == "11:00"
    assert slots[0]["location"] == ""
    assert slots[0]["is_work_busy_only"] is True
    dumped = json.dumps(slots)
    assert "Personalakte" not in dumped
    assert "chef@firma.de" not in dumped


def test_sanitize_work_busy_slots_and_compute_free_slots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = dt.date.today() + dt.timedelta(days=3)
    target_iso = target.isoformat()

    raw_work = [
        {
            "subject": "Vertraulich 1:1",
            "attendees": ["boss@work.com"],
            "start": f"{target_iso}T09:00:00",
            "end": f"{target_iso}T11:00:00",
            "show_as": "busy",
        },
        {
            "subject": "Blocker Frei",
            "start": f"{target_iso}T13:00:00",
            "end": f"{target_iso}T14:00:00",
            "show_as": "free",
        },
    ]
    sanitized = calendar_sync.sanitize_work_busy_slots(raw_work, source="power_automate")
    assert len(sanitized) == 1
    assert sanitized[0]["title"] == "Arbeit: Belegt"
    assert "Vertraulich" not in json.dumps(sanitized)

    private_ev = [{"title": "Arzt", "date": target_iso, "start_time": "14:00", "end_time": "15:00"}]
    free = calendar_sync.compute_free_slots(
        private_ev + sanitized,
        target_date=target_iso,
        work_start="08:00",
        work_end="18:00",
        min_slot_minutes=30,
        buffer_minutes=0,
    )
    assert [(s["start_time"], s["end_time"]) for s in free] == [
        ("08:00", "09:00"),
        ("11:00", "14:00"),
        ("15:00", "18:00"),
    ]

    blocks = calendar_sync.suggest_time_blocks(
        free,
        [{"title": "Klausur korrigieren", "kind": "To-Do", "priority": "Hoch", "estimated_minutes": 60}],
        max_suggestions=2,
    )
    assert len(blocks) == 1
    assert blocks[0]["start_time"] == "08:00"
    assert blocks[0]["end_time"] == "09:00"


def test_export_ics_event_deterministic_uid(tmp_path: Path) -> None:
    ev = {"title": "Zahnarzt", "date": "2026-12-01", "start_time": "15:00", "end_time": "16:00"}
    p1 = calendar_sync.export_ics_event(ev, outbox_dir=tmp_path / "o1")
    p2 = calendar_sync.export_ics_event(ev, outbox_dir=tmp_path / "o2")
    uid1 = [ln for ln in p1.read_text(encoding="utf-8").splitlines() if ln.startswith("UID:")][0]
    uid2 = [ln for ln in p2.read_text(encoding="utf-8").splitlines() if ln.startswith("UID:")][0]
    assert uid1 == uid2
    assert "@notion-local-assistant" in uid1


# --------------------------------------------------------------------------------------
# 3. telegram_ui: HTML-Escaping, Plaintext-Fallback & Autorisierung in beiden Bots
# --------------------------------------------------------------------------------------

def test_telegram_ui_escape_keyboard_and_400_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    assert telegram_ui.esc("<A & B>") == "&lt;A &amp; B&gt;"
    kb = telegram_ui.build_inline_keyboard([[("✅ Done", "done:1"), ("🔗 Link", "https://example.org")]])
    assert kb["inline_keyboard"][0][0] == {"text": "✅ Done", "callback_data": "done:1"}
    assert kb["inline_keyboard"][0][1] == {"text": "🔗 Link", "url": "https://example.org"}

    calls: list[dict[str, Any]] = []

    def fake_urlopen(req: Any, timeout: int = 15) -> _DummyHTTPResponse:
        body = json.loads(req.data.decode("utf-8"))
        calls.append(body)
        if body.get("parse_mode") == "HTML":
            raise error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b"can't parse entities"))
        return _DummyHTTPResponse({"ok": True, "result": {"message_id": 1}})

    monkeypatch.setattr(telegram_ui.request, "urlopen", fake_urlopen)
    ok = telegram_ui.send_message("tok", "12345", "<b>Hallo</b> & <broken", parse_mode="HTML")
    assert ok is True
    assert len(calls) == 2
    assert "parse_mode" not in calls[1]
    assert calls[1]["text"] == "Hallo & <broken"


def test_unauthorized_chat_blocked_in_both_bots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot:test")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "99999")
    monkeypatch.setenv("TELEGRAM_RSS_CHAT_ID", "99999")
    monkeypatch.setenv("ASSISTANT_DB_PATH", str(tmp_path / "auth_test.db"))

    # Haupt-Bot: process_telegram_updates darf fremde Chat-ID weder beantworten noch speichern
    sent_main: list[str] = []
    monkeypatch.setattr(assistant, "_send_telegram_message", lambda *a, **k: sent_main.append(str(a)) or True)
    monkeypatch.setattr(assistant, "save_local_state", lambda *a, **k: None)
    monkeypatch.setattr(
        assistant.request,
        "urlopen",
        lambda req, timeout=15: _DummyHTTPResponse({
            "ok": True,
            "result": [{"update_id": 10, "message": {"chat": {"id": 123456}, "text": "/todos"}}],
        }),
    )
    cfg = assistant.load_config()
    main_state: dict[str, Any] = {"telegram_offset": 0, "tasks": [], "items": [], "calendar_events": []}
    assistant.process_telegram_updates(cfg, main_state, notion=None)
    assert sent_main == []
    assert "last_telegram_chat_id" not in main_state

    # RSS-Bot: process_updates darf fremde Chat-ID weder beantworten noch speichern
    sent_rss: list[str] = []
    monkeypatch.setattr(rss_bot, "send_message", lambda *a, **k: sent_rss.append(str(a)) or True)
    monkeypatch.setattr(rss_bot, "save_rss_state", lambda st: None)
    monkeypatch.setattr(
        rss_bot.urllib.request,
        "urlopen",
        lambda req, timeout=15: _DummyHTTPResponse({
            "ok": True,
            "result": [{"update_id": 20, "message": {"chat": {"id": 123456}, "text": "/themen"}}],
        }),
    )
    rss_state: dict[str, Any] = {"rss_telegram_offset": 0}
    rss_bot.process_updates("rss:test", rss_state)
    assert sent_rss == []
    assert "last_rss_chat_id" not in rss_state


# --------------------------------------------------------------------------------------
# 4. NotionClient: Retry 429, Pagination has_more & Schema-Fallback
# --------------------------------------------------------------------------------------

def test_notion_client_retries_pagination_and_schema_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import integrations.notion_client as nc_mod

    monkeypatch.setattr(nc_mod.time, "sleep", lambda s: None)
    client = NotionClient("secret_test")

    # 429 Retry + 2-Seiten-Pagination
    attempt_counter = {"n": 0}

    def fake_urlopen(req: Any, timeout: int = 30) -> _DummyHTTPResponse:
        attempt_counter["n"] += 1
        if attempt_counter["n"] == 1:
            raise error.HTTPError(
                req.full_url, 429, "Too Many Requests", {"Retry-After": "0"}, io.BytesIO(b"rate limited")
            )
        body = json.loads(req.data.decode("utf-8")) if req.data else {}
        if body.get("start_cursor") == "cur2":
            return _DummyHTTPResponse({
                "results": [{
                    "id": "p2",
                    "properties": {
                        "Aufgabenbezeichnung": {"title": [{"plain_text": "Zweite Aufgabe"}]},
                        "Status": {"status": {"name": "Nicht begonnen"}},
                        "Art": {"select": {"name": "To-Do"}},
                        "Prioritaet": {"select": {"name": "Hoch"}},
                    },
                }],
                "has_more": False,
            })
        return _DummyHTTPResponse({
            "results": [{
                "id": "p1",
                "properties": {
                    "Aufgabenbezeichnung": {"title": [{"plain_text": "Erste Aufgabe"}]},
                    "Status": {"status": {"name": "Nicht begonnen"}},
                    "Art": {"select": {"name": "To-Do"}},
                    "Prioritaet": {"select": {"name": "Mittel"}},
                },
            }],
            "has_more": True,
            "next_cursor": "cur2",
        })

    monkeypatch.setattr(nc_mod.request, "urlopen", fake_urlopen)
    tasks = client.query_open_tasks("db123")
    assert [t["title"] for t in tasks] == ["Erste Aufgabe", "Zweite Aufgabe"]
    assert attempt_counter["n"] == 3

    # Schema-Fallback bei fehlenden optionalen Spalten ("Bereich", "Dauer (Min)")
    seen_payloads: list[dict[str, Any]] = []

    def fake_post_page(req: Any, timeout: int = 30) -> _DummyHTTPResponse:
        body = json.loads(req.data.decode("utf-8"))
        seen_payloads.append(body)
        props = body.get("properties", {})
        if "Bereich" in props or "Dauer (Min)" in props:
            raise error.HTTPError(
                req.full_url, 400, "Bad Request", {}, io.BytesIO(b"Bereich is not a property that exists.")
            )
        return _DummyHTTPResponse({"id": "created-page-id"})

    monkeypatch.setattr(nc_mod.request, "urlopen", fake_post_page)
    pid = client.create_task_page(
        tasks_db_id="db123",
        title="Test",
        kind="To-Do",
        priority="Mittel",
        due_date="2026-11-25",
        next_step="Los",
        area="Hochschule/Lehre",
        estimated_minutes=45,
    )
    assert pid == "created-page-id"
    assert len(seen_payloads) == 2
    assert "Bereich" not in seen_payloads[1]["properties"]
    assert "Dauer (Min)" not in seen_payloads[1]["properties"]


# --------------------------------------------------------------------------------------
# 5. assistant.py & agent_tools.py: /slots, Callbacks (done/defer/block/undo), Pings, Briefing
# --------------------------------------------------------------------------------------

class _FakeNotion:
    def __init__(self) -> None:
        self.done: list[str] = []
        self.reopened: list[str] = []
        self.updated: list[tuple[str, str | None]] = []
        self.archived: list[str] = []

    def mark_task_done(self, pid: str) -> None:
        self.done.append(pid)

    def reopen_task(self, pid: str) -> None:
        self.reopened.append(pid)

    def update_task(self, page_id: str, due_date: str | None = None, priority: str | None = None) -> None:
        self.updated.append((page_id, due_date))

    def archive_task(self, pid: str) -> None:
        self.archived.append(pid)

    def search_workspace(self, q: str, limit: int = 3) -> list[dict[str, Any]]:
        return [{"title": "Vorlesung Skript", "url": "https://notion.so/1"}]


def test_slots_callbacks_and_undo_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "777")
    store = StateStore(tmp_path / "flow.db")
    cfg = assistant.load_config()
    tomorrow = dt.date.today() + dt.timedelta(days=1)

    store.replace_work_busy_slots(
        [{"date": tomorrow.isoformat(), "start_time": "10:00", "end_time": "12:00"}],
        source="file",
    )

    state: dict[str, Any] = {
        "tasks": [
            {
                "kind": "To-Do",
                "title": "Folien aktualisieren",
                "priority": "Hoch",
                "due_date": None,
                "status": "Nicht begonnen",
                "notion_page_id": "np-1",
                "estimated_minutes": 45,
            }
        ],
        "calendar_events": [],
        "items": [],
    }
    monkeypatch.setattr(assistant, "fetch_google_calendar_events", lambda days_ahead=7: [])

    slots_html, slots_kb = assistant._build_slots_reply(tomorrow, cfg, state, store)
    assert "Arbeit: Belegt" in slots_html
    assert "08:00–09:55" in slots_html
    assert "12:05–20:00" in slots_html
    assert slots_kb is not None
    block_cb = slots_kb["inline_keyboard"][0][0]["callback_data"]
    assert block_cb.startswith("block:")

    sent_msgs: list[str] = []
    monkeypatch.setattr(assistant, "_send_telegram_message", lambda api, cid, txt, **kw: sent_msgs.append(txt) or True)
    monkeypatch.setattr(telegram_ui, "answer_callback_query", lambda *a, **k: True)
    monkeypatch.setattr(assistant, "save_local_state", lambda *a, **k: None)
    monkeypatch.setattr(assistant, "export_ics_event", lambda ev: assistant.BASE_DIR / "calendar_outbox" / "tmp_test.ics")
    monkeypatch.setattr(assistant, "create_google_calendar_event", lambda ev: "https://calendar.google.com/ev1")
    deleted_ids: list[str] = []
    monkeypatch.setattr(assistant, "delete_google_calendar_event", lambda eid: deleted_ids.append(eid) or True)

    fake_notion = _FakeNotion()
    state_path = tmp_path / "local_state.json"

    # 1) block:<tok> trägt Fokus-Slot ein & legt Undo an
    assert assistant._handle_telegram_callback_query(
        {"id": "c1", "data": block_cb, "message": {"chat": {"id": "777"}}},
        "tok", "https://api.telegram.org/bottok", cfg, state, state_path, fake_notion, store,
    )
    assert any("Fokus: Folien aktualisieren" == ev["title"] for ev in state["calendar_events"])

    # 2) done:1 erledigt Task -> undo:last öffnet ihn wieder
    assert assistant._handle_telegram_callback_query(
        {"id": "c2", "data": "done:1", "message": {"chat": {"id": "777"}}},
        "tok", "https://api.telegram.org/bottok", cfg, state, state_path, fake_notion, store,
    )
    assert state["tasks"][0]["status"] == "Erledigt"
    assert fake_notion.done == ["np-1"]

    assert assistant._handle_telegram_callback_query(
        {"id": "c3", "data": "undo:last", "message": {"chat": {"id": "777"}}},
        "tok", "https://api.telegram.org/bottok", cfg, state, state_path, fake_notion, store,
    )
    assert state["tasks"][0]["status"] == "Nicht begonnen"
    assert fake_notion.reopened == ["np-1"]

    # 3) defer:1 verschiebt Fälligkeit auf morgen und zählt hoch
    assert assistant._handle_telegram_callback_query(
        {"id": "c4", "data": "defer:1", "message": {"chat": {"id": "777"}}},
        "tok", "https://api.telegram.org/bottok", cfg, state, state_path, fake_notion, store,
    )
    assert state["tasks"][0]["due_date"] == tomorrow.isoformat()
    assert store.get_task_defer_count("np-1") == 1


def test_briefing_includes_anonymous_work_blocks_without_leaking_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSISTANT_DB_PATH", str(tmp_path / "brief.db"))
    slots_file = tmp_path / "work_busy.json"
    slots_file.write_text(
        json.dumps([
            {
                "subject": "STRENG GEHEIMES BUDGET MEETING",
                "date": dt.date.today().isoformat(),
                "start_time": "10:00",
                "end_time": "11:00",
            }
        ]),
        encoding="utf-8",
    )
    monkeypatch.setenv("WORK_BUSY_SLOTS_FILE", str(slots_file))
    monkeypatch.setattr(assistant, "fetch_google_calendar_events", lambda days_ahead=3: [])

    cfg = assistant.load_config()
    state: dict[str, Any] = {"tasks": [], "items": [], "calendar_events": []}
    text = assistant.generate_briefing(cfg, state, notify=False, mode="morning")
    assert "anonyme Belegt-Blöcke" in text
    assert "STRENG GEHEIM" not in text


def test_pre_meeting_ping_deduplicated_and_rss_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "pings_rss.db"
    monkeypatch.setenv("ASSISTANT_DB_PATH", str(db_path))
    store = StateStore(db_path)

    tz = calendar_sync._berlin_tz()
    now = dt.datetime.now(tz) if tz else dt.datetime.now()
    soon = now + dt.timedelta(minutes=25)
    # Guard against midnight wrap around 23:40
    if soon.date() == now.date():
        monkeypatch.setattr(
            calendar_sync,
            "fetch_google_calendar_events",
            lambda days_ahead=1: [{
                "title": "Vorlesung KI",
                "date": soon.date().isoformat(),
                "start_time": soon.strftime("%H:%M"),
                "location": "Hörsaal B",
            }],
        )
        fake_notion = _FakeNotion()
        state = {"tasks": [{"title": "Vorlesung Folien prüfen", "status": "Nicht begonnen"}]}
        pings1 = agent_tools.check_upcoming_meeting_pings({}, state, fake_notion, store, lead_minutes=40)
        pings2 = agent_tools.check_upcoming_meeting_pings({}, state, fake_notion, store, lead_minutes=40)
        assert len(pings1) == 1
        assert "Vorlesung KI" in pings1[0]
        assert "Hörsaal B" in pings1[0]
        assert pings2 == []

    # RSS Callback: rss_bad: und rss_task:
    monkeypatch.setenv("TELEGRAM_RSS_CHAT_ID", "888")
    monkeypatch.setattr(telegram_ui, "answer_callback_query", lambda *a, **k: True)
    sent_rss: list[str] = []
    monkeypatch.setattr(rss_bot, "send_message", lambda tok, cid, txt, **kw: sent_rss.append(txt) or True)

    rss_store = rss_bot.get_store()
    rss_store.put_callback_payload(
        "rbad1",
        "rss_item",
        {"topic_id": "parkinson", "title": "Clickbait News", "link": "https://x.test/1", "summary": "Irrelevant"},
    )
    rss_bot.handle_callback_query(
        "tok",
        {"id": "rcb1", "data": "rss_bad:rbad1", "message": {"chat": {"id": "888"}}},
        {},
    )
    assert "Clickbait News" in rss_store.get_recent_negative_feedback("parkinson")

    # rss_task: erstellt Notion Task & legt "rss" Undo-Eintrag an
    created_tasks: list[dict[str, Any]] = []

    class _FakeRSSNotion:
        def __init__(self, tok: str) -> None:
            self.tok = tok

        def create_task_page(self, **kw: Any) -> str:
            created_tasks.append(kw)
            return "rss-task-page-1"

    monkeypatch.setenv("NOTION_TOKEN", "secret_rss")
    monkeypatch.setattr(rss_bot, "NotionClient", _FakeRSSNotion)
    rss_store.put_callback_payload(
        "rtask1",
        "rss_item",
        {"topic_id": "parkinson", "title": "Wichtige Phase-2-Studie", "link": "https://x.test/2", "summary": "Parkinson Therapie"},
    )
    rss_bot.handle_callback_query(
        "tok",
        {"id": "rcb2", "data": "rss_task:rtask1", "message": {"chat": {"id": "888"}}},
        {},
    )
    assert len(created_tasks) == 1
    assert "Wichtige Phase-2-Studie" in created_tasks[0]["title"]
    undo_entry = rss_store.pop_last_undo("rss", "888")
    assert undo_entry is not None
    assert undo_entry["action_type"] == "archive_task"
    assert undo_entry["payload"] == {"notion_page_id": "rss-task-page-1", "title": "Wichtige Phase-2-Studie"}

    # BM25 /suche Fallback in rss_bot
    rss_store.add_knowledge_record({
        "topic_id": "covid",
        "key": "c1",
        "title": "Abwassermonitoring zeigt leichte Welle",
        "summary": "Neue Daten vom RKI und rv-enigma.",
        "link": "https://covid.test/1",
        "relevance": "Hoch",
    })
    monkeypatch.setattr(rss_bot, "query_notion_knowledge", lambda **kw: [])
    search_reply = rss_bot.handle_suche("888", "Abwassermonitoring")
    assert "Abwassermonitoring" in search_reply
