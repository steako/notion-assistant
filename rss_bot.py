#!/usr/bin/env python3
"""
rss_bot.py – Dedizierter interaktiver Telegram-Bot für den RSS- & Bluesky-Wissensspeicher.

Trennt News, wissenschaftliche Studien und Rabattaktionen sauber von persönlichen
Terminen, To-Dos und Kalender-Benachrichtigungen.

Verwendet TELEGRAM_RSSBOT_TOKEN aus .env (Fallback: TELEGRAM_BOT_TOKEN).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core import agent_tools
from core.storage import StateStore, atomic_write_json, safe_read_json
from integrations import telegram_ui
from integrations.notion_client import NotionClient

# Lokale Module einbinden
try:
    import assistant
except Exception:
    assistant = None

_rss_monitor_import_error: str | None = None
try:
    import rss_monitor
except Exception as _r_err:
    rss_monitor = None
    _rss_monitor_import_error = str(_r_err)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [RSSBot] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rss_bot")

STATE_PATH = BASE_DIR / "rss_state.json"
FEEDS_CONFIG_PATH = BASE_DIR / "feeds.json"

_STORE_CACHE: dict[str, StateStore] = {}


def get_store(db_path: Path | None = None) -> StateStore:
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


# --------------------------------------------------------------------------------------
# Konfiguration & Hilfsfunktionen
# --------------------------------------------------------------------------------------

def get_tokens() -> tuple[str, str]:
    """Gibt (rss_token, main_token) zurück."""
    if assistant is not None:
        assistant.load_dotenv_file(BASE_DIR / ".env")
    rss_token = os.environ.get("TELEGRAM_RSSBOT_TOKEN", "").strip()
    main_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    return rss_token, main_token


def get_active_token() -> str:
    rss_token, main_token = get_tokens()
    return rss_token or main_token


def load_rss_state() -> dict[str, Any]:
    data = safe_read_json(STATE_PATH, {"seen": {}, "last_rss_chat_id": "", "rss_telegram_offset": 0})
    if not isinstance(data, dict):
        data = {}
    data.setdefault("seen", {})
    data.setdefault("last_rss_chat_id", "")
    data.setdefault("rss_telegram_offset", 0)
    return data


def save_rss_state(state: dict[str, Any]) -> None:
    try:
        atomic_write_json(STATE_PATH, state, indent=1)
    except Exception as exc:
        log.warning("Fehler beim Speichern von rss_state.json: %s", exc)


def send_message(
    token: str,
    chat_id: str,
    text: str,
    parse_mode: str | None = "HTML",
    reply_markup: dict[str, Any] | None = None,
) -> bool:
    """Sendet eine Nachricht an Telegram über die einheitliche HTML-Transport-Schicht."""
    return telegram_ui.send_message(
        token=token,
        chat_id=chat_id,
        text=text,
        parse_mode=parse_mode,
        reply_markup=reply_markup,
    )


def load_feeds_config() -> dict[str, Any]:
    """Lädt die Themen-Konfiguration robust (über rss_monitor oder direkt aus feeds.json)."""
    if rss_monitor is not None and hasattr(rss_monitor, "load_feeds_config"):
        try:
            return rss_monitor.load_feeds_config()
        except Exception:
            pass
    if FEEDS_CONFIG_PATH.exists():
        data = safe_read_json(FEEDS_CONFIG_PATH, {"topics": []})
        if isinstance(data, dict):
            return data
    return {"topics": []}


# --------------------------------------------------------------------------------------
# Notion-Abfragen für den Wissensspeicher
# --------------------------------------------------------------------------------------

def query_notion_knowledge(filter_payload: dict[str, Any] | None = None, page_size: int = 5) -> list[dict[str, Any]]:
    """Fragt die Wissensspeicher-Datenbank in Notion ab."""
    token = os.environ.get("NOTION_TOKEN", "").strip()
    if not token:
        return []

    feeds_cfg = load_feeds_config()
    db_id = (
        feeds_cfg.get("notion", {}).get("knowledge_database_id")
        or feeds_cfg.get("notion_database_id")
        or "c7bac0f9-b222-4aa6-a0b4-2510fb987b7b"
    ).replace("-", "")

    payload: dict[str, Any] = {"page_size": page_size, "sorts": [{"property": "Erfasst", "direction": "descending"}]}
    if filter_payload:
        payload["filter"] = filter_payload

    try:
        client = NotionClient(token)
        res = client._request("POST", f"/databases/{db_id}/query", payload)
        return res.get("results", [])
    except Exception as exc:
        log.warning("Notion-Abfragefehler: %s", exc)
        return []


def format_notion_results(results: list[dict[str, Any]], title_header: str) -> str:
    """Formatiert Notion-Seiten lesbar für Telegram HTML."""
    esc = telegram_ui.esc
    if not results:
        return f"{title_header}\n\nKeine passenden Einträge im Wissensspeicher gefunden."

    lines = [f"{title_header}\n"]
    for idx, page in enumerate(results, 1):
        props = page.get("properties", {})
        title_list = props.get("Titel", {}).get("title", [])
        title = title_list[0]["plain_text"] if title_list else "Ohne Titel"

        thema_sel = props.get("Thema", {}).get("select") or {}
        thema = thema_sel.get("name", "")

        rel_sel = props.get("Relevanz", {}).get("select") or {}
        rel = rel_sel.get("name", "")

        link = props.get("Link", {}).get("url") or ""

        summary_list = props.get("Zusammenfassung", {}).get("rich_text", [])
        summary = summary_list[0]["plain_text"] if summary_list else ""

        badge = f"[{esc(thema)}]" if thema else ""
        if rel:
            badge += f" ({esc(rel)})"

        title_html = f'<a href="{esc(link)}">{esc(title)}</a>' if link else f"<b>{esc(title)}</b>"
        lines.append(f"{idx}. {badge} {title_html}")
        if summary:
            first_sentence = summary.split(". ")[0] + ("." if not summary.endswith(".") else "")
            lines.append(f"   <i>{esc(first_sentence[:220])}</i>")
        lines.append("")

    return "\n".join(lines).strip()


# --------------------------------------------------------------------------------------
# Befehls-Handler
# --------------------------------------------------------------------------------------

def handle_start_or_help(chat_id: str) -> str:
    return (
        "📡 <b>Willkommen beim RSS- &amp; Wissensspeicher-Bot!</b>\n\n"
        "Ich sammle, bewerte und dedupliziere automatisch wissenschaftliche Studien, "
        "Durchbrüche, Bluesky-Signale und Rabattaktionen – getrennt von deinen privaten "
        "Kalenderterminen und Aufgaben.\n\n"
        "📋 <b>Verfügbare Befehle:</b>\n"
        "• <code>/themen</code> – Übersicht aller Themen &amp; Status\n"
        "• <code>/an &lt;id&gt;</code> – Thema aktivieren (z. B. <code>/an parkinson</code>)\n"
        "• <code>/aus &lt;id&gt;</code> – Thema pausieren\n"
        "• <code>/neu &lt;Name&gt;</code> – Neues Thema mit KI-Recherche anlegen\n"
        "• <code>/loeschen &lt;id&gt;</code> – Thema dauerhaft entfernen\n"
        "• <code>/run</code> – Sofortigen RSS-/Bluesky-Suchlauf starten\n"
        "• <code>/synthese</code> – Themenübergreifende Wochen-Synthese erstellen\n"
        "• <code>/covid</code> – Aktuelle Infektionsschätzung von @rv-enigma\n"
        "• <code>/deals</code> – Aktuelle Rabattaktionen für Studenten/Dozenten\n"
        "• <code>/suche &lt;Begriff&gt;</code> – Notion &amp; lokalen Wissensspeicher durchsuchen\n"
        "• <code>/status</code> – Systemstatus, Feed-Health &amp; nächste Termine\n"
        "• <code>/update</code> – Neuesten Code von GitHub laden &amp; Bot neu starten\n\n"
        "💡 <i>Tipp: Du kannst mir auch Fragen zu den Themen stellen, Sprachnachrichten "
        "oder PDF-Studien schicken und Funde direkt per Button als Task/Idee übernehmen!</i>"
    )


def handle_themen(chat_id: str) -> str:
    esc = telegram_ui.esc
    cfg = load_feeds_config()
    topics = cfg.get("topics", [])
    if not topics:
        return "Keine Themen in feeds.json konfiguriert."

    lines = ["📡 <b>Konfigurierte Themengebiete:</b>\n"]
    for t in topics:
        status = "🟢 Aktiv" if t.get("enabled", True) else "⚪ Pausiert"
        emoji = t.get("emoji", "📌")
        name = t.get("name", "Unbenannt")
        tid = t.get("id", "")
        sources_cnt = len(t.get("sources", []))
        min_rel = t.get("min_relevance", "Mittel")

        lines.append(f"{esc(emoji)} <b>{esc(name)}</b> (<code>{esc(tid)}</code>) – {status}")
        lines.append(f"   ↳ {sources_cnt} Quellen | Min-Relevanz: {esc(min_rel)}")
        focus = t.get("focus", "")
        if focus:
            short_focus = focus[:120] + ("..." if len(focus) > 120 else "")
            lines.append(f"   ↳ <i>{esc(short_focus)}</i>")
        lines.append("")

    lines.append("Steuerung:")
    lines.append("• <code>/an &lt;id&gt;</code> / <code>/aus &lt;id&gt;</code> zum Ein- oder Ausschalten")
    lines.append("• <code>/neu &lt;Name&gt;</code> um ein neues Thema per KI zu erstellen")
    lines.append("• <code>/run</code> für einen sofortigen Suchlauf")
    return "\n".join(lines)


def handle_toggle(chat_id: str, topic_id: str, enable: bool) -> str:
    esc = telegram_ui.esc
    if not topic_id.strip():
        return f"Bitte Themen-ID angeben, z. B. <code>/{'an' if enable else 'aus'} parkinson</code>."
    if rss_monitor is not None and hasattr(rss_monitor, "cmd_topic_set_enabled"):
        ok = rss_monitor.cmd_topic_set_enabled(topic_id.strip(), enable)
    else:
        cfg = load_feeds_config()
        ok = False
        for t in cfg.get("topics", []):
            if t.get("id") == topic_id.strip():
                t["enabled"] = enable
                ok = True
                break
        if ok:
            try:
                atomic_write_json(FEEDS_CONFIG_PATH, cfg, indent=2)
            except Exception:
                ok = False
    if ok:
        status_str = "aktiviert 🟢" if enable else "pausiert ⚪"
        return f"Thema <code>{esc(topic_id)}</code> wurde {status_str}."
    return f"Thema <code>{esc(topic_id)}</code> nicht gefunden. Nutze <code>/themen</code> für die Liste."


def handle_neu(chat_id: str, new_name: str) -> str:
    esc = telegram_ui.esc
    if not new_name.strip():
        return "Bitte gib einen Namen an, z. B. <code>/neu Multiple Sklerose</code>."
    if rss_monitor is None:
        return "Fehler: rss_monitor nicht verfügbar."

    import argparse as _ap
    ns = _ap.Namespace(name=new_name.strip(), id=None, emoji=None, focus=None, rss=None, auto=True)
    ok = rss_monitor.cmd_topic_add(ns)
    if ok:
        return (
            f"✅ Thema <b>{esc(new_name)}</b> wurde erfolgreich angelegt!\n"
            "Quellen aus Google News, Europe PMC und Bluesky wurden automatisch konfiguriert "
            "und aktiviert. Starte mit <code>/run</code> den ersten Suchlauf."
        )
    return f"❌ Konnte Thema <b>{esc(new_name)}</b> nicht anlegen."


def handle_loeschen(chat_id: str, topic_id: str) -> str:
    esc = telegram_ui.esc
    if not topic_id.strip():
        return "Bitte Themen-ID angeben, z. B. <code>/loeschen mein_thema</code>."
    if rss_monitor is None:
        return "Fehler: rss_monitor nicht verfügbar."

    ok = rss_monitor.cmd_topic_remove(topic_id.strip(), confirm=True)
    if ok:
        return f"🗑️ Thema <code>{esc(topic_id)}</code> wurde gelöscht."
    return f"❌ Thema <code>{esc(topic_id)}</code> konnte nicht gelöscht werden (nicht gefunden?)."


def handle_run_async(chat_id: str, token: str) -> None:
    """Führt rss_monitor.run() in einem separaten Thread aus."""
    if rss_monitor is None:
        err_msg = f" ({telegram_ui.esc(_rss_monitor_import_error)})" if _rss_monitor_import_error else ""
        send_message(
            token,
            chat_id,
            f"❌ Fehler: rss_monitor nicht verfügbar{err_msg}.\n\n"
            "💡 Tipp: Bitte auf dem Server Abhängigkeiten installieren:\n"
            "<code>pip install -r requirements.txt</code>",
        )
        return

    # Lock prüfen (LOCK_PATH oder LOCK_FILE)
    lock_path = getattr(rss_monitor, "LOCK_PATH", getattr(rss_monitor, "LOCK_FILE", BASE_DIR / "rss_monitor.lock"))
    try:
        if lock_path.exists():
            pid = int(lock_path.read_text(encoding="utf-8").strip())
            if hasattr(rss_monitor, "is_pid_running") and rss_monitor.is_pid_running(pid):
                send_message(token, chat_id, f"⚠️ Ein Suchlauf läuft bereits im Hintergrund (PID {pid}). Bitte warte kurz.")
                return
    except Exception as exc:
        log.warning("Fehler bei Lock-Prüfung: %s", exc)

    send_message(
        token,
        chat_id,
        "📡 <b>RSS- &amp; Bluesky-Suchlauf gestartet!</b>\n"
        "Ich prüfe Google News, Fachmagazine, Preprints, ATProto-Bluesky und Rabatte. "
        "Ergebnisse werden nach Abschluss gemeldet...",
    )

    def _worker():
        try:
            if hasattr(rss_monitor, "acquire_lock") and not rss_monitor.acquire_lock():
                send_message(token, chat_id, "⚠️ Ein anderer Suchlauf ist noch aktiv. Bitte warte kurz.")
                return

            try:
                import argparse as _ap
                ns = _ap.Namespace(topic=None, dry_run=False, no_notify=False, no_notion=False, verbose=False)
                res = rss_monitor.run(ns)
                log.info("Manueller Telegram-Run beendet mit Code %s", res)

                st = load_rss_state()
                last_run = (st.get("runs") or [{}])[0]
                new_cnt = last_run.get("new", 0)
                if new_cnt == 0:
                    send_message(
                        token,
                        chat_id,
                        "✅ <b>Suchlauf abgeschlossen.</b>\n"
                        "Es wurden keine neuen relevanten Studien oder Artikel gefunden – alle Quellen sind aktuell.",
                    )
            finally:
                if hasattr(rss_monitor, "release_lock"):
                    rss_monitor.release_lock()
        except Exception as exc:
            log.exception("Fehler beim manuellen Suchlauf: %s", exc)
            send_message(token, chat_id, f"❌ Fehler beim Suchlauf: {telegram_ui.esc(exc)}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def handle_covid(chat_id: str) -> str:
    """Ruft die neuesten Covid-Zahlen von @rv-enigma aus Notion ab."""
    filter_payload = {
        "property": "Quelle",
        "rich_text": {"contains": "rv-enigma"},
    }
    pages = query_notion_knowledge(filter_payload=filter_payload, page_size=5)
    return format_notion_results(pages, "🦠 <b>Neueste COVID-19 Schätzungen (@rv-enigma auf Bluesky):</b>")


def handle_deals(chat_id: str) -> str:
    """Ruft die neuesten Rabattaktionen aus Notion ab."""
    filter_payload = {
        "property": "Thema",
        "select": {"equals": "Rabatte Studierende/Dozierende"},
    }
    pages = query_notion_knowledge(filter_payload=filter_payload, page_size=5)
    return format_notion_results(pages, "🎓 <b>Aktuelle Rabatte für Studierende &amp; Dozierende (≥ 50% / Free):</b>")


def handle_suche(chat_id: str, query: str) -> str:
    """Durchsucht Notion-Wissensspeicher + lokalen BM25-Index hybrid nach Begriffen."""
    esc = telegram_ui.esc
    if not query.strip():
        return "Bitte Suchbegriff angeben, z. B. <code>/suche Long COVID</code> oder <code>/suche Stammzellen</code>."

    q = query.strip()
    filter_payload = {
        "or": [
            {"property": "Titel", "title": {"contains": q}},
            {"property": "Zusammenfassung", "rich_text": {"contains": q}},
        ]
    }
    pages = query_notion_knowledge(filter_payload=filter_payload, page_size=5)
    local_hits = get_store().search_knowledge_bm25(q, limit=5)

    notion_links = {
        (p.get("properties", {}).get("Link", {}).get("url") or "")
        for p in pages
    }
    extra_local = [h for h in local_hits if (h.get("link") or "") not in notion_links or not pages]

    if pages:
        msg = format_notion_results(pages, f"🔍 <b>Suchergebnisse im Wissensspeicher für „{esc(q)}“:</b>")
        if extra_local:
            msg += "\n\n🗂️ <b>Weitere Treffer im lokalen Archiv (BM25):</b>\n"
            for idx, lh in enumerate(extra_local[:3], 1):
                link = lh.get("link") or ""
                title_h = f'<a href="{esc(link)}">{esc(lh.get("title", ""))}</a>' if link else f"<b>{esc(lh.get('title', ''))}</b>"
                msg += f"{idx}. [{esc(lh.get('topic_id', ''))}] {title_h}\n"
        return msg

    if extra_local:
        lines = [f"🔍 <b>Lokale Wissensspeicher-Treffer für „{esc(q)}“:</b>\n"]
        for idx, lh in enumerate(extra_local[:5], 1):
            link = lh.get("link") or ""
            title_h = f'<a href="{esc(link)}">{esc(lh.get("title", ""))}</a>' if link else f"<b>{esc(lh.get('title', ''))}</b>"
            badge = f"[{esc(lh.get('topic_id', ''))}]"
            if lh.get("relevance"):
                badge += f" ({esc(lh['relevance'])})"
            lines.append(f"{idx}. {badge} {title_h}")
            if lh.get("summary"):
                lines.append(f"   <i>{esc(lh['summary'][:200])}</i>")
            lines.append("")
        return "\n".join(lines).strip()

    return f"🔍 <b>Suchergebnisse im Wissensspeicher für „{esc(q)}“:</b>\n\nKeine passenden Einträge gefunden."


def handle_synthese(chat_id: str) -> str:
    """Erstellt eine themenübergreifende Wochen-Synthese über gespeicherte RSS-Funde."""
    if rss_monitor is None or not hasattr(rss_monitor, "generate_weekly_rss_synthesis"):
        return "❌ Fehler: Synthese-Funktion (rss_monitor) nicht verfügbar."
    cfg = assistant.load_config() if assistant is not None else {}
    token = os.environ.get("NOTION_TOKEN", "").strip()
    notion = NotionClient(token) if token else None
    html_out, _ = rss_monitor.generate_weekly_rss_synthesis(cfg=cfg, store=get_store(), notion=notion, days=7)
    return html_out


def handle_status(chat_id: str) -> str:
    """Gibt Systemstatus, Anzahl Einträge, Feed-Health und Zeitpläne aus."""
    esc = telegram_ui.esc
    cfg = load_feeds_config()
    topics = cfg.get("topics", [])
    active_cnt = sum(1 for t in topics if t.get("enabled", True))
    paused_cnt = len(topics) - active_cnt

    all_pages = query_notion_knowledge(page_size=1)
    last_entry = "Unbekannt"
    if all_pages:
        created = all_pages[0].get("properties", {}).get("Erfasst", {}).get("date", {}).get("start")
        if created:
            last_entry = created[:16].replace("T", " ")

    rss_st = load_rss_state()
    seen_cnt = len(rss_st.get("seen", {})) // 2

    import platform as _plat
    _host = _plat.node() or "unknown"
    _commit = ""
    try:
        _cproc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True, timeout=3)
        if _cproc.returncode == 0:
            _commit = _cproc.stdout.strip()
    except Exception:
        pass
    _env_label = f"<code>{esc(sys.platform)}</code> @ <code>{esc(_host)}</code> (PID <code>{os.getpid()}</code>" + (
        f", Commit <code>{esc(_commit)}</code>" if _commit else ""
    ) + ")"

    lines = [
        "📊 <b>Wissensspeicher- & RSS-Status:</b>\n",
        f"• 🖥️ <b>Host:</b> {_env_label}",
        f"• Themen: <b>{active_cnt} aktiv</b>, {paused_cnt} pausiert",
        f"• Gesehene Artikel im Cache: <b>{seen_cnt}</b>",
        f"• Letzter Notion-Eintrag: <b>{esc(last_entry)}</b>",
    ]

    degraded = get_store().get_degraded_feeds()
    if degraded:
        lines.append(f"• ⚠️ <b>Degradierte Feeds:</b> {len(degraded)} (Circuit Breaker)")
        for df in degraded[:3]:
            lines.append(f"   ↳ {esc(df.get('source_name', '?'))}: {df.get('fail_count', 0)} Fehler")

    sched_lines = []
    if assistant is not None:
        try:
            sched_st = assistant._load_scheduler_state()
            now_berlin = assistant._scheduler_now("Europe/Berlin")
            time_str = now_berlin.strftime("%H:%M Uhr (%d.%m.%Y)")
            sched_lines.append(f"🕒 <b>Serverzeit (Berlin):</b> {esc(time_str)}")
            job_names = {
                "rss_morning": "📰 RSS-Monitor Morgen (07:30 Uhr)",
                "briefing_morning": "☀️ Morgen-Briefing (08:00 Uhr)",
                "rss_evening": "📰 RSS-Monitor Abend (18:30 Uhr)",
                "briefing_evening": "🌙 Abend-Briefing (19:30 Uhr)",
                "weekly_synthesis": "🧠 Wochen-Synthese (So 18:15 Uhr)",
            }
            for jid, jlabel in job_names.items():
                last = sched_st.get(jid, "noch nicht gelaufen")
                sched_lines.append(f"• <b>{jlabel}</b>\n   ↳ Zuletzt: <code>{esc(last)}</code>")
        except Exception:
            pass

    if sched_lines:
        lines.extend(["", "⏰ <b>Automatischer Zeitplan &amp; Status:</b>"] + sched_lines)
    else:
        lines.extend([
            "",
            "⏰ <b>Automatischer Zeitplan (im Daemon integriert, Europe/Berlin):</b>",
            "• Morgens: <b>07:30 Uhr</b> (RSS) | <b>08:00 Uhr</b> (Briefing)",
            "• Abends: <b>18:30 Uhr</b> (RSS) | <b>19:30 Uhr</b> (Briefing)",
        ])

    if _rss_monitor_import_error:
        lines.extend([
            "",
            f"⚠️ <i>Hinweis: rss_monitor Modul: {esc(_rss_monitor_import_error)}</i>",
            "<i>(Tipp: Auf dem Server <code>pip install -r requirements.txt</code> ausführen)</i>",
        ])

    lines.extend([
        "",
        '🔗 <a href="https://app.notion.com/p/c7bac0f9b2224aa6a0b42510fb987b7b">Wissensspeicher in Notion öffnen</a>',
    ])
    return "\n".join(lines)


def handle_natural_language(chat_id: str, text: str, token: str) -> None:
    """Behandelt Freitext-Fragen mit Jev-Choice-Router, Multi-Turn-Memory und Gemini-Tiering."""
    import jev_client

    store = get_store()
    store.append_chat_turn("rss", chat_id, "user", text)

    # 1. Jev-Choice Intent-Router (schnell, günstig, deterministisch ohne Textgenerierung)
    jclient = jev_client.JevClient()
    if jclient.is_configured:
        try:
            options = {
                "covid": "Fragen zu Covid-Zahlen, Inzidenz, Infektionslage oder Schätzungen von rv-enigma",
                "deals": "Fragen zu Studenten- oder Dozenten-Rabatten, Aktionen, Deals, Freebies",
                "suche": "Suche nach Begriffen, Studien oder Artikeln im Wissensspeicher",
                "themen": "Übersicht, Aktivierung oder Verwaltung von RSS-Themen",
                "synthese": "Zusammenfassung oder Wochen-Synthese über alle Themen hinweg",
                "run": "Suchlauf manuell starten oder aktualisieren",
                "general_question": "Freie inhaltliche Frage zu den Forschungsthemen oder Feedback",
            }
            choice, conf, _ = jclient.evaluate_choice(
                state=text[:250],
                instructions="Welche Aktion oder welches Thema fragt der Nutzer im RSS-Bot an?",
                options=options,
                question_id="rss_bot_intent",
            )
            if conf >= 0.70:
                if choice == "covid":
                    reply = handle_covid(chat_id)
                    send_message(token, chat_id, reply)
                    store.append_chat_turn("rss", chat_id, "assistant", reply[:400])
                    return
                elif choice == "deals":
                    reply = handle_deals(chat_id)
                    send_message(token, chat_id, reply)
                    store.append_chat_turn("rss", chat_id, "assistant", reply[:400])
                    return
                elif choice == "themen":
                    reply = handle_themen(chat_id)
                    send_message(token, chat_id, reply)
                    store.append_chat_turn("rss", chat_id, "assistant", reply[:400])
                    return
                elif choice == "synthese":
                    reply = handle_synthese(chat_id)
                    send_message(token, chat_id, reply)
                    store.append_chat_turn("rss", chat_id, "assistant", reply[:400])
                    return
                elif choice == "run":
                    handle_run_async(chat_id, token)
                    return
                elif choice == "suche":
                    clean_q = re.sub(r"(?i)^(suche|finde|zeig mir)\s+", "", text).strip()
                    reply = handle_suche(chat_id, clean_q or text)
                    send_message(token, chat_id, reply)
                    store.append_chat_turn("rss", chat_id, "assistant", reply[:400])
                    return
        except Exception as jexc:
            log.warning("Jev RSS-Bot Router Fehler (%s) – nutze Heuristik.", jexc)

    # 2. Deterministische Heuristik als Fallback
    lower = text.lower()
    if any(k in lower for k in ["covid", "corona", "infiziert", "zahlen", "inzidenz"]):
        reply = handle_covid(chat_id)
        send_message(token, chat_id, reply)
        store.append_chat_turn("rss", chat_id, "assistant", reply[:400])
        return
    if any(k in lower for k in ["rabatt", "deal", "student", "dozent", "kostenlos", "freebie"]):
        reply = handle_deals(chat_id)
        send_message(token, chat_id, reply)
        store.append_chat_turn("rss", chat_id, "assistant", reply[:400])
        return
    if "synthese" in lower or "wochenrückblick" in lower:
        reply = handle_synthese(chat_id)
        send_message(token, chat_id, reply)
        store.append_chat_turn("rss", chat_id, "assistant", reply[:400])
        return
    if lower.startswith("suche ") or lower.startswith("such "):
        q = re.sub(r"^suche?\s+", "", text, flags=re.IGNORECASE)
        reply = handle_suche(chat_id, q)
        send_message(token, chat_id, reply)
        store.append_chat_turn("rss", chat_id, "assistant", reply[:400])
        return

    # 3. Mit Multi-Turn-Verlauf + lokalen Wissens-Hits an Gemini/Ollama übergeben
    if assistant is not None:
        try:
            cfg = assistant.load_config()
            history = store.get_recent_chat_turns("rss", chat_id, limit=6)
            history_block = "\n".join(f"{h['role'].upper()}: {h['content'][:250]}" for h in history[:-1])
            local_ctx = store.search_knowledge_bm25(text, limit=3)
            ctx_block = "\n".join(
                f"- [{c.get('topic_id')}] {c.get('title')}: {(c.get('summary') or '')[:160]}"
                for c in local_ctx
            )
            prompt = (
                "Du bist der wissenschaftliche RSS- und Wissensspeicher-Assistent von Stefan.\n"
                + (f"Letzter Gesprächsverlauf:\n{history_block}\n\n" if history_block else "")
                + (f"Passende Wissensspeicher-Treffer:\n{ctx_block}\n\n" if ctx_block else "")
                + f"Aktuelle Nachricht von Stefan:\n\"{text}\"\n\n"
                "Antworte Stefan präzise, freundlich und sachlich auf Deutsch (nutze optional HTML-Tags wie <b> oder <i>, kein Markdown)."
            )
            reply = assistant.call_llm_text(prompt, cfg, task="chat_simple")
            send_message(token, chat_id, reply)
            store.append_chat_turn("rss", chat_id, "assistant", reply[:500])
            return
        except Exception as exc:
            log.warning("Fehler bei LLM-Antwort: %s", exc)

    send_message(
        token,
        chat_id,
        "Ich habe deine Nachricht erhalten. Nutze <code>/themen</code> für die Übersicht oder <code>/help</code> für alle Befehle.",
    )


# --------------------------------------------------------------------------------------
# Inline-Button Callback-Handler
# --------------------------------------------------------------------------------------

def handle_callback_query(token: str, cb: dict[str, Any], state: dict[str, Any]) -> None:
    """Verarbeitet Inline-Buttons aus RSS-Benachrichtigungen (Task, Idee, Irrelevant)."""
    cb_id = str(cb.get("id") or "")
    data = str(cb.get("data") or "")
    msg = cb.get("message") or {}
    chat_id = str(msg.get("chat", {}).get("id", ""))

    if not chat_id or not telegram_ui.is_authorized_chat(
        chat_id,
        env_keys=("TELEGRAM_RSS_CHAT_ID", "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USERS"),
    ):
        telegram_ui.answer_callback_query(token, cb_id, "⛔ Nicht autorisiert.", show_alert=True)
        return

    if ":" not in data:
        telegram_ui.answer_callback_query(token, cb_id, "Unbekannte Aktion.")
        return

    action, tok = data.split(":", 1)
    store = get_store()
    entry = store.get_callback_payload(tok)
    if not entry:
        telegram_ui.answer_callback_query(token, cb_id, "Dieser Button ist abgelaufen.", show_alert=True)
        return

    _, payload = entry
    title = payload.get("title", "RSS-Fund")
    link = payload.get("link", "")
    topic_id = payload.get("topic_id", "")
    summary = payload.get("summary", "")

    if action == "rss_bad":
        store.add_rss_feedback(topic_id=topic_id, title=title, link=link, feedback="irrelevant")
        telegram_ui.answer_callback_query(token, cb_id, "👎 Gemerkt – Filter lernt dazu!")
        send_message(
            token,
            chat_id,
            f"👎 <b>Feedback gespeichert:</b> „{telegram_ui.esc(title[:90])}“ wird künftig für Thema <code>{telegram_ui.esc(topic_id)}</code> als negatives Beispiel berücksichtigt.",
        )
        return

    if action in ("rss_task", "rss_idea"):
        notion_tok = os.environ.get("NOTION_TOKEN", "").strip()
        area = agent_tools.detect_area(f"{topic_id} {title} {summary}")
        if notion_tok and assistant is not None:
            try:
                cfg = assistant.load_config()
                notion = NotionClient(notion_tok)
                if action == "rss_task":
                    page_id = notion.create_task_page(
                        tasks_db_id=cfg["notion"]["tasks_database_id"],
                        title=f"Prüfen: {title[:70]}",
                        kind="To-Do",
                        priority="Mittel",
                        due_date=None,
                        next_step=f"Artikel/Studie lesen: {link}" if link else summary[:160],
                        area=area,
                        estimated_minutes=20,
                    )
                    store.push_undo("rss", chat_id, "archive_task", {"notion_page_id": page_id, "title": title})
                    telegram_ui.answer_callback_query(token, cb_id, "📌 Als Notion-Task gespeichert!")
                    send_message(
                        token,
                        chat_id,
                        f"📌 <b>Aufgabe angelegt:</b> {telegram_ui.esc(title[:80])}\n<i>Bereich:</i> {telegram_ui.esc(area)}",
                    )
                else:
                    page_id = notion.create_task_page(
                        tasks_db_id=cfg["notion"]["tasks_database_id"],
                        title=f"Idee: {title[:70]}",
                        kind="KI-Vorschlag / Idee",
                        priority="Niedrig",
                        due_date=None,
                        next_step=f"Quelle: {link}" if link else summary[:160],
                        area=area,
                    )
                    store.push_undo("rss", chat_id, "archive_task", {"notion_page_id": page_id, "title": title})
                    telegram_ui.answer_callback_query(token, cb_id, "🎬 Als Idee in Notion gespeichert!")
                    send_message(
                        token,
                        chat_id,
                        f"🎬 <b>Idee vorgemerkt:</b> {telegram_ui.esc(title[:80])}",
                    )
                return
            except Exception as exc:
                log.warning("RSS-Button Notion-Erstellung fehlgeschlagen: %s", exc)

        telegram_ui.answer_callback_query(token, cb_id, "ℹ️ Notion nicht aktiv – nur lokal vorgemerkt.")
        send_message(token, chat_id, f"📌 Vorgemerkt ({telegram_ui.esc(action)}): {telegram_ui.esc(title)}")
        return

    telegram_ui.answer_callback_query(token, cb_id, "Unbekannte Aktion.")


# --------------------------------------------------------------------------------------
# Telegram Polling
# --------------------------------------------------------------------------------------

def process_updates(token: str, state: dict[str, Any], poll_timeout: int = 15) -> int:
    """Fragt neue Telegram-Nachrichten & Callbacks via Long-Polling ab."""
    offset = int(state.get("rss_telegram_offset", 0))
    api_base = f"https://api.telegram.org/bot{token}"

    try:
        req = urllib.request.Request(f"{api_base}/getUpdates?offset={offset}&timeout={poll_timeout}")
        with urllib.request.urlopen(req, timeout=poll_timeout + 10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.debug("Polling-Timeout oder Verbindungsfehler: %s", exc)
        return 0

    if not data.get("ok"):
        return 0

    results = data.get("result", [])
    count = 0

    for update in results:
        update_id = update["update_id"]
        state["rss_telegram_offset"] = max(int(state.get("rss_telegram_offset", 0)), update_id + 1)
        save_rss_state(state)
        count += 1

        if "callback_query" in update:
            try:
                handle_callback_query(token, update["callback_query"], state)
            except Exception as exc:
                log.exception("Fehler bei Callback-Query in Update %s: %s", update_id, exc)
            continue

        msg = update.get("message") or update.get("edited_message")
        if not msg:
            continue

        chat_id = str(msg.get("chat", {}).get("id", ""))
        if not chat_id:
            continue

        # Autorisierung VOR jeglicher State-Mutation oder Befehlsausführung prüfen
        if not telegram_ui.is_authorized_chat(
            chat_id,
            env_keys=("TELEGRAM_RSS_CHAT_ID", "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USERS"),
        ):
            log.warning("Unautorisierter RSS-Bot-Zugriff von Chat-ID %s blockiert.", chat_id)
            continue

        try:
            # Chat-ID für Benachrichtigungen merken
            if state.get("last_rss_chat_id") != chat_id:
                state["last_rss_chat_id"] = chat_id
                save_rss_state(state)
                log.info("Chat-ID %s für RSS-Bot registriert.", chat_id)

            raw_text = (msg.get("text") or "").strip()
            voice = msg.get("voice") or msg.get("audio")
            doc = msg.get("document")

            # Sprachnachricht transkribieren
            if voice and assistant is not None:
                file_id = voice.get("file_id")
                if file_id:
                    tmp_audio = telegram_ui.download_telegram_file(token, file_id, BASE_DIR / "temp", prefix="voice")
                    if tmp_audio:
                        try:
                            cfg = assistant.load_config()
                            raw_text = assistant.transcribe_audio(tmp_audio, cfg)
                            send_message(token, chat_id, f"🎤 <i>Erkannt:</i> „{telegram_ui.esc(raw_text)}“")
                        except Exception as exc:
                            log.warning("Fehler bei Audio-Transkription: %s", exc)
                        finally:
                            try:
                                tmp_audio.unlink()
                            except Exception:
                                pass

            # Dokument-Upload (PDF/TXT/MD -> Studienanalyse)
            if doc and not raw_text:
                file_id = doc.get("file_id")
                mime = doc.get("mime_type") or "application/pdf"
                fname = doc.get("file_name") or "dokument"
                caption = (msg.get("caption") or "").strip()
                if file_id and (mime == "application/pdf" or fname.lower().endswith((".pdf", ".txt", ".md"))):
                    tmp_doc = telegram_ui.download_telegram_file(token, file_id, BASE_DIR / "temp", prefix="doc")
                    if tmp_doc:
                        try:
                            cfg = assistant.load_config() if assistant is not None else {}
                            if fname.lower().endswith((".txt", ".md")):
                                txt_content = tmp_doc.read_text(encoding="utf-8", errors="replace")[:4000]
                                handle_natural_language(chat_id, f"{caption}\n\nDokument ({fname}):\n{txt_content}", token)
                            else:
                                ref = assistant.get_date_reference_block() if assistant is not None else ""
                                res_mm = agent_tools.analyze_multimodal_file(tmp_doc, mime, caption, cfg, ref)
                                send_message(
                                    token,
                                    chat_id,
                                    f"📄 <b>{telegram_ui.esc(res_mm.get('title', fname))}</b>\n\n"
                                    f"{telegram_ui.esc(res_mm.get('summary', ''))}\n"
                                    f"<i>Bereich: {telegram_ui.esc(res_mm.get('area', ''))}</i>",
                                )
                                notion_tok = os.environ.get("NOTION_TOKEN", "").strip()
                                if notion_tok and assistant is not None:
                                    NotionClient(notion_tok).create_inbox_page(
                                        inbox_db_id=cfg["notion"]["inbox_database_id"],
                                        title=res_mm.get("title", fname),
                                        source="Telegram PDF",
                                        raw_text=res_mm.get("raw_text", ""),
                                        summary=res_mm.get("summary", ""),
                                        suggestions_text="",
                                        area=res_mm.get("area"),
                                    )
                        except Exception as exc:
                            log.warning("Fehler bei Dokument-Verarbeitung: %s", exc)
                            send_message(token, chat_id, f"❌ Dokument konnte nicht analysiert werden: {telegram_ui.esc(exc)}")
                        finally:
                            try:
                                tmp_doc.unlink()
                            except Exception:
                                pass
                        continue

            if not raw_text:
                continue

            log.info("Empfangen von %s: %s", chat_id, raw_text[:60])

            parts = raw_text.strip().split()
            cmd = parts[0].lower().split("@")[0]  # strip @RSS8472Bot
            arg = " ".join(parts[1:]).strip()

            if cmd in ("/start", "/help", "hilfe", "start"):
                send_message(token, chat_id, handle_start_or_help(chat_id))
            elif cmd in ("/themen", "/topics", "themen"):
                send_message(token, chat_id, handle_themen(chat_id))
            elif cmd in ("/an", "/enable", "/aktivieren"):
                send_message(token, chat_id, handle_toggle(chat_id, arg, enable=True))
            elif cmd in ("/aus", "/disable", "/deaktivieren", "/pause"):
                send_message(token, chat_id, handle_toggle(chat_id, arg, enable=False))
            elif cmd in ("/neu", "/add"):
                send_message(token, chat_id, handle_neu(chat_id, arg))
            elif cmd in ("/loeschen", "/remove", "/del"):
                send_message(token, chat_id, handle_loeschen(chat_id, arg))
            elif cmd in ("/run", "/sync", "sync", "run"):
                handle_run_async(chat_id, token)
            elif cmd in ("/synthese", "/synthesis", "synthese"):
                send_message(token, chat_id, handle_synthese(chat_id))
            elif cmd in ("/covid", "/corona", "covid"):
                send_message(token, chat_id, handle_covid(chat_id))
            elif cmd in ("/deals", "/rabatte", "deals", "rabatte"):
                send_message(token, chat_id, handle_deals(chat_id))
            elif cmd in ("/suche", "/search", "suche"):
                send_message(token, chat_id, handle_suche(chat_id, arg))
            elif cmd in ("/status", "status"):
                send_message(token, chat_id, handle_status(chat_id))
            elif cmd in ("/update", "/upgrade", "update", "/pull"):
                send_message(token, chat_id, "⏳ Prüfe GitHub auf Updates...")
                if assistant is not None:
                    needs_restart, update_msg = assistant.perform_self_update()
                    send_message(token, chat_id, update_msg)
                    if needs_restart:
                        assistant.trigger_restart()
                else:
                    send_message(token, chat_id, "❌ Fehler: assistant-Modul nicht geladen.")
            else:
                handle_natural_language(chat_id, raw_text, token)
        except Exception as exc:
            log.exception("Fehler bei Verarbeitung von Nachricht aus Update %s: %s", update_id, exc)
            try:
                send_message(token, chat_id, f"❌ Ein interner Fehler ist aufgetreten: {telegram_ui.esc(exc)}")
            except Exception:
                pass

    return count


def run_bot_daemon(poll_timeout: int = 15) -> None:
    """Dauerhafte Abfrageschleife für den RSS-Bot."""
    token = get_active_token()
    if not token:
        log.error("Kein Telegram-Bot-Token konfiguriert (TELEGRAM_RSSBOT_TOKEN oder TELEGRAM_BOT_TOKEN fehlt).")
        return

    telegram_ui.register_bot_commands(token, telegram_ui.RSS_BOT_COMMANDS)
    log.info("RSS-Bot Daemon gestartet (Token: %s...). Warte auf Nachrichten...", token[:10])
    while True:
        try:
            state = load_rss_state()
            process_updates(token, state, poll_timeout=poll_timeout)
        except Exception as exc:
            log.warning("Fehler in Polling-Schleife: %s", exc)
            time.sleep(5)


_background_thread: threading.Thread | None = None

def start_in_background(poll_timeout: int = 15) -> threading.Thread | None:
    """Startet den RSS-Bot in einem Hintergrund-Thread (z. B. aus assistant.py run_daemon)."""
    global _background_thread
    token = get_active_token()
    if not token:
        log.warning("Kein RSS-Bot Token gefunden – Hintergrund-Thread wird nicht gestartet.")
        return None

    if _background_thread is not None and _background_thread.is_alive():
        log.info("RSS-Bot Hintergrund-Thread läuft bereits.")
        return _background_thread

    def _loop():
        log.info("RSS-Bot Hintergrund-Thread aktiv.")
        run_bot_daemon(poll_timeout=poll_timeout)

    _background_thread = threading.Thread(target=_loop, daemon=True, name="RSSBotDaemon")
    _background_thread.start()
    return _background_thread


# --------------------------------------------------------------------------------------
# CLI Entry Point
# --------------------------------------------------------------------------------------

def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Dedizierter Telegram-Bot für den RSS-Wissensspeicher")
    parser.add_argument("--once", action="store_true", help="Einmalig prüfen und beenden")
    parser.add_argument("--timeout", type=int, default=15, help="Long-Polling Timeout in Sekunden")
    args = parser.parse_args()

    token = get_active_token()
    if not token:
        print("FEHLER: TELEGRAM_RSSBOT_TOKEN ist nicht gesetzt!")
        sys.exit(1)

    if args.once:
        state = load_rss_state()
        cnt = process_updates(token, state, poll_timeout=args.timeout)
        print(f"{cnt} Update(s) verarbeitet.")
    else:
        run_bot_daemon(poll_timeout=args.timeout)


if __name__ == "__main__":
    main()
