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

# Lokale Module einbinden
try:
    import assistant
except ImportError:
    assistant = None

try:
    import rss_monitor
except ImportError:
    rss_monitor = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [RSSBot] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rss_bot")

STATE_PATH = BASE_DIR / "rss_state.json"
FEEDS_CONFIG_PATH = BASE_DIR / "feeds.json"


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
    if not STATE_PATH.exists():
        return {"seen": {}, "last_rss_chat_id": "", "rss_telegram_offset": 0}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
        data.setdefault("seen", {})
        data.setdefault("last_rss_chat_id", "")
        data.setdefault("rss_telegram_offset", 0)
        return data
    except Exception as exc:
        log.warning("Fehler beim Lesen von rss_state.json: %s", exc)
        return {"seen": {}, "last_rss_chat_id": "", "rss_telegram_offset": 0}


def save_rss_state(state: dict[str, Any]) -> None:
    try:
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(STATE_PATH)
    except Exception as exc:
        log.warning("Fehler beim Speichern von rss_state.json: %s", exc)


def send_message(token: str, chat_id: str, text: str, parse_mode: str | None = None) -> bool:
    """Sendet eine Nachricht an Telegram. Teilt lange Texte automatisch in Chunks."""
    if not token or not chat_id:
        return False
    
    success = True
    for i in range(0, len(text), 3900):
        chunk = text[i:i + 3900]
        payload_data: dict[str, Any] = {"chat_id": chat_id, "text": chunk}
        if parse_mode:
            payload_data["parse_mode"] = parse_mode
        try:
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=json.dumps(payload_data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if not data.get("ok"):
                    success = False
        except urllib.error.HTTPError as he:
            log.warning("Telegram HTTP-Fehler (%s) an %s: %s", he.code, chat_id, he.reason)
            return False
        except Exception as exc:
            log.warning("Telegram-Fehler an %s: %s", chat_id, exc)
            return False
    return success


# --------------------------------------------------------------------------------------
# Notion-Abfragen für den Wissensspeicher
# --------------------------------------------------------------------------------------

def query_notion_knowledge(filter_payload: dict[str, Any] | None = None, page_size: int = 5) -> list[dict[str, Any]]:
    """Fragt die Wissensspeicher-Datenbank in Notion ab."""
    token = os.environ.get("NOTION_TOKEN", "").strip()
    if not token or rss_monitor is None:
        return []
    
    feeds_cfg = rss_monitor.load_feeds_config()
    db_id = feeds_cfg.get("notion_database_id", "c7bac0f9-b222-4aa6-a0b4-2510fb987b7b").replace("-", "")
    
    payload: dict[str, Any] = {"page_size": page_size}
    if filter_payload:
        payload["filter"] = filter_payload
    payload["sorts"] = [{"property": "Erfasst", "direction": "descending"}]

    try:
        req = urllib.request.Request(
            f"https://api.notion.com/v1/databases/{db_id}/query",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("results", [])
    except Exception as exc:
        log.warning("Notion-Abfragefehler: %s", exc)
        return []


def format_notion_results(results: list[dict[str, Any]], title_header: str) -> str:
    """Formatiert Notion-Seiten lesbar für Telegram."""
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
        
        badge = f"[{thema}]" if thema else ""
        if rel:
            badge += f" ({rel})"
            
        lines.append(f"{idx}. {badge} *{title}*")
        if summary:
            # Erste 1-2 Sätze
            first_sentence = summary.split(". ")[0] + ("." if not summary.endswith(".") else "")
            lines.append(f"   _{first_sentence}_")
        if link:
            lines.append(f"   🔗 {link}")
        lines.append("")
        
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------------------
# Befehls-Handler
# --------------------------------------------------------------------------------------

def handle_start_or_help(chat_id: str) -> str:
    return (
        "📡 *Willkommen beim RSS- & Wissensspeicher-Bot!*\n\n"
        "Ich sammle, bewerte und dedupliziere automatisch wissenschaftliche Studien, "
        "Durchbrüche, Bluesky-Signale und Rabattaktionen – getrennt von deinen privaten "
        "Kalenderterminen und Aufgaben.\n\n"
        "📋 *Verfügbare Befehle:*\n"
        "• `/themen` – Übersicht aller Themen & Status\n"
        "• `/an <id>` – Thema aktivieren (z. B. `/an parkinson`)\n"
        "• `/aus <id>` – Thema pausieren\n"
        "• `/neu <Name>` – Neues Thema mit KI-Recherche anlegen\n"
        "• `/loeschen <id>` – Thema dauerhaft entfernen\n"
        "• `/run` – Sofortigen RSS-/Bluesky-Suchlauf starten\n"
        "• `/covid` – Aktuelle Infektionsschätzung von @rv-enigma\n"
        "• `/deals` – Aktuelle Rabattaktionen für Studenten/Dozenten\n"
        "• `/suche <Begriff>` – Notion-Wissensspeicher durchsuchen\n"
        "• `/status` – Systemstatus, Eintragszahl & nächste Termine\n"
        "• `/update` – Neuesten Code von GitHub laden & Bot neu starten\n\n"
        "💡 _Tipp: Du kannst mir auch einfach Fragen zu den Themen stellen "
        "oder Sprachnachrichten senden!_"
    )


def handle_themen(chat_id: str) -> str:
    if rss_monitor is None:
        return "Fehler: rss_monitor Modul nicht geladen."
    cfg = rss_monitor.load_feeds_config()
    topics = cfg.get("topics", [])
    if not topics:
        return "Keine Themen in feeds.json konfiguriert."

    lines = ["📡 *Konfigurierte Themengebiete:*\n"]
    for t in topics:
        status = "🟢 Aktiv" if t.get("enabled", True) else "⚪ Pausiert"
        emoji = t.get("emoji", "📌")
        name = t.get("name", "Unbenannt")
        tid = t.get("id", "")
        sources_cnt = len(t.get("sources", []))
        min_rel = t.get("min_relevance", "Mittel")

        lines.append(f"{emoji} *{name}* (`{tid}`) – {status}")
        lines.append(f"   ↳ {sources_cnt} Quellen | Min-Relevanz: {min_rel}")
        focus = t.get("focus", "")
        if focus:
            short_focus = focus[:120] + ("..." if len(focus) > 120 else "")
            lines.append(f"   ↳ _{short_focus}_")
        lines.append("")

    lines.append("Steuerung:")
    lines.append("• `/an <id>` / `/aus <id>` zum Ein- oder Ausschalten")
    lines.append("• `/neu <Name>` um ein neues Thema per KI zu erstellen")
    lines.append("• `/run` für einen sofortigen Suchlauf")
    return "\n".join(lines)


def handle_toggle(chat_id: str, topic_id: str, enable: bool) -> str:
    if not topic_id.strip():
        return f"Bitte Themen-ID angeben, z. B. `/{'an' if enable else 'aus'} parkinson`."
    if rss_monitor is None:
        return "Fehler: rss_monitor nicht verfügbar."
    ok = rss_monitor.cmd_topic_set_enabled(topic_id.strip(), enable)
    if ok:
        status_str = "aktiviert 🟢" if enable else "pausiert ⚪"
        return f"Thema `{topic_id}` wurde {status_str}."
    return f"Thema `{topic_id}` nicht gefunden. Nutze `/themen` für die Liste."


def handle_neu(chat_id: str, new_name: str) -> str:
    if not new_name.strip():
        return "Bitte gib einen Namen an, z. B. `/neu Multiple Sklerose`."
    if rss_monitor is None:
        return "Fehler: rss_monitor nicht verfügbar."
    
    import argparse as _ap
    ns = _ap.Namespace(name=new_name.strip(), id=None, emoji=None, focus=None, rss=None, auto=True)
    ok = rss_monitor.cmd_topic_add(ns)
    if ok:
        return (
            f"✅ Thema *{new_name}* wurde erfolgreich angelegt!\n"
            f"Quellen aus Google News, Europe PMC und Bluesky wurden automatisch konfiguriert "
            f"und aktiviert. Starte mit `/run` den ersten Suchlauf."
        )
    return f"❌ Konnte Thema *{new_name}* nicht anlegen."


def handle_loeschen(chat_id: str, topic_id: str) -> str:
    if not topic_id.strip():
        return "Bitte Themen-ID angeben, z. B. `/loeschen mein_thema`."
    if rss_monitor is None:
        return "Fehler: rss_monitor nicht verfügbar."
    
    ok = rss_monitor.cmd_topic_remove(topic_id.strip(), confirm=True)
    if ok:
        return f"🗑️ Thema `{topic_id}` wurde gelöscht."
    return f"❌ Thema `{topic_id}` konnte nicht gelöscht werden (nicht gefunden?)."


def handle_run_async(chat_id: str, token: str) -> None:
    """Führt rss_monitor.run() in einem separaten Thread aus."""
    if rss_monitor is None:
        send_message(token, chat_id, "Fehler: rss_monitor nicht verfügbar.")
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
        "📡 *RSS- & Bluesky-Suchlauf gestartet!*\n"
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

                # Prüfen, ob neue Funde gemeldet wurden:
                st = load_rss_state()
                last_run = (st.get("runs") or [{}])[0]
                new_cnt = last_run.get("new", 0)
                if new_cnt == 0:
                    send_message(
                        token,
                        chat_id,
                        "✅ *Suchlauf abgeschlossen.*\n"
                        "Es wurden keine neuen relevanten Studien oder Artikel gefunden – alle Quellen sind aktuell.",
                    )
            finally:
                if hasattr(rss_monitor, "release_lock"):
                    rss_monitor.release_lock()
        except Exception as exc:
            log.exception("Fehler beim manuellen Suchlauf: %s", exc)
            send_message(token, chat_id, f"❌ Fehler beim Suchlauf: {exc}")

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def handle_covid(chat_id: str) -> str:
    """Ruft die neuesten Covid-Zahlen von @rv-enigma aus Notion ab."""
    filter_payload = {
        "property": "Quelle",
        "rich_text": {"contains": "rv-enigma"},
    }
    pages = query_notion_knowledge(filter_payload=filter_payload, page_size=5)
    return format_notion_results(pages, "🦠 *Neueste COVID-19 Schätzungen (@rv-enigma auf Bluesky):*")


def handle_deals(chat_id: str) -> str:
    """Ruft die neuesten Rabattaktionen aus Notion ab."""
    filter_payload = {
        "property": "Thema",
        "select": {"equals": "Rabatte Studierende/Dozierende"},
    }
    pages = query_notion_knowledge(filter_payload=filter_payload, page_size=5)
    return format_notion_results(pages, "🎓 *Aktuelle Rabatte für Studierende & Dozierende (≥ 50% / Free):*")


def handle_suche(chat_id: str, query: str) -> str:
    """Durchsucht den Wissensspeicher nach Begriffen."""
    if not query.strip():
        return "Bitte Suchbegriff angeben, z. B. `/suche Long COVID` oder `/suche Stammzellen`."
    
    q = query.strip()
    filter_payload = {
        "or": [
            {"property": "Titel", "title": {"contains": q}},
            {"property": "Zusammenfassung", "rich_text": {"contains": q}},
        ]
    }
    pages = query_notion_knowledge(filter_payload=filter_payload, page_size=5)
    return format_notion_results(pages, f"🔍 *Suchergebnisse im Wissensspeicher für „{q}“:*")


def handle_status(chat_id: str) -> str:
    """Gibt Systemstatus, Anzahl Einträge und Zeitpläne aus."""
    if rss_monitor is None:
        return "Fehler: rss_monitor nicht geladen."
    
    cfg = rss_monitor.load_feeds_config()
    topics = cfg.get("topics", [])
    active_cnt = sum(1 for t in topics if t.get("enabled", True))
    paused_cnt = len(topics) - active_cnt

    # Letzter Lauf aus Digest oder Wissensspeicher
    all_pages = query_notion_knowledge(page_size=1)
    last_entry = "Unbekannt"
    if all_pages:
        created = all_pages[0].get("properties", {}).get("Erfasst", {}).get("date", {}).get("start")
        if created:
            last_entry = created[:16].replace("T", " ")

    rss_st = load_rss_state()
    seen_cnt = len(rss_st.get("seen", {})) // 2  # Link + Title Hash pro Eintrag

    lines = [
        "📊 *Wissensspeicher- & RSS-Status:*\n",
        f"• Themen: *{active_cnt} aktiv*, {paused_cnt} pausiert",
        f"• Gesehene Artikel im Cache: *{seen_cnt}*",
        f"• Letzter Notion-Eintrag: *{last_entry}*",
    ]

    sched_lines = []
    if assistant is not None:
        try:
            sched_st = assistant._load_scheduler_state()
            now_berlin = assistant._scheduler_now("Europe/Berlin")
            time_str = now_berlin.strftime("%H:%M Uhr (%d.%m.%Y)")
            sched_lines.append(f"🕒 *Serverzeit (Berlin):* {time_str}")
            job_names = {
                "rss_morning": "📰 RSS-Monitor Morgen (07:30 Uhr)",
                "briefing_morning": "☀️ Morgen-Briefing (08:00 Uhr)",
                "rss_evening": "📰 RSS-Monitor Abend (18:30 Uhr)",
                "briefing_evening": "🌙 Abend-Briefing (19:30 Uhr)",
            }
            for jid, jlabel in job_names.items():
                last = sched_st.get(jid, "noch nicht gelaufen")
                sched_lines.append(f"• *{jlabel}*\n   ↳ Zuletzt: `{last}`")
        except Exception:
            pass

    if sched_lines:
        lines.extend(["", "⏰ *Automatischer Zeitplan & Status:*"] + sched_lines)
    else:
        lines.extend([
            "",
            "⏰ *Automatischer Zeitplan (im Daemon integriert, Europe/Berlin):*",
            "• Morgens: *07:30 Uhr* (RSS) | *08:00 Uhr* (Briefing)",
            "• Abends: *18:30 Uhr* (RSS) | *19:30 Uhr* (Briefing)",
        ])

    lines.extend([
        "",
        "🔗 [Wissensspeicher in Notion öffnen](https://app.notion.com/p/c7bac0f9b2224aa6a0b42510fb987b7b)",
    ])
    return "\n".join(lines)


def handle_natural_language(chat_id: str, text: str, token: str) -> None:
    """Behandelt Freitext-Fragen oder Notizen mit schnellem Jev-Choice-Router und Gemini-Tiering."""
    import jev_client

    # 1. Jev-Choice Intent-Router (schnell, günstig, deterministisch ohne Textgenerierung)
    jclient = jev_client.JevClient()
    if jclient.is_configured:
        try:
            options = {
                "covid": "Fragen zu Covid-Zahlen, Inzidenz, Infektionslage oder Schätzungen von rv-enigma",
                "deals": "Fragen zu Studenten- oder Dozenten-Rabatten, Aktionen, Deals, Freebies",
                "suche": "Suche nach Begriffen, Studien oder Artikeln im Wissensspeicher",
                "themen": "Übersicht, Aktivierung oder Verwaltung von RSS-Themen",
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
                    send_message(token, chat_id, handle_covid(chat_id))
                    return
                elif choice == "deals":
                    send_message(token, chat_id, handle_deals(chat_id))
                    return
                elif choice == "themen":
                    send_message(token, chat_id, handle_themen(chat_id))
                    return
                elif choice == "run":
                    handle_run_async(chat_id, token)
                    return
                elif choice == "suche":
                    clean_q = re.sub(r"(?i)^(suche|finde|zeig mir)\s+", "", text).strip()
                    send_message(token, chat_id, handle_suche(chat_id, clean_q or text))
                    return
        except Exception as jexc:
            log.warning("Jev RSS-Bot Router Fehler (%s) – nutze Heuristik.", jexc)

    # 2. Deterministische Heuristik als Fallback
    lower = text.lower()
    if any(k in lower for k in ["covid", "corona", "infiziert", "zahlen", "inzidenz"]):
        send_message(token, chat_id, handle_covid(chat_id))
        return
    if any(k in lower for k in ["rabatt", "deal", "student", "dozent", "kostenlos", "freebie"]):
        send_message(token, chat_id, handle_deals(chat_id))
        return
    if lower.startswith("suche ") or lower.startswith("such "):
        q = re.sub(r"^suche?\s+", "", text, flags=re.IGNORECASE)
        send_message(token, chat_id, handle_suche(chat_id, q))
        return

    # 3. Mit Gemini 3.5 Flash-Lite (Tier 'chat_simple', thinking=minimal) oder Ollama beantworten
    if assistant is not None:
        try:
            cfg = assistant.load_config()
            prompt = (
                "Du bist der wissenschaftliche RSS- und Wissensspeicher-Assistent von Stefan. "
                "Stefan schickt dir folgende Nachricht auf Telegram:\n"
                f"\"{text}\"\n\n"
                "Antworte Stefan präzise, freundlich und sachlich auf Deutsch. "
                "Wenn er nach neuen Erkenntnissen oder Studien zu Parkinson, Long COVID, ME/CFS "
                "oder Rabatten fragt, erkläre kurz die Relevanz oder verweise auf die Befehle /suche, /themen oder /run."
            )
            reply = assistant.call_llm_text(prompt, cfg, task="chat_simple")
            send_message(token, chat_id, reply)
            return
        except Exception as exc:
            log.warning("Fehler bei LLM-Antwort: %s", exc)

    send_message(
        token,
        chat_id,
        "Ich habe deine Nachricht erhalten. Nutze `/themen` für die Übersicht oder `/help` für alle Befehle.",
    )


# --------------------------------------------------------------------------------------
# Telegram Polling
# --------------------------------------------------------------------------------------

def process_updates(token: str, state: dict[str, Any], poll_timeout: int = 15) -> int:
    """Fragt neue Telegram-Nachrichten via Long-Polling ab."""
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

        msg = update.get("message") or update.get("edited_message")
        if not msg:
            continue

        chat_id = str(msg.get("chat", {}).get("id", ""))
        if not chat_id:
            continue

        try:
            # Chat-ID für Benachrichtigungen merken
            if state.get("last_rss_chat_id") != chat_id:
                state["last_rss_chat_id"] = chat_id
                save_rss_state(state)
                log.info("Chat-ID %s für RSS-Bot registriert.", chat_id)

            raw_text = (msg.get("text") or "").strip()
            voice = msg.get("voice") or msg.get("audio")

            # Sprachnachricht transkribieren
            if voice and assistant is not None:
                file_id = voice.get("file_id")
                if file_id:
                    try:
                        f_req = urllib.request.Request(f"{api_base}/getFile?file_id={file_id}")
                        with urllib.request.urlopen(f_req, timeout=15) as f_resp:
                            f_info = json.loads(f_resp.read().decode("utf-8"))
                            f_path = f_info.get("result", {}).get("file_path")
                        if f_path:
                            dl_url = f"https://api.telegram.org/file/bot{token}/{f_path}"
                            tmp_audio = BASE_DIR / "temp" / f"voice_{int(time.time())}.ogg"
                            tmp_audio.parent.mkdir(parents=True, exist_ok=True)
                            with urllib.request.urlopen(dl_url, timeout=30) as dl, open(tmp_audio, "wb") as out:
                                out.write(dl.read())
                            cfg = assistant.load_config()
                            raw_text = assistant.transcribe_audio(tmp_audio, cfg)
                            send_message(token, chat_id, f"🎤 _Erkannt:_ „{raw_text}“")
                            try:
                                tmp_audio.unlink()
                            except Exception:
                                pass
                    except Exception as exc:
                        log.warning("Fehler bei Audio-Download/Transkription: %s", exc)

            if not raw_text:
                continue

            log.info("Empfangen von %s: %s", chat_id, raw_text[:60])

            # Befehle auswerten
            lower = raw_text.strip().lower()
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
                send_message(token, chat_id, f"❌ Ein interner Fehler ist aufgetreten: {exc}")
            except Exception:
                pass

    return count


def run_bot_daemon(poll_timeout: int = 15) -> None:
    """Dauerhafte Abfrageschleife für den RSS-Bot."""
    token = get_active_token()
    if not token:
        log.error("Kein Telegram-Bot-Token konfiguriert (TELEGRAM_RSSBOT_TOKEN oder TELEGRAM_BOT_TOKEN fehlt).")
        return

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
