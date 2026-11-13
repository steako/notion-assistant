"""
Einheitliche Telegram-UI & Transport-Schicht für Haupt-Bot und RSS-Bot.

Funktionen:
- Sicheres HTML-Formatting mit automatischem Plaintext-Fallback bei HTTP 400
- Chunking langer Nachrichten (> 3900 Zeichen) + Retries bei 429/5xx
- InlineKeyboardMarkup-Helfer, Callback-Query-Antworten & Message-Editing
- Registrierung der Bot-Menüs (`setMyCommands`)
- Einheitliche Chat-ID-Autorisierung für Haupt- und RSS-Bot
- Download von Voice-, Audio-, Foto- und Dokument-Dateien
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib import error, request

log = logging.getLogger("integrations.telegram")


def esc(text: Any) -> str:
    """Escapet Text sicher für Telegram parse_mode='HTML'."""
    return html.escape(str(text or ""), quote=False)


def strip_html_tags(text: str) -> str:
    """Entfernt HTML-Tags für den Plaintext-Fallback."""
    clean = re.sub(r"<[^>]+>", "", text)
    return html.unescape(clean)


def build_inline_keyboard(rows: list[list[tuple[str, str]]]) -> dict[str, Any]:
    """
    Erzeugt ein Telegram InlineKeyboardMarkup.
    Jeder Button ist ein Tupel `(label, callback_data)` oder `(label, url)` (wenn mit http startend).
    """
    keyboard: list[list[dict[str, str]]] = []
    for row in rows:
        btn_row: list[dict[str, str]] = []
        for label, data in row:
            if data.startswith(("http://", "https://")):
                btn_row.append({"text": label[:64], "url": data})
            else:
                btn_row.append({"text": label[:64], "callback_data": data[:64]})
        if btn_row:
            keyboard.append(btn_row)
    return {"inline_keyboard": keyboard}


def get_allowed_chat_ids(env_keys: tuple[str, ...] = ("TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USERS")) -> set[str]:
    """Liest konfigurierte erlaubte Chat-IDs (einzeln oder kommagetrennt)."""
    allowed: set[str] = set()
    for key in env_keys:
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        for part in re.split(r"[,;\s]+", raw):
            p = part.strip()
            if p and (p.lstrip("-").isdigit()):
                allowed.add(p)
    return allowed


def is_authorized_chat(
    chat_id: str | int,
    env_keys: tuple[str, ...] = ("TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USERS"),
) -> bool:
    """
    Prüft, ob eine Chat-ID berechtigt ist.
    Wenn keine erlaubten IDs konfiguriert sind, wird True zurückgegeben (Erstkonfiguration).
    """
    allowed = get_allowed_chat_ids(env_keys)
    if not allowed:
        return True
    return str(chat_id).strip() in allowed


def _tg_post(token: str, method: str, payload: dict[str, Any], timeout: int = 15) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = json.dumps(payload).encode("utf-8")
    last_exc: Exception | None = None
    for attempt in range(3):
        req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as he:
            body = ""
            try:
                body = he.read().decode("utf-8", errors="replace")
            except Exception:
                body = str(he.reason)
            if he.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise RuntimeError(f"Telegram HTTP {he.code}: {body}") from he
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(0.8 * (attempt + 1))
                continue
            raise RuntimeError(f"Telegram Fehler ({method}): {last_exc}") from exc
    return {"ok": False}


def send_message(
    token: str,
    chat_id: str | int,
    text: str,
    parse_mode: str | None = "HTML",
    reply_markup: dict[str, Any] | None = None,
    disable_web_page_preview: bool = True,
) -> bool:
    """
    Sendet eine Nachricht an Telegram:
    - Teilt Texte > 3900 Zeichen in Chunks
    - Hängt `reply_markup` an den letzten Chunk
    - Fällt bei HTML-Parse-Fehlern (HTTP 400) automatisch auf Plaintext zurück
    """
    if not token or not chat_id or not text:
        return False

    chunks = [text[i : i + 3900] for i in range(0, len(text), 3900)]
    all_ok = True
    for idx, chunk in enumerate(chunks):
        is_last = idx == len(chunks) - 1
        payload: dict[str, Any] = {
            "chat_id": str(chat_id),
            "text": chunk,
            "disable_web_page_preview": disable_web_page_preview,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if is_last and reply_markup:
            payload["reply_markup"] = reply_markup

        try:
            res = _tg_post(token, "sendMessage", payload)
            if not res.get("ok"):
                all_ok = False
        except RuntimeError as exc:
            msg_str = str(exc)
            # Fallback auf Plaintext wenn Entity-Parsing fehlschlägt
            if "HTTP 400" in msg_str and parse_mode:
                log.warning("Telegram HTML-Fallback wegen HTTP 400: %s", msg_str[:160])
                fallback_payload = dict(payload)
                fallback_payload.pop("parse_mode", None)
                fallback_payload["text"] = strip_html_tags(chunk)
                try:
                    res2 = _tg_post(token, "sendMessage", fallback_payload)
                    if not res2.get("ok"):
                        all_ok = False
                except Exception as exc2:
                    log.warning("Telegram Plaintext-Fallback fehlgeschlagen: %s", exc2)
                    all_ok = False
            else:
                log.warning("Telegram-Versand fehlgeschlagen: %s", msg_str[:200])
                all_ok = False
    return all_ok


def edit_message_text(
    token: str,
    chat_id: str | int,
    message_id: int,
    text: str,
    parse_mode: str | None = "HTML",
    reply_markup: dict[str, Any] | None = None,
) -> bool:
    """Aktualisiert eine bestehende Telegram-Nachricht (z. B. nach Inline-Button-Klick)."""
    if not token or not chat_id or not message_id:
        return False
    payload: dict[str, Any] = {
        "chat_id": str(chat_id),
        "message_id": int(message_id),
        "text": text[:4000],
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    try:
        res = _tg_post(token, "editMessageText", payload)
        return bool(res.get("ok"))
    except RuntimeError as exc:
        if "HTTP 400" in str(exc) and parse_mode:
            fallback = dict(payload)
            fallback.pop("parse_mode", None)
            fallback["text"] = strip_html_tags(text[:4000])
            try:
                res2 = _tg_post(token, "editMessageText", fallback)
                return bool(res2.get("ok"))
            except Exception:
                return False
        log.debug("editMessageText Hinweis: %s", exc)
        return False


def answer_callback_query(
    token: str,
    callback_query_id: str,
    text: str = "",
    show_alert: bool = False,
) -> bool:
    if not token or not callback_query_id:
        return False
    payload: dict[str, Any] = {
        "callback_query_id": callback_query_id,
        "show_alert": show_alert,
    }
    if text:
        payload["text"] = text[:190]
    try:
        res = _tg_post(token, "answerCallbackQuery", payload)
        return bool(res.get("ok"))
    except Exception as exc:
        log.debug("answerCallbackQuery Fehler: %s", exc)
        return False


def register_bot_commands(token: str, commands: list[tuple[str, str]]) -> bool:
    """Registriert das blaue `/`-Befehlsmenü in Telegram."""
    if not token:
        return False
    payload = {
        "commands": [{"command": cmd.lstrip("/"), "description": desc[:256]} for cmd, desc in commands]
    }
    try:
        res = _tg_post(token, "setMyCommands", payload, timeout=10)
        return bool(res.get("ok"))
    except Exception as exc:
        log.debug("setMyCommands Hinweis: %s", exc)
        return False


def download_telegram_file(token: str, file_id: str, dest_dir: Path, prefix: str = "tg") -> Path | None:
    """Lädt eine Datei (Voice, Audio, Foto, Dokument) über Telegram getFile herunter."""
    if not token or not file_id:
        return None
    try:
        info_url = f"https://api.telegram.org/bot{token}/getFile?file_id={file_id}"
        with request.urlopen(request.Request(info_url), timeout=15) as resp:
            info = json.loads(resp.read().decode("utf-8"))
        remote_path = (info.get("result") or {}).get("file_path")
        if not remote_path:
            return None
        ext = Path(remote_path).suffix or ".bin"
        dest_dir.mkdir(parents=True, exist_ok=True)
        local_path = dest_dir / f"{prefix}_{int(time.time() * 1000)}{ext}"
        dl_url = f"https://api.telegram.org/file/bot{token}/{remote_path}"
        with request.urlopen(request.Request(dl_url), timeout=45) as dl_resp:
            local_path.write_bytes(dl_resp.read())
        return local_path
    except Exception as exc:
        log.warning("Telegram Datei-Download fehlgeschlagen: %s", exc)
        return None


MAIN_BOT_COMMANDS: list[tuple[str, str]] = [
    ("morgen", "Tages-Briefing mit Terminen, freien Slots & Top-Fokus"),
    ("abend", "Abend-Checkliste & Rückblick"),
    ("todos", "Offene To-Dos mit Schnell-Buttons"),
    ("kalender", "Termine der nächsten 7 Tage"),
    ("slots", "Freie Zeitfenster (inkl. Arbeitskalender-Blocker)"),
    ("ideen", "Proaktive KI-Vorschläge aus Notizen"),
    ("review", "Wöchentliches Review erstellen"),
    ("suche", "Notion & Wissensspeicher durchsuchen"),
    ("mails", "Markierte E-Mails auswerten"),
    ("undo", "Letzte Änderung rückgängig machen"),
    ("status", "System-, Kalender- & Scheduler-Status"),
    ("help", "Hilfe & Befehlsübersicht"),
]

RSS_BOT_COMMANDS: list[tuple[str, str]] = [
    ("themen", "RSS-Themengebiete & Status"),
    ("run", "Suchlauf sofort starten"),
    ("suche", "Wissensspeicher durchsuchen"),
    ("deals", "Aktuelle Studenten-/Dozenten-Rabatte"),
    ("covid", "Neueste COVID-19 Schätzungen (@rv-enigma)"),
    ("synthese", "Wochen-Synthese über alle Themen"),
    ("an", "Thema aktivieren (/an <id>)"),
    ("aus", "Thema pausieren (/aus <id>)"),
    ("neu", "Neues Thema anlegen (/neu <Name>)"),
    ("status", "RSS- & Feed-Health-Status"),
    ("help", "Befehlsübersicht"),
]
