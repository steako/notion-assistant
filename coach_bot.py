#!/usr/bin/env python3
"""
coach_bot.py – Persönlicher Telegram Coach-Bot für Stefan Kollmeier.

Funktionen:
- Tägliche Coaching-Impulse (08:15 Uhr) & freiwilliger Abend-Check-in (20:00 Uhr)
  sowie wöchentliches Coaching-Review (Sonntags 18:30 Uhr).
- Nutzt das verdichtete LPP-Persönlichkeitsprofil, persönliche Ziele (/ziele + Notion),
  offene Notion-Tasks und heutige Kalender-Auslastung/Free-Slots.
- Modell-Tiering: Schnelle Antworten via Gemini 3.5 Flash-Lite (`coach_chat`),
  tiefe Impulse, Reviews & Sparring via Gemini 4 Hydrogen (`coach_deep`, Fallback 3.8 Flash).
- Interaktive Inline-Buttons (Task anlegen, 30m Slot blocken, Reflexion sichern, Feedback).
- Unterstützt Text, Sprachnachrichten, Fotos/PDFs und direkten LPP-Import (.zip/.docx/.pdf).

Verwendet TELEGRAM_COACHBOT_TOKEN aus .env.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import secrets
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core import agent_tools, coaching_core
from core.storage import StateStore, atomic_write_json, safe_read_json
from integrations import telegram_ui
from integrations.notion_client import NotionClient

try:
    import assistant
except Exception:
    assistant = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [CoachBot] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("coach_bot")

STATE_PATH = BASE_DIR / "coach_state.json"
_STORE_CACHE: dict[str, StateStore] = {}
_background_thread: threading.Thread | None = None


def get_store(db_path: Path | None = None) -> StateStore:
    target = db_path or Path(os.environ.get("ASSISTANT_DB_PATH") or (BASE_DIR / "assistant.db"))
    key = str(target.resolve())
    if key not in _STORE_CACHE:
        st = StateStore(target)
        _STORE_CACHE[key] = st
    return _STORE_CACHE[key]


def _load_env_and_cfg() -> dict[str, Any]:
    if assistant is not None:
        return assistant.load_config()
    cfg_path = BASE_DIR / "config.json"
    if cfg_path.exists():
        return safe_read_json(cfg_path, {})
    return {}


def get_active_token() -> str:
    if assistant is not None:
        assistant.load_dotenv_file(BASE_DIR / ".env")
    return os.environ.get("TELEGRAM_COACHBOT_TOKEN", "").strip()


def load_coach_state() -> dict[str, Any]:
    data = safe_read_json(STATE_PATH, {"last_coach_chat_id": "", "coach_telegram_offset": 0})
    if not isinstance(data, dict):
        data = {}
    data.setdefault("last_coach_chat_id", "")
    data.setdefault("coach_telegram_offset", 0)
    return data


def save_coach_state(state: dict[str, Any]) -> None:
    try:
        atomic_write_json(STATE_PATH, state, indent=1)
    except Exception as exc:
        log.warning("Konnte coach_state.json nicht speichern: %s", exc)


def resolve_target_chat_id(state: dict[str, Any] | None = None) -> str:
    for key in ("TELEGRAM_COACH_CHAT_ID", "TELEGRAM_CHAT_ID"):
        val = os.environ.get(key, "").strip()
        if val:
            first = val.split(",")[0].strip()
            if first:
                return first
    if state and state.get("last_coach_chat_id"):
        return str(state["last_coach_chat_id"]).strip()
    local_st = load_coach_state()
    if local_st.get("last_coach_chat_id"):
        return str(local_st["last_coach_chat_id"]).strip()
    allowed = telegram_ui.get_allowed_chat_ids(("TELEGRAM_COACH_CHAT_ID", "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USERS"))
    if allowed:
        return sorted(allowed)[0]
    return ""


def send_message(
    token: str,
    chat_id: str,
    text: str,
    parse_mode: str | None = "HTML",
    reply_markup: dict[str, Any] | None = None,
) -> bool:
    return telegram_ui.send_message(
        token=token,
        chat_id=chat_id,
        text=text,
        parse_mode=parse_mode,
        reply_markup=reply_markup,
    )


def _get_notion_client() -> NotionClient | None:
    ntok = os.environ.get("NOTION_TOKEN", "").strip()
    return NotionClient(ntok) if ntok else None


def _call_llm_json(prompt: str, cfg: dict[str, Any], task: str = "coach_deep", system_instruction: str | None = None) -> dict[str, Any]:
    if assistant is not None:
        return assistant.call_llm_json(prompt, cfg, system_instruction=system_instruction, task=task)
    raise RuntimeError("assistant Modul nicht geladen.")


def _call_llm_text(prompt: str, cfg: dict[str, Any], task: str = "coach_chat", system_instruction: str | None = None) -> str:
    if assistant is not None:
        return assistant.call_llm_text(prompt, cfg, system_instruction=system_instruction, task=task)
    raise RuntimeError("assistant Modul nicht geladen.")


# --------------------------------------------------------------------------------------
# Impuls-, Check-in- & Review-Generierung
# --------------------------------------------------------------------------------------

def build_daily_impulse(
    cfg: dict[str, Any],
    store: StateStore,
    notion: NotionClient | None = None,
    alternative_angle: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Erstellt einen personalisierten Coaching-Impuls auf Basis von LPP, Zielen, Tasks & Kalender."""
    profile = coaching_core.load_coach_profile(store)
    profile_ctx = coaching_core.format_coach_profile_context(profile)
    goals = coaching_core.list_goals(store)
    goals_lines = "\n".join(
        f"• [{g.get('area', 'Allgemein')} / {g.get('horizon', 'Quartal')}] {g.get('title')}"
        for g in goals[:12]
    ) or "Noch keine lokalen /ziele angelegt."

    notion_ctx = coaching_core.fetch_notion_coaching_context(cfg, notion, store)
    day_load = coaching_core.fetch_day_load_context(cfg, store)
    recent_logs = store.get_recent_coaching_logs("coach", days=7, limit=8)
    recent_summary = "\n".join(
        f"- {l['date']} ({l['kind']}, Thema: {l.get('topic') or '-'}, Feedback: {l.get('feedback') or '-'}): {(l.get('prompt_summary') or l.get('content') or '')[:120]}"
        for l in recent_logs
    ) or "Keine kürzlichen Impulse."

    today_iso = dt.date.today().isoformat()
    alt_hint = (
        "\nWICHTIG: Stefan hat einen ANDEREN Blickwinkel angefordert. Wähle gezielt ein anderes LPP-Handlungsfeld als zuletzt!\n"
        if alternative_angle
        else ""
    )

    prompt = f"""Heute ist {today_iso}.
Du bist der persönliche, psychologisch fundierte Executive- & Lebens-Coach von Stefan Kollmeier.
{alt_hint}
=== PERSÖNLICHKEITS- & LPP-PROFIL ===
{profile_ctx}

=== AKTIVE ZIELE (/ziele) ===
{goals_lines}

=== NOTION KONTEXT (About Stefan, Ziele, Memories, Tasks) ===
About Stefan:
{notion_ctx.get('about_text') or '(nicht geladen)'}

Notion Ziele:
{notion_ctx.get('goals_text') or '(keine separate Ziel-Seite konfiguriert)'}

Offene Top-Tasks:
{json.dumps(notion_ctx.get('open_tasks', [])[:8], ensure_ascii=False)}

Mehrfach verschobene Tasks (Prokrastinations-/Überlastungssignal):
{', '.join(notion_ctx.get('deferred_tasks', [])) or 'Keine'}

=== HEUTIGE KALENDER-AUSLASTUNG ===
Private Termine heute: {', '.join(day_load.get('private_events_today', [])) or 'Keine'}
Arbeitskalender-Blocker heute: {day_load.get('work_busy_count', 0)} Blöcke
Freie Zeitfenster heute: {json.dumps(day_load.get('free_slots', [])[:4], ensure_ascii=False)}

=== LETZTE COACHING-IMPULSE (Wiederholungen vermeiden!) ===
{recent_summary}

AUFGABE:
Entwickle EINEN messerscharfen, warmherzigen, konkret umsetzbaren Tages-Coaching-Impuls für Stefan (max. 220 Wörter im Impuls-Text).
Verbinde gezielt ein passendes LPP-Muster/Handlungsfeld mit Stefans aktuellen Zielen und seiner heutigen Auslastung.
Kein generisches Blabla – nutze ggf. eine treffende Metapher, aber liefere zwingend eine kleine, heute machbare Umsetzungsbrücke (5–20 Min. oder 1 klare Entscheidung).

Antworte AUSSCHLIESSLICH mit diesem JSON-Schema:
{{
  "theme": "Kurzer Themen-Tag (z. B. Umsetzung & WIP-Limit | Klare Kante | Kontext-Log | Regeneration)",
  "impulse_title": "Prägnante Überschrift (max. 65 Zeichen)",
  "impulse_text": "Der Coaching-Impuls (2-3 knappe Absätze, persönlich an Stefan gerichtet)",
  "micro_action": "Konkreter 1-Satz-Aktionsschritt oder Reflexionsimpuls für heute",
  "suggested_task_title": "Kurzer Titel falls Stefan das als Notion-Task speichern möchte",
  "suggested_block_title": "Kurzer Titel für einen 30-Min-Kalenderblock (z. B. Fokus: ... oder Regenerations-Anker)"
}}"""

    sys_inst = (
        "Du bist ein hochkompetenter, empathischer und strukturstarker Coach (LINC Personality Profiler zertifiziert). "
        "Du schützt Stefan vor Verzettelung, Überlastung und Harmoniefallen, ohne seine Kreativität einzuengen. "
        "Antworte ausschließlich mit validem JSON."
    )

    try:
        data = _call_llm_json(prompt, cfg, task="coach_deep", system_instruction=sys_inst)
    except Exception as exc:
        log.warning("Coach-Impuls LLM Fehler (%s) -> deterministischer Fallback.", exc)
        data = {
            "theme": "Fokus & WIP-Limit",
            "impulse_title": "Ein Thema wirklich zu Ende denken",
            "impulse_text": (
                "Bei hoher Kreativität und vielen parallelen Baustellen entsteht innere Spannung oft nicht durch "
                "einzelne Aufgaben, sondern durch zu viele offene Schleifen gleichzeitig. Wähle heute bewusst EINE "
                "Sache, die du vom Denkraum in den Entscheidungs- oder Umsetzungsraum holst – und schütze danach "
                "eine kleine Erholungspause im Kalender."
            ),
            "micro_action": "Definiere in 3 Minuten den kleinsten konkreten nächsten Schritt für dein wichtigstes Ziel.",
            "suggested_task_title": "Fokus-Schritt für Top-Ziel festlegen",
            "suggested_block_title": "Fokus- & Entlastungsblock (30m)",
        }

    esc = telegram_ui.esc
    theme = str(data.get("theme") or "Coaching-Impuls")
    title = str(data.get("impulse_title") or "Dein Tages-Impuls")
    body = str(data.get("impulse_text") or "")
    micro = str(data.get("micro_action") or "")

    html_msg = (
        f"🧭 <b>Coaching-Impuls</b> <i>[{esc(theme)}]</i>\n"
        f"<b>{esc(title)}</b>\n\n"
        f"{esc(body)}\n\n"
        f"🎯 <b>Heute konkret:</b> {esc(micro)}"
    )
    return html_msg, data


def send_daily_impulse(
    chat_id: str | None = None,
    cfg: dict[str, Any] | None = None,
    token: str | None = None,
    alternative_angle: bool = False,
) -> tuple[bool, str]:
    cfg = cfg or _load_env_and_cfg()
    tok = token or get_active_token()
    st = load_coach_state()
    cid = chat_id or resolve_target_chat_id(st)
    if not tok or not cid:
        return False, "Coach-Bot Token oder Chat-ID fehlt."

    store = get_store()
    notion = _get_notion_client()
    html_msg, data = build_daily_impulse(cfg, store, notion, alternative_angle=alternative_angle)

    log_id = store.add_coaching_log(
        bot="coach",
        kind="morning_impulse",
        topic=str(data.get("theme") or ""),
        prompt_summary=str(data.get("impulse_title") or ""),
        content=f"{data.get('impulse_text', '')}\nHeute konkret: {data.get('micro_action', '')}",
    )

    cb_tok = secrets.token_hex(6)
    store.put_callback_payload(cb_tok, "coach_impulse", {
        "log_id": log_id,
        "theme": data.get("theme", ""),
        "title": data.get("impulse_title", "Coaching-Impuls"),
        "micro_action": data.get("micro_action", ""),
        "task_title": data.get("suggested_task_title") or data.get("impulse_title") or "Coaching-Schritt",
        "block_title": data.get("suggested_block_title") or "Fokus- & Regenerationsblock",
    })
    kb = telegram_ui.build_inline_keyboard([
        [("🎯 Task anlegen", f"coach_task:{cb_tok}"), ("⏸️ 30m Slot blocken", f"coach_block:{cb_tok}")],
        [("📝 Reflexion sichern", f"coach_save:{cb_tok}"), ("👍 Hilfreich", f"coach_like:{cb_tok}"), ("🔄 Anderer Fokus", f"coach_alt:{cb_tok}")],
    ])
    ok = send_message(tok, cid, html_msg, reply_markup=kb)
    return ok, html_msg


def send_evening_checkin(
    chat_id: str | None = None,
    cfg: dict[str, Any] | None = None,
    token: str | None = None,
) -> tuple[bool, str]:
    cfg = cfg or _load_env_and_cfg()
    tok = token or get_active_token()
    st = load_coach_state()
    cid = chat_id or resolve_target_chat_id(st)
    if not tok or not cid:
        return False, "Coach-Bot Token oder Chat-ID fehlt."

    store = get_store()
    today_iso = dt.date.today().isoformat()
    recent = store.get_recent_coaching_logs("coach", days=1, kind="morning_impulse", limit=1)
    today_imp = recent[0] if recent and recent[0].get("date") == today_iso else None

    esc = telegram_ui.esc
    if today_imp:
        imp_title = today_imp.get("prompt_summary") or today_imp.get("topic") or "heutiger Impuls"
        ref_line = f"Heute ging es um: <b>{esc(imp_title)}</b>\n\n"
    else:
        ref_line = ""

    html_msg = (
        f"🌙 <b>Abend-Check-in ({today_iso})</b>\n\n"
        f"{ref_line}"
        "1 kurze Frage zum Runterfahren:\n"
        "<i>Wo hast du heute bewusst Fokus gehalten oder dich rechtzeitig abgegrenzt – und was darfst du für heute getrost loslassen?</i>\n\n"
        "Antworte gerne kurz per Text/Sprachnotiz oder tippe auf einen Button:"
    )

    log_id = store.add_coaching_log(
        bot="coach",
        kind="evening_checkin",
        topic=today_imp.get("topic", "") if today_imp else "Abend-Check-in",
        prompt_summary="Abend-Check-in",
        content=telegram_ui.strip_html_tags(html_msg),
    )
    cb_tok = secrets.token_hex(6)
    store.put_callback_payload(cb_tok, "coach_checkin", {"log_id": log_id})
    kb = telegram_ui.build_inline_keyboard([
        [("✅ Umgesetzt", f"coach_done:{cb_tok}"), ("🔄 Teilweise / Morgen", f"coach_partial:{cb_tok}")],
        [("😮‍💨 Tag war zu voll", f"coach_overloaded:{cb_tok}")],
    ])
    ok = send_message(tok, cid, html_msg, reply_markup=kb)
    return ok, html_msg


def generate_weekly_coaching_review(
    cfg: dict[str, Any],
    store: StateStore,
    notion: NotionClient | None = None,
) -> tuple[str, str | None]:
    """Erstellt ein wöchentliches Coaching-Review (Telegram HTML + optional Notion Documents Seite)."""
    profile = coaching_core.load_coach_profile(store)
    goals = coaching_core.list_goals(store, include_done=True)
    logs = store.get_recent_coaching_logs("coach", days=7, limit=25)
    notion_ctx = coaching_core.fetch_notion_coaching_context(cfg, notion, store)

    logs_txt = "\n".join(
        f"- {l['date']} [{l['kind']}] Thema={l.get('topic', '')} | Feedback={l.get('feedback', '-')}: {l['content'][:200]}"
        for l in logs
    ) or "Keine Coaching-Logs in den letzten 7 Tagen."

    goals_txt = "\n".join(
        f"- [{g.get('status')}] ({g.get('area')}) {g.get('title')}" for g in goals
    ) or "Keine lokalen Ziele hinterlegt."

    prompt = f"""Erstelle ein tiefgehendes, wertschätzendes Wochen-Coaching-Review für Stefan Kollmeier (max. 320 Wörter).
Berücksichtige:
1. LPP-Muster & Handlungsfelder: {json.dumps(profile.get('lpp_summary', {}).get('development_areas', []), ensure_ascii=False)}
2. Ziele (/ziele): {goals_txt}
3. Coaching-Woche (Impulse, Check-ins, Reflexionen): {logs_txt}
4. Mehrfach verschobene Notion-Tasks (Engpässe): {', '.join(notion_ctx.get('deferred_tasks', [])) or 'Keine'}

Gliedere übersichtlich in:
- Was diese Woche gut gelungen ist / Stärken-Einsatz
- Muster & Engpässe (ehrlich, wohlwollend: z. B. WIP-Limit, Abgrenzung, Konfliktkante)
- 1–2 konkrete Leitplanken für die kommende Woche

Gib reinen deutschen Text ohne Markdown-Codeblöcke zurück (einfache Absätze und Aufzählungszeichen '•')."""

    try:
        review_body = _call_llm_text(prompt, cfg, task="coach_deep")
    except Exception as exc:
        review_body = f"Wochen-Review-Entwurf konnte nicht via LLM generiert werden ({exc}).\nLogs der Woche: {len(logs)} Einträge."

    today = dt.date.today()
    doc_url: str | None = None
    if notion is not None:
        try:
            md_lines = [f"## Coaching Wochen-Review ({today:%d.%m.%Y})", *review_body.splitlines()]
            doc_url = notion.create_document_page(
                cfg["notion"]["documents_database_id"],
                title=f"Coaching Review {today:%d.%m.%Y}",
                category="Coaching",
                markdown_lines=md_lines,
                emoji="🧭",
            )
        except Exception as exc:
            log.warning("Notion Coaching-Review Dokument fehlgeschlagen: %s", exc)

    esc = telegram_ui.esc
    html_out = f"🧭 <b>Wöchentliches Coaching-Review ({today:%d.%m.%Y})</b>\n\n{esc(review_body)}"
    if doc_url:
        html_out += f'\n\n📄 <a href="{esc(doc_url)}">In Notion öffnen</a>'
    return html_out, doc_url


# --------------------------------------------------------------------------------------
# Befehls-Handler
# --------------------------------------------------------------------------------------

def handle_start_or_help() -> str:
    return (
        "🧭 <b>Willkommen bei deinem persönlichen Coach-Bot!</b>\n\n"
        "Ich liefere dir tägliche Coaching-Impulse und Sparring auf Basis deines "
        "<b>LPP-Persönlichkeitsprofils</b>, deiner <b>Ziele</b>, deiner offenen <b>Notion-Tasks</b> "
        "und deiner <b>Kalender-Auslastung</b>.\n\n"
        "📋 <b>Befehle:</b>\n"
        "• <code>/impuls</code> – Heutigen Coaching-Impuls (Gemini 4 Hydrogen) abrufen\n"
        "• <code>/checkin</code> – Abend-Check-in &amp; Reflexion starten\n"
        "• <code>/ziele</code> – Aktive Ziele anzeigen\n"
        "• <code>/ziele add &lt;Ziel&gt;</code> | <code>/ziele done &lt;Nr&gt;</code> | <code>/ziele del &lt;Nr&gt;</code>\n"
        "• <code>/profil</code> – LPP-Profil &amp; Fokus ansehen (oder <code>/profil &lt;Notiz&gt;</code> ergänzen)\n"
        "• <code>/tief &lt;Thema&gt;</code> – Tiefes Coaching-Sparring zu Entscheidung/Konflikt/Fokus\n"
        "• <code>/review</code> – Wöchentliches Coaching-Review\n"
        "• <code>/status</code> – Status, Zeitplan &amp; Profilstand\n"
        "• <code>/update</code> – Neuesten Code von GitHub laden &amp; neu starten\n\n"
        "💡 <i>Tipp: Du kannst mir jederzeit Sprachnachrichten, Reflexionen oder neue LPP-Unterlagen (.zip/.docx/.pdf) schicken!</i>"
    )


def handle_ziele(arg: str, store: StateStore) -> str:
    esc = telegram_ui.esc
    raw = arg.strip()
    if raw.lower().startswith("add "):
        title = raw[4:].strip()
        if not title:
            return "Bitte Zieltext angeben: <code>/ziele add Klare 1:1-Routine im Team etablieren</code>"
        area = agent_tools.detect_area(title)
        g = coaching_core.add_goal(store, title=title, area=area)
        return f"🎯 <b>Ziel gespeichert:</b> {esc(g['title'])}\n<i>Bereich: {esc(area)}</i>"

    if raw.lower().startswith("done "):
        q = raw[5:].strip()
        g = coaching_core.complete_goal(store, q)
        if g:
            return f"✅ <b>Ziel abgeschlossen:</b> {esc(g['title'])}"
        return f"⚠️ Kein aktives Ziel passend zu <code>{esc(q)}</code> gefunden."

    if raw.lower().startswith(("del ", "remove ", "loeschen ")):
        q = raw.split(maxsplit=1)[1].strip()
        g = coaching_core.delete_goal(store, q)
        if g:
            return f"🗑️ <b>Ziel entfernt:</b> {esc(g['title'])}"
        return f"⚠️ Kein aktives Ziel passend zu <code>{esc(q)}</code> gefunden."

    goals = coaching_core.list_goals(store)
    if not goals:
        return (
            "🎯 <b>Aktive Ziele:</b> Noch keine lokalen Ziele hinterlegt.\n\n"
            "Ergänze ein Ziel mit:\n<code>/ziele add Weniger parallele Baustellen, 1 Projekt pro Woche abschließen</code>\n"
            "<i>(Notion-Ziele aus „About Stefan“ &amp; „Memories“ werden zusätzlich automatisch einbezogen.)</i>"
        )
    lines = ["🎯 <b>Deine aktiven Coaching-Ziele:</b>\n"]
    for idx, g in enumerate(goals, 1):
        lines.append(f"{idx}. [{esc(g.get('area', 'Allgemein'))}] <b>{esc(g.get('title'))}</b>")
    lines.append("\nBefehle: <code>/ziele add ...</code> · <code>/ziele done &lt;Nr&gt;</code> · <code>/ziele del &lt;Nr&gt;</code>")
    return "\n".join(lines)


def handle_profil(arg: str, store: StateStore) -> str:
    esc = telegram_ui.esc
    if arg.strip():
        note = arg[4:].strip() if arg.lower().startswith("add ") else arg.strip()
        coaching_core.append_coach_note(note, store)
        return f"✅ <b>Coaching-Profil ergänzt:</b> „{esc(note)}“"

    prof = coaching_core.load_coach_profile(store)
    lpp = prof.get("lpp_summary", {})
    lines = [
        f"🧠 <b>Coaching- &amp; LPP-Profil ({esc(prof.get('name', 'Stefan'))})</b>\n",
        f"• <b>Motive:</b> {esc(' | '.join(lpp.get('motives', [])) or 'Keine')}",
        f"• <b>Kernstärken:</b> {esc(', '.join(lpp.get('core_strengths', [])) or 'Keine')}",
    ]
    dev_areas = lpp.get("development_areas") or []
    if dev_areas:
        lines.append("\n<b>LPP-Handlungsfelder:</b>")
        for da in dev_areas:
            lines.append(f"  • {esc(da.get('title', ''))}")
    notes = prof.get("custom_notes") or []
    if notes:
        lines.append("\n<b>Eigene Ergänzungen:</b>")
        for n in notes[-4:]:
            lines.append(f"  • ({esc(n.get('at', '')[:10])}) {esc(n.get('text', ''))}")
    lines.append("\n<i>Ergänzen per <code>/profil &lt;Notiz&gt;</code> oder durch Senden einer LPP-ZIP/DOCX/PDF-Datei.</i>")
    return "\n".join(lines)


def handle_status(cfg: dict[str, Any], store: StateStore) -> str:
    esc = telegram_ui.esc
    prof = coaching_core.load_coach_profile(store)
    goals = coaching_core.list_goals(store)
    logs = store.get_recent_coaching_logs("coach", days=7, limit=20)
    sched_st = assistant._load_scheduler_state() if assistant is not None else {}

    import platform as _plat
    import subprocess as _sub
    _commit = ""
    try:
        _cp = _sub.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True, timeout=3)
        if _cp.returncode == 0:
            _commit = _cp.stdout.strip()
    except Exception:
        pass

    lines = [
        "🧭 <b>Coach-Bot Status</b>\n",
        f"• 🖥️ <b>Host:</b> <code>{esc(sys.platform)}</code> @ <code>{esc(_plat.node())}</code> (PID <code>{os.getpid()}</code>" + (f", Commit <code>{esc(_commit)}</code>" if _commit else "") + ")",
        f"• 🧠 <b>LPP-Profil:</b> {len(prof.get('lpp_summary', {}).get('development_areas', []))} Handlungsfelder geladen",
        f"• 🎯 <b>Lokale Ziele:</b> {len(goals)} aktiv",
        f"• 📝 <b>Coaching-Logs (7 Tage):</b> {len(logs)}",
        "\n⏰ <b>Zeitplan (Europe/Berlin):</b>",
        f"• 08:15 Uhr Morgen-Impuls (Zuletzt: <code>{esc(sched_st.get('coach_morning', 'noch nicht'))}</code>)",
        f"• 20:00 Uhr Abend-Check-in (Zuletzt: <code>{esc(sched_st.get('coach_evening', 'noch nicht'))}</code>)",
        f"• So 18:30 Uhr Wochen-Review (Zuletzt: <code>{esc(sched_st.get('coach_weekly', 'noch nicht'))}</code>)",
    ]
    return "\n".join(lines)


def handle_natural_language(chat_id: str, text: str, token: str, cfg: dict[str, Any], force_deep: bool = False) -> None:
    store = get_store()
    notion = _get_notion_client()
    store.append_chat_turn("coach", chat_id, "user", text)

    deep_triggers = ("strategie", "konflikt", "entscheidung", "überlast", "lpp", "warum", "dilemma", "führung", "abgrenz")
    use_deep = force_deep or any(k in text.lower() for k in deep_triggers) or len(text) > 350
    task_tier = "coach_deep" if use_deep else "coach_chat"

    profile = coaching_core.load_coach_profile(store)
    profile_ctx = coaching_core.format_coach_profile_context(profile)
    goals = coaching_core.list_goals(store)
    goals_txt = ", ".join(g.get("title", "") for g in goals[:8]) or "Keine lokalen Ziele"
    history = store.get_recent_chat_turns("coach", chat_id, limit=8)
    history_block = "\n".join(f"{h['role'].upper()}: {h['content'][:300]}" for h in history[:-1])

    notion_hint = ""
    if use_deep and notion is not None:
        nc = coaching_core.fetch_notion_coaching_context(cfg, notion, store)
        notion_hint = f"\nOffene Top-Tasks: {', '.join(t['title'] for t in nc.get('open_tasks', [])[:5])}"

    system_instr = (
        "Du bist der persönliche Coach von Stefan Kollmeier auf Telegram.\n"
        f"Profil-Basis:\n{profile_ctx}\n"
        f"Aktive Ziele: {goals_txt}{notion_hint}\n\n"
        "Antworte auf fehlerfreiem Deutsch, warm, klar, konkret und ohne leere Floskeln. "
        "Vermeide Markdown-Codeblöcke; nutze schlichte Absätze und bei Bedarf einen konkreten nächsten Schritt."
    )
    prompt = (f"Bisheriger Gesprächsverlauf:\n{history_block}\n\n" if history_block else "") + f"Nachricht von Stefan:\n{text}"

    try:
        reply = _call_llm_text(prompt, cfg, task=task_tier, system_instruction=system_instr)
    except Exception as exc:
        log.warning("Coach-Bot LLM Fehler: %s", exc)
        reply = "Ich habe deinen Gedanken gespeichert. Woran möchtest du heute als Erstes konkret ansetzen?"

    store.append_chat_turn("coach", chat_id, "assistant", reply[:600])
    if len(text) >= 25:
        store.add_coaching_log(bot="coach", kind="reflection", topic="Dialog", prompt_summary=text[:120], content=f"Stefan: {text[:800]}\nCoach: {reply[:800]}")

    send_message(token, chat_id, telegram_ui.esc(reply))


# --------------------------------------------------------------------------------------
# Callback Query Handler
# --------------------------------------------------------------------------------------

def handle_callback_query(token: str, cb: dict[str, Any], cfg: dict[str, Any]) -> None:
    cb_id = str(cb.get("id") or "")
    data = str(cb.get("data") or "")
    msg = cb.get("message") or {}
    chat_id = str(msg.get("chat", {}).get("id", ""))

    if not chat_id or not telegram_ui.is_authorized_chat(
        chat_id, env_keys=("TELEGRAM_COACH_CHAT_ID", "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USERS")
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
    log_id = int(payload.get("log_id") or 0)
    esc = telegram_ui.esc

    if action == "coach_task":
        notion = _get_notion_client()
        title = str(payload.get("task_title") or payload.get("title") or "Coaching-Schritt")
        micro = str(payload.get("micro_action") or "")
        area = agent_tools.detect_area(f"{title} {micro}")
        if notion is not None and cfg.get("notion", {}).get("tasks_database_id"):
            try:
                page_id = notion.create_task_page(
                    tasks_db_id=cfg["notion"]["tasks_database_id"],
                    title=title[:90],
                    kind="To-Do",
                    priority="Mittel",
                    due_date=dt.date.today().isoformat(),
                    next_step=micro[:300],
                    area=area,
                    estimated_minutes=20,
                )
                store.push_undo("coach", chat_id, "archive_task", {"notion_page_id": page_id, "title": title})
                telegram_ui.answer_callback_query(token, cb_id, "🎯 Als Notion-Task angelegt!")
                send_message(token, chat_id, f"🎯 <b>Notion-Task angelegt:</b> {esc(title)}\n<i>Fällig heute · Bereich: {esc(area)}</i>")
                return
            except Exception as exc:
                log.warning("Coach-Task Notion Fehler: %s", exc)
        telegram_ui.answer_callback_query(token, cb_id, "ℹ️ Notion nicht verbunden – lokal vorgemerkt.")
        send_message(token, chat_id, f"🎯 <b>Lokal vorgemerkt:</b> {esc(title)}")
        return

    if action == "coach_block":
        block_title = str(payload.get("block_title") or "Coaching Fokus-/Pausenblock")
        ev = coaching_core.book_coaching_slot(cfg, store, title=block_title, duration_minutes=30)
        if ev:
            telegram_ui.answer_callback_query(token, cb_id, "⏸️ 30m Slot geblockt!")
            send_message(
                token,
                chat_id,
                f"⏸️ <b>Zeitblock reserviert:</b> {esc(ev['title'])}\n"
                f"📅 {esc(ev['date'])} · {esc(ev['start_time'])}–{esc(ev['end_time'])} Uhr",
            )
        else:
            telegram_ui.answer_callback_query(token, cb_id, "Kein freier 30m Slot gefunden.", show_alert=True)
        return

    if action == "coach_save":
        notion = _get_notion_client()
        title = str(payload.get("title") or "Coaching-Impuls")
        micro = str(payload.get("micro_action") or "")
        if notion is not None and cfg.get("notion", {}).get("inbox_database_id"):
            try:
                notion.create_inbox_page(
                    inbox_db_id=cfg["notion"]["inbox_database_id"],
                    title=f"Coaching: {title[:75]}",
                    source="Telegram",
                    raw_text=micro or title,
                    summary=f"Thema: {payload.get('theme', '')} | {micro}",
                    suggestions_text=micro,
                    area="Familie/Privat",
                )
                telegram_ui.answer_callback_query(token, cb_id, "📝 In Notion AI Inbox gesichert!")
                send_message(token, chat_id, f"📝 <b>In Notion AI Inbox gespeichert:</b> {esc(title)}")
                return
            except Exception as exc:
                log.warning("Coach-Save Notion Fehler: %s", exc)
        store.add_coaching_log(bot="coach", kind="saved_reflection", topic=str(payload.get("theme") or ""), content=f"{title}: {micro}")
        telegram_ui.answer_callback_query(token, cb_id, "📝 Lokal gespeichert!")
        return

    if action == "coach_like":
        if log_id:
            store.set_coaching_feedback(log_id, "helpful")
        telegram_ui.answer_callback_query(token, cb_id, "👍 Danke – merke ich mir!")
        return

    if action == "coach_alt":
        if log_id:
            store.set_coaching_feedback(log_id, "alt_requested")
        telegram_ui.answer_callback_query(token, cb_id, "🔄 Suche einen anderen Fokus...")
        send_daily_impulse(chat_id=chat_id, cfg=cfg, token=token, alternative_angle=True)
        return

    if action in ("coach_done", "coach_partial", "coach_overloaded"):
        fb_map = {"coach_done": "done", "coach_partial": "partial", "coach_overloaded": "overloaded"}
        if log_id:
            store.set_coaching_feedback(log_id, fb_map[action])
        if action == "coach_done":
            telegram_ui.answer_callback_query(token, cb_id, "✅ Stark! Schönen Feierabend.")
            send_message(token, chat_id, "✅ <b>Klasse gemacht!</b> Gönn dir jetzt bewusste Erholung ohne Grübeln.")
        elif action == "coach_partial":
            telegram_ui.answer_callback_query(token, cb_id, "🔄 Kein Stress – morgen kleiner einsteigen.")
            send_message(token, chat_id, "🔄 <b>Alles gut.</b> Morgen reicht ein noch kleinerer 5-Minuten-Schritt – schließ den Tag jetzt ruhig ab.")
        else:
            telegram_ui.answer_callback_query(token, cb_id, "😮‍💨 Verstanden – morgen Fokus auf Entlastung.")
            send_message(token, chat_id, "😮‍💨 <b>Danke für die ehrliche Rückmeldung.</b> Ich schlage dir morgen gezielt Entlastung, WIP-Limit und Abgrenzung vor. Schlaf gut!")
        return

    telegram_ui.answer_callback_query(token, cb_id, "Unbekannte Aktion.")


# --------------------------------------------------------------------------------------
# Telegram Polling
# --------------------------------------------------------------------------------------

def process_updates(token: str, state: dict[str, Any], poll_timeout: int = 15) -> int:
    offset = int(state.get("coach_telegram_offset", 0))
    api_base = f"https://api.telegram.org/bot{token}"

    try:
        req = urllib.request.Request(f"{api_base}/getUpdates?offset={offset}&timeout={poll_timeout}")
        with urllib.request.urlopen(req, timeout=poll_timeout + 10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.debug("CoachBot Polling Hinweis: %s", exc)
        return 0

    if not data.get("ok"):
        return 0

    cfg = _load_env_and_cfg()
    count = 0

    for update in data.get("result", []):
        update_id = update["update_id"]
        state["coach_telegram_offset"] = max(int(state.get("coach_telegram_offset", 0)), update_id + 1)
        save_coach_state(state)
        count += 1

        if "callback_query" in update:
            try:
                handle_callback_query(token, update["callback_query"], cfg)
            except Exception as exc:
                log.exception("CoachBot Callback Fehler in Update %s: %s", update_id, exc)
            continue

        msg = update.get("message") or update.get("edited_message")
        if not msg:
            continue

        chat_id = str(msg.get("chat", {}).get("id", ""))
        if not chat_id or not telegram_ui.is_authorized_chat(
            chat_id, env_keys=("TELEGRAM_COACH_CHAT_ID", "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USERS")
        ):
            log.warning("Unautorisierter CoachBot-Zugriff von Chat-ID %s blockiert.", chat_id)
            continue

        try:
            if state.get("last_coach_chat_id") != chat_id:
                state["last_coach_chat_id"] = chat_id
                save_coach_state(state)

            raw_text = (msg.get("text") or "").strip()
            voice = msg.get("voice") or msg.get("audio")
            doc = msg.get("document")

            if voice and assistant is not None:
                file_id = voice.get("file_id")
                if file_id:
                    tmp_audio = telegram_ui.download_telegram_file(token, file_id, BASE_DIR / "temp", prefix="coach_voice")
                    if tmp_audio:
                        try:
                            raw_text = assistant.transcribe_audio(tmp_audio, cfg)
                            send_message(token, chat_id, f"🎤 <i>Erkannt:</i> „{telegram_ui.esc(raw_text)}“")
                        finally:
                            tmp_audio.unlink(missing_ok=True)

            if doc and not raw_text:
                file_id = doc.get("file_id")
                fname = str(doc.get("file_name") or "dokument")
                mime = str(doc.get("mime_type") or "application/pdf")
                caption = (msg.get("caption") or "").strip()
                if file_id:
                    tmp_doc = telegram_ui.download_telegram_file(token, file_id, BASE_DIR / "temp", prefix="coach_doc")
                    if tmp_doc:
                        try:
                            lower_n = fname.lower()
                            if lower_n.endswith((".zip", ".docx")) or ("lpp" in lower_n or "tiefenprofil" in lower_n or "feedback" in lower_n):
                                _, n_files = coaching_core.import_lpp_into_profile(
                                    tmp_doc, cfg, store=get_store(), call_llm_json_fn=_call_llm_json if assistant else None
                                )
                                if n_files > 0:
                                    send_message(token, chat_id, f"✅ <b>LPP-/Coaching-Unterlagen importiert ({n_files} Datei(en))!</b>\nNutze <code>/profil</code> oder <code>/impuls</code>.")
                                    continue
                            ref = assistant.get_date_reference_block() if assistant is not None else ""
                            res_mm = agent_tools.analyze_multimodal_file(tmp_doc, mime, caption, cfg, ref)
                            handle_natural_language(
                                chat_id,
                                f"Dokument '{fname}' ({res_mm.get('title', '')}): {res_mm.get('summary', '')}\n{caption}",
                                token,
                                cfg,
                            )
                            continue
                        finally:
                            tmp_doc.unlink(missing_ok=True)

            if not raw_text:
                continue

            parts = raw_text.strip().split()
            cmd = parts[0].lower().split("@")[0]
            arg = " ".join(parts[1:]).strip()
            store = get_store()

            if cmd in ("/start", "/help", "/hilfe", "start", "hilfe"):
                send_message(token, chat_id, handle_start_or_help())
            elif cmd in ("/impuls", "/impulse", "impuls"):
                send_daily_impulse(chat_id=chat_id, cfg=cfg, token=token)
            elif cmd in ("/checkin", "/check-in", "checkin"):
                send_evening_checkin(chat_id=chat_id, cfg=cfg, token=token)
            elif cmd in ("/ziele", "/ziel", "ziele"):
                send_message(token, chat_id, handle_ziele(arg, store))
            elif cmd in ("/profil", "/profile", "profil"):
                send_message(token, chat_id, handle_profil(arg, store))
            elif cmd in ("/review", "review"):
                send_message(token, chat_id, "🧭 Erstelle wöchentliches Coaching-Review (Hydrogen)...")
                html_out, _ = generate_weekly_coaching_review(cfg, store, _get_notion_client())
                send_message(token, chat_id, html_out)
            elif cmd in ("/tief", "/deep"):
                if not arg:
                    send_message(token, chat_id, "Bitte Thema angeben, z. B. <code>/tief Wie setze ich im Team eine klare Grenze?</code>")
                else:
                    handle_natural_language(chat_id, arg, token, cfg, force_deep=True)
            elif cmd in ("/status", "status"):
                send_message(token, chat_id, handle_status(cfg, store))
            elif cmd in ("/update", "/upgrade", "update"):
                send_message(token, chat_id, "⏳ Prüfe GitHub auf Updates...")
                if assistant is not None:
                    needs_restart, update_msg = assistant.perform_self_update()
                    send_message(token, chat_id, update_msg)
                    if needs_restart:
                        assistant.trigger_restart()
                else:
                    send_message(token, chat_id, "❌ assistant-Modul nicht geladen.")
            else:
                handle_natural_language(chat_id, raw_text, token, cfg)
        except Exception as exc:
            log.exception("CoachBot Fehler in Update %s: %s", update_id, exc)
            send_message(token, chat_id, f"❌ Interner Fehler: {telegram_ui.esc(exc)}")

    return count


def run_bot_daemon(poll_timeout: int = 15) -> None:
    token = get_active_token()
    if not token:
        log.error("TELEGRAM_COACHBOT_TOKEN nicht gesetzt.")
        return

    telegram_ui.register_bot_commands(token, telegram_ui.COACH_BOT_COMMANDS)
    log.info("Coach-Bot Daemon gestartet (Token: %s...). Warte auf Nachrichten...", token[:10])
    while True:
        try:
            state = load_coach_state()
            process_updates(token, state, poll_timeout=poll_timeout)
        except Exception as exc:
            log.warning("CoachBot Polling-Fehler: %s", exc)
            time.sleep(5)


def start_in_background(poll_timeout: int = 15) -> threading.Thread | None:
    global _background_thread
    token = get_active_token()
    if not token:
        log.warning("Kein TELEGRAM_COACHBOT_TOKEN gesetzt – Coach-Bot startet nicht.")
        return None
    if _background_thread is not None and _background_thread.is_alive():
        return _background_thread

    def _loop():
        run_bot_daemon(poll_timeout=poll_timeout)

    _background_thread = threading.Thread(target=_loop, daemon=True, name="CoachBotDaemon")
    _background_thread.start()
    return _background_thread


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Persönlicher Telegram Coach-Bot")
    sub = parser.add_subparsers(dest="subcmd")
    imp_p = sub.add_parser("import-lpp", help="LPP-ZIP/DOCX/PDF-Dateien ins lokale Profil importieren")
    imp_p.add_argument("path", help="Pfad zur ZIP, Datei oder Ordner")

    parser.add_argument("--once", action="store_true", help="Einmalig Updates pollen")
    parser.add_argument("--impuls", action="store_true", help="Sofort einen Coaching-Impuls senden")
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    cfg = _load_env_and_cfg()
    if args.subcmd == "import-lpp":
        prof, n = coaching_core.import_lpp_into_profile(
            Path(args.path),
            cfg,
            store=get_store(),
            call_llm_json_fn=_call_llm_json if assistant and (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) else None,
        )
        print(f"Import abgeschlossen: {n} Datei(en) verarbeitet -> {coaching_core.COACH_PROFILE_PATH}")
        print(f"Handlungsfelder: {len(prof.get('lpp_summary', {}).get('development_areas', []))}")
        return

    token = get_active_token()
    if not token:
        print("FEHLER: TELEGRAM_COACHBOT_TOKEN ist nicht gesetzt!")
        sys.exit(1)

    if args.impuls:
        ok, _ = send_daily_impulse(cfg=cfg, token=token)
        print("Impuls gesendet." if ok else "Impuls konnte nicht gesendet werden (Chat-ID prüfen).")
        return

    if args.once:
        cnt = process_updates(token, load_coach_state(), poll_timeout=args.timeout)
        print(f"{cnt} Update(s) verarbeitet.")
    else:
        run_bot_daemon(poll_timeout=args.timeout)


if __name__ == "__main__":
    main()
