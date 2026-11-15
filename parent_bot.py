#!/usr/bin/env python3
"""
parent_bot.py – Dedizierter Telegram Elterncoaching-Bot für Stefan.

Fokus: Umgang mit Stress, Konflikten & ADHS im Familienalltag.
- 3 tägliche Check-ins & Impulse (Europe/Berlin):
  * 07:15 Uhr: Morgen-Check-in (Übergang in den Tag, Reiz- & Stress-Prävention)
  * 12:45 Uhr: Mittags-Check-in (Übergang Schule/Arbeit -> Nachmittag, Co-Regulation)
  * 20:15 Uhr: Abend-Reflexion (Tagesabschluss, Konflikt-/ADHS-Muster, Entlastung)
- /akut für sofortige Deeskalation & konkrete Formulierungshilfen (Gemini 4 Hydrogen).
- Datenschutz: Alle Gespräche, Stresswerte und Familienprofile bleiben standardmäßig
  rein lokal in assistant.db & data/parent_profile.json (kein automatischer Notion-Export).
- Unterstützt Text, Sprachnachrichten und Fotos/PDFs.

Verwendet TELEGRAM_PARENTBOT_TOKEN aus .env.
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
    format="%(asctime)s [%(levelname)s] [ParentBot] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("parent_bot")

STATE_PATH = BASE_DIR / "parent_state.json"
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
    return os.environ.get("TELEGRAM_PARENTBOT_TOKEN", "").strip()


def load_parent_state() -> dict[str, Any]:
    data = safe_read_json(STATE_PATH, {"last_parent_chat_id": "", "parent_telegram_offset": 0})
    if not isinstance(data, dict):
        data = {}
    data.setdefault("last_parent_chat_id", "")
    data.setdefault("parent_telegram_offset", 0)
    return data


def save_parent_state(state: dict[str, Any]) -> None:
    try:
        atomic_write_json(STATE_PATH, state, indent=1)
    except Exception as exc:
        log.warning("Konnte parent_state.json nicht speichern: %s", exc)


def resolve_target_chat_id(state: dict[str, Any] | None = None) -> str:
    for key in ("TELEGRAM_PARENT_CHAT_ID", "TELEGRAM_CHAT_ID"):
        val = os.environ.get(key, "").strip()
        if val:
            first = val.split(",")[0].strip()
            if first:
                return first
    if state and state.get("last_parent_chat_id"):
        return str(state["last_parent_chat_id"]).strip()
    local_st = load_parent_state()
    if local_st.get("last_parent_chat_id"):
        return str(local_st["last_parent_chat_id"]).strip()
    allowed = telegram_ui.get_allowed_chat_ids(("TELEGRAM_PARENT_CHAT_ID", "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USERS"))
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


def _call_llm_json(prompt: str, cfg: dict[str, Any], task: str = "coach_deep", system_instruction: str | None = None) -> dict[str, Any]:
    if assistant is not None:
        return assistant.call_llm_json(prompt, cfg, system_instruction=system_instruction, task=task)
    raise RuntimeError("assistant Modul nicht geladen.")


def _call_llm_text(prompt: str, cfg: dict[str, Any], task: str = "coach_chat", system_instruction: str | None = None) -> str:
    if assistant is not None:
        return assistant.call_llm_text(prompt, cfg, system_instruction=system_instruction, task=task)
    raise RuntimeError("assistant Modul nicht geladen.")


# --------------------------------------------------------------------------------------
# Geplante Check-ins & Impulse (Morgen, Mittag, Abend)
# --------------------------------------------------------------------------------------

_PHASE_META = {
    "morning": {
        "emoji": "🌅",
        "label": "Morgen-Check-in (Elterncoach)",
        "focus": "Ruhiger Start in den Tag, ADHS-freundliche Morgen-Übergänge, Zeitpuffer & eigene Gelassenheit vor Abfahrt/Schulstart.",
        "question": "Wie ist dein Energie- und Stresslevel gerade vor dem Start?",
    },
    "midday": {
        "emoji": "🌤️",
        "label": "Mittags-Check-in (Übergang & Reset)",
        "focus": "Übergang von Schule/Arbeit in den Nachmittag: Reizüberflutung abfedern, Dopamin-/Pausen-Puffer einbauen, Konflikte beim Nachmittagsstart entschärfen.",
        "question": "Kurzer Mittags-Reset: Wie voll ist dein Akku für den Nachmittag?",
    },
    "evening": {
        "emoji": "🌙",
        "label": "Abend-Reflexion (Elterncoach)",
        "focus": "Entlastender Tagesabschluss, Erkennen von Stress-/ADHS-/Konfliktmustern ohne Schuldgefühle, 1 Mini-Erfolg festhalten & ruhig herunterfahren.",
        "question": "Was hat heute Druck rausgenommen – und was darfst du für heute loslassen?",
    },
}


def build_parent_checkin(
    phase: str,
    cfg: dict[str, Any],
    store: StateStore,
) -> tuple[str, dict[str, Any]]:
    """Erstellt einen auf Stress, Konflikte & ADHS zugeschnittenen Tages-Impuls (morning|midday|evening)."""
    phase = phase if phase in _PHASE_META else "morning"
    meta = _PHASE_META[phase]

    parent_prof = coaching_core.load_parent_profile(store)
    parent_ctx = coaching_core.format_parent_profile_context(parent_prof)
    recent_logs = store.get_recent_coaching_logs("parent", days=5, limit=8)
    recent_summary = "\n".join(
        f"- {l['date']} [{l['kind']}] Stress={l.get('stress_level') or '-'} | Feedback={l.get('feedback') or '-'}: {(l.get('prompt_summary') or l.get('content') or '')[:120]}"
        for l in recent_logs
    ) or "Noch keine kürzlichen Einträge."

    today_iso = dt.date.today().isoformat()
    prompt = f"""Heute ist {today_iso}. Tagesphase: {phase.upper()} ({meta['label']}).
Schwerpunkt dieser Phase: {meta['focus']}

=== FAMILIEN- & ADHS-KONTEXT ===
{parent_ctx}

=== LETZTE CHECK-INS & REFLEXIONEN (Wiederholungen vermeiden!) ===
{recent_summary}

AUFGABE:
Erstelle einen kurzen, praxisnahen, neurodiversitäts-sensiblen Elterncoaching-Impuls für diese Tageszeit (max. 170 Wörter im Impuls-Text).
Wichtig:
- Konzentriere dich strikt auf Umgang mit Stress, Konflikten und ADHS im Familienalltag.
- Liefere eine sofort nutzbare Mikro-Strategie (z. B. 10-Sekunden-Co-Regulation, konkreter Satz für schwierige Situationen, oder klare visuelle/zeitliche Strukturhilfe).
- Keine Vorwürfe, kein perfektes Eltern-Ideal, sondern echte Entlastung & Klarheit.

Antworte AUSSCHLIESSLICH mit diesem JSON-Objekt:
{{
  "topic": "Kurzes Schlagwort (z. B. Co-Regulation | ADHS-Übergang | Klare Grenze | Reizschutz)",
  "headline": "Prägnanter Titel (max. 65 Zeichen)",
  "impulse_text": "2 kurze Absätze praktischer Impuls auf Deutsch",
  "ready_phrase": "1 konkreter Satz zum wörtlichen Aussprechen im Alltag",
  "checkin_question": "1 kurze Check-in-Frage passend zur Tageszeit",
  "optional_task_title": "Optionaler konkreter Orga-Task (oder leer lassen)"
}}"""

    sys_inst = (
        "Du bist ein erfahrener, bindungsorientierter und ADHS-kundiger Elterncoach. "
        "Du unterstützt Eltern praxisnah bei Stressregulation, Konfliktlösung und ADHS-Alltagsstrukturen. "
        "Antworte ausschließlich mit validem JSON."
    )

    try:
        data = _call_llm_json(prompt, cfg, task="coach_deep", system_instruction=sys_inst)
    except Exception as exc:
        log.warning("Parent-Bot Check-in LLM Fehler (%s) -> Fallback.", exc)
        data = {
            "topic": "Co-Regulation & Struktur",
            "headline": "Erst Verbindung und Ruhe, dann die Ansage",
            "impulse_text": (
                "Bei ADHS und Konfliktstress eskalieren Situationen oft dann, wenn gleichzeitig geredet, "
                "erklärt und gedrängt wird. Nimm dir heute vor schwierigen Übergängen bewusst 10 Sekunden, "
                "senke die Schultern und gib nur einen einzigen, klaren Schritt statt einer Kette."
            ),
            "ready_phrase": "Ich sehe, das ist gerade viel. Wir machen jetzt genau diesen einen Schritt zusammen.",
            "checkin_question": meta["question"],
            "optional_task_title": "",
        }

    esc = telegram_ui.esc
    topic = str(data.get("topic") or "Elterncoaching")
    headline = str(data.get("headline") or meta["label"])
    body = str(data.get("impulse_text") or "")
    phrase = str(data.get("ready_phrase") or "")
    q_txt = str(data.get("checkin_question") or meta["question"])

    html_msg = (
        f"{meta['emoji']} <b>{esc(meta['label'])}</b> <i>[{esc(topic)}]</i>\n"
        f"<b>{esc(headline)}</b>\n\n"
        f"{esc(body)}\n\n"
        + (f"💬 <b>Satz für heute:</b> <i>„{esc(phrase)}“</i>\n\n" if phrase else "")
        + f"❓ {esc(q_txt)}"
    )
    return html_msg, data


def send_parent_checkin(
    phase: str = "morning",
    chat_id: str | None = None,
    cfg: dict[str, Any] | None = None,
    token: str | None = None,
) -> tuple[bool, str]:
    cfg = cfg or _load_env_and_cfg()
    tok = token or get_active_token()
    st = load_parent_state()
    cid = chat_id or resolve_target_chat_id(st)
    if not tok or not cid:
        return False, "Parent-Bot Token oder Chat-ID fehlt."

    store = get_store()
    html_msg, data = build_parent_checkin(phase, cfg, store)
    log_id = store.add_coaching_log(
        bot="parent",
        kind=f"{phase}_checkin",
        topic=str(data.get("topic") or ""),
        prompt_summary=str(data.get("headline") or ""),
        content=f"{data.get('impulse_text', '')}\nSatz: {data.get('ready_phrase', '')}",
    )

    cb_tok = secrets.token_hex(6)
    store.put_callback_payload(cb_tok, "parent_checkin", {
        "log_id": log_id,
        "phase": phase,
        "topic": data.get("topic", ""),
        "headline": data.get("headline", ""),
        "task_title": data.get("optional_task_title") or "",
    })

    if phase == "evening":
        rows = [
            [("✅ Ruhig gemeistert", f"p_fb_calm:{cb_tok}"), ("⚡ Konflikt gehabt", f"p_fb_conflict:{cb_tok}")],
            [("🔴 Sehr stressig", f"p_stress_3:{cb_tok}"), ("💾 Strategie merken", f"p_save_strat:{cb_tok}")],
        ]
    else:
        rows = [
            [("🟢 Ruhig", f"p_stress_1:{cb_tok}"), ("🟡 Angespannt", f"p_stress_2:{cb_tok}"), ("🔴 Überlastet", f"p_stress_3:{cb_tok}")],
            [("🆘 Akut-Hilfe Konflikt/ADHS", f"p_akut_help:{cb_tok}")],
        ]
    kb = telegram_ui.build_inline_keyboard(rows)
    ok = send_message(tok, cid, html_msg, reply_markup=kb)
    return ok, html_msg


def generate_akut_help(situation: str, cfg: dict[str, Any], store: StateStore) -> str:
    """Erstellt eine strukturierte Sofort-Deeskalationskarte für Konflikt-/Stress-/ADHS-Situationen."""
    parent_prof = coaching_core.load_parent_profile(store)
    parent_ctx = coaching_core.format_parent_profile_context(parent_prof)

    prompt = f"""AKUTE ELTERNSITUATION (Soforthilfe gebraucht – kurz, ruhig, sofort lesbar):
Situation: "{situation}"

Kontext:
{parent_ctx}

Liefere eine kompakte Soforthilfe in genau 3 kurzen Punkten (max. 160 Wörter insgesamt):
1. 🫁 Eigener Reset (10 Sekunden): Was du jetzt körperlich/innerlich tust, bevor du sprichst.
2. 💬 Wörtlicher Satz (ADHS- & konflikttauglich): Ein ruhiger, kurzer, klarer Satz ohne Rechtfertigung oder Machtkampf.
3. 🪜 Nächster Mini-Schritt: Eine konkrete, kleine Handlungsoption für die nächsten 2–5 Minuten.

Schlichter deutscher Text ohne Markdown-Codeblöcke."""

    sys_inst = (
        "Du bist ein ruhiger Krisen- & ADHS-Elterncoach. Keine langen Theorien, keine Vorwürfe – "
        "nur sofort wirksame Co-Regulation, Klarheit und Deeskalation."
    )
    try:
        reply = _call_llm_text(prompt, cfg, task="coach_deep", system_instruction=sys_inst)
    except Exception as exc:
        log.warning("Parent-Bot Akut LLM Fehler: %s", exc)
        reply = (
            "1. 🫁 Reset: 2x tief ausatmen, Schultern senken, Tempo rausnehmen.\n"
            "2. 💬 Satz: „Stopp, wir machen kurz Pause. Ich bleibe ruhig und wir schauen gleich in 2 Minuten weiter.“\n"
            "3. 🪜 Mini-Schritt: Reize reduzieren (Licht/Ton/Publikum weg) und nur EINE konkrete Sache ansprechen."
        )

    store.add_coaching_log(
        bot="parent",
        kind="akut_help",
        topic="Akut-Situation",
        prompt_summary=situation[:140],
        content=f"Situation: {situation}\n\nAntwort:\n{reply}",
    )
    return reply


def generate_weekly_patterns(cfg: dict[str, Any], store: StateStore) -> str:
    """Wertet lokale Check-ins, Stresswerte & Akutsituationen der letzten 7 Tage aus (rein lokal)."""
    logs = store.get_recent_coaching_logs("parent", days=7, limit=30)
    if not logs:
        return "📊 <b>Wochen-Muster (lokal):</b> Noch keine Einträge in den letzten 7 Tagen."

    stress_vals = [l["stress_level"] for l in logs if l.get("stress_level") is not None]
    avg_stress = f"{sum(stress_vals)/len(stress_vals):.1f}/3" if stress_vals else "noch nicht erfasst"

    logs_txt = "\n".join(
        f"- {l['date']} [{l['kind']}] Stress={l.get('stress_level') or '-'} | FB={l.get('feedback') or '-'}: {l['content'][:180]}"
        for l in logs
    )
    prompt = f"""Analysiere diese rein lokalen Elterncoaching-Logs der letzten 7 Tage (Durchschnitts-Stress: {avg_stress}):
{logs_txt}

Erstelle eine ermutigende, praxisnahe Muster-Auswertung (max. 240 Wörter):
- Wann war die Anspannung am höchsten (Morgen / Mittag / Abend)?
- Welche Strategien haben geholfen?
- 2 konkrete ADHS-/Stress-Anpassungen für die nächste Woche.
Schlichtes Deutsch ohne Markdown-Codeblöcke."""

    try:
        analysis = _call_llm_text(prompt, cfg, task="coach_deep")
    except Exception as exc:
        analysis = f"Lokale Übersicht: {len(logs)} Einträge, Ø Stresslevel: {avg_stress} ({exc})."

    esc = telegram_ui.esc
    return (
        f"🧩 <b>Wöchentliche Eltern-Muster (rein lokal, Ø Stress: {esc(avg_stress)})</b>\n\n"
        f"{esc(analysis)}"
    )


# --------------------------------------------------------------------------------------
# Befehls-Handler
# --------------------------------------------------------------------------------------

def handle_start_or_help() -> str:
    return (
        "👨‍👩‍👧‍👦 <b>Willkommen beim Elterncoaching-Bot!</b>\n\n"
        "Dein Fokus: <b>Stressregulation, ruhige Konfliktlösung &amp; ADHS im Alltag</b>.\n"
        "<i>Datenschutz: Alle Gespräche und Familiennotizen bleiben rein lokal in deiner Server-Datenbank.</i>\n\n"
        "📋 <b>Befehle:</b>\n"
        "• <code>/akut &lt;Situation&gt;</code> – Sofort-Deeskalation &amp; konkreter Satz zum Aussprechen\n"
        "• <code>/impuls [morgen|mittag|abend]</code> – Aktuellen Check-in-Impuls abrufen\n"
        "• <code>/checkin</code> – Stress- &amp; Tages-Check-in starten\n"
        "• <code>/muster</code> – Wöchentliche Stress- &amp; Konfliktmuster (lokal)\n"
        "• <code>/profil</code> – Familien- &amp; ADHS-Kontext ansehen oder mit <code>/profil &lt;Info&gt;</code> ergänzen\n"
        "• <code>/status</code> – Status &amp; Check-in-Zeiten (07:15 / 12:45 / 20:15)\n"
        "• <code>/update</code> – Neuesten Code laden &amp; neu starten\n\n"
        "🎤 <i>Tipp: Du kannst mir im Alltag jederzeit direkt per Text oder Sprachnachricht schreiben, wenn es gerade hochkocht!</i>"
    )


def handle_profil(arg: str, store: StateStore) -> str:
    esc = telegram_ui.esc
    if arg.strip():
        note = arg[4:].strip() if arg.lower().startswith("add ") else arg.strip()
        coaching_core.append_parent_note(note, store)
        return f"✅ <b>Familienprofil lokal ergänzt:</b> „{esc(note)}“"

    prof = coaching_core.load_parent_profile(store)
    fc = prof.get("family_context", {})
    lines = [
        "👨‍👩‍👧‍👦 <b>Familien- &amp; ADHS-Profil (rein lokal)</b>\n",
        f"• <b>Kontext:</b> {esc(fc.get('children_summary', ''))}",
        f"• <b>Schwerpunkte:</b> {esc(', '.join(fc.get('focus_areas', [])))}",
    ]
    if fc.get("working_strategies"):
        lines.append("\n<b>Bewährte Strategien:</b>")
        for s in fc["working_strategies"][:5]:
            lines.append(f"  • {esc(s)}")
    notes = prof.get("custom_notes") or []
    if notes:
        lines.append("\n<b>Eigene Ergänzungen:</b>")
        for n in notes[-5:]:
            lines.append(f"  • ({esc(n.get('at', '')[:10])}) {esc(n.get('text', ''))}")
    lines.append("\n<i>Ergänze jederzeit Alter, ADHS-Details oder funktionierende Tricks via <code>/profil &lt;Text&gt;</code>.</i>")
    return "\n".join(lines)


def handle_status(store: StateStore) -> str:
    esc = telegram_ui.esc
    logs = store.get_recent_coaching_logs("parent", days=7, limit=30)
    stress_vals = [l["stress_level"] for l in logs if l.get("stress_level") is not None]
    avg_s = f"{sum(stress_vals)/len(stress_vals):.1f}/3" if stress_vals else "noch keine Bewertung"
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
        "👨‍👩‍👧‍👦 <b>Elterncoaching-Bot Status</b>\n",
        f"• 🖥️ <b>Host:</b> <code>{esc(sys.platform)}</code> @ <code>{esc(_plat.node())}</code> (PID <code>{os.getpid()}</code>" + (f", Commit <code>{esc(_commit)}</code>" if _commit else "") + ")",
        f"• 🔒 <b>Datenschutz:</b> Rein lokale Speicherung (SQLite + gitignored Profil)",
        f"• 📊 <b>Check-ins (7 Tage):</b> {len(logs)} | <b>Ø Stress:</b> {esc(avg_s)}",
        "\n⏰ <b>Tägliche Check-ins (Europe/Berlin):</b>",
        f"• 07:15 Uhr Morgen-Check-in (Zuletzt: <code>{esc(sched_st.get('parent_morning', 'noch nicht'))}</code>)",
        f"• 12:45 Uhr Mittags-Check-in (Zuletzt: <code>{esc(sched_st.get('parent_midday', 'noch nicht'))}</code>)",
        f"• 20:15 Uhr Abend-Reflexion (Zuletzt: <code>{esc(sched_st.get('parent_evening', 'noch nicht'))}</code>)",
    ]
    return "\n".join(lines)


def handle_natural_language(chat_id: str, text: str, token: str, cfg: dict[str, Any]) -> None:
    store = get_store()
    store.append_chat_turn("parent", chat_id, "user", text)

    deep_triggers = ("akut", "eskal", "schrei", "meltdown", "wut", "konflikt", "adhs", "überfordert", "streit", "hausaufgaben", "schlaf", "verzweifelt")
    use_deep = any(k in text.lower() for k in deep_triggers) or len(text) > 300
    task_tier = "coach_deep" if use_deep else "coach_chat"

    prof = coaching_core.load_parent_profile(store)
    parent_ctx = coaching_core.format_parent_profile_context(prof)
    history = store.get_recent_chat_turns("parent", chat_id, limit=8)
    history_block = "\n".join(f"{h['role'].upper()}: {h['content'][:300]}" for h in history[:-1])

    system_instr = (
        "Du bist der einfühlsame, ADHS-kundige Elterncoach von Stefan auf Telegram.\n"
        f"Familien- & Persönlichkeitskontext:\n{parent_ctx}\n\n"
        "REGELN:\n"
        "1. Fokussiere auf Stressregulation, Co-Regulation, Konflikt-Deeskalation und ADHS-gerechte Struktur.\n"
        "2. Gib immer erst Entlastung/Validierung, dann 1–2 sofort anwendbare Sätze oder Mini-Schritte.\n"
        "3. Antworte kompakt auf Deutsch ohne Markdown-Codeblöcke."
    )
    prompt = (f"Bisheriger Gesprächsverlauf:\n{history_block}\n\n" if history_block else "") + f"Nachricht:\n{text}"

    try:
        reply = _call_llm_text(prompt, cfg, task=task_tier, system_instruction=system_instr)
    except Exception as exc:
        log.warning("Parent-Bot LLM Fehler: %s", exc)
        reply = "Ich bin da. Nimm kurz einen tiefen Atemzug – beschreibe mir in 1–2 Sätzen, was gerade passiert ist."

    store.append_chat_turn("parent", chat_id, "assistant", reply[:600])
    if len(text) >= 20:
        store.add_coaching_log(
            bot="parent",
            kind="dialog",
            topic="Eltern-Sparring",
            prompt_summary=text[:120],
            content=f"Elternteil: {text[:800]}\nCoach: {reply[:800]}",
        )

    cb_tok = secrets.token_hex(6)
    store.put_callback_payload(cb_tok, "parent_reply", {"text": text[:160], "reply": reply[:240]})
    kb = telegram_ui.build_inline_keyboard([
        [("💾 Strategie lokal merken", f"p_save_strat:{cb_tok}"), ("📌 Als Task nach Notion", f"p_notion_task:{cb_tok}")],
    ])
    send_message(token, chat_id, telegram_ui.esc(reply), reply_markup=kb)


# --------------------------------------------------------------------------------------
# Callback Query Handler
# --------------------------------------------------------------------------------------

def handle_callback_query(token: str, cb: dict[str, Any], cfg: dict[str, Any]) -> None:
    cb_id = str(cb.get("id") or "")
    data = str(cb.get("data") or "")
    msg = cb.get("message") or {}
    chat_id = str(msg.get("chat", {}).get("id", ""))

    if not chat_id or not telegram_ui.is_authorized_chat(
        chat_id, env_keys=("TELEGRAM_PARENT_CHAT_ID", "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USERS")
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

    if action in ("p_stress_1", "p_stress_2", "p_stress_3"):
        lvl = int(action.split("_")[-1])
        lbl = {1: "ruhig (1/3)", 2: "angespannt (2/3)", 3: "hoch/überlastet (3/3)"}[lvl]
        if log_id:
            store.set_coaching_feedback(log_id, f"stress_{lvl}", stress_level=lvl)
        telegram_ui.answer_callback_query(token, cb_id, f"Stresslevel gespeichert: {lbl}")
        if lvl == 1:
            send_message(token, chat_id, "🟢 <b>Schön zu hören!</b> Nutze die Ruhe für einen entspannten kleinen Verbindungsmoment.")
        elif lvl == 2:
            send_message(token, chat_id, "🟡 <b>Anspannung früh bemerkt – stark!</b> Tipp: Nimm vor dem nächsten Übergang 30 Sekunden Pause, schalte Reize runter und sprich in ganz kurzen Sätzen.")
        else:
            send_message(
                token,
                chat_id,
                "🔴 <b>Hoher Druck erkannt – zuerst DU, dann die Situation:</b>\n"
                "1. Ansprüche für heute auf absolutes Minimum senken.\n"
                "2. Nichts Grundsätzliches jetzt ausdiskutieren.\n"
                "3. Schreib mir einfach per Sprachnachricht, was gerade konkret drückt – oder nutze <code>/akut</code>.",
            )
        return

    if action == "p_akut_help":
        telegram_ui.answer_callback_query(token, cb_id, "🆘 Soforthilfe kommt...")
        reply = generate_akut_help("Akute Stress-/Konfliktsituation im ADHS-Familienalltag", cfg, store)
        send_message(token, chat_id, f"🆘 <b>Soforthilfe-Karte:</b>\n\n{esc(reply)}")
        return

    if action in ("p_fb_calm", "p_fb_conflict"):
        fb = "calm" if action == "p_fb_calm" else "conflict"
        if log_id:
            store.set_coaching_feedback(log_id, fb, stress_level=1 if fb == "calm" else 2)
        telegram_ui.answer_callback_query(token, cb_id, "Danke für dein Check-in!")
        if fb == "calm":
            send_message(token, chat_id, "✅ <b>Stark!</b> Was hat heute besonders geholfen? Schreib mir gern 1 Satz dazu, dann merken wir uns die Strategie.")
        else:
            send_message(token, chat_id, "⚡ <b>Konflikte gehören bei ADHS &amp; Stress dazu – kein Versagen.</b> Wenn du magst, schreib kurz per Voice oder Text, wie es losging, und wir schauen gemeinsam ohne Schuldgefühl drauf.")
        return

    if action == "p_save_strat":
        note_txt = payload.get("headline") or payload.get("reply") or payload.get("text") or "Bewährte Strategie"
        coaching_core.append_parent_note(f"Gemerkt: {note_txt[:260]}", store)
        telegram_ui.answer_callback_query(token, cb_id, "💾 Strategie lokal im Profil gemerkt!")
        return

    if action == "p_notion_task":
        ntok = os.environ.get("NOTION_TOKEN", "").strip()
        title = str(payload.get("task_title") or payload.get("text") or "Familien-Aufgabe")[:80]
        if ntok and cfg.get("notion", {}).get("tasks_database_id"):
            try:
                NotionClient(ntok).create_task_page(
                    tasks_db_id=cfg["notion"]["tasks_database_id"],
                    title=title,
                    kind="To-Do",
                    priority="Mittel",
                    due_date=None,
                    next_step="Aus Elterncoaching übernommen",
                    area="Familie/Privat",
                    estimated_minutes=15,
                )
                telegram_ui.answer_callback_query(token, cb_id, "📌 Als Task in Notion gespeichert!")
                send_message(token, chat_id, f"📌 <b>Aufgabe nach Notion übertragen:</b> {esc(title)}")
                return
            except Exception as exc:
                log.warning("Parent-Bot manueller Notion-Export Fehler: %s", exc)
        telegram_ui.answer_callback_query(token, cb_id, "Notion nicht verbunden.", show_alert=True)
        return

    telegram_ui.answer_callback_query(token, cb_id, "Unbekannte Aktion.")


# --------------------------------------------------------------------------------------
# Telegram Polling
# --------------------------------------------------------------------------------------

def process_updates(token: str, state: dict[str, Any], poll_timeout: int = 15) -> int:
    offset = int(state.get("parent_telegram_offset", 0))
    api_base = f"https://api.telegram.org/bot{token}"

    try:
        req = urllib.request.Request(f"{api_base}/getUpdates?offset={offset}&timeout={poll_timeout}")
        with urllib.request.urlopen(req, timeout=poll_timeout + 10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.debug("ParentBot Polling Hinweis: %s", exc)
        return 0

    if not data.get("ok"):
        return 0

    cfg = _load_env_and_cfg()
    count = 0

    for update in data.get("result", []):
        update_id = update["update_id"]
        state["parent_telegram_offset"] = max(int(state.get("parent_telegram_offset", 0)), update_id + 1)
        save_parent_state(state)
        count += 1

        if "callback_query" in update:
            try:
                handle_callback_query(token, update["callback_query"], cfg)
            except Exception as exc:
                log.exception("ParentBot Callback Fehler in Update %s: %s", update_id, exc)
            continue

        msg = update.get("message") or update.get("edited_message")
        if not msg:
            continue

        chat_id = str(msg.get("chat", {}).get("id", ""))
        if not chat_id or not telegram_ui.is_authorized_chat(
            chat_id, env_keys=("TELEGRAM_PARENT_CHAT_ID", "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USERS")
        ):
            log.warning("Unautorisierter ParentBot-Zugriff von Chat-ID %s blockiert.", chat_id)
            continue

        try:
            if state.get("last_parent_chat_id") != chat_id:
                state["last_parent_chat_id"] = chat_id
                save_parent_state(state)

            raw_text = (msg.get("text") or "").strip()
            voice = msg.get("voice") or msg.get("audio")
            doc = msg.get("document")
            photo = msg.get("photo")

            if voice and assistant is not None:
                file_id = voice.get("file_id")
                if file_id:
                    tmp_audio = telegram_ui.download_telegram_file(token, file_id, BASE_DIR / "temp", prefix="parent_voice")
                    if tmp_audio:
                        try:
                            raw_text = assistant.transcribe_audio(tmp_audio, cfg)
                            send_message(token, chat_id, f"🎤 <i>Erkannt:</i> „{telegram_ui.esc(raw_text)}“")
                        finally:
                            tmp_audio.unlink(missing_ok=True)

            if photo and not raw_text:
                photo_id = photo[-1]["file_id"]
                caption = (msg.get("caption") or "").strip()
                tmp_img = telegram_ui.download_telegram_file(token, photo_id, BASE_DIR / "temp", prefix="parent_img")
                if tmp_img:
                    try:
                        ext = tmp_img.suffix.lower()
                        mime = "image/png" if ext == ".png" else "image/jpeg"
                        ref = assistant.get_date_reference_block() if assistant is not None else ""
                        res_mm = agent_tools.analyze_multimodal_file(tmp_img, mime, caption, cfg, ref)
                        handle_natural_language(
                            chat_id,
                            f"Bild-Kontext ({res_mm.get('title', '')}): {res_mm.get('summary', '')}\n{caption}",
                            token,
                            cfg,
                        )
                        continue
                    finally:
                        tmp_img.unlink(missing_ok=True)

            if doc and not raw_text:
                file_id = doc.get("file_id")
                fname = str(doc.get("file_name") or "dokument")
                mime = str(doc.get("mime_type") or "application/pdf")
                caption = (msg.get("caption") or "").strip()
                if file_id:
                    tmp_doc = telegram_ui.download_telegram_file(token, file_id, BASE_DIR / "temp", prefix="parent_doc")
                    if tmp_doc:
                        try:
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
            elif cmd in ("/akut", "/sos", "/hilfe_akut", "akut"):
                sit = arg or "Akute Überforderung / Konflikt im Familienalltag"
                reply = generate_akut_help(sit, cfg, store)
                send_message(token, chat_id, f"🆘 <b>Soforthilfe:</b>\n\n{telegram_ui.esc(reply)}")
            elif cmd in ("/impuls", "/impulse", "impuls"):
                phase_map = {"morgen": "morning", "morning": "morning", "mittag": "midday", "midday": "midday", "abend": "evening", "evening": "evening"}
                phase = phase_map.get(arg.lower())
                if not phase:
                    hour = dt.datetime.now().hour
                    phase = "morning" if hour < 11 else ("midday" if hour < 17 else "evening")
                send_parent_checkin(phase=phase, chat_id=chat_id, cfg=cfg, token=token)
            elif cmd in ("/checkin", "/check-in", "checkin"):
                hour = dt.datetime.now().hour
                phase = "morning" if hour < 11 else ("midday" if hour < 17 else "evening")
                send_parent_checkin(phase=phase, chat_id=chat_id, cfg=cfg, token=token)
            elif cmd in ("/muster", "/review", "muster"):
                send_message(token, chat_id, generate_weekly_patterns(cfg, store))
            elif cmd in ("/profil", "/profile", "profil"):
                send_message(token, chat_id, handle_profil(arg, store))
            elif cmd in ("/status", "status"):
                send_message(token, chat_id, handle_status(store))
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
            log.exception("ParentBot Fehler in Update %s: %s", update_id, exc)
            send_message(token, chat_id, f"❌ Interner Fehler: {telegram_ui.esc(exc)}")

    return count


def run_bot_daemon(poll_timeout: int = 15) -> None:
    token = get_active_token()
    if not token:
        log.error("TELEGRAM_PARENTBOT_TOKEN nicht gesetzt.")
        return

    telegram_ui.register_bot_commands(token, telegram_ui.PARENT_BOT_COMMANDS)
    log.info("Elterncoaching-Bot Daemon gestartet (Token: %s...). Warte auf Nachrichten...", token[:10])
    while True:
        try:
            state = load_parent_state()
            process_updates(token, state, poll_timeout=poll_timeout)
        except Exception as exc:
            log.warning("ParentBot Polling-Fehler: %s", exc)
            time.sleep(5)


def start_in_background(poll_timeout: int = 15) -> threading.Thread | None:
    global _background_thread
    token = get_active_token()
    if not token:
        log.warning("Kein TELEGRAM_PARENTBOT_TOKEN gesetzt – Eltern-Bot startet nicht.")
        return None
    if _background_thread is not None and _background_thread.is_alive():
        return _background_thread

    def _loop():
        run_bot_daemon(poll_timeout=poll_timeout)

    _background_thread = threading.Thread(target=_loop, daemon=True, name="ParentBotDaemon")
    _background_thread.start()
    return _background_thread


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Dedizierter Telegram Elterncoaching-Bot")
    parser.add_argument("--once", action="store_true", help="Einmalig Updates pollen")
    parser.add_argument("--checkin", choices=["morning", "midday", "evening"], help="Check-in sofort senden")
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    token = get_active_token()
    if not token:
        print("FEHLER: TELEGRAM_PARENTBOT_TOKEN ist nicht gesetzt!")
        sys.exit(1)

    cfg = _load_env_and_cfg()
    if args.checkin:
        ok, _ = send_parent_checkin(phase=args.checkin, cfg=cfg, token=token)
        print("Check-in gesendet." if ok else "Check-in konnte nicht gesendet werden (Chat-ID prüfen).")
        return

    if args.once:
        cnt = process_updates(token, load_parent_state(), poll_timeout=args.timeout)
        print(f"{cnt} Update(s) verarbeitet.")
    else:
        run_bot_daemon(poll_timeout=args.timeout)


if __name__ == "__main__":
    main()
