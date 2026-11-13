#!/usr/bin/env python3
"""
RSS-Monitor mit wachsendem Wissensspeicher in Notion.

Ablauf pro Lauf (geplant 2x täglich, siehe setup_scheduler.ps1):
1. Für jedes aktive Thema aus `feeds.json` werden RSS-Feeds und Europe-PMC-Abfragen gelesen.
2. Bereits gesehene Artikel (Link- oder Titel-Fingerprint in `rss_state.json`) werden übersprungen.
3. Die übrigen Kandidaten werden zusammen mit dem bisherigen Wissensstand des Themas
   (zuletzt erfasste Einträge aus der Notion-Datenbank "Wissensspeicher") an das LLM
   (Gemini 3.8 Flash, Fallback lokal Ollama) gegeben. Es wählt nur relevante UND wirklich neue
   Funde aus, fasst Meldungen zur selben Nachricht zusammen und erkennt Weiterentwicklungen.
4. Neue Funde werden als Einträge im Wissensspeicher angelegt, pro Lauf entsteht eine
   Digest-Seite in "Documents"; optional Benachrichtigung per Telegram und Windows.

Nutzung:
    python rss_monitor.py run                    # alle Themen
    python rss_monitor.py run --topic parkinson  # nur ein Thema
    python rss_monitor.py run --dry-run          # nichts schreiben, nur anzeigen
    python rss_monitor.py topics                 # Themen und Quellen anzeigen
    python rss_monitor.py test-feeds             # Erreichbarkeit aller Quellen prüfen
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

try:
    import feedparser  # type: ignore
except ImportError:
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "feedparser", "-q"], timeout=60)
        import feedparser  # type: ignore
    except Exception:
        feedparser = None

try:
    import requests
except ImportError:
    try:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "requests", "-q"], timeout=60)
        import requests
    except Exception:
        requests = None

try:
    import trafilatura  # type: ignore
except ImportError:
    trafilatura = None

import assistant  # bestehender Assistent: Config/.env, NotionClient, LLM-Auswahl, Benachrichtigungen
import jev_client
import llm_core
import preprocess
from core.storage import StateStore, atomic_write_json, safe_read_json
from integrations import telegram_ui

BASE_DIR = Path(__file__).resolve().parent
FEEDS_PATH = BASE_DIR / "feeds.json"
STATE_PATH = BASE_DIR / "rss_state.json"
LOCK_PATH = BASE_DIR / "rss_monitor.lock"
LOCK_FILE = LOCK_PATH
LOG_PATH = BASE_DIR / "rss_monitor.log"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NotionRSSMonitor/1.0 (+stefan.kollmeier@hs-rm.de)"
RELEVANCE_ORDER = {"Niedrig": 0, "Mittel": 1, "Hoch": 2}

log = logging.getLogger("rss_monitor")

_STORE_CACHE: dict[str, StateStore] = {}


def get_store(db_path: Path | None = None) -> StateStore:
    """Gibt den zentralen SQLite-StateStore (inkl. Legacy-Migration) zurück."""
    target = db_path or Path(os.environ.get("ASSISTANT_DB_PATH") or (BASE_DIR / "assistant.db"))
    key = str(target.resolve())
    if key not in _STORE_CACHE:
        st = StateStore(target)
        try:
            st.migrate_from_legacy_json(BASE_DIR)
        except Exception as exc:
            log.debug("Legacy-Migration Hinweis: %s", exc)
        _STORE_CACHE[key] = st
    return _STORE_CACHE[key]


# --------------------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    if not log.handlers:
        fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
        if sys.stderr is not None:  # unter pythonw.exe gibt es keine Konsole
            sh = logging.StreamHandler(sys.stderr)
            sh.setFormatter(fmt)
            log.addHandler(sh)


def load_feeds_config() -> dict[str, Any]:
    with FEEDS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_state() -> dict[str, Any]:
    data = safe_read_json(STATE_PATH, {"seen": {}, "knowledge": {}, "runs": []})
    if not isinstance(data, dict):
        data = {}
    data.setdefault("seen", {})
    data.setdefault("knowledge", {})
    data.setdefault("runs", [])
    return data


def save_state(state: dict[str, Any]) -> None:
    atomic_write_json(STATE_PATH, state, indent=1)

def sha(text: str) -> str:
    return preprocess.sha(text)


def strip_html(text: str) -> str:
    return preprocess.strip_html_and_boilerplate(text)


def normalize_title(title: str) -> str:
    return preprocess.normalize_title(title)


def item_fingerprints(item: dict[str, Any]) -> list[str]:
    return preprocess.compute_item_fingerprints(item)


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if h:
                kernel32.CloseHandle(h)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except Exception:
        return False


def acquire_lock() -> bool:
    if LOCK_PATH.exists():
        try:
            old_pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
        except Exception:
            old_pid = 0
        if old_pid and is_pid_running(old_pid):
            age = time.time() - LOCK_PATH.stat().st_mtime
            if age < 3 * 3600:
                return False
            log.warning("Veraltete Lock-Datei von PID %d (%.0f min) wird verdrängt.", old_pid, age / 60)
        else:
            log.info("Vorherige Lock-Datei von inaktiver PID %d wird entfernt.", old_pid)
    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
    return True


def release_lock() -> None:
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------------------
# Quellen
# --------------------------------------------------------------------------------------

_feed_cache: dict[str, list[dict[str, Any]]] = {}


def _http_get_bytes(url: str, timeout: int = 30) -> bytes:
    """Ruft eine URL über requests oder das native urllib.request ab."""
    if requests is not None:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse_feed(content: bytes) -> Any:
    """Parst RSS/Atom entweder per feedparser oder per nativem ElementTree XML-Fallback."""
    if feedparser is not None:
        return feedparser.parse(content)

    import xml.etree.ElementTree as ET
    import email.utils

    class _Entry:
        def __init__(self, title: str, link: str, summary: str, published_parsed: Any = None, src_name: str | None = None):
            self.title = title
            self.link = link
            self.summary = summary
            self.published_parsed = published_parsed
            self.updated_parsed = published_parsed
            self.source = {"title": src_name} if src_name else {}

        def get(self, k: str, default: Any = None) -> Any:
            return getattr(self, k, default)

    class _Feed:
        def __init__(self, entries: list[Any]):
            self.entries = entries

    entries: list[Any] = []
    try:
        root = ET.fromstring(content)
        # 1. RSS 2.0 (<channel><item>...)
        for item in root.findall(".//item"):
            t = (item.findtext("title") or "").strip()
            l = (item.findtext("link") or "").strip()
            s = (item.findtext("description") or item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded") or "").strip()
            src = (item.findtext("source") or "").strip()
            pub = item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date")
            ts = None
            if pub:
                try:
                    ts = email.utils.parsedate(pub)
                except Exception:
                    pass
            entries.append(_Entry(t, l, s, published_parsed=ts, src_name=src))

        # 2. Atom (<feed><entry>...)
        if not entries:
            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry") or root.findall(".//entry"):
                t = (entry.findtext("{http://www.w3.org/2005/Atom}title") or entry.findtext("title") or "").strip()
                l_el = entry.find("{http://www.w3.org/2005/Atom}link") or entry.find("link")
                l = l_el.get("href", "") if l_el is not None else ""
                s = (entry.findtext("{http://www.w3.org/2005/Atom}summary") or entry.findtext("{http://www.w3.org/2005/Atom}content") or entry.findtext("summary") or "").strip()
                upd = entry.findtext("{http://www.w3.org/2005/Atom}updated") or entry.findtext("{http://www.w3.org/2005/Atom}published")
                ts = None
                if upd:
                    try:
                        clean_dt = upd.replace("Z", "+00:00")
                        ts = dt.datetime.fromisoformat(clean_dt).timetuple()
                    except Exception:
                        pass
                entries.append(_Entry(t, l, s, published_parsed=ts))
    except Exception as exc:
        log.warning("XML Fallback Parser Fehler: %s", exc)
    return _Feed(entries)


def fetch_rss(source: dict[str, Any], max_age_days: int, max_items: int) -> list[dict[str, Any]]:
    url = source["url"]
    if url not in _feed_cache:
        content = _http_get_bytes(url, timeout=30)
        parsed = _parse_feed(content)
        entries = []
        for e in parsed.entries:
            ts = e.get("published_parsed") or e.get("updated_parsed")
            published = dt.datetime(*ts[:6], tzinfo=dt.timezone.utc) if ts else None
            src_name = (e.get("source") or {}).get("title") if isinstance(e.get("source"), dict) else None
            raw_desc = e.get("summary") or e.get("description") or ""
            title_text = strip_html(e.get("title") or "")
            if not title_text and raw_desc:
                # Für Microblogging-Feeds (Bluesky, Mastodon etc.) ohne separaten Titel: erste Zeile nutzen
                first_line = strip_html(raw_desc.split("<br")[0].split("\n")[0])
                title_text = first_line[:110] if first_line else strip_html(raw_desc)[:110]
            entries.append({
                "title": title_text,
                "link": e.get("link", ""),
                "summary": strip_html(raw_desc)[:1200],
                "published": published,
                "publisher": src_name,
            })
        _feed_cache[url] = entries

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max_age_days)
    keywords = [k.lower() for k in source.get("keywords", [])]
    out = []
    for e in _feed_cache[url]:
        if e["published"] and e["published"] < cutoff:
            continue
        if keywords:
            hay = (e["title"] + " " + e["summary"]).lower()
            if not any(k in hay for k in keywords):
                continue
        out.append(dict(e, source=source["name"] + (f" / {e['publisher']}" if e.get("publisher") else "")))
        if len(out) >= max_items:
            break
    return out


def fetch_europepmc(source: dict[str, Any], max_age_days: int, max_items: int) -> list[dict[str, Any]]:
    today = dt.date.today()
    start = today - dt.timedelta(days=max_age_days)
    query = f"({source['query']}) AND FIRST_PDATE:[{start.isoformat()} TO {today.isoformat()}]"
    resp = requests.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": query, "format": "json", "resultType": "core",
                "pageSize": max_items, "sort": "P_PDATE_D desc"},
        headers={"User-Agent": USER_AGENT},
        timeout=40,
    )
    resp.raise_for_status()
    out = []
    for r in resp.json().get("resultList", {}).get("result", []):
        doi = r.get("doi")
        link = f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/{r.get('source')}/{r.get('id')}"
        journal = (r.get("journalInfo") or {}).get("journal", {}).get("title") or r.get("bookOrReportDetails", {}).get("publisher") or r.get("source")
        pub_types = ", ".join((r.get("pubTypeList") or {}).get("pubType", [])[:3])
        pdate = r.get("firstPublicationDate")
        published = dt.datetime.fromisoformat(pdate).replace(tzinfo=dt.timezone.utc) if pdate else None
        out.append({
            "title": strip_html(r.get("title", "")),
            "link": link,
            "summary": f"[{journal}; {pub_types}] " + strip_html(r.get("abstractText", ""))[:800],
            "published": published,
            "source": f"{source['name']} / {journal}",
        })
    return out


def fetch_atproto(source: dict[str, Any], max_age_days: int, max_items: int) -> list[dict[str, Any]]:
    """
    Durchsucht Bluesky über die offizielle ATProto / XRPC API (app.bsky.feed.searchPosts).
    Unterstützt einzelne Suchanfragen ('query') oder Listen ('queries').
    """
    endpoint = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
    queries = source.get("queries") or ([source["query"]] if source.get("query") else [])
    if not queries:
        return []

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max_age_days)
    since_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    min_likes = int(source.get("min_likes", 0))

    seen_uris: set[str] = set()
    out: list[dict[str, Any]] = []

    for q in queries:
        params: dict[str, Any] = {
            "q": q,
            "limit": min(50, max_items * 2),
            "sort": source.get("sort", "latest"),
            "since": since_str,
        }
        if source.get("lang"):
            params["lang"] = source["lang"]

        try:
            resp = requests.get(endpoint, params=params, headers={"User-Agent": USER_AGENT}, timeout=25)
            if not resp.ok:
                log.warning("Bluesky ATProto HTTP %s für Query '%s'", resp.status_code, q)
                continue
            data = resp.json()
        except Exception as exc:
            log.warning("Bluesky ATProto Fehler für Query '%s': %s", q, exc)
            continue

        for post in data.get("posts", []):
            uri = post.get("uri", "")
            if uri in seen_uris:
                continue
            seen_uris.add(uri)

            likes = post.get("likeCount", 0)
            if likes < min_likes:
                continue

            rec = post.get("record", {})
            text = (rec.get("text") or "").strip()
            if not text:
                continue

            created_raw = rec.get("createdAt")
            published = None
            if created_raw:
                try:
                    clean_ts = re.sub(r"\.\d+", "", created_raw).replace("Z", "+00:00")
                    published = dt.datetime.fromisoformat(clean_ts)
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=dt.timezone.utc)
                except Exception:
                    pass

            author_obj = post.get("author", {})
            handle = author_obj.get("handle") or "bluesky"
            display_name = author_obj.get("displayName") or handle
            rkey = uri.split("/")[-1]
            post_link = f"https://bsky.app/profile/{handle}/post/{rkey}"

            first_line = text.splitlines()[0].strip()
            title_text = first_line[:120] if first_line else text[:120]

            summary_parts = [f"Bluesky @{handle} ({display_name}): {text}"]
            embed = rec.get("embed", {})
            if isinstance(embed, dict):
                external = embed.get("external")
                if isinstance(external, dict) and external.get("uri"):
                    ext_title = external.get("title") or "Link"
                    ext_desc = (external.get("description") or "")[:200]
                    summary_parts.append(f"[Geteilter Link: {ext_title} ({external['uri']}) – {ext_desc}]")

            out.append({
                "title": title_text,
                "link": post_link,
                "summary": " ".join(summary_parts)[:1200],
                "published": published,
                "source": f"{source.get('name', 'Bluesky')} / @{handle}",
            })
            if len(out) >= max_items:
                break
        if len(out) >= max_items:
            break

    return out


def collect_candidates(
    topic: dict[str, Any],
    settings: dict[str, Any],
    state: dict[str, Any],
    store: StateStore | None = None,
) -> tuple[list[dict[str, Any]], int]:
    max_age = int(settings.get("max_age_days", 10))
    max_items = int(settings.get("max_items_per_source", 25))
    undated_policy = settings.get("undated_policy", "keep")
    max_topic_candidates = int(settings.get("max_candidates_per_topic", 60))
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max_age)
    seen = state["seen"]
    candidates: list[dict[str, Any]] = []
    run_fps: set[str] = set()
    errors = 0
    for source in topic.get("sources", []):
        src_name = str(source.get("name") or "Quelle")
        source_key = f"{topic['id']}::{src_name}"
        if store is not None and store.is_feed_tripped(source_key):
            log.info("[%s] Quelle '%s' vorübergehend pausiert (Circuit Breaker).", topic["id"], src_name)
            continue
        try:
            stype = source.get("type", "rss").lower()
            if stype == "europepmc":
                items = fetch_europepmc(source, max_age, max_items)
            elif stype in ("atproto", "bluesky"):
                items = fetch_atproto(source, max_age, max_items)
            else:
                items = fetch_rss(source, max_age, max_items)
            if store is not None:
                store.record_feed_success(source_key, topic["id"], src_name)
        except Exception as exc:
            errors += 1
            if store is not None:
                tripped = store.record_feed_failure(source_key, topic["id"], src_name, str(exc))
                if tripped:
                    log.warning("[%s] Quelle '%s' nach wiederholten Fehlern für 12h pausiert: %s", topic["id"], src_name, exc)
                    continue
            log.warning("[%s] Quelle '%s' fehlgeschlagen: %s", topic["id"], src_name, exc)
            continue
        fresh = 0
        for it in items:
            clean_it = preprocess.filter_and_preprocess_feed_item(
                it,
                cutoff=cutoff,
                undated_policy=undated_policy,
                max_summary_length=1200,
            )
            if clean_it is None:
                continue
            fps = clean_it.get("fingerprints") or item_fingerprints(clean_it)
            if not fps or any(fp in seen or fp in run_fps for fp in fps):
                continue
            run_fps.update(fps)
            clean_it["fingerprints"] = fps
            clean_it["source"] = it.get("source") or source.get("name")
            candidates.append(clean_it)
            fresh += 1
        log.info("[%s] %-45s %3d Einträge, %3d neu", topic["id"], src_name[:45], len(items), fresh)

    # Schutz vor Batch-Spitzen: Begrenze auf die neuesten N Kandidaten pro Thema
    capped = preprocess.cap_candidates(candidates, max_total=max_topic_candidates)
    if len(capped) < len(candidates):
        log.info("[%s] Kandidaten durch Cap von %d auf %d begrenzt.", topic["id"], len(candidates), len(capped))
    return capped, errors


def fetch_article_fulltext(url: str, max_chars: int = 2400) -> str:
    """Extrahiert optional den Artikel-Haupttext via trafilatura (überspringt Redirect/Social-Wrapper)."""
    if trafilatura is None or not url:
        return ""
    lower_u = url.lower()
    if any(skip in lower_u for skip in ("news.google.com", "bsky.app", "youtube.com", "youtu.be")):
        return ""
    try:
        raw_bytes = _http_get_bytes(url, timeout=12)
        html_str = raw_bytes.decode("utf-8", errors="replace")
        extracted = trafilatura.extract(html_str, include_comments=False, include_tables=False) or ""
        return extracted.strip()[:max_chars]
    except Exception as exc:
        log.debug("Volltext-Extraktion Hinweis (%s): %s", url[:60], exc)
        return ""


def enrich_candidates_fulltext(
    candidates: list[dict[str, Any]],
    settings: dict[str, Any],
    max_enrich: int = 8,
) -> int:
    """Reichert Kandidaten mit kurzer Summary vor dem LLM-Aufruf um echten Artikeltext an."""
    if trafilatura is None or not settings.get("enrich_fulltext", True):
        return 0
    enriched = 0
    for c in candidates:
        if enriched >= max_enrich:
            break
        cur_sum = (c.get("summary") or "").strip()
        if len(cur_sum) >= 260:
            continue
        ft = fetch_article_fulltext(c.get("link", ""))
        if ft and len(ft) > len(cur_sum) + 80:
            c["summary"] = ft[:1500]
            c["fulltext_enriched"] = True
            enriched += 1
    return enriched


# --------------------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------------------

def get_gemini_api_key() -> tuple[str, str]:
    """
    Ermittelt den Gemini API-Key für den RSS-Monitor.
    Bevorzugt GEMINI_FREE_API_KEY (für öffentliche Feeds, 100% kostenlos im Free Tier),
    fällt sonst auf GEMINI_API_KEY bzw. GOOGLE_API_KEY zurück.
    Gibt (key, key_name) zurück.
    """
    free_key = os.environ.get("GEMINI_FREE_API_KEY", "").strip()
    if free_key:
        return free_key, "GEMINI_FREE_API_KEY"
    main_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    if main_key:
        return main_key, "GEMINI_API_KEY"
    return "", ""


def llm_json(prompt: str, system: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Gemini (bevorzugt Free Tier) mit Task-Tiering ('rss_summary') und Fallback auf Ollama."""
    key, key_name = get_gemini_api_key()
    provider = "gemini" if key else "ollama"

    # Modell und Thinking-Level auflösen (Standard: gemini-3.5-flash-lite, minimal)
    model, thinking_level = llm_core.resolve_task_model("rss_summary", cfg)

    if provider == "gemini":
        try:
            from google import genai
            client = genai.Client(api_key=key)
            resp = llm_core.execute_gemini_call(
                prompt=prompt,
                client=client,
                task="rss_summary",
                model=model,
                thinking_level=thinking_level,
                system_instruction=system,
                response_mime_type="application/json",
            )
            return json.loads(resp.text or "{}")
        except Exception as exc:
            log.warning("Gemini-Fehler mit %s (%s).", key_name, exc)
            main_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
            if key_name == "GEMINI_FREE_API_KEY" and main_key and main_key != key:
                try:
                    log.info("Versuche Fallback auf GEMINI_API_KEY...")
                    client = genai.Client(api_key=main_key)
                    resp = llm_core.execute_gemini_call(
                        prompt=prompt,
                        client=client,
                        task="rss_summary",
                        model=model,
                        thinking_level=thinking_level,
                        system_instruction=system,
                        response_mime_type="application/json",
                    )
                    return json.loads(resp.text or "{}")
                except Exception as exc2:
                    log.warning("Gemini-Fehler mit GEMINI_API_KEY (%s) -> Fallback auf Ollama.", exc2)
            else:
                log.info("Fallback auf Ollama...")

    return llm_core.call_ollama_json_safe(
        prompt=prompt,
        system=system,
        ollama_cfg=cfg.get("ollama", {}),
    )


SYSTEM_PROMPT = (
    "Du bist ein sorgfältiger, skeptischer Fachredakteur, der für Stefan Kollmeier einen "
    "persönlichen Wissensspeicher pflegt. Du bewertest neue Meldungen streng nach Relevanz, "
    "Evidenzqualität und Neuheit gegenüber dem bereits bekannten Wissen. Du erfindest nichts: "
    "Alle Angaben stammen ausschließlich aus den gelieferten Kandidaten. Schreibe auf Deutsch. "
    "Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt."
)


def build_prompt(
    topic: dict[str, Any],
    known: list[dict[str, Any]],
    batch: list[dict[str, Any]],
    ignored_examples: list[str] | None = None,
) -> str:
    known_lines = "\n".join(
        f"- [{k.get('key')}] ({k.get('date', '')}, {k.get('category', '')}) {k.get('title', '')}"
        for k in known
    ) or "(noch leer – erster Lauf)"
    cand_lines = []
    for i, c in enumerate(batch):
        date = c["published"].date().isoformat() if c.get("published") else "?"
        cand_lines.append(f"[{i}] {c['source']} | {date} | {c['title']}\n    {c.get('summary', '')[:900]}")
    cats = ", ".join(topic.get("categories", ["Sonstiges"]))
    today = dt.date.today().isoformat()
    neg_block = ""
    if ignored_examples:
        neg_block = (
            "\nVOM NUTZER ALS IRRELEVANT MARKIERT (ähnliche Meldungen bitte konsequent verwerfen):\n"
            + "\n".join(f"- {ex}" for ex in ignored_examples[:8])
            + "\n"
        )
    return f"""Heute ist {today}.

THEMA: {topic['name']}
FOKUS / KRITERIEN:
{topic['focus']}

ERLAUBTE KATEGORIEN: {cats}
{neg_block}
BEREITS BEKANNTES WISSEN (schon gemeldet – NICHT erneut melden):
{known_lines}

NEUE KANDIDATEN:
{chr(10).join(cand_lines)}

AUFGABE:
1. Wähle nur Kandidaten, die klar zum Fokus passen und substanziell sind.
2. Fasse Kandidaten, die über dieselbe Nachricht/Studie/Aktion berichten, zu EINEM Fund zusammen (alle IDs angeben).
3. Prüfe jeden Fund gegen das bekannte Wissen. Melde ihn nur, wenn er inhaltlich neu ist. Eine echte
   Weiterentwicklung eines bekannten Eintrags (z. B. neue Studienphase, Ergebnisse, Zulassung, Verlängerung/Änderung
   eines Angebots) darf gemeldet werden – dann 'update_of' auf dessen Schlüssel setzen und im Text sagen, was neu ist.
4. relevance: "Hoch" = bedeutsam/praxisrelevant/großer Fortschritt bzw. sehr attraktives Angebot; "Mittel" = solide und
   interessant; "Niedrig" = nur am Rande relevant.
5. key: kurzer, eindeutiger, sprechender Schlüssel in Kleinbuchstaben mit Bindestrichen (z. B. "tavapadon-fda-zulassung").

Antworte als JSON:
{{
  "findings": [
    {{
      "candidate_ids": [0, 3],
      "title": "prägnanter deutscher Titel (max. 110 Zeichen)",
      "summary": "2–4 Sätze: Was ist neu? Evidenz (Studientyp, Teilnehmerzahl, Phase) bzw. Konditionen/Zielgruppe. Warum relevant?",
      "category": "eine der erlaubten Kategorien",
      "relevance": "Hoch|Mittel|Niedrig",
      "key": "eindeutiger-schluessel",
      "update_of": null,
      "discount": null,
      "valid_until": null
    }}
  ],
  "overview": "1–2 Sätze Einordnung der neuen Funde (leer, wenn keine)"
}}
Wenn nichts Neues und Relevantes dabei ist: {{"findings": [], "overview": ""}}"""


def jev_prefilter_candidates(
    topic: dict[str, Any],
    candidates: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """
    Filtert und taggt RSS-Kandidaten vorab mit TypeSafe AI (Jev System One).

    - Bewertet Relevanz per Noul (oder Score)
    - Taggt Kategorie per Choice
    - Nur Kandidaten mit Relevanz >= Threshold gehen an Gemini weiter
    - Verworfene Kandidaten werden zurückgegeben, damit sie als 'gesehen' markiert werden (Kostenschutz!)
    - Bei Jev-Fehler: Fallback gemäß JEV_FAILURE_MODE ('gemini' | 'abort')
    """
    client = jev_client.JevClient()
    stats = {"in": len(candidates), "passed": 0, "dropped": 0, "errors": 0}

    # Falls Jev nicht konfiguriert oder deaktiviert ist: alle an Gemini durchreichen
    if not client.is_configured or not settings.get("jev_enabled", True):
        log.info("[%s] Jev-Vorfilter inaktiv (kein TYPESAFE_API_KEY oder jev_enabled=false) -> alle %d Kandidaten an LLM.",
                 topic["id"], len(candidates))
        return candidates, [], stats

    try:
        threshold = float(os.environ.get("JEV_RELEVANCE_THRESHOLD", settings.get("jev_relevance_threshold", 0.7)))
    except (ValueError, TypeError):
        threshold = 0.7

    try:
        min_cat_conf = float(os.environ.get("JEV_CATEGORY_MIN_CONFIDENCE", settings.get("jev_category_min_confidence", 0.4)))
    except (ValueError, TypeError):
        min_cat_conf = 0.4

    failure_mode = os.environ.get("JEV_FAILURE_MODE", settings.get("jev_failure_mode", "gemini")).strip().lower()

    passed: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    # Kategorien für Jev-Choice vorbereiten
    allowed_cats = topic.get("categories", ["Sonstiges"])
    options: dict[str, str | None] = {cat: f"Kategorie {cat}" for cat in allowed_cats}
    if "Sonstiges" not in options:
        options["Sonstiges"] = "Keine der spezifischen Kategorien"

    focus_desc = topic.get("focus", "")[:300]

    for item in candidates:
        state_text = f"Titel: {item.get('title', '')}\nQuelle: {item.get('source', '')}\nText: {item.get('summary', '')[:600]}"
        questions = {
            "is_relevant": {
                "type": "noul",
                "instructions": f"Passt diese Meldung substanziell zum Thema '{topic['name']}' und den folgenden Kriterien?\nKriterien: {focus_desc}",
                "criteria": {
                    "true": "Die Meldung ist eine relevante, substanzielle Neuerung/Studie/Aktion zum Thema.",
                    "false": "Die Meldung ist irrelevant, reines Marketing, Randnotiz oder passt nicht zum Fokus.",
                },
            },
            "category": {
                "type": "choice",
                "instructions": f"Welche Kategorie beschreibt die Meldung am besten?",
                "criteria": options,
            },
        }

        try:
            res = client.system_one(state=state_text, questions=questions)
            answers = res.get("answers", {})

            # 1. Noul Relevanz
            rel_prob = float(answers.get("is_relevant", {}).get("noul", 0.0))
            item["jev_relevance"] = rel_prob

            # 2. Choice Kategorie
            cat_ans = answers.get("category", {})
            cat_choice = str(cat_ans.get("choice") or "Sonstiges")
            cat_conf = float(cat_ans.get("confidence", 0.0))
            item["jev_category"] = cat_choice if cat_conf >= min_cat_conf else "Sonstiges"
            item["jev_category_confidence"] = cat_conf

            if rel_prob >= threshold:
                passed.append(item)
                stats["passed"] += 1
            else:
                dropped.append(item)
                stats["dropped"] += 1

        except Exception as exc:
            stats["errors"] += 1
            log.warning("[%s] Jev-Vorfilterfehler bei '%s': %s", topic["id"], item.get("title", "")[:40], exc)
            if failure_mode == "abort":
                log.info("[%s] JEV_FAILURE_MODE=abort -> Breche Vorfilter ab.", topic["id"])
                # Bisherige nicht verwerfen, aber Abbrechen
                return passed, dropped, stats
            # Fallback 'gemini': Kandidat durchlassen
            passed.append(item)
            stats["passed"] += 1

    log.info(
        "jev_filter topic=%s in=%d passed=%d dropped=%d errors=%d threshold=%.2f model=%s",
        topic["id"], stats["in"], stats["passed"], stats["dropped"], stats["errors"], threshold, client.model,
    )
    return passed, dropped, stats


def evaluate_topic(
    topic: dict[str, Any],
    candidates: list[dict[str, Any]],
    known: list[dict[str, Any]],
    settings: dict[str, Any],
    cfg: dict[str, Any],
    store: StateStore | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Gibt (Funde, Overviews, erfolgreich bewertete Kandidaten) zurück."""
    batch_size = int(settings.get("max_candidates_per_llm_call", 35))
    min_rel = RELEVANCE_ORDER.get(topic.get("min_relevance") or settings.get("min_relevance", "Mittel"), 1)
    allowed_cats = set(topic.get("categories", []))
    known_ctx = list(known)
    findings: list[dict[str, Any]] = []
    overviews: list[str] = []
    evaluated: list[dict[str, Any]] = []
    known_keys = {k.get("key") for k in known_ctx}
    ignored_examples = store.get_recent_negative_feedback(topic["id"]) if store is not None else []

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        try:
            result = llm_json(build_prompt(topic, known_ctx, batch, ignored_examples=ignored_examples), SYSTEM_PROMPT, cfg)
        except Exception as exc:
            log.error("[%s] LLM-Bewertung fehlgeschlagen (%s) – Batch wird beim nächsten Lauf erneut geprüft.", topic["id"], exc)
            continue
        evaluated.extend(batch)
        if result.get("overview"):
            overviews.append(str(result["overview"]).strip())
        for f in result.get("findings", []) or []:
            ids = [i for i in f.get("candidate_ids", []) if isinstance(i, int) and 0 <= i < len(batch)]
            if not ids or not f.get("title"):
                continue
            rel = f.get("relevance") if f.get("relevance") in RELEVANCE_ORDER else "Mittel"
            if RELEVANCE_ORDER[rel] < min_rel:
                continue
            key = re.sub(r"[^a-z0-9-]+", "-", str(f.get("key") or normalize_title(f["title"])).lower()).strip("-")[:80]
            if key in known_keys:
                log.info("[%s] Überspringe bekannten Schlüssel %s", topic["id"], key)
                continue
            srcs = [batch[i] for i in ids]
            primary = next((s for s in srcs if "news.google.com" not in s["link"]), srcs[0])
            dates = [s["published"] for s in srcs if s.get("published")]

            # Kategorie: Bevorzuge deterministisches Jev-Tagging, falls vorhanden
            jev_cat = primary.get("jev_category")
            if jev_cat and jev_cat in allowed_cats:
                cat = jev_cat
            else:
                cat = f.get("category") if f.get("category") in allowed_cats else "Sonstiges"

            finding = {
                "topic_id": topic["id"],
                "topic_name": topic["name"],
                "title": str(f["title"]).strip()[:180],
                "summary": str(f.get("summary") or "").strip(),
                "category": cat,
                "relevance": rel,
                "key": key,
                "update_of": f.get("update_of"),
                "discount": f.get("discount"),
                "valid_until": f.get("valid_until") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(f.get("valid_until") or "")) else None,
                "link": primary["link"],
                "sources": [{"name": s["source"], "link": s["link"]} for s in srcs],
                "published": min(dates).date().isoformat() if dates else None,
            }
            findings.append(finding)
            known_keys.add(key)
            known_ctx.insert(0, {"key": key, "title": finding["title"], "date": dt.date.today().isoformat(), "category": cat})

    findings.sort(key=lambda x: -RELEVANCE_ORDER[x["relevance"]])
    return findings, overviews, evaluated


# --------------------------------------------------------------------------------------
# Notion
# --------------------------------------------------------------------------------------

def rt(text: str, link: str | None = None, bold: bool = False) -> dict[str, Any]:
    obj: dict[str, Any] = {"type": "text", "text": {"content": text[:1990]}}
    if link:
        obj["text"]["link"] = {"url": link}
    if bold:
        obj["annotations"] = {"bold": True}
    return obj


def load_known_from_notion(notion: assistant.NotionClient, db_id: str, topic_name: str, limit: int) -> list[dict[str, Any]]:
    known: list[dict[str, Any]] = []
    cursor = None
    while len(known) < limit:
        payload: dict[str, Any] = {
            "filter": {"property": "Thema", "select": {"equals": topic_name}},
            "sorts": [{"timestamp": "created_time", "direction": "descending"}],
            "page_size": min(100, limit - len(known)),
        }
        if cursor:
            payload["start_cursor"] = cursor
        res = notion._request("POST", f"/databases/{db_id}/query", payload)
        for page in res.get("results", []):
            p = page.get("properties", {})
            title = "".join(t.get("plain_text", "") for t in p.get("Titel", {}).get("title", []))
            key = "".join(t.get("plain_text", "") for t in p.get("Schlüssel", {}).get("rich_text", []))
            date = ((p.get("Erfasst", {}) or {}).get("date") or {}).get("start", "")
            cat = ((p.get("Kategorie", {}) or {}).get("select") or {}).get("name", "")
            known.append({"key": key or normalize_title(title)[:60], "title": title, "date": date, "category": cat})
        if not res.get("has_more"):
            break
        cursor = res.get("next_cursor")
    return known


def create_knowledge_entry(notion: assistant.NotionClient, db_id: str, f: dict[str, Any], emoji: str) -> str:
    now = dt.datetime.now().astimezone()
    summary = f["summary"]
    if f.get("update_of"):
        summary = f"Update zu „{f['update_of']}“: " + summary
    props: dict[str, Any] = {
        "Titel": {"title": [rt(f["title"])]},
        "Thema": {"select": {"name": f["topic_name"]}},
        "Kategorie": {"select": {"name": f["category"]}},
        "Relevanz": {"select": {"name": f["relevance"]}},
        "Zusammenfassung": {"rich_text": [rt(summary)]},
        "Link": {"url": f["link"]},
        "Quelle": {"rich_text": [rt(", ".join(dict.fromkeys(s["name"] for s in f["sources"])))]},
        "Erfasst": {"date": {"start": now.isoformat(timespec="minutes")}},
        "Schlüssel": {"rich_text": [rt(f["key"])]},
    }
    if f.get("published"):
        props["Veröffentlicht"] = {"date": {"start": f["published"]}}
    if f.get("discount"):
        props["Rabatt"] = {"rich_text": [rt(str(f["discount"]))]}
    if f.get("valid_until"):
        props["Gültig bis"] = {"date": {"start": f["valid_until"]}}

    children: list[dict[str, Any]] = [
        {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [rt(summary)]}},
        {"object": "block", "type": "heading_3", "heading_3": {"rich_text": [rt("Quellen")]}},
    ]
    for s in f["sources"][:10]:
        children.append({"object": "block", "type": "bulleted_list_item",
                         "bulleted_list_item": {"rich_text": [rt(s["name"], link=s["link"])]}})
    res = notion._request("POST", "/pages", {
        "parent": {"database_id": db_id},
        "icon": {"type": "emoji", "emoji": emoji},
        "properties": props,
        "children": children,
    })
    return res["id"]


def create_digest_page(
    notion: assistant.NotionClient,
    documents_db_id: str,
    knowledge_db_id: str,
    results: list[dict[str, Any]],
) -> str:
    now = dt.datetime.now()
    title = f"RSS-Digest {now:%d.%m.%Y} – {now:%H:%M}"
    total = sum(len(r["findings"]) for r in results)
    blocks: list[dict[str, Any]] = [
        {"object": "block", "type": "callout", "callout": {
            "icon": {"type": "emoji", "emoji": "📡"},
            "rich_text": [rt(f"{total} neue Funde. Alle Einträge im "),
                          {"type": "mention", "mention": {"database": {"id": knowledge_db_id}}},
                          rt(".")],
        }},
    ]
    for r in results:
        if not r["findings"]:
            continue
        topic = r["topic"]
        blocks.append({"object": "block", "type": "heading_2", "heading_2": {
            "rich_text": [rt(f"{topic.get('emoji', '')} {topic['name']} ({len(r['findings'])})".strip())]}})
        if r.get("overview"):
            blocks.append({"object": "block", "type": "quote", "quote": {"rich_text": [rt(r["overview"])]}})
        for f in r["findings"]:
            badge = {"Hoch": "🔴", "Mittel": "🟡", "Niedrig": "⚪"}.get(f["relevance"], "")
            extra = f" · {f['discount']}" if f.get("discount") else ""
            extra += f" · bis {f['valid_until']}" if f.get("valid_until") else ""
            parts = [rt(f"{badge} "), rt(f["title"], link=f["link"], bold=True),
                     rt(f" ({f['category']}{extra})\n{f['summary']}")]
            if f.get("notion_page_id"):
                parts += [rt("\n→ "), {"type": "mention", "mention": {"page": {"id": f["notion_page_id"]}}}]
            blocks.append({"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": parts}})

    res = notion._request("POST", "/pages", {
        "parent": {"database_id": documents_db_id},
        "icon": {"type": "emoji", "emoji": "📡"},
        "properties": {
            "Name": {"title": [rt(title)]},
            "Category": {"select": {"name": "Research"}},
        },
        "children": blocks[:100],
    })
    return res.get("url", res["id"])


# --------------------------------------------------------------------------------------
# Benachrichtigung
# --------------------------------------------------------------------------------------

def send_telegram_msg(
    token: str,
    chat_id: str,
    text: str,
    parse_mode: str | None = "HTML",
    reply_markup: dict[str, Any] | None = None,
) -> bool:
    """Sendet Nachricht an Telegram über die einheitliche Transport-Schicht."""
    return telegram_ui.send_message(
        token=token,
        chat_id=chat_id,
        text=text,
        parse_mode=parse_mode,
        reply_markup=reply_markup,
    )


def notify(
    results: list[dict[str, Any]],
    digest_url: str | None,
    settings: dict[str, Any],
    store: StateStore | None = None,
) -> None:
    total = sum(len(r["findings"]) for r in results)
    if total == 0:
        return

    esc = telegram_ui.esc
    lines = [f"📡 <b>RSS-Monitor: {total} neue Funde</b>"]
    action_buttons: list[list[tuple[str, str]]] = []
    btn_idx = 0

    for r in results:
        if not r["findings"]:
            continue
        emoji = r["topic"].get("emoji", "")
        lines.append(f"\n{esc(emoji)} <b>{esc(r['topic']['name'])}</b>")
        for f in r["findings"][:6]:
            badge = {"Hoch": "🔴", "Mittel": "🟡", "Niedrig": "⚪"}.get(f["relevance"], "•")
            extra = f" ({esc(f['discount'])})" if f.get("discount") else ""
            title_html = esc(f["title"])
            link = f.get("link") or ""
            title_part = f'<a href="{esc(link)}">{title_html}</a>' if link else title_html
            summary_short = esc((f.get("summary") or "")[:180])
            btn_idx += 1
            lines.append(f"{badge} <b>#{btn_idx}</b> {title_part}{extra}")
            if summary_short:
                lines.append(f"   <i>{summary_short}</i>")

            if store is not None and btn_idx <= 3:
                raw_key = f"{f.get('topic_id', '')}|{f.get('key', '')}|{f.get('title', '')}"
                tok = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:10]
                store.put_callback_payload(
                    tok,
                    "rss_finding",
                    {
                        "topic_id": f.get("topic_id", ""),
                        "topic_name": f.get("topic_name", ""),
                        "title": f.get("title", ""),
                        "summary": f.get("summary", ""),
                        "link": link,
                        "category": f.get("category", ""),
                    },
                )
                action_buttons.append([
                    (f"📌 #{btn_idx} Task", f"rss_task:{tok}"),
                    (f"🎬 #{btn_idx} Idee", f"rss_idea:{tok}"),
                    (f"👎 #{btn_idx}", f"rss_bad:{tok}"),
                ])
        if len(r["findings"]) > 6:
            lines.append(f"   … und {len(r['findings']) - 6} weitere")

    if digest_url:
        lines.append(f'\n📄 <a href="{esc(digest_url)}">Vollständigen Digest in Notion öffnen</a>')
    text = "\n".join(lines)
    markup = telegram_ui.build_inline_keyboard(action_buttons) if action_buttons else None

    if settings.get("notify_telegram", True):
        rss_token = os.environ.get("TELEGRAM_RSSBOT_TOKEN", "").strip()
        main_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

        # Chat-ID ermitteln (RSS-spezifisch oder Haupt-Chat-ID)
        chat_id = os.environ.get("TELEGRAM_RSS_CHAT_ID", "").strip() or os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not chat_id and store is not None:
            chat_id = str(store.get_kv("last_rss_chat_id", "") or store.get_kv("last_telegram_chat_id", "") or "")
        if not chat_id and STATE_PATH.exists():
            rss_st = safe_read_json(STATE_PATH, {})
            if isinstance(rss_st, dict):
                chat_id = str(rss_st.get("last_rss_chat_id") or "")
        if not chat_id:
            loc_st = safe_read_json(BASE_DIR / "local_state.json", {})
            if isinstance(loc_st, dict):
                chat_id = str(loc_st.get("last_telegram_chat_id") or "")

        sent = False
        # 1. Bevorzuge den dedizierten RSS-Bot
        if rss_token and chat_id:
            if send_telegram_msg(rss_token, chat_id, text, parse_mode="HTML", reply_markup=markup):
                sent = True
                log.info("Telegram-Benachrichtigung über RSS-Bot gesendet.")
            else:
                log.warning("Versand über RSS-Bot fehlgeschlagen (Nutzer muss @RSS8472Bot evtl. erst einmal /start senden).")

        # 2. Fallback auf den Haupt-Bot, falls kein RSS-Bot oder ungesendet
        if not sent and main_token and chat_id:
            send_telegram_msg(main_token, chat_id, text, parse_mode="HTML", reply_markup=markup)
            log.info("Telegram-Benachrichtigung über Haupt-Bot gesendet (Fallback).")
        elif not (rss_token or main_token) or not chat_id:
            log.info("Telegram nicht konfiguriert (Token oder Chat-ID fehlt).")

    if settings.get("notify_windows", True):
        summary = ", ".join(f"{r['topic']['name']}: {len(r['findings'])}" for r in results if r["findings"])
        assistant.show_windows_notification("📡 Neue Funde im Wissensspeicher", summary)


# --------------------------------------------------------------------------------------
# Hauptablauf & Wochen-Synthese
# --------------------------------------------------------------------------------------

def prune_seen(state: dict[str, Any], retention_days: int) -> None:
    cutoff = (dt.datetime.now() - dt.timedelta(days=retention_days)).isoformat()
    state["seen"] = {k: v for k, v in state["seen"].items() if v >= cutoff}


def generate_weekly_rss_synthesis(
    cfg: dict[str, Any] | None = None,
    store: StateStore | None = None,
    notion: Any = None,
    days: int = 7,
) -> tuple[str, str | None]:
    """Erstellt eine themenübergreifende Wochen-Synthese aus den gespeicherten RSS-Funden."""
    cfg = cfg or assistant.load_config()
    store = store or get_store()
    items = store.get_recent_knowledge(days=days, limit=60)
    if not items:
        return "📡 <b>Wochen-Synthese:</b> In den letzten 7 Tagen wurden keine neuen RSS-Funde gespeichert.", None

    bullet_input = "\n".join(
        f"- [{it.get('topic_id')}] ({it.get('relevance')}) {it.get('title')}: {(it.get('summary') or '')[:220]}"
        for it in items
    )
    prompt = f"""Analysiere die folgenden {len(items)} RSS- und Studienfunde der letzten {days} Tage für Stefan Kollmeier:

{bullet_input}

Erstelle eine prägnante Wochen-Synthese als JSON:
{{
  "headline": "Kurze Überschrift der Woche",
  "trends": ["3-5 wichtigste Erkenntnisse oder Trends über alle Themen hinweg"],
  "content_ideas": ["1-2 konkrete Ideen für Vorlesung, YouTube ('Life on Pause') oder Recherche"]
}}"""
    system = "Du bist Stefans wissenschaftlicher Forschungs-Redakteur. Antworte AUSSCHLIESSLICH mit validem JSON."
    try:
        res = llm_json(prompt, system, cfg)
    except Exception:
        res = {
            "headline": f"Wochen-Synthese ({len(items)} Funde)",
            "trends": [f"[{it.get('topic_id')}] {it.get('title')}" for it in items[:5]],
            "content_ideas": [],
        }

    esc = telegram_ui.esc
    lines = [f"🧠 <b>{esc(res.get('headline') or 'Wochen-Synthese')}</b> ({len(items)} Funde)\n"]
    trends = res.get("trends") or []
    if trends:
        lines.append("📌 <b>Wichtigste Trends & Erkenntnisse:</b>")
        for tr in trends[:6]:
            lines.append(f"• {esc(tr)}")
    ideas = res.get("content_ideas") or []
    if ideas:
        lines.append("\n🎬 <b>Impulse für Lehre &amp; Content:</b>")
        for id_item in ideas[:3]:
            lines.append(f"• {esc(id_item)}")

    doc_url: str | None = None
    if notion is not None:
        md_lines = [
            f"## {res.get('headline') or 'Wochen-Synthese'}",
            *[f"- {tr}" for tr in trends],
            "## Impulse für Lehre & Content",
            *[f"- {id_item}" for id_item in ideas],
        ]
        try:
            doc_url = notion.create_document_page(
                cfg["notion"]["documents_database_id"],
                title=f"RSS Wochen-Synthese {dt.date.today():%d.%m.%Y}",
                category="Research",
                markdown_lines=md_lines,
                emoji="🧠",
            )
            lines.append(f'\n📄 <a href="{esc(doc_url)}">Synthese in Notion öffnen</a>')
        except Exception as exc:
            log.warning("Synthese Notion-Dokument fehlgeschlagen: %s", exc)

    return "\n".join(lines), doc_url


def run(args: argparse.Namespace) -> int:
    cfg = assistant.load_config()
    feeds_cfg = load_feeds_config()
    settings = feeds_cfg.get("settings", {})
    ncfg = feeds_cfg["notion"]
    state = load_state()
    state.setdefault("seen", {})
    state.setdefault("knowledge", {})
    state.setdefault("runs", [])
    store = get_store()

    token = os.environ.get("NOTION_TOKEN", "").strip()
    notion = assistant.NotionClient(token) if token and not args.no_notion else None
    if notion is None and not args.dry_run:
        log.warning("Kein NOTION_TOKEN – Funde werden nur lokal gespeichert.")

    topics = [t for t in feeds_cfg["topics"] if t.get("enabled", True)]
    if args.topic:
        topics = [t for t in feeds_cfg["topics"] if t["id"] in args.topic]
        if not topics:
            log.error("Unbekanntes Thema: %s", args.topic)
            return 2

    key, key_name = get_gemini_api_key()
    if key:
        log.info("LLM-Provider: Gemini (%s) – Modell: %s", key_name, cfg.get("llm", {}).get("model", "gemini-3.8-flash"))
    else:
        log.info("LLM-Provider: Lokales Ollama (kein Gemini-Key gesetzt)")

    results: list[dict[str, Any]] = []
    now_iso = dt.datetime.now().isoformat(timespec="seconds")
    for topic in topics:
        log.info("=== %s %s ===", topic.get("emoji", ""), topic["name"])
        candidates, _ = collect_candidates(topic, settings, state, store=store)
        if not candidates:
            log.info("[%s] Keine neuen Kandidaten.", topic["id"])
            results.append({"topic": topic, "findings": [], "overview": ""})
            continue

        log.info("[%s] Starte Vorfilterung für %d Kandidaten...", topic["id"], len(candidates))
        passed_cands, dropped_cands, filter_stats = jev_prefilter_candidates(topic, candidates, settings)

        # Verworfene Kandidaten sofort als gesehen markieren (Schutz vor doppelter Bearbeitung)
        if not args.dry_run and dropped_cands:
            for dc in dropped_cands:
                for fp in dc.get("fingerprints", []):
                    state["seen"][fp] = now_iso
            save_state(state)

        if not passed_cands:
            log.info("[%s] Alle %d Kandidaten durch Vorfilter/Schwellenwert verworfen.", topic["id"], len(candidates))
            results.append({"topic": topic, "findings": [], "overview": ""})
            continue

        enriched_cnt = enrich_candidates_fulltext(passed_cands, settings)
        if enriched_cnt:
            log.info("[%s] %d Artikel mit Volltext (trafilatura) angereichert.", topic["id"], enriched_cnt)

        known: list[dict[str, Any]] = []
        limit = int(settings.get("known_context_items", 150))
        if notion:
            try:
                known = load_known_from_notion(notion, ncfg["knowledge_database_id"], topic["name"], limit)
            except Exception as exc:
                log.warning("[%s] Wissensstand aus Notion nicht ladbar (%s) – nutze lokalen Cache.", topic["id"], exc)
                known = state["knowledge"].get(topic["id"], [])[:limit]
        else:
            known = state["knowledge"].get(topic["id"], [])[:limit]

        # Schätzung der Token-Einsparung durch Jev-Vorfilterung (ca. 400 Tokens pro verworfenem Kandidat in Prompt/Thinking)
        saved_est = filter_stats.get("dropped", 0) * 400
        if saved_est > 0:
            log.info("savings_est topic=%s dropped_tokens≈%d dropped_items=%d", topic["id"], saved_est, filter_stats.get("dropped", 0))

        log.info("[%s] Bewerte %d gefilterte Kandidaten gegen %d bekannte Einträge ...", topic["id"], len(passed_cands), len(known))
        findings, overviews, evaluated = evaluate_topic(topic, passed_cands, known, settings, cfg, store=store)
        log.info("[%s] %d neue relevante Funde.", topic["id"], len(findings))

        written: list[dict[str, Any]] = []
        for f in findings:
            log.info("  + [%s] %s", f["relevance"], f["title"])
            if args.dry_run:
                written.append(f)
                continue
            if notion:
                try:
                    f["notion_page_id"] = create_knowledge_entry(notion, ncfg["knowledge_database_id"], f, topic.get("emoji", "📌"))
                except Exception as exc:
                    log.error("  Notion-Eintrag fehlgeschlagen: %s", exc)
                    continue
            written.append(f)
            state["knowledge"].setdefault(topic["id"], []).insert(0, {
                "key": f["key"], "title": f["title"], "date": dt.date.today().isoformat(),
                "category": f["category"], "link": f["link"],
            })
            store.add_knowledge_record(f)

        if not args.dry_run:
            # Erfolgreich von Gemini bewertete Kandidaten als gesehen markieren
            for c in evaluated:
                for fp in c.get("fingerprints", []):
                    state["seen"][fp] = now_iso
            state["knowledge"][topic["id"]] = state["knowledge"].get(topic["id"], [])[:500]
            save_state(state)

        results.append({"topic": topic, "findings": written, "overview": " ".join(overviews)})

    total = sum(len(r["findings"]) for r in results)
    digest_url = None
    if total and not args.dry_run and notion and settings.get("create_digest_page", True):
        try:
            digest_url = create_digest_page(notion, cfg["notion"]["documents_database_id"], ncfg["knowledge_database_id"], results)
            log.info("Digest-Seite erstellt: %s", digest_url)
        except Exception as exc:
            log.error("Digest-Seite fehlgeschlagen: %s", exc)

    if not args.dry_run:
        prune_seen(state, int(settings.get("seen_retention_days", 180)))
        state["runs"] = ([{"at": now_iso, "new": total, "digest": digest_url}] + state["runs"])[:60]
        save_state(state)
        if not args.no_notify:
            notify(results, digest_url, settings, store=store)

    log.info("Fertig: %d neue Funde.", total)
    return 0


# --------------------------------------------------------------------------------------
# Themen-Verwaltung (CLI & API)
# --------------------------------------------------------------------------------------

def save_feeds_config(cfg: dict[str, Any]) -> None:
    atomic_write_json(FEEDS_PATH, cfg, indent=2)


def slugify_topic_id(name: str) -> str:
    s = name.lower()
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:30] or "thema"


def find_topic(feeds_cfg: dict[str, Any], topic_id: str) -> tuple[int, dict[str, Any] | None]:
    tid_lower = topic_id.strip().lower()
    for idx, t in enumerate(feeds_cfg.get("topics", [])):
        if t["id"].lower() == tid_lower or t["name"].lower() == tid_lower:
            return idx, t
    return -1, None


def cmd_topic_list(verbose: bool = False) -> None:
    feeds_cfg = load_feeds_config()
    topics = feeds_cfg.get("topics", [])
    print(f"\nVerfügbare RSS-Themengebiete ({len(topics)}):")
    print("=" * 70)
    for i, t in enumerate(topics, 1):
        status = "🟢 aktiv" if t.get("enabled", True) else "⚪ inaktiv"
        print(f"[{i}] {t.get('emoji', '📌')} {t['name']} (ID: {t['id']}) – {status}")
        sources = t.get("sources", [])
        src_names = ", ".join(s.get("name", "") for s in sources[:4])
        if len(sources) > 4:
            src_names += f" (+{len(sources)-4} weitere)"
        print(f"    Quellen ({len(sources)}): {src_names or '(keine)'}")
        focus_preview = (t.get("focus") or "").replace("\n", " ").strip()
        if len(focus_preview) > 130:
            focus_preview = focus_preview[:127] + "..."
        if focus_preview:
            print(f"    Fokus: {focus_preview}")
        if verbose:
            print(f"    Mindest-Relevanz: {t.get('min_relevance', 'Mittel')}")
            for sidx, s in enumerate(sources, 1):
                u = s.get("url") or s.get("query") or ""
                print(f"      [{sidx}] {s.get('type', 'rss').upper():9s} {s.get('name', ''):30s} {u[:65]}")
        print()
    print("Tipp: Themen aktivieren/deaktivieren mit:  python rss_monitor.py topic enable <id>")
    print("      Neues Thema anlegen mit:            python rss_monitor.py topic add \"Themenname\" --auto")
    print("      Detailansicht aller Quellen:        python rss_monitor.py topic list -v\n")


def cmd_topic_set_enabled(topic_id: str, enabled: bool) -> bool:
    feeds_cfg = load_feeds_config()
    idx, t = find_topic(feeds_cfg, topic_id)
    if t is None:
        print(f"❌ Thema '{topic_id}' wurde nicht gefunden.")
        return False
    t["enabled"] = enabled
    save_feeds_config(feeds_cfg)
    status_str = "🟢 aktiviert" if enabled else "⚪ deaktiviert"
    print(f"Thema '{t['name']}' (ID: {t['id']}) ist jetzt {status_str}.")
    return True


def cmd_topic_remove(topic_id: str, confirm: bool = False) -> bool:
    feeds_cfg = load_feeds_config()
    idx, t = find_topic(feeds_cfg, topic_id)
    if t is None:
        print(f"❌ Thema '{topic_id}' wurde nicht gefunden.")
        return False
    if not confirm:
        ans = input(f"Möchtest du das Thema '{t['name']}' (ID: {t['id']}) wirklich löschen? [j/N]: ").strip().lower()
        if ans not in ("j", "ja", "y", "yes"):
            print("Abgebrochen.")
            return False
    del feeds_cfg["topics"][idx]
    save_feeds_config(feeds_cfg)
    print(f"✅ Thema '{t['name']}' (ID: {t['id']}) wurde entfernt.")
    return True


def generate_topic_with_ai(topic_name: str, hint: str = "") -> dict[str, Any]:
    """Nutzt Gemini (oder lokales Ollama), um automatisch eine optimale Themenkonfiguration zu generieren."""
    import urllib.parse
    safe_q_de = urllib.parse.quote(f"{topic_name} (Studie OR Durchbruch OR Behandlung OR Zulassung)")
    safe_q_en = urllib.parse.quote(f"{topic_name} (study OR trial OR breakthrough OR treatment)")

    prompt = f"""Erstelle für ein RSS-Monitoring-System eine optimale Themenkonfiguration für das Thema:
"{topic_name}"
Zusätzlicher Hinweis/Fokus: "{hint or 'Wichtige neue Studien, Durchbrüche, Behandlungen/Therapien und Neuzulassungen'}"

Gib ein valides JSON-Objekt mit exakt dieser Struktur zurück:
{{
  "name": "{topic_name}",
  "emoji": "passendes Emoji (z. B. 🧬, 🧠, 💻, 🌿)",
  "focus": "2-4 Sätze: Was soll gemeldet werden (z. B. Phase-2/3-Studien, Neuzulassungen, belastbare Studienergebnisse, echte Durchbrüche)? Was soll ignoriert werden (z. B. Clickbait, Spendenaufrufe, Lifestyle-Tipps ohne neue Fakten)?",
  "categories": ["Studie", "Klinische Studie", "Durchbruch", "Behandlung/Therapie", "Zulassung", "Sonstiges"],
  "sources": [
    {{
      "type": "rss",
      "name": "Google News (DE)",
      "url": "https://news.google.com/rss/search?q={safe_q_de}+when:4d&hl=de&gl=DE&ceid=DE:de"
    }},
    {{
      "type": "rss",
      "name": "Google News (EN)",
      "url": "https://news.google.com/rss/search?q={safe_q_en}+when:4d&hl=en-US&gl=US&ceid=US:en"
    }}
  ]
}}
Zusätzliche Quellen-Regeln:
- Falls es ein medizinisches/biowissenschaftliches Thema ist, füge Medical Xpress (https://medicalxpress.com/rss-feed/search/?search={urllib.parse.quote(topic_name)}) oder eine Europe-PMC-Quelle hinzu (type: "europepmc", name: "Europe PMC", query: "TITLE:({topic_name}) AND (PUB_TYPE:\\"Randomized Controlled Trial\\" OR PUB_TYPE:\\"Review\\")").
- Alle URLs müssen syntaktisch korrekt und URL-encodiert sein."""

    system = "Du bist ein Konfigurations-Assistent für ein wissenschaftliches RSS-Monitoring-System. Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt."
    cfg = assistant.load_config()
    raw = llm_json(prompt, system, cfg)
    return raw


def cmd_topic_add(args: argparse.Namespace) -> bool:
    feeds_cfg = load_feeds_config()
    name = (getattr(args, "name", None) or "").strip()

    # Interaktiver Modus falls kein Name angegeben
    if not name:
        try:
            name = input("Themen-Name (z. B. Multiple Sklerose, Quantencomputing): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAbgebrochen.")
            return False
        if not name:
            print("❌ Kein Name angegeben. Abbruch.")
            return False

    tid = getattr(args, "id", None) or slugify_topic_id(name)
    existing_idx, existing = find_topic(feeds_cfg, tid)
    if existing is not None:
        print(f"❌ Ein Thema mit ID '{tid}' (Name: '{existing['name']}') existiert bereits.")
        return False

    emoji = getattr(args, "emoji", None) or "📌"
    focus = getattr(args, "focus", None) or ""
    sources: list[dict[str, Any]] = []

    # Explizit angegebene RSS-Quellen
    raw_rss = getattr(args, "rss", None) or []
    for r in raw_rss:
        if "=" in r:
            s_name, s_url = r.split("=", 1)
        else:
            s_name, s_url = "RSS", r
        sources.append({"type": "rss", "name": s_name.strip(), "url": s_url.strip()})

    auto = getattr(args, "auto", False)
    # Wenn interaktiv und keine Quellen angegeben -> Frage nach KI-Generierung
    if not auto and not sources and not focus and sys.stdin.isatty():
        try:
            ans = input("Passende Feeds, Fokus & Emoji automatisch per KI generieren? [J/n]: ").strip().lower()
            if ans in ("", "j", "ja", "y", "yes"):
                auto = True
        except (KeyboardInterrupt, EOFError):
            return False

    if auto:
        print(f"Generiere Konfiguration für '{name}' via KI ...")
        try:
            ai_data = generate_topic_with_ai(name, hint=focus)
            if ai_data.get("emoji"):
                emoji = ai_data["emoji"]
            if ai_data.get("focus"):
                focus = ai_data["focus"]
            if ai_data.get("sources"):
                sources = ai_data["sources"]
            categories = ai_data.get("categories") or ["Studie", "Durchbruch", "Behandlung/Therapie", "Zulassung", "Sonstiges"]
        except Exception as exc:
            print(f"⚠️ KI-Generierung fehlgeschlagen ({exc}) – erstelle Standard-Vorlage.")
            import urllib.parse
            q_de = urllib.parse.quote(f"{name} (Studie OR Durchbruch OR Behandlung)")
            q_en = urllib.parse.quote(f"{name} (study OR breakthrough OR treatment)")
            sources = [
                {"type": "rss", "name": "Google News (DE)", "url": f"https://news.google.com/rss/search?q={q_de}+when:4d&hl=de&gl=DE&ceid=DE:de"},
                {"type": "rss", "name": "Google News (EN)", "url": f"https://news.google.com/rss/search?q={q_en}+when:4d&hl=en-US&gl=US&ceid=US:en"},
            ]
            categories = ["Studie", "Klinische Studie", "Durchbruch", "Behandlung/Therapie", "Zulassung", "Sonstiges"]
            focus = f"{name}: wichtige neue Studien, Durchbrüche, Behandlungen und Neuzulassungen."
    else:
        categories = ["Studie", "Klinische Studie", "Durchbruch", "Behandlung/Therapie", "Zulassung", "Sonstiges"]
        if not focus:
            focus = f"{name}: wichtige neue Studien, Durchbrüche, Behandlungen und Neuzulassungen."
        if not sources:
            import urllib.parse
            q_de = urllib.parse.quote(f"{name} (Studie OR Durchbruch OR Behandlung)")
            sources = [{"type": "rss", "name": "Google News (DE)", "url": f"https://news.google.com/rss/search?q={q_de}+when:4d&hl=de&gl=DE&ceid=DE:de"}]

    new_topic = {
        "id": tid,
        "name": name,
        "emoji": emoji,
        "enabled": True,
        "focus": focus,
        "categories": categories,
        "sources": sources,
    }

    print("\n--- Neue Themen-Konfiguration ---")
    print(f"ID:      {new_topic['id']}")
    print(f"Name:    {new_topic['emoji']} {new_topic['name']}")
    print(f"Fokus:   {new_topic['focus']}")
    print(f"Quellen ({len(sources)}):")
    for s in sources:
        u = s.get("url") or s.get("query") or ""
        print(f"  - [{s.get('type', 'rss')}] {s.get('name')}: {u[:75]}")

    feeds_cfg.setdefault("topics", []).append(new_topic)
    save_feeds_config(feeds_cfg)
    print(f"\n✅ Thema '{new_topic['name']}' erfolgreich angelegt und aktiviert!")
    print(f"Starte einen Test-Lauf mit:  python rss_monitor.py run --topic {tid}")
    return True


def cmd_topic_edit(args: argparse.Namespace) -> bool:
    feeds_cfg = load_feeds_config()
    idx, t = find_topic(feeds_cfg, args.topic_id)
    if t is None:
        print(f"❌ Thema '{args.topic_id}' wurde nicht gefunden.")
        return False
    changed = False
    if getattr(args, "name", None):
        t["name"] = args.name.strip()
        changed = True
    if getattr(args, "emoji", None):
        t["emoji"] = args.emoji.strip()
        changed = True
    if getattr(args, "focus", None):
        t["focus"] = args.focus.strip()
        changed = True
    if getattr(args, "min_relevance", None):
        t["min_relevance"] = args.min_relevance.strip()
        changed = True
    if getattr(args, "enabled", None) is not None:
        t["enabled"] = bool(args.enabled)
        changed = True
    if changed:
        save_feeds_config(feeds_cfg)
        print(f"✅ Thema '{t['name']}' (ID: {t['id']}) aktualisiert.")
    else:
        print("Keine Änderungen übergeben.")
    return True


def cmd_topic_add_source(args: argparse.Namespace) -> bool:
    feeds_cfg = load_feeds_config()
    idx, t = find_topic(feeds_cfg, args.topic_id)
    if t is None:
        print(f"❌ Thema '{args.topic_id}' wurde nicht gefunden.")
        return False
    name = (getattr(args, "name", None) or "").strip() or "RSS Feed"
    url = getattr(args, "url", "").strip()
    stype = getattr(args, "type", "rss").strip().lower()
    keywords = [k.strip() for k in (getattr(args, "keywords", None) or []) if k.strip()]
    src_obj: dict[str, Any] = {"type": stype, "name": name}
    if stype == "europepmc":
        src_obj["query"] = url
    elif stype in ("atproto", "bluesky"):
        src_obj["queries"] = [q.strip() for q in url.split(",") if q.strip()]
    else:
        src_obj["url"] = url
    if keywords:
        src_obj["keywords"] = keywords
    t.setdefault("sources", []).append(src_obj)
    save_feeds_config(feeds_cfg)
    print(f"✅ Quelle '{name}' zu Thema '{t['name']}' hinzugefügt.")
    return True


def cmd_topic_remove_source(args: argparse.Namespace) -> bool:
    feeds_cfg = load_feeds_config()
    idx, t = find_topic(feeds_cfg, args.topic_id)
    if t is None:
        print(f"❌ Thema '{args.topic_id}' wurde nicht gefunden.")
        return False
    sources = t.get("sources", [])
    target = args.source_ref.strip()
    removed = None
    if target.isdigit():
        sidx = int(target) - 1
        if 0 <= sidx < len(sources):
            removed = sources.pop(sidx)
    if removed is None:
        for sidx, s in enumerate(sources):
            if s.get("name", "").lower() == target.lower():
                removed = sources.pop(sidx)
                break
    if removed is not None:
        save_feeds_config(feeds_cfg)
        print(f"✅ Quelle '{removed.get('name')}' aus Thema '{t['name']}' entfernt.")
        return True
    print(f"❌ Quelle '{target}' im Thema '{t['name']}' nicht gefunden.")
    return False


def cmd_topics() -> None:
    cmd_topic_list(verbose=False)


def cmd_test_feeds() -> None:
    feeds_cfg = load_feeds_config()
    for t in feeds_cfg["topics"]:
        print(f"{t.get('emoji', '')} {t['name']}")
        for s in t.get("sources", []):
            try:
                stype = s.get("type", "rss").lower()
                if stype == "europepmc":
                    items = fetch_europepmc(s, 10, 25)
                elif stype in ("atproto", "bluesky"):
                    items = fetch_atproto(s, 10, 25)
                else:
                    items = fetch_rss(s, 10, 25)
                print(f"   OK   {len(items):3d}  {s['name']}")
            except Exception as exc:
                print(f"   FEHLER     {s['name']}: {exc}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="RSS-Monitor mit Notion-Wissensspeicher")
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. run
    p_run = sub.add_parser("run", help="Feeds prüfen, bewerten und neue Funde speichern")
    p_run.add_argument("--topic", action="append", help="Nur dieses Thema (id, mehrfach möglich)")
    p_run.add_argument("--dry-run", action="store_true", help="Nichts speichern, nur anzeigen")
    p_run.add_argument("--no-notify", action="store_true", help="Keine Telegram-/Windows-Benachrichtigung")
    p_run.add_argument("--no-notion", action="store_true", help="Nicht nach Notion schreiben")
    p_run.add_argument("-v", "--verbose", action="store_true")

    # 2. topic (Themen verwalten)
    p_topic = sub.add_parser("topic", help="Themen anzeigen, aktivieren, anlegen oder anpassen")
    sub_topic = p_topic.add_subparsers(dest="topic_action", required=False)

    p_t_list = sub_topic.add_parser("list", help="Alle Themen anzeigen")
    p_t_list.add_argument("-v", "--verbose", action="store_true", help="Quellen-URLs & Details anzeigen")

    p_t_enable = sub_topic.add_parser("enable", help="Thema aktivieren")
    p_t_enable.add_argument("topic_id", help="Themen-ID oder Name")

    p_t_disable = sub_topic.add_parser("disable", help="Thema deaktivieren")
    p_t_disable.add_argument("topic_id", help="Themen-ID oder Name")

    p_t_add = sub_topic.add_parser("add", help="Neues Thema anlegen")
    p_t_add.add_argument("name", nargs="?", help="Name des Themas (z. B. 'Multiple Sklerose')")
    p_t_add.add_argument("--id", help="Eindeutige ID (Standard: automatisch aus Name)")
    p_t_add.add_argument("--emoji", help="Emoji-Symbol (z. B. 🧬)")
    p_t_add.add_argument("--focus", help="Fokus-Beschreibung (was soll erfasst werden)")
    p_t_add.add_argument("--rss", action="append", help="RSS-Quelle im Format 'Name=URL' oder 'URL'")
    p_t_add.add_argument("--auto", action="store_true", help="Feeds, Fokus und Emoji automatisch per KI generieren")

    p_t_remove = sub_topic.add_parser("remove", help="Thema entfernen")
    p_t_remove.add_argument("topic_id", help="Themen-ID oder Name")
    p_t_remove.add_argument("-y", "--yes", action="store_true", help="Ohne Sicherheitsnachfrage löschen")

    p_t_edit = sub_topic.add_parser("edit", help="Themen-Eigenschaften bearbeiten")
    p_t_edit.add_argument("topic_id", help="Themen-ID oder Name")
    p_t_edit.add_argument("--name", help="Neuer Anzeigename")
    p_t_edit.add_argument("--emoji", help="Neues Emoji")
    p_t_edit.add_argument("--focus", help="Neue Fokus-Kriterien")
    p_t_edit.add_argument("--min-relevance", choices=["Hoch", "Mittel", "Niedrig"], help="Mindest-Relevanz")

    p_t_add_src = sub_topic.add_parser("add-source", help="Feed-Quelle zu einem Thema hinzufügen")
    p_t_add_src.add_argument("topic_id", help="Themen-ID")
    p_t_add_src.add_argument("--url", required=True, help="RSS-URL oder Europe-PMC-Query")
    p_t_add_src.add_argument("--name", help="Name der Quelle (z. B. 'FAZ Wissen')")
    p_t_add_src.add_argument("--type", choices=["rss", "europepmc", "atproto", "bluesky"], default="rss", help="Quellen-Typ")
    p_t_add_src.add_argument("--keywords", nargs="+", help="Optionale Keyword-Filter")

    p_t_rem_src = sub_topic.add_parser("remove-source", help="Quelle aus Thema entfernen")
    p_t_rem_src.add_argument("topic_id", help="Themen-ID")
    p_t_rem_src.add_argument("source_ref", help="Nummer der Quelle (aus 'topic list -v') oder Name")

    # 3. Aliase / Hilfsbefehle
    sub.add_parser("topics", help="Kurzübersicht aller Themen")
    sub.add_parser("test-feeds", help="Erreichbarkeit aller Quellen prüfen")
    sub.add_parser("bot", help="Dedizierten Telegram-Bot für den RSS-Wissensspeicher starten")
    p_syn = sub.add_parser("synthese", help="Themenübergreifende Wochen-Synthese generieren")
    p_syn.add_argument("--days", type=int, default=7, help="Anzahl Tage (Standard: 7)")

    args = parser.parse_args()

    if args.command == "bot":
        import rss_bot
        rss_bot.main()
        return

    if args.command == "synthese":
        cfg = assistant.load_config()
        token = os.environ.get("NOTION_TOKEN", "").strip()
        notion = assistant.NotionClient(token) if token else None
        html_out, _ = generate_weekly_rss_synthesis(cfg=cfg, store=get_store(), notion=notion, days=args.days)
        print(telegram_ui.strip_html_tags(html_out))
        return

    if args.command in ("topics", "topic"):
        action = getattr(args, "topic_action", None)
        if not action or action == "list":
            cmd_topic_list(verbose=getattr(args, "verbose", False))
            return
        if action == "enable":
            cmd_topic_set_enabled(args.topic_id, True)
            return
        if action == "disable":
            cmd_topic_set_enabled(args.topic_id, False)
            return
        if action == "add":
            cmd_topic_add(args)
            return
        if action == "remove":
            cmd_topic_remove(args.topic_id, confirm=getattr(args, "yes", False))
            return
        if action == "edit":
            cmd_topic_edit(args)
            return
        if action == "add-source":
            cmd_topic_add_source(args)
            return
        if action == "remove-source":
            cmd_topic_remove_source(args)
            return
        return

    if args.command == "test-feeds":
        cmd_test_feeds()
        return

    setup_logging(getattr(args, "verbose", False))
    if not acquire_lock():
        log.warning("Ein anderer Lauf ist noch aktiv – Abbruch.")
        sys.exit(1)
    try:
        sys.exit(run(args))
    except Exception:
        log.exception("Unerwarteter Fehler")
        sys.exit(1)
    finally:
        release_lock()


if __name__ == "__main__":
    main()
