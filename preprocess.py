"""
Deterministisches Preprocessing für RSS- und Web-Feeds.

Strikt deterministisch ohne KI-Calls:
1. Strippt HTML-Tags, <script>, <style>, <nav>, <footer>, <aside>, HTML-Entities und typische Boilerplate.
2. Kanonisiert URLs (entfernt utm_*, fbclid, gclid, mc_*, ref etc., Fragmente, Trailing Slashes).
3. Berechnet Fingerprints über GUID/ID, kanonische URL, normalisierten Titel und Content-Hash.
4. Filtert nach Alter (Cutoff) und undatierten Einträgen (Policy: 'drop' | 'keep' | 'first_seen').
5. Begrenzt Kandidatenzahlen pro Thema (Cap).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# Typische Marketing-, Tracking- und Session-Parameter
TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "utm_reader",
    "fbclid",
    "gclid",
    "gclsrc",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
    "igshid",
    "_ga",
    "_gl",
    "spm",
    "yclid",
    "twclid",
}

# Typische Feed-Boilerplate-Phrasen, die Text aufblähen
BOILERPLATE_PATTERNS = [
    r"(?i)the post\s+.*?\s+appeared first on\s+.*?(\.|$)",
    r"(?i)der beitrag\s+.*?\s+erschien zuerst auf\s+.*?(\.|$)",
    r"(?i)continue reading(\s+on\s+.*?)?(\.|$)",
    r"(?i)weiterlesen(\s+auf\s+.*?)?(\.|$)",
    r"(?i)read more(\s+on\s+.*?)?(\.|$)",
    r"(?i)hier weiterlesen.*?(\.|$)",
    r"(?i)mehr dazu bei\s+.*?(\.|$)",
    r"(?i)originalquelle:\s*https?://\S+",
    r"(?i)quelle:\s*https?://\S+",
]


def strip_html_and_boilerplate(raw_text: str | None) -> str:
    """
    Bereinigt Text deterministisch:
    - Entfernt <script>, <style>, <nav>, <footer>, <aside>, <noscript>, <svg>, <header>
    - Entfernt restliche HTML-Tags
    - Dekodiert HTML-Entities (&amp;, &quot;, &#x27; ...)
    - Entfernt typische Feed-Boilerplate-Sätze
    - Normalisiert Whitespace
    """
    if not raw_text:
        return ""

    text = raw_text

    # 1. Ganze Tags inkl. Inhalt für irrelevante Container entfernen
    for block_tag in ("script", "style", "nav", "footer", "aside", "noscript", "svg", "header", "form"):
        text = re.sub(rf"(?is)<{block_tag}\b[^>]*>.*?</{block_tag}>", " ", text)

    # 2. Übrige HTML-Tags entfernen
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. HTML-Entities auflösen (mehrfach, falls doppelt encodiert)
    for _ in range(2):
        prev = text
        text = html.unescape(text)
        if text == prev:
            break

    # 4. Boilerplate-Floskeln entfernen
    for pat in BOILERPLATE_PATTERNS:
        text = re.sub(pat, " ", text)

    # 5. Whitespace normalisieren
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonicalize_url(raw_url: str | None) -> str:
    """
    Normalisiert eine URL deterministisch für Deduplizierung:
    - Entfernt Query-Tracking-Parameter (utm_*, fbclid, gclid, mc_*, ref etc.)
    - Entfernt Fragmente (#section)
    - Normalisiert Scheme und Host auf Kleinbuchstaben
    - Entfernt redundante Standard-Ports (:80, :443)
    - Entfernt redundante Trailing Slashes auf Root-Ebene
    """
    if not raw_url:
        return ""

    url = raw_url.strip()
    if not url:
        return ""

    try:
        parsed = urlparse(url)
    except Exception:
        return url.split("#")[0].split("?utm")[0].strip()

    scheme = (parsed.scheme or "http").lower()
    netloc = (parsed.netloc or "").lower()

    # Standard-Ports bereinigen
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # Query-Parameter filtern
    clean_params: list[tuple[str, str]] = []
    if parsed.query:
        for k, v in parse_qsl(parsed.query, keep_blank_values=False):
            k_lower = k.lower()
            if k_lower in TRACKING_QUERY_PARAMS or k_lower.startswith("utm_"):
                continue
            clean_params.append((k, v))

    clean_query = urlencode(sorted(clean_params)) if clean_params else ""

    path = parsed.path or ""
    # Trailing Slash nur bei Nicht-Wurzel-Pfaden optional vereinheitlichen
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Fragment immer entfernen
    canon = urlunparse((scheme, netloc, path, parsed.params, clean_query, ""))
    return canon


def normalize_title(title: str | None) -> str:
    """Normalisiert Titel für robustes Deduplizieren."""
    if not title:
        return ""
    t = title.lower()
    # " - Publisher"-Suffix (z. B. Google News "- Spiegel Online")
    t = re.sub(r"\s+[-–|]\s+[^-–|]{2,60}$", "", t)
    t = re.sub(r"[^a-z0-9äöüß ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def compute_item_fingerprints(item: dict[str, Any]) -> list[str]:
    """
    Berechnet robuste Fingerprints für einen Feed-Artikel:
    - g: GUID/ID (falls vorhanden, stabil über Redirects)
    - l: Kanonische URL
    - t: Normalisierter Titel (wenn lang genug)
    - c: Content-Hash (Titel + die ersten 160 Zeichen der Zusammenfassung)
    """
    fps: list[str] = []

    # 1. Explizite GUID/ID
    guid = str(item.get("id") or item.get("guid") or "").strip()
    if guid:
        fps.append("g:" + sha(guid))

    # 2. Kanonische URL
    raw_link = item.get("link") or ""
    c_link = canonicalize_url(raw_link)
    if c_link:
        fps.append("l:" + sha(c_link))

    # 3. Normalisierter Titel
    nt = normalize_title(item.get("title", ""))
    if len(nt) > 15:
        fps.append("t:" + sha(nt))

    # 4. Content-Hash (erkennt Duplikate mit leicht veränderter URL/Überschrift)
    summary_snip = normalize_title(item.get("summary", ""))[:160]
    if len(nt) > 10 and len(summary_snip) > 20:
        fps.append("c:" + sha(nt + "|" + summary_snip))

    return fps


def filter_and_preprocess_feed_item(
    raw_item: dict[str, Any],
    cutoff: dt.datetime | None = None,
    undated_policy: str = "keep",
    max_summary_length: int = 1200,
) -> dict[str, Any] | None:
    """
    Bereinigt und prüft einen einzelnen Artikel.
    Gibt bereinigtes Dict zurück oder None, wenn der Artikel alt oder ungültig ist.

    undated_policy:
      - 'keep': Undatierte Artikel werden behalten (Default, bisheriges Verhalten).
      - 'drop': Undatierte Artikel werden verworfen (Schutz vor alten Batch-Spitzen).
    """
    # Titel & Summary deterministisch bereinigen
    title = strip_html_and_boilerplate(raw_item.get("title") or "")
    raw_summary = raw_item.get("summary") or raw_item.get("description") or ""
    summary = strip_html_and_boilerplate(raw_summary)[:max_summary_length]

    # Falls Microblogging ohne separaten Titel: erste Zeile nutzen
    if not title and summary:
        first_line = summary.split("\n")[0].split(". ")[0].strip()
        title = first_line[:110] if first_line else summary[:110]

    if not title:
        return None

    # Datum prüfen
    published = raw_item.get("published")
    if published is not None and not isinstance(published, dt.datetime):
        # Falls ISO-String
        try:
            published = dt.datetime.fromisoformat(str(published))
        except Exception:
            published = None

    if published and published.tzinfo is None:
        published = published.replace(tzinfo=dt.timezone.utc)

    if cutoff is not None:
        if published is not None:
            if published < cutoff:
                return None
        else:
            if undated_policy == "drop":
                return None

    link = canonicalize_url(raw_item.get("link") or "")

    clean_item: dict[str, Any] = {
        "title": title,
        "link": link,
        "summary": summary,
        "published": published,
        "publisher": raw_item.get("publisher"),
        "source": raw_item.get("source"),
    }
    if raw_item.get("id"):
        clean_item["id"] = raw_item["id"]
    if raw_item.get("guid"):
        clean_item["guid"] = raw_item["guid"]

    clean_item["fingerprints"] = compute_item_fingerprints(clean_item)
    return clean_item


def cap_candidates(
    candidates: list[dict[str, Any]],
    max_total: int = 60,
) -> list[dict[str, Any]]:
    """
    Begrenzt Kandidaten deterministisch auf die neuesten Einträge.
    Undatierte Einträge werden stabil am Ende eingeordnet.
    """
    if len(candidates) <= max_total:
        return candidates

    def sort_key(it: dict[str, Any]) -> tuple[int, dt.datetime]:
        pub = it.get("published")
        if isinstance(pub, dt.datetime):
            return (1, pub)
        return (0, dt.datetime.min.replace(tzinfo=dt.timezone.utc))

    sorted_candidates = sorted(candidates, key=sort_key, reverse=True)
    return sorted_candidates[:max_total]
