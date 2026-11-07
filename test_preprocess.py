"""
Tests für deterministisches Preprocessing:
1. HTML-Bereinigung, Script/Style/Nav-Entfernung, Entities & Boilerplate
2. URL-Kanonisierung (Tracking-Parameter, Fragmente, Normalisierung)
3. Fingerprints (GUID, URL, Titel, Content-Hash)
4. Zeitfenster & Datums-Filter (Cutoff, undated_policy)
5. Kandidaten-Begrenzung (Cap)
"""

from __future__ import annotations

import datetime as dt
import preprocess


def test_strip_html_and_boilerplate_removes_scripts_and_tags():
    raw = (
        "<html><head><script>alert('bad');</script><style>body {color:red}</style></head>"
        "<body><nav>Home | About</nav><p>Wichtige <b>Studie</b> zu Parkinson &amp; Long COVID.</p>"
        "<aside>Werbung</aside><footer>Copyright 2026</footer>"
        "<p>The post Parkinson Breakthrough appeared first on MedicalNews.</p></body></html>"
    )
    clean = preprocess.strip_html_and_boilerplate(raw)
    assert "bad" not in clean
    assert "color:red" not in clean
    assert "Home | About" not in clean
    assert "Werbung" not in clean
    assert "appeared first on" not in clean
    assert "Wichtige Studie zu Parkinson & Long COVID." in clean


def test_strip_html_handles_empty_and_entities():
    assert preprocess.strip_html_and_boilerplate(None) == ""
    assert preprocess.strip_html_and_boilerplate("   ") == ""
    assert preprocess.strip_html_and_boilerplate("&quot;Test&#x27;s&quot;") == '"Test\'s"'


def test_canonicalize_url_strips_tracking():
    url = (
        "https://WWW.example.com:443/article/?utm_source=twitter&utm_medium=social"
        "&fbclid=IwAR123&real_param=42&ref=newsletter#comments"
    )
    canon = preprocess.canonicalize_url(url)
    assert "utm_source" not in canon
    assert "fbclid" not in canon
    assert "comments" not in canon
    assert ":443" not in canon
    assert "real_param=42" in canon
    assert canon == "https://www.example.com/article?real_param=42"


def test_compute_item_fingerprints_variety():
    item = {
        "id": "doi:10.1038/s41582-026-00123-x",
        "link": "https://nature.com/articles/s41582-026-00123-x?utm_source=rss",
        "title": "Phase-3 trial of new Parkinson drug shows motor symptom improvement - Nature",
        "summary": "Randomized double-blind placebo-controlled study with 420 patients over 52 weeks.",
    }
    fps = preprocess.compute_item_fingerprints(item)
    types = [fp[:2] for fp in fps]
    assert "g:" in types  # GUID/ID
    assert "l:" in types  # Canonical Link
    assert "t:" in types  # Normalized Title
    assert "c:" in types  # Content Hash


def test_filter_and_preprocess_feed_item_cutoff_and_policy():
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(days=7)

    recent_item = {
        "title": "Neues Medikament zugelassen",
        "link": "https://example.org/news/1",
        "published": now - dt.timedelta(days=2),
    }
    old_item = {
        "title": "Alte Meldung vor einem Monat",
        "link": "https://example.org/news/2",
        "published": now - dt.timedelta(days=30),
    }
    undated_item = {
        "title": "Undatierter Artikel",
        "link": "https://example.org/news/3",
        "published": None,
    }

    # Frischer Artikel kommt durch
    assert preprocess.filter_and_preprocess_feed_item(recent_item, cutoff=cutoff) is not None

    # Alter Artikel wird verworfen
    assert preprocess.filter_and_preprocess_feed_item(old_item, cutoff=cutoff) is None

    # Undatierter Artikel bei 'keep' behalten
    assert preprocess.filter_and_preprocess_feed_item(undated_item, cutoff=cutoff, undated_policy="keep") is not None

    # Undatierter Artikel bei 'drop' verworfen
    assert preprocess.filter_and_preprocess_feed_item(undated_item, cutoff=cutoff, undated_policy="drop") is None


def test_cap_candidates():
    now = dt.datetime.now(dt.timezone.utc)
    cands = [
        {"title": f"Item {i}", "published": now - dt.timedelta(hours=i)}
        for i in range(20)
    ]
    capped = preprocess.cap_candidates(cands, max_total=5)
    assert len(capped) == 5
    # Neueste zuerst
    assert capped[0]["title"] == "Item 0"
    assert capped[-1]["title"] == "Item 4"
