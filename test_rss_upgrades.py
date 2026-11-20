"""
Unit-Tests für die RSS-Monitor-Erweiterungen:
1. Quellen-Diversifizierung in preprocess.cap_candidates(..., diversify_by_source=True)
2. Keyword- & Exclude-Keyword-Filterung (_matches_source_filters)
3. PubMed E-Utilities Connector (fetch_pubmed)
4. ClinicalTrials.gov API v2 Connector (fetch_clinicaltrials)
5. collect_candidates (Dispatch, _source_name, skip_jev, Round-Robin-Cap)
6. jev_prefilter_candidates (skip_jev-Bypass, voller Fokus > 300 Zeichen)
7. Validierung der feeds.json-Konfiguration
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import preprocess
import rss_monitor


def test_cap_candidates_diversify_by_source():
    now = dt.datetime.now(dt.timezone.utc)
    # 20 sehr frische Einträge aus Quelle A (z. B. Bluesky) + 2 etwas ältere aus Quelle B (z. B. npj Parkinson's)
    cands = [
        {"title": f"A-{i}", "published": now - dt.timedelta(minutes=i), "_source_name": "Bluesky"}
        for i in range(20)
    ] + [
        {"title": "B-0", "published": now - dt.timedelta(days=2), "_source_name": "npj Parkinson's"},
        {"title": "B-1", "published": now - dt.timedelta(days=3), "_source_name": "npj Parkinson's"},
    ]

    # Ohne Diversifizierung verdrängt Bluesky Quelle B komplett
    pure_recent = preprocess.cap_candidates(cands, max_total=6, diversify_by_source=False)
    assert all(c["_source_name"] == "Bluesky" for c in pure_recent)

    # Mit Diversifizierung bekommt Quelle B direkt in den ersten Runden Slots
    diversified = preprocess.cap_candidates(cands, max_total=6, diversify_by_source=True)
    assert len(diversified) == 6
    b_titles = [c["title"] for c in diversified if c["_source_name"] == "npj Parkinson's"]
    assert b_titles == ["B-0", "B-1"]


def test_matches_source_filters_keywords_and_excludes():
    src = {
        "keywords": ["student", "gratis", "vollversion"],
        "exclude_keywords": ["aldi", "gewinnspiel"],
    }
    assert rss_monitor._matches_source_filters({"title": "Office Pro gratis für Studenten", "summary": ""}, src) is True
    assert rss_monitor._matches_source_filters({"title": "Vollversion CAD kostenlos", "summary": "Aktion"}, src) is True
    # Fehlendes Positiv-Keyword
    assert rss_monitor._matches_source_filters({"title": "Allgemeine Hochschulnews", "summary": "Bericht"}, src) is False
    # Treffer auf Exclude-Keyword trotz Positiv-Keyword
    assert rss_monitor._matches_source_filters({"title": "Gratis Chips bei ALDI Nord", "summary": "student"}, src) is False
    assert rss_monitor._matches_source_filters({"title": "Software Gewinnspiel für Studenten", "summary": "gratis"}, src) is False

    # Kurze Keywords (<= 4 Zeichen) dürfen keine Teilwort-Treffer wie "Sparabo" -> "abo" auslösen
    short_src = {"keywords": ["abo", "ai", "ki"]}
    assert rss_monitor._matches_source_filters({"title": "Hasseröder im Sparabo", "summary": "Email"}, short_src) is False
    assert rss_monitor._matches_source_filters({"title": "Neues KI Abo für Studierende", "summary": ""}, short_src) is True


@patch("rss_monitor.requests.get")
def test_fetch_pubmed_parses_xml(mock_get):
    esearch_resp = MagicMock()
    esearch_resp.raise_for_status.return_value = None
    esearch_resp.json.return_value = {"esearchresult": {"idlist": ["43034713"]}}

    xml_payload = b"""<?xml version="1.0" encoding="utf-8"?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>43034713</PMID>
          <Article>
            <Journal>
              <ISOAbbreviation>Mov Disord</ISOAbbreviation>
            </Journal>
            <ArticleTitle>Phase 3 trial of <i>foslevodopa</i> in Parkinson's disease.</ArticleTitle>
            <Abstract>
              <AbstractText Label="METHODS">Randomized double-blind study with 240 patients.</AbstractText>
              <AbstractText Label="RESULTS">Significant improvement in ON time.</AbstractText>
            </Abstract>
            <PublicationTypeList>
              <PublicationType>Randomized Controlled Trial</PublicationType>
            </PublicationTypeList>
          </Article>
        </MedlineCitation>
        <PubmedData>
          <History>
            <PubMedPubDate PubStatus="entrez">
              <Year>2026</Year><Month>11</Month><Day>15</Day>
            </PubMedPubDate>
          </History>
          <ArticleIdList>
            <ArticleId IdType="doi">10.1002/mds.12345</ArticleId>
          </ArticleIdList>
        </PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>"""
    efetch_resp = MagicMock()
    efetch_resp.raise_for_status.return_value = None
    efetch_resp.content = xml_payload

    mock_get.side_effect = [esearch_resp, efetch_resp]

    items = rss_monitor.fetch_pubmed(
        {"name": "PubMed", "query": "Parkinson*[Title]"},
        max_age_days=10,
        max_items=10,
    )
    assert len(items) == 1
    it = items[0]
    assert it["title"] == "Phase 3 trial of foslevodopa in Parkinson's disease."
    assert it["link"] == "https://pubmed.ncbi.nlm.nih.gov/43034713/"
    assert "Mov Disord; Randomized Controlled Trial" in it["summary"]
    assert "METHODS: Randomized double-blind study" in it["summary"]
    assert it["published"] == dt.datetime(2026, 11, 15, tzinfo=dt.timezone.utc)
    assert it["source"] == "PubMed / Mov Disord"


@patch("rss_monitor.requests.get")
def test_fetch_clinicaltrials_parses_and_filters_phases(mock_get):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT07154511", "briefTitle": "Phase 2 Trial of Tributyrin"},
                    "statusModule": {"overallStatus": "RECRUITING", "lastUpdatePostDateStruct": {"date": "2026-11-13"}},
                    "designModule": {"studyType": "INTERVENTIONAL", "phases": ["PHASE2"], "enrollmentInfo": {"count": 60}},
                    "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Mass General Brigham"}},
                    "armsInterventionsModule": {"interventions": [{"name": "Tributyrin"}]},
                    "descriptionModule": {"briefSummary": "Evaluating gut-brain axis modulation in PD."},
                }
            },
            {
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT06778135", "briefTitle": "Observational Imaging"},
                    "statusModule": {"overallStatus": "RECRUITING", "lastUpdatePostDateStruct": {"date": "2026-11-12"}},
                    "designModule": {"studyType": "OBSERVATIONAL", "phases": ["NA"]},
                    "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Uni Hospital"}},
                    "armsInterventionsModule": {"interventions": []},
                    "descriptionModule": {"briefSummary": "MRI imaging study."},
                }
            },
        ]
    }
    mock_get.return_value = resp

    items = rss_monitor.fetch_clinicaltrials(
        {"name": "ClinicalTrials.gov", "query": "Parkinson Disease", "phases": ["PHASE2", "PHASE3"]},
        max_age_days=10,
        max_items=10,
    )
    assert len(items) == 1
    assert items[0]["title"] == "Phase 2 Trial of Tributyrin"
    assert items[0]["link"] == "https://clinicaltrials.gov/study/NCT07154511"
    assert "PHASE2" in items[0]["summary"]
    assert "Tributyrin" in items[0]["summary"]
    assert items[0]["published"] == dt.datetime(2026, 11, 13, tzinfo=dt.timezone.utc)


def test_collect_candidates_dispatches_and_propagates_skip_jev(monkeypatch):
    now = dt.datetime.now(dt.timezone.utc)
    monkeypatch.setattr(
        rss_monitor,
        "fetch_pubmed",
        lambda src, age, lim: [
            {
                "title": "New alpha-synuclein biomarker validation",
                "link": "https://pubmed.ncbi.nlm.nih.gov/123/",
                "summary": "Large multicenter cohort.",
                "published": now - dt.timedelta(days=1),
                "source": "PubMed / Lancet Neurol",
            },
            {
                "title": "Excluded conference abstract",
                "link": "https://pubmed.ncbi.nlm.nih.gov/124/",
                "summary": "Erratum notice",
                "published": now - dt.timedelta(days=1),
                "source": "PubMed / X",
            },
        ],
    )
    monkeypatch.setattr(
        rss_monitor,
        "fetch_clinicaltrials",
        lambda src, age, lim: [
            {
                "title": "Phase 3 Trial of Drug X",
                "link": "https://clinicaltrials.gov/study/NCT999",
                "summary": "Randomized placebo-controlled trial.",
                "published": now - dt.timedelta(days=2),
                "source": "CT.gov / Pharma",
            }
        ],
    )

    topic = {
        "id": "parkinson",
        "name": "Parkinson",
        "sources": [
            {"type": "pubmed", "name": "PubMed", "query": "PD", "exclude_keywords": ["erratum"], "skip_jev": True},
            {"type": "clinicaltrials", "name": "CT.gov", "query": "PD"},
        ],
    }
    cands, errs = rss_monitor.collect_candidates(topic, {"max_age_days": 10, "max_candidates_per_topic": 50}, {"seen": {}})
    assert errs == 0
    assert len(cands) == 2
    by_src = {c["_source_name"]: c for c in cands}
    assert by_src["PubMed"].get("skip_jev") is True
    assert by_src["CT.gov"].get("skip_jev") is None


def test_jev_prefilter_bypasses_skip_jev_and_passes_full_focus(monkeypatch):
    long_focus = ("A" * 350) + " SONDERFALL_AM_ENDE"
    seen_instructions: list[str] = []

    class _FakeJev:
        is_configured = True
        model = "jev-test"

        def system_one(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
            seen_instructions.append(questions["is_relevant"]["instructions"])
            return {
                "answers": {
                    "is_relevant": {"noul": 0.60},
                    "category": {"choice": "Studie", "confidence": 0.9},
                }
            }

    monkeypatch.setattr(rss_monitor.jev_client, "JevClient", lambda: _FakeJev())
    topic = {"id": "long_covid", "name": "Long COVID", "focus": long_focus, "categories": ["Studie", "Infektionslage"]}
    cands = [
        {"title": "Covid Lage 1 von 50", "source": "RV-Enigma", "summary": "R=1.0", "skip_jev": True},
        {"title": "Long COVID Kohorte", "source": "GNews", "summary": "1200 Patienten"},
    ]
    passed, dropped, stats = rss_monitor.jev_prefilter_candidates(
        topic,
        cands,
        {"jev_enabled": True, "jev_relevance_threshold": 0.45},
    )
    assert len(passed) == 2
    assert len(dropped) == 0
    assert stats["bypassed"] == 1
    assert passed[0].get("jev_bypassed") is True
    assert len(seen_instructions) == 1
    assert "SONDERFALL_AM_ENDE" in seen_instructions[0]


def test_feeds_json_schema_valid():
    feeds_path = Path(__file__).resolve().parent / "feeds.json"
    cfg = json.loads(feeds_path.read_text(encoding="utf-8"))
    settings = cfg.get("settings", {})
    assert settings["max_candidates_per_topic"] >= 100
    assert 0.35 <= settings["jev_relevance_threshold"] <= 0.60

    valid_types = {"rss", "europepmc", "pubmed", "clinicaltrials", "atproto", "bluesky"}
    for topic in cfg.get("topics", []):
        if not topic.get("enabled", True):
            continue
        assert topic["sources"], f"Topic {topic['id']} has no sources"
        seen_names: set[str] = set()
        for src in topic["sources"]:
            stype = src.get("type", "rss").lower()
            assert stype in valid_types, f"Invalid type {stype} in {topic['id']}"
            assert src["name"] not in seen_names, f"Duplicate source name {src['name']} in {topic['id']}"
            seen_names.add(src["name"])
            if stype == "rss":
                assert src["url"].startswith("http")
                assert "mydealz.de/rss" not in src["url"]
            elif stype in ("europepmc", "pubmed", "clinicaltrials"):
                assert src.get("query") or src.get("term") or src.get("condition")
            elif stype in ("atproto", "bluesky"):
                assert src.get("queries") or src.get("query")


def test_telegram_split_message_chunks_preserves_html_links():
    from integrations import telegram_ui

    item_block = (
        '🔴 <b>#1</b> <a href="https://example.org/study/12345">Sehr langer Studientitel zu Parkinson</a>\n'
        '   <i>Ausführliche Zusammenfassung der Phase-2-Studie über 52 Wochen mit 240 Patienten und positiven Ergebnissen.</i>\n'
        '   🔗 <a href="https://example.org/study/12345">Mehr lesen</a>'
    )
    long_text = "\n".join([item_block] * 30)
    chunks = telegram_ui._split_message_chunks(long_text, max_len=1000)
    assert len(chunks) > 1
    for ch in chunks:
        assert len(ch) <= 1000
        # Jedes geöffnete <a>-Tag muss im selben Chunk auch wieder geschlossen sein
        assert ch.count("<a ") == ch.count("</a>")
        assert ch.count("<i>") == ch.count("</i>")


def test_notify_sends_full_summary_and_read_more_link(monkeypatch):
    sent_texts: list[str] = []
    monkeypatch.setenv("TELEGRAM_RSSBOT_TOKEN", "bot:test")
    monkeypatch.setenv("TELEGRAM_RSS_CHAT_ID", "12345")
    monkeypatch.setattr(
        rss_monitor,
        "send_telegram_msg",
        lambda token, chat_id, text, **kw: sent_texts.append(text) or True,
    )

    long_summary = (
        "Eine randomisierte Phase-2-Studie mit 180 Teilnehmenden untersuchte den Wirkstoff XY über 24 Wochen. "
        "Der primäre Endpunkt zeigte eine statistisch signifikante Verbesserung der motorischen Scores im Vergleich zu Placebo. "
        "Schwerwiegende Nebenwirkungen traten nicht häufiger auf als in der Kontrollgruppe; Phase-3-Planung läuft."
    )
    assert len(long_summary) > 180

    results = [
        {
            "topic": {"id": "parkinson", "name": "Parkinson", "emoji": "🧠"},
            "findings": [
                {
                    "topic_id": "parkinson",
                    "topic_name": "Parkinson",
                    "title": "Phase-2-Erfolg für Wirkstoff XY",
                    "summary": long_summary,
                    "relevance": "Hoch",
                    "key": "wirkstoff-xy-phase2",
                    "link": "https://pubmed.ncbi.nlm.nih.gov/99999999/",
                }
            ],
            "overview": "",
        },
        {
            "topic": {"id": "edu_deals", "name": "Rabatte Studierende/Dozierende", "emoji": "🎓"},
            "findings": [],
            "overview": "",
        },
    ]
    rss_monitor.notify(results, "https://notion.so/digest", {"notify_telegram": True, "notify_windows": False})
    assert len(sent_texts) == 1
    msg = sent_texts[0]
    assert long_summary in msg
    assert '🔗 <a href="https://pubmed.ncbi.nlm.nih.gov/99999999/">Mehr lesen</a>' in msg
    assert "Rabatte Studierende/Dozierende" in msg
    assert "keine neuen Funde in diesem Lauf" in msg
