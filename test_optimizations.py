"""
Unit-Tests für:
1. JevClient (Noul, Choice, Score, Schwellenwert-Filterung mit gemocktem Requests)
2. Jev-Datenschutz-Guardrail (Maskierung von PII, Opt-in Toggle)
3. StepBudget im Chat-Agenten (max_iterations, Token-Budget-Bremse)
4. LLM-Core Modell-Tiering & Thinking-Level Validierung
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

import chat_guard
import jev_client
import llm_core


# ------------------------------------------------------------------------------
# 1. Jev Client & Schwellenwerte
# ------------------------------------------------------------------------------

def test_jev_client_not_configured():
    client = jev_client.JevClient(api_key="")
    assert not client.is_configured
    with pytest.raises(jev_client.JevError, match="nicht konfiguriert"):
        client.system_one("some text", {})


@patch("requests.post")
def test_jev_evaluate_noul_threshold(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "jev-1.13.0",
        "answers": {
            "rel": {"type": "noul", "noul": 0.85}
        },
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    mock_post.return_value = mock_resp

    client = jev_client.JevClient(api_key="ts_test_key", model="jev-1.13.0")
    val = client.evaluate_noul("Studie zu Parkinson", "Ist das relevant?", question_id="rel")
    assert val == 0.85
    assert val >= 0.70  # Schwellenwert erreicht


@patch("requests.post")
def test_jev_evaluate_choice_tagging(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "jev-1.13.0",
        "answers": {
            "cat": {
                "type": "choice",
                "choice": "Studie",
                "confidence": 0.92,
                "probabilities": {"Studie": 0.92, "Sonstiges": 0.08},
            }
        },
    }
    mock_post.return_value = mock_resp

    client = jev_client.JevClient(api_key="ts_test_key")
    choice, conf, probs = client.evaluate_choice(
        state="RCT zu Parkinson",
        instructions="Kategorie?",
        options={"Studie": "Forschung", "Sonstiges": "Rest"},
        question_id="cat",
    )
    assert choice == "Studie"
    assert conf == 0.92
    assert probs["Studie"] == 0.92


@patch("requests.post")
def test_jev_retry_on_503(mock_post):
    mock_fail = MagicMock()
    mock_fail.status_code = 503
    mock_fail.text = "Service Temporarily Unavailable"

    mock_ok = MagicMock()
    mock_ok.status_code = 200
    mock_ok.json.return_value = {
        "model": "jev-1.13.0",
        "answers": {"q": {"type": "noul", "noul": 0.9}},
    }

    mock_post.side_effect = [mock_fail, mock_ok]

    client = jev_client.JevClient(api_key="ts_test_key")
    val = client.evaluate_noul("text", "frage", question_id="q")
    assert val == 0.9
    assert mock_post.call_count == 2


# ------------------------------------------------------------------------------
# 2. Datenschutz-Guardrail
# ------------------------------------------------------------------------------

def test_redact_pii_masks_sensitive_data():
    raw = "Kontakt: stefan.kollmeier@hs-rm.de, Tel: +49 171 1234567, IBAN DE12345678901234567890, Passwort: secret123"
    redacted = jev_client.redact_pii_for_external_router(raw)
    assert "stefan.kollmeier@hs-rm.de" not in redacted
    assert "[EMAIL]" in redacted
    assert "+49" not in redacted
    assert "[TELEFON]" in redacted
    assert "DE1234" not in redacted
    assert "[IBAN]" in redacted
    assert "secret123" not in redacted


def test_is_jev_chat_allowed_defaults_to_false():
    with patch.dict(os.environ, {}, clear=True):
        assert not jev_client.is_jev_chat_allowed()

    with patch.dict(os.environ, {"JEV_ALLOW_CHAT_ROUTING": "true"}):
        assert jev_client.is_jev_chat_allowed()

    with patch.dict(os.environ, {"JEV_ALLOW_CHAT_ROUTING": "false"}):
        assert not jev_client.is_jev_chat_allowed()


# ------------------------------------------------------------------------------
# 3. Chat-Guard: StepBudget & Harte Loop-Bremse
# ------------------------------------------------------------------------------

def test_step_budget_limits_iterations():
    budget = chat_guard.StepBudget(max_iterations=3, max_estimated_tokens=5000)
    assert budget.step(100) == 1
    assert budget.step(100) == 2
    assert budget.step(100) == 3
    # 4. Schritt überschreitet das Limit
    with pytest.raises(chat_guard.ChatBudgetExceeded, match="max_iterations=3"):
        budget.step(100)


def test_step_budget_limits_token_growth():
    budget = chat_guard.StepBudget(max_iterations=10, max_estimated_tokens=100)
    # 100 Tokens ~= 400 Zeichen
    budget.step(extra_text_len=200)
    with pytest.raises(chat_guard.ChatBudgetExceeded, match="Token-Budget-Bremse"):
        budget.step(extra_text_len=300)


def test_slim_payloads():
    fat_task = {
        "notion_page_id": "uuid-1234",
        "kind": "To-Do",
        "title": "Dokumente fertigstellen",
        "priority": "Hoch",
        "due_date": "2026-11-10",
        "next_step": "Kapitel 4 überarbeiten und Formatierung prüfen",
        "status": "In Bearbeitung",
        "raw_json_dump": {"huge": "payload", "tokens": 500},
    }
    slim = chat_guard.slim_task(fat_task)
    assert "notion_page_id" not in slim
    assert "raw_json_dump" not in slim
    assert slim["title"] == "Dokumente fertigstellen"
    assert slim["priority"] == "Hoch"
    assert slim["due"] == "2026-11-10"


# ------------------------------------------------------------------------------
# 4. LLM-Core Thinking & Tiering Regeln
# ------------------------------------------------------------------------------

def test_llm_core_resolve_task_model():
    model, thinking = llm_core.resolve_task_model("rss_summary")
    assert model == "gemini-3.5-flash-lite"
    assert thinking == "minimal"

    c_model, c_thinking = llm_core.resolve_task_model("chat_complex")
    assert c_model == "gemini-3.8-flash"
    assert c_thinking == "low"  # Niemals minimal bei 3.8 Flash!


def test_llm_core_prevents_minimal_on_gemini_3_8_flash():
    # Wenn fälschlicherweise 3.8 Flash mit minimal konfiguriert wird, muss es auf low korrigiert werden
    cfg = {
        "llm": {
            "tiers": {
                "custom_task": {"model": "gemini-3.8-flash", "thinking_level": "minimal"}
            }
        }
    }
    model, thinking = llm_core.resolve_task_model("custom_task", cfg)
    assert model == "gemini-3.8-flash"
    assert thinking == "low"
