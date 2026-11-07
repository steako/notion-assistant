"""
TypeSafe AI (Jev) - System-One Decision Client.

Implementiert schnelle, deterministische Entscheidungen (Noul, Choice, Score)
ohne Textgenerierung über den Endpoint POST https://api.typesafe.ai/v1/systemone.

Eigenschaften:
- Gepinnte Modellversion (Standard: jev-1.13.0, steuerbar per JEV_MODEL).
- Strenger Timeout und max. 1 Retry bei transienten Fehlern (429/5xx).
- Datenschutz-Guardrail: Prüft JEV_ALLOW_CHAT_ROUTING und anonymisiert PII.
- Fallback-Mechanismus (kontrollierter Abbruch oder Übergabe an LLM).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
import requests

log = logging.getLogger("jev_client")

DEFAULT_JEV_MODEL = "jev-1.13.0"
DEFAULT_JEV_TIMEOUT_S = 10
DEFAULT_JEV_BASE_URL = "https://api.typesafe.ai/v1"


class JevError(Exception):
    """Basis-Fehlerklasse für TypeSafe Jev API Fehler."""

    def __init__(self, message: str, status_code: int | None = None, response_body: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class JevClient:
    """Client für TypeSafe AI System One (Jev)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = (api_key or os.environ.get("TYPESAFE_API_KEY", "")).strip()
        self.model = model or os.environ.get("JEV_MODEL", DEFAULT_JEV_MODEL).strip() or DEFAULT_JEV_MODEL
        try:
            self.timeout = int(timeout or os.environ.get("JEV_TIMEOUT_S", DEFAULT_JEV_TIMEOUT_S))
        except (ValueError, TypeError):
            self.timeout = DEFAULT_JEV_TIMEOUT_S
        self.base_url = (base_url or os.environ.get("JEV_BASE_URL", DEFAULT_JEV_BASE_URL)).rstrip("/")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def system_one(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, dict[str, Any]],
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """
        Sendet einen System-One-Aufruf an die TypeSafe API.

        Gibt das JSON-Dictionary mit 'answers', 'model' und 'usage' zurück.
        Wirft JevError bei Fehlschlägen.
        """
        if not self.api_key:
            raise JevError("TYPESAFE_API_KEY ist nicht konfiguriert.")

        payload = {
            "model": model_override or self.model,
            "state": state,
            "questions": questions,
        }

        url = f"{self.base_url}/systemone"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "NotionAssistant-Jev/1.0",
        }

        # Bis zu 2 Versuche (1 Retry bei transientem 429 oder 5xx)
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()

                # Bei Client-Fehlern (400, 401, 403, 404) nicht wiederholen
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise JevError(
                        f"TypeSafe API Client-Fehler HTTP {resp.status_code}: {resp.text}",
                        status_code=resp.status_code,
                        response_body=resp.text,
                    )

                if attempt == 0 and resp.status_code in (429, 500, 502, 503, 504):
                    log.warning("Jev HTTP %d auf Versuch 1 – versuche einmaligen Retry...", resp.status_code)
                    continue

                raise JevError(
                    f"TypeSafe API Server-Fehler HTTP {resp.status_code}: {resp.text}",
                    status_code=resp.status_code,
                    response_body=resp.text,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt == 0:
                    log.warning("Jev Netzwerk-Fehler (%s) – versuche einmaligen Retry...", exc)
                    continue
                break

        raise JevError(f"TypeSafe API Anfrage fehlgeschlagen: {last_exc}")

    # --------------------------------------------------------------------------
    # Komfort-Helfer für die 3 System-One Primitives
    # --------------------------------------------------------------------------

    def evaluate_noul(
        self,
        state: str | dict[str, Any],
        instructions: str,
        criteria_true: str | None = None,
        criteria_false: str | None = None,
        question_id: str = "q",
    ) -> float:
        """Führt eine binäre Noul-Bewertung aus und gibt die Wahrscheinlichkeit (0.0 bis 1.0) zurück."""
        q_def: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria_true or criteria_false:
            q_def["criteria"] = {
                "true": criteria_true or "Yes",
                "false": criteria_false or "No",
            }

        res = self.system_one(state=state, questions={question_id: q_def})
        answer = res.get("answers", {}).get(question_id, {})
        val = answer.get("noul")
        if val is None:
            raise JevError(f"Ungültige Noul-Antwort: {res}")
        return float(val)

    def evaluate_choice(
        self,
        state: str | dict[str, Any],
        instructions: str,
        options: dict[str, str | None],
        question_id: str = "choice",
    ) -> tuple[str, float, dict[str, float]]:
        """
        Führt eine Choice-Klassifikation aus.
        Gibt (selected_choice, confidence, probabilities_dict) zurück.
        """
        q_def: dict[str, Any] = {
            "type": "choice",
            "instructions": instructions,
            "criteria": options,
        }
        res = self.system_one(state=state, questions={question_id: q_def})
        answer = res.get("answers", {}).get(question_id, {})
        choice = answer.get("choice")
        conf = float(answer.get("confidence", 0.0))
        probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
        if choice is None:
            raise JevError(f"Ungültige Choice-Antwort: {res}")
        return str(choice), conf, probs

    def evaluate_score(
        self,
        state: str | dict[str, Any],
        instructions: str,
        levels: list[str],
        question_id: str = "score",
    ) -> tuple[float, float, dict[str, float]]:
        """
        Führt ein Score-Rating gegen geordnete Rubrikstufen durch.
        Gibt (score, confidence, probabilities_dict) zurück.
        """
        q_def: dict[str, Any] = {
            "type": "score",
            "instructions": instructions,
            "criteria": levels,
        }
        res = self.system_one(state=state, questions={question_id: q_def})
        answer = res.get("answers", {}).get(question_id, {})
        sc = float(answer.get("score", 0.0))
        conf = float(answer.get("confidence", 0.0))
        probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()}
        return sc, conf, probs


# ------------------------------------------------------------------------------
# Datenschutz-Guardrail für Jev (US-Hosting)
# ------------------------------------------------------------------------------

EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"(\+?[0-9]{1,3}[-.\s]?)?(\(?\d{3,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,9}")
IBAN_REGEX = re.compile(r"\b[A-Z]{2}[0-9]{2}(?:[ ]?[0-9]{4}){4}(?:[ ]?[0-9]{1,4})?\b")


def redact_pii_for_external_router(text: str, max_chars: int = 200) -> str:
    """
    Maskiert sensible personenbezogene Daten (E-Mails, Telefonnummern, IBANs)
    und kürzt den Text auf kurze Routing-Metadaten vor Weitergabe an externe US-Dienste.
    """
    if not text:
        return ""
    cleaned = EMAIL_REGEX.sub("[EMAIL]", text)
    cleaned = IBAN_REGEX.sub("[IBAN]", cleaned)
    cleaned = PHONE_REGEX.sub("[TELEFON]", cleaned)
    # Entferne evtl. Tokens / Passwörter / Secrets
    cleaned = re.sub(r"(?i)(passwort|password|geheim|secret|token)\s*[:=]\s*\S+", r"\1: [REDACTED]", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:max_chars]


def is_jev_chat_allowed() -> bool:
    """
    Prüft, ob Jev im Chat-Modus genutzt werden darf.
    Standard: FALSE (Opt-in erforderlich, da Notion/Chat persönliche Daten enthalten).
    """
    val = os.environ.get("JEV_ALLOW_CHAT_ROUTING", "false").strip().lower()
    return val in ("true", "1", "yes", "on")
