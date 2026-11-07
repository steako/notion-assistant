"""
Zentraler LLM- und Gemini-Core mit Modell-Tiering, Thinking-Level-Steuerung
und strukturiertem Token-Logging.

Tasks & Tiers:
- 'rss_summary': gemini-3.5-flash-lite, thinking minimal (schnell, günstig, deterministisch)
- 'extract':     gemini-3.5-flash-lite, thinking minimal
- 'intent':      gemini-3.5-flash-lite, thinking minimal
- 'chat_simple': gemini-3.5-flash-lite, thinking minimal
- 'chat_complex': gemini-3.8-flash, thinking low (Eskalationspfad für mehrstufige Analysen)

Gemini 3.x Regeln:
- Für 3.5 Flash-Lite: thinking_level = "minimal" | "low" | "medium" | "high"
- Für 3.8 Flash: thinking_level = "low" | "medium" | "high" (MINIMAL ist nicht erlaubt und erzeugt einen 400-Fehler)
- Niemals thinking_budget zusammen mit thinking_level senden
- Keine veralteten Sampling-Parameter (frequency_penalty, etc.)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib import request

log = logging.getLogger("llm_core")

# Standard-Modelle
MODEL_FLASH_LITE = "gemini-3.5-flash-lite"
MODEL_FLASH = "gemini-3.8-flash"

# Task-Zuordnungen
TASK_CONFIGS: dict[str, dict[str, str]] = {
    "rss_summary": {"model": MODEL_FLASH_LITE, "thinking_level": "minimal"},
    "extract": {"model": MODEL_FLASH_LITE, "thinking_level": "minimal"},
    "intent": {"model": MODEL_FLASH_LITE, "thinking_level": "minimal"},
    "chat_simple": {"model": MODEL_FLASH_LITE, "thinking_level": "minimal"},
    "chat_complex": {"model": MODEL_FLASH, "thinking_level": "low"},
    "default": {"model": MODEL_FLASH_LITE, "thinking_level": "minimal"},
}


def resolve_task_model(task: str, cfg: dict[str, Any] | None = None) -> tuple[str, str]:
    """
    Ermittelt das optimale Modell und das erlaubte Thinking-Level für einen Task.
    Gibt (model_name, thinking_level) zurück.
    """
    # 1. Config-Overrides
    cfg = cfg or {}
    llm_cfg = cfg.get("llm", {})
    tiers = llm_cfg.get("tiers", {})

    task_spec = tiers.get(task) or TASK_CONFIGS.get(task, TASK_CONFIGS["default"])
    model = task_spec.get("model", MODEL_FLASH_LITE)
    thinking_level = task_spec.get("thinking_level", "minimal")

    # WICHTIGE SICHERHEITSREGEL: 3.8 Flash akzeptiert KEIN "minimal"
    if "3.8-flash" in model and thinking_level == "minimal":
        thinking_level = "low"

    return model, thinking_level


def build_gemini_config(
    model: str,
    thinking_level: str | None = None,
    system_instruction: str | None = None,
    response_mime_type: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Erzeugt ein sauberes types.GenerateContentConfig für die neue google-genai SDK."""
    from google.genai import types

    # Validiere thinking_level gegen das Ziel-Modell
    clean_level = thinking_level or "minimal"
    if "3.8-flash" in model and clean_level == "minimal":
        clean_level = "low"

    thinking_config = types.ThinkingConfig(thinking_level=clean_level)

    kwargs: dict[str, Any] = {
        "thinking_config": thinking_config,
    }
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    if response_mime_type:
        kwargs["response_mime_type"] = response_mime_type
    if temperature is not None:
        kwargs["temperature"] = temperature

    return types.GenerateContentConfig(**kwargs)


def log_gemini_usage(task: str, model: str, thinking_level: str, response: Any) -> None:
    """Loggt Token-Verbrauch inkl. Thinking-Tokens strukturiert."""
    try:
        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt_tok = getattr(usage, "prompt_token_count", 0) or 0
            cand_tok = getattr(usage, "candidates_token_count", 0) or 0
            total_tok = getattr(usage, "total_token_count", 0) or 0
            thought_tok = getattr(usage, "thoughts_token_count", 0) or 0
            log.info(
                "gemini_call task=%s model=%s thinking=%s in_tok=%d out_tok=%d thought_tok=%d total_tok=%d",
                task, model, thinking_level, prompt_tok, cand_tok, thought_tok, total_tok,
            )
            return
    except Exception:
        pass
    log.info("gemini_call task=%s model=%s thinking=%s status=ok", task, model, thinking_level)


def execute_gemini_call(
    prompt: str | list[Any],
    client: Any,
    task: str,
    model: str,
    thinking_level: str,
    system_instruction: str | None = None,
    response_mime_type: str | None = None,
    temperature: float | None = None,
) -> Any:
    """Führt den eigentlichen Call gegen client.models.generate_content aus."""
    config = build_gemini_config(
        model=model,
        thinking_level=thinking_level,
        system_instruction=system_instruction,
        response_mime_type=response_mime_type,
        temperature=temperature,
    )
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    log_gemini_usage(task, model, thinking_level, resp)
    return resp


def call_ollama_json_safe(
    prompt: str,
    system: str | None,
    ollama_cfg: dict[str, Any],
    model_override: str | None = None,
) -> dict[str, Any]:
    """Sicherer lokaler Ollama-Aufruf mit Timeout und korrekten Imports."""
    base_url = os.environ.get("OLLAMA_BASE_URL", ollama_cfg.get("base_url", "http://localhost:11434"))
    # Wenn ein Gemini-Modell übergeben wurde, nicht an Ollama weiterreichen
    if model_override and ("gemini" in model_override.lower()):
        model_override = None

    models = [model_override] if model_override else [
        os.environ.get("OLLAMA_MODEL", ollama_cfg.get("model", "devstral:24b")),
        *ollama_cfg.get("fallback_models", []),
    ]
    models = [m for m in models if m and "gemini" not in m.lower()]
    if not models:
        models = ["qwen2.5-coder:7b"]

    last_exc: Exception | None = None
    # Maximal die ersten 2 Modelle versuchen (Schutz vor Timeout-Kaskaden)
    for m in models[:2]:
        payload = {
            "model": m,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": system or "Antworte ausschließlich mit einem validen JSON-Objekt."},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            req = request.Request(
                f"{base_url.rstrip('/')}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=90) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body.get("message", {}).get("content", "{}")
                return json.loads(content)
        except Exception as exc:
            last_exc = exc
            log.warning("Ollama-Modell %s fehlgeschlagen: %s", m, exc)
            continue
    raise RuntimeError(f"Kein lokales Ollama-Modell erreichbar: {last_exc}")
