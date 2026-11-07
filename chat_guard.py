"""
Chat-Guard & schlanke Notion-Payloads für den Assistenten.

1. StepBudget: Harte Loop-Bremse für Telegram/Notion-Chat
   - Begrenzt Iterationen (max_iterations, Standard: 3)
   - Begrenzt geschätztes Token-/Zeichen-Budget
   - Verhindert Endlosschleifen bei Tool-/Parsing-Fehlern

2. Schlanke Payloads:
   - Extrahiert nur die für den Prompt notwendigen Properties aus Notion-Objekten
     (keine vollen Page-Dumps mit internen IDs, URLs, Relationen, Metadaten)
   - Kürzt und filtert Kalenderevents, Notizen und Workspace-Suchtreffer
"""

from __future__ import annotations

import json
from typing import Any


class ChatBudgetExceeded(Exception):
    """Wird ausgelöst, wenn das Step- oder Token-Budget eines Chat-Turns überschritten wird."""
    pass


class StepBudget:
    """Kontrolliert die maximale Anzahl von Iterationen und Schritten pro Benutzer-Turn."""

    def __init__(self, max_iterations: int = 3, max_estimated_tokens: int = 12000) -> None:
        self.max_iterations = max_iterations
        self.max_estimated_tokens = max_estimated_tokens
        self.iterations = 0
        self.accumulated_chars = 0

    def step(self, extra_text_len: int = 0) -> int:
        """Registriert einen neuen Schritt/Call. Wirft ChatBudgetExceeded bei Überschreitung."""
        self.iterations += 1
        self.accumulated_chars += extra_text_len
        est_tokens = self.accumulated_chars // 4

        if self.iterations > self.max_iterations:
            raise ChatBudgetExceeded(
                f"Harte Loop-Bremse aktiv: max_iterations={self.max_iterations} überschritten (Turn {self.iterations})."
            )
        if est_tokens > self.max_estimated_tokens:
            raise ChatBudgetExceeded(
                f"Token-Budget-Bremse aktiv: geschätzt {est_tokens} Tokens überschreiten {self.max_estimated_tokens}."
            )
        return self.iterations

    def can_proceed(self) -> bool:
        return self.iterations < self.max_iterations


# ------------------------------------------------------------------------------
# Schlanke Notion- & Kontext-Payloads (deterministisch)
# ------------------------------------------------------------------------------

def slim_task(task: dict[str, Any]) -> dict[str, Any]:
    """Reduziert ein Notion-Task-Dict auf die für das Prompting wesentlichen Felder."""
    out: dict[str, Any] = {
        "title": (task.get("title") or "").strip(),
    }
    if task.get("kind"):
        out["kind"] = task["kind"]
    if task.get("priority") and task["priority"] != "Mittel":
        out["priority"] = task["priority"]
    if task.get("due_date"):
        out["due"] = task["due_date"]
    if task.get("next_step"):
        out["step"] = (task["next_step"] or "")[:120].strip()
    return out


def slim_calendar_event(event: dict[str, Any]) -> dict[str, Any]:
    """Reduziert ein Google-Kalender-Event auf Titel, Datum und Uhrzeit."""
    out: dict[str, Any] = {
        "title": (event.get("title") or "").strip(),
        "date": event.get("date") or "",
    }
    if event.get("start_time"):
        out["start"] = event["start_time"]
    if event.get("end_time"):
        out["end"] = event["end_time"]
    return out


def slim_recent_note(note: dict[str, Any]) -> dict[str, Any]:
    """Kompakte Notizdarstellung."""
    title = (note.get("title") or "").strip()
    summary = (note.get("summary") or "").strip()
    return {
        "title": title[:80],
        "summary": summary[:200],
    }


def slim_search_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Kompakter Notion-Suchtreffer (ohne Block-Arrays)."""
    return {
        "title": (hit.get("title") or "Unbenannt")[:100],
        "type": hit.get("type", "page"),
    }


def truncate_context_json(data: Any, max_chars: int = 3500) -> str:
    """Serialisiert ein Objekt zu JSON und begrenzt die Gesamtlänge strikt."""
    raw = json.dumps(data, ensure_ascii=False)
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 15] + '... [gekürzt]"}'
