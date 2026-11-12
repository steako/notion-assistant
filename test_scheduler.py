import datetime as dt
import time
from unittest.mock import MagicMock, patch
import pytest

import assistant


def test_generate_briefing_notify_empty_todos_no_index_error():
    """Stellt sicher, dass notify=True bei leeren To-Dos und Vorschlägen keinen IndexError wirft."""
    cfg = assistant.load_config()
    state = {
        "tasks": [],
        "items": [],
        "calendar_events": [{"title": "Zahnarzt morgen", "date": (dt.date.today() + dt.timedelta(days=1)).isoformat()}],
    }

    # Sollte ohne IndexError durchlaufen
    with patch("assistant.show_windows_notification") as mock_notify:
        text = assistant.generate_briefing(cfg, state, notify=True, mode="morning", send_telegram=False)
        assert "Guten Morgen" in text
        mock_notify.assert_called_once()
        args = mock_notify.call_args[0]
        assert "Zahnarzt morgen" in args[1]


def test_generate_briefing_notify_all_empty():
    """Stellt sicher, dass notify=True bei komplett leerem Zustand nicht abstürzt."""
    cfg = assistant.load_config()
    state = {"tasks": [], "items": [], "calendar_events": []}

    with patch("assistant.show_windows_notification") as mock_notify:
        text = assistant.generate_briefing(cfg, state, notify=True, mode="evening", send_telegram=False)
        assert "Abend-Checkliste" in text
        mock_notify.assert_not_called()


def test_generate_briefing_raise_on_telegram_failure():
    """Prüft, dass raise_on_telegram_failure=True eine Exception wirft, wenn Token/Chat fehlt oder Senden fehlschlägt."""
    cfg = assistant.load_config()
    state = {"tasks": [], "items": [], "calendar_events": []}

    with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}):
        with pytest.raises(RuntimeError, match="fehlt"):
            assistant.generate_briefing(cfg, state, notify=False, send_telegram=True, raise_on_telegram_failure=True)


def test_scheduler_retry_cooldown_on_failure():
    """Testet, dass fehlgeschlagene Jobs einen 5-Minuten Retry-Cooldown erhalten und nicht sofort den Status als erledigt markieren."""
    assistant.SCHEDULER_RETRY_COOLDOWN.clear()
    cfg = {
        "schedule": {
            "enabled": True,
            "timezone": "Europe/Berlin",
            "jobs": [
                {"id": "briefing_morning", "time": "00:00", "catchup_until": "23:59", "command": []}
            ]
        }
    }
    state = {"tasks": [], "items": [], "calendar_events": []}
    running = {}

    with patch("assistant._load_scheduler_state", return_value={}), \
         patch("assistant._save_scheduler_state") as mock_save, \
         patch("assistant.generate_briefing", side_effect=RuntimeError("Netzwerkfehler")):

        assistant.run_scheduled_jobs(cfg, running, state=state)
        # Warten bis der Worker-Thread beendet ist
        t = running.get("briefing_morning")
        if t:
            t.join(timeout=2)

        # Status darf NICHT als erledigt abgespeichert worden sein
        mock_save.assert_not_called()
        # Cooldown muss aktiv sein
        assert "briefing_morning" in assistant.SCHEDULER_RETRY_COOLDOWN
        assert assistant.SCHEDULER_RETRY_COOLDOWN["briefing_morning"] > time.time()
