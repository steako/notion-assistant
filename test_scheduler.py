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


def test_perform_self_update_restarts_when_already_up_to_date():
    """Prüft, dass perform_self_update auch bei 'Already up to date' needs_restart=True zurückgibt."""
    import subprocess
    proc_mock = MagicMock()
    proc_mock.returncode = 0
    proc_mock.stdout = "Already up to date."
    proc_mock.stderr = ""

    with patch("subprocess.run", return_value=proc_mock):
        needs_restart, msg = assistant.perform_self_update()
        assert needs_restart is True
        assert "bereits auf dem neuesten Stand" in msg
        assert "Starte Dienst neu" in msg


def test_process_telegram_status_command():
    """Prüft, dass /status direkt den Assistenten-Statusblock sendet und nicht an den Chat-Router fällt."""
    cfg = assistant.load_config()
    state = {"tasks": [], "items": [], "calendar_events": []}
    mock_updates = {
        "ok": True,
        "result": [
            {
                "update_id": 999,
                "message": {
                    "chat": {"id": 12345},
                    "text": "/status",
                }
            }
        ]
    }

    with patch("assistant.load_config", return_value=cfg), \
         patch("assistant.save_local_state"), \
         patch("assistant._send_telegram_message") as mock_send, \
         patch("assistant.classify_intent") as mock_classify:

        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "mock_token", "TELEGRAM_ALLOWED_USERS": "12345"}):
            with patch("urllib.request.urlopen") as mock_url:
                import io, json
                mock_resp = io.BytesIO(json.dumps(mock_updates).encode("utf-8"))
                mock_url.return_value.__enter__.return_value.read.return_value = mock_resp.getvalue()
                mock_url.return_value.__enter__.return_value.status = 200

                assistant.process_telegram_updates(cfg, state, notion=None)

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][2]
        assert "Assistent & Scheduler Status" in sent_text
        # Intent Router darf für /status NIEMALS aufgerufen werden!
        mock_classify.assert_not_called()


def test_rss_bot_handle_status_without_rss_monitor():
    """Stellt sicher, dass rss_bot.handle_status auch funktioniert, wenn rss_monitor None ist."""
    import rss_bot
    with patch.object(rss_bot, "rss_monitor", None), \
         patch.object(rss_bot, "query_notion_knowledge", return_value=[]), \
         patch.object(rss_bot, "load_rss_state", return_value={"seen": {}}):

        status_msg = rss_bot.handle_status("12345")
        assert "Wissensspeicher- & RSS-Status" in status_msg
        assert "Fehler: rss_monitor nicht geladen" not in status_msg

        themen_msg = rss_bot.handle_themen("12345")
        assert "Konfigurierte Themengebiete" in themen_msg


