import datetime as dt
import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import assistant
import coach_bot
import llm_core
import parent_bot
from core import coaching_core
from core.storage import StateStore


def test_llm_tiering_and_hydrogen_fallback():
    cfg = assistant.load_config()
    m_chat, t_chat = llm_core.resolve_task_model("coach_chat", cfg)
    m_deep, t_deep = llm_core.resolve_task_model("coach_deep", cfg)
    assert m_chat == "gemini-3.5-flash-lite"
    assert t_chat == "minimal"
    assert m_deep == "gemini-4-hydrogen"
    assert t_deep == "low"

    # Fallback von gemini-4-hydrogen -> gemini-3.8-flash wenn API Fehler wirft
    mock_client = MagicMock()
    ok_resp = MagicMock()
    ok_resp.text = "Fallback OK"
    mock_client.models.generate_content.side_effect = [
        RuntimeError("404 Model gemini-4-hydrogen not found"),
        ok_resp,
    ]
    res = llm_core.execute_gemini_call("Hallo", mock_client, "coach_deep", "gemini-4-hydrogen", "low")
    assert res.text == "Fallback OK"
    assert mock_client.models.generate_content.call_count == 2
    second_call_kwargs = mock_client.models.generate_content.call_args_list[1].kwargs
    assert second_call_kwargs["model"] == "gemini-3.8-flash"


def test_storage_coaching_logs(tmp_path):
    store = StateStore(tmp_path / "test.db")
    lid = store.add_coaching_log(
        bot="parent",
        kind="midday_checkin",
        topic="ADHS-Übergang",
        prompt_summary="Ruhiger Start in den Nachmittag",
        content="Kurzer Atem-Reset vor den Hausaufgaben.",
        stress_level=2,
    )
    assert lid > 0
    store.set_coaching_feedback(lid, "stress_3", stress_level=3)
    logs = store.get_recent_coaching_logs("parent", days=7)
    assert len(logs) == 1
    assert logs[0]["feedback"] == "stress_3"
    assert logs[0]["stress_level"] == 3


def test_goals_and_lpp_docx_zip_import(tmp_path):
    store = StateStore(tmp_path / "goals.db")
    g1 = coaching_core.add_goal(store, "Maximal 2 Hauptprojekte parallel", area="Hochschule/Lehre")
    g2 = coaching_core.add_goal(store, "Abends 30 Min Handy-Pause", area="Familie/Privat")
    assert len(coaching_core.list_goals(store)) == 2
    assert g1["id"] != g2["id"]

    done = coaching_core.complete_goal(store, "1")
    assert done is not None and done["status"] == "done"
    assert len(coaching_core.list_goals(store)) == 1

    # Synthetische DOCX in ZIP verpacken und LPP-Importer testen
    docx_buf = io.BytesIO()
    with zipfile.ZipFile(docx_buf, "w") as zdoc:
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            "<w:p><w:r><w:t>Ordnungsorientierung (G)</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>IST: minimal</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>SOLL: hoch</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        zdoc.writestr("word/document.xml", xml)

    zip_path = tmp_path / "lpp_test.zip"
    with zipfile.ZipFile(zip_path, "w") as zouter:
        zouter.writestr("Feedback.docx", docx_buf.getvalue())

    with patch.object(coaching_core, "COACH_PROFILE_PATH", tmp_path / "coach_profile.json"):
        prof, cnt = coaching_core.import_lpp_into_profile(zip_path, cfg={}, store=store)
        assert cnt == 1
        assert "Feedback.docx" in prof["lpp_summary"]["sources"]
        assert any("Ordnungsorientierung" in g for g in prof["lpp_summary"]["ist_soll_gaps"])


def test_coach_bot_tokens_isolated_and_impulse_flow(tmp_path):
    with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "main_tok", "TELEGRAM_COACHBOT_TOKEN": ""}, clear=False):
        assert coach_bot.get_active_token() == ""
    with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "main_tok", "TELEGRAM_PARENTBOT_TOKEN": ""}, clear=False):
        assert parent_bot.get_active_token() == ""

    store = StateStore(tmp_path / "coach.db")
    cfg = assistant.load_config()
    mock_llm_resp = {
        "theme": "Umsetzung & WIP-Limit",
        "impulse_title": "Ein Thema vom Denkraum in die Tat",
        "impulse_text": "Du hast viele starke Ideen. Wähle heute bewusst eine davon aus.",
        "micro_action": "Definiere den nächsten 15-Minuten-Schritt.",
        "suggested_task_title": "Pilot-Schritt festlegen",
        "suggested_block_title": "Fokus-Block",
    }

    with patch.object(coach_bot, "get_store", return_value=store), \
         patch.object(coach_bot, "_get_notion_client", return_value=None), \
         patch.object(coach_bot, "_call_llm_json", return_value=mock_llm_resp) as mock_llm, \
         patch.object(coach_bot, "send_message", return_value=True) as mock_send, \
         patch("core.coaching_core.fetch_day_load_context", return_value={"date": "2026-11-15", "private_events_today": [], "work_busy_count": 0, "free_slots": []}):

        ok, html_msg = coach_bot.send_daily_impulse(chat_id="999", cfg=cfg, token="coach_tok")
        assert ok is True
        assert "Ein Thema vom Denkraum in die Tat" in html_msg
        assert mock_llm.call_args.kwargs["task"] == "coach_deep"
        mock_send.assert_called_once()
        # Inline keyboard sollte angehängt sein
        assert mock_send.call_args.kwargs["reply_markup"] is not None

        # Feedback-Callback testen
        kb = mock_send.call_args.kwargs["reply_markup"]["inline_keyboard"]
        like_cb_data = kb[1][1]["callback_data"]
        with patch("integrations.telegram_ui.answer_callback_query") as mock_ans:
            coach_bot.handle_callback_query(
                "coach_tok",
                {"id": "cb1", "data": like_cb_data, "message": {"chat": {"id": 999}}},
                cfg,
            )
            mock_ans.assert_called_once()
        logs = store.get_recent_coaching_logs("coach", days=1)
        assert logs[0]["feedback"] == "helpful"


def test_parent_bot_phases_akut_and_privacy(tmp_path):
    store = StateStore(tmp_path / "parent.db")
    cfg = assistant.load_config()
    mock_parent_json = {
        "topic": "ADHS-Übergang",
        "headline": "Übergang Schule nach Hause entschärfen",
        "impulse_text": "Erst 15 Minuten Ankommens-Pause ohne Fragen oder Anforderungen.",
        "ready_phrase": "Wir essen erst was Kleines, dann schauen wir in Ruhe auf den Plan.",
        "checkin_question": "Wie ist dein Akku für den Nachmittag?",
        "optional_task_title": "",
    }

    with patch.object(parent_bot, "get_store", return_value=store), \
         patch.object(parent_bot, "_call_llm_json", return_value=mock_parent_json), \
         patch.object(parent_bot, "send_message", return_value=True) as mock_send, \
         patch("integrations.notion_client.NotionClient.create_inbox_page") as mock_notion_inbox:

        for phase in ("morning", "midday", "evening"):
            ok, msg = parent_bot.send_parent_checkin(phase=phase, chat_id="777", cfg=cfg, token="parent_tok")
            assert ok is True
            assert "ADHS-Übergang" in msg

        assert mock_send.call_count == 3
        # Datenschutz: Sensible Eltern-Checkins schreiben niemals automatisch nach Notion
        mock_notion_inbox.assert_not_called()

        # Stresslevel-Callback testen
        first_kb = mock_send.call_args_list[1].kwargs["reply_markup"]["inline_keyboard"]
        stress_cb = first_kb[0][2]["callback_data"]  # p_stress_3:...
        with patch("integrations.telegram_ui.answer_callback_query"):
            parent_bot.handle_callback_query(
                "parent_tok",
                {"id": "cb2", "data": stress_cb, "message": {"chat": {"id": 777}}},
                cfg,
            )
        logs = store.get_recent_coaching_logs("parent", days=1)
        assert any(l.get("stress_level") == 3 for l in logs)

    # /akut Aufruf testen (rein lokal + coach_deep)
    with patch.object(parent_bot, "get_store", return_value=store), \
         patch.object(parent_bot, "_call_llm_text", return_value="1. Atmen\n2. Ruhig bleiben\n3. 1 Schritt") as mock_txt:
        res = parent_bot.generate_akut_help("Hausaufgaben-Streit eskaliert gerade", cfg, store)
        assert "Atmen" in res
        assert mock_txt.call_args.kwargs["task"] == "coach_deep"


def test_scheduler_runs_coach_and_parent_jobs(tmp_path):
    assistant.SCHEDULER_RETRY_COOLDOWN.clear()
    cfg = {
        "schedule": {
            "enabled": True,
            "timezone": "Europe/Berlin",
            "jobs": [
                {"id": "coach_morning", "time": "00:00", "catchup_until": "23:59"},
                {"id": "parent_midday", "time": "00:00", "catchup_until": "23:59"},
            ],
        }
    }
    running = {}
    saved_states = []

    with patch.dict("os.environ", {"TELEGRAM_COACHBOT_TOKEN": "c_tok", "TELEGRAM_PARENTBOT_TOKEN": "p_tok"}), \
         patch("assistant._load_scheduler_state", return_value={}), \
         patch("assistant._save_scheduler_state", side_effect=lambda s: saved_states.append(dict(s))), \
         patch("coach_bot.send_daily_impulse", return_value=(True, "ok")) as mock_coach, \
         patch("parent_bot.send_parent_checkin", return_value=(True, "ok")) as mock_parent:

        assistant.run_scheduled_jobs(cfg, running, state={"tasks": [], "items": []})
        for t in list(running.values()):
            t.join(timeout=3)

        mock_coach.assert_called_once()
        mock_parent.assert_called_once_with(phase="midday", cfg=cfg, token="p_tok")
        assert len(saved_states) >= 1
