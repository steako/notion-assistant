"""
Gehärteter Notion-API-Client mit Retries (429/5xx), Pagination und Schema-Schutz.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib import error, request

log = logging.getLogger("integrations.notion")


class NotionClient:
    """Robuster Notion-API-Client für den Hintergrund- und Dialogbetrieb."""

    def __init__(self, token: str) -> None:
        self.token = token
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            req = request.Request(
                f"{self.base_url}{path}",
                data=data,
                headers=self.headers,
                method=method,
            )
            try:
                with request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempt < max_retries - 1:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    wait_s = float(retry_after) if retry_after and str(retry_after).isdigit() else (1.5 * (attempt + 1))
                    log.warning("Notion 429 Rate Limit – warte %.1fs (Versuch %d/%d)...", wait_s, attempt + 1, max_retries)
                    time.sleep(min(wait_s, 10.0))
                    continue
                if exc.code in (500, 502, 503, 504) and attempt < max_retries - 1:
                    wait_s = 1.2 * (attempt + 1)
                    log.warning("Notion HTTP %d – Retry in %.1fs...", exc.code, wait_s)
                    time.sleep(wait_s)
                    continue
                raise RuntimeError(f"Notion API Fehler ({exc.code}): {details}") from exc
            except error.URLError as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise RuntimeError(f"Notion Netzwerkfehler: {last_exc}") from exc

        raise RuntimeError(f"Notion Anfrage fehlgeschlagen: {last_exc}")

    def _post_page_safe(self, payload: dict[str, Any], optional_prop_keys: list[str] | None = None) -> dict[str, Any]:
        """Legt eine Seite an; entfernt optionale Properties automatisch, falls die DB-Spalte nicht existiert."""
        try:
            return self._request("POST", "/pages", payload)
        except RuntimeError as exc:
            err_msg = str(exc)
            if optional_prop_keys and "400" in err_msg and "property" in err_msg.lower():
                clean_payload = json.loads(json.dumps(payload))
                props = clean_payload.get("properties", {})
                removed = False
                for k in optional_prop_keys:
                    if k in props:
                        props.pop(k, None)
                        removed = True
                if removed:
                    log.debug("Notion Schema-Fallback: entferne optionale Spalten %s", optional_prop_keys)
                    return self._request("POST", "/pages", clean_payload)
            raise

    def query_new_inbox_items(self, inbox_db_id: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            payload: dict[str, Any] = {
                "filter": {"property": "Status", "status": {"equals": "Neu"}},
                "page_size": 100,
            }
            if cursor:
                payload["start_cursor"] = cursor
            res = self._request("POST", f"/databases/{inbox_db_id}/query", payload)
            results.extend(res.get("results", []))
            if not res.get("has_more"):
                break
            cursor = res.get("next_cursor")
        return results

    def get_page_blocks_text(self, page_id: str, max_blocks: int = 60) -> str:
        res = self._request("GET", f"/blocks/{page_id}/children?page_size={min(max_blocks, 100)}")
        lines: list[str] = []
        for block in res.get("results", []):
            btype = block.get("type", "")
            bdata = block.get(btype, {})
            rich = bdata.get("rich_text", [])
            text = "".join(rt.get("plain_text", "") for rt in rich).strip()
            if text:
                lines.append(text)
        return "\n".join(lines).strip()

    def create_inbox_page(
        self,
        inbox_db_id: str,
        title: str,
        source: str,
        raw_text: str,
        summary: str,
        suggestions_text: str,
        area: str | None = None,
    ) -> str:
        props: dict[str, Any] = {
            "Name": {"title": [{"text": {"content": title[:100]}}]},
            "Status": {"status": {"name": "Ausgewertet"}},
            "Quelle": {"select": {"name": source[:100]}},
            "Transkript": {"rich_text": [{"text": {"content": raw_text[:1900]}}]},
            "Zusammenfassung": {"rich_text": [{"text": {"content": summary[:1900]}}]},
            "Vorschlaege": {"rich_text": [{"text": {"content": suggestions_text[:1900]}}]},
        }
        optional_keys: list[str] = []
        if area:
            props["Bereich"] = {"select": {"name": area[:80]}}
            optional_keys.append("Bereich")

        payload: dict[str, Any] = {
            "parent": {"database_id": inbox_db_id},
            "icon": {"type": "emoji", "emoji": "🎙️"},
            "properties": props,
        }
        res = self._post_page_safe(payload, optional_prop_keys=optional_keys)
        return res["id"]

    def update_inbox_page(
        self,
        page_id: str,
        raw_text: str,
        summary: str,
        suggestions_text: str,
    ) -> None:
        payload: dict[str, Any] = {
            "properties": {
                "Status": {"status": {"name": "Ausgewertet"}},
                "Transkript": {"rich_text": [{"text": {"content": raw_text[:1900]}}]},
                "Zusammenfassung": {"rich_text": [{"text": {"content": summary[:1900]}}]},
                "Vorschlaege": {"rich_text": [{"text": {"content": suggestions_text[:1900]}}]},
            }
        }
        self._request("PATCH", f"/pages/{page_id}", payload)

    def create_task_page(
        self,
        tasks_db_id: str,
        title: str,
        kind: str,
        priority: str,
        due_date: str | None,
        next_step: str,
        inbox_page_id: str | None = None,
        area: str | None = None,
        estimated_minutes: int | None = None,
    ) -> str:
        props: dict[str, Any] = {
            "Aufgabenbezeichnung": {"title": [{"text": {"content": title[:100]}}]},
            "Status": {"status": {"name": "Nicht begonnen"}},
            "Art": {"select": {"name": kind}},
            "Prioritaet": {"select": {"name": priority if priority in ("Hoch", "Mittel", "Niedrig") else "Mittel"}},
            "Schritt": {"rich_text": [{"text": {"content": next_step[:1900]}}]},
        }
        if due_date and len(due_date) == 10 and due_date != "null":
            props["Fällig"] = {"date": {"start": due_date}}
        if inbox_page_id:
            props["Notiz"] = {"relation": [{"id": inbox_page_id}]}

        optional_keys: list[str] = []
        if area:
            props["Bereich"] = {"select": {"name": area[:80]}}
            optional_keys.append("Bereich")
        if estimated_minutes and estimated_minutes > 0:
            props["Dauer (Min)"] = {"number": int(estimated_minutes)}
            optional_keys.append("Dauer (Min)")

        icon_emoji = "✅" if kind == "To-Do" else ("📅" if kind == "Erinnerung" else "💡")
        payload: dict[str, Any] = {
            "parent": {"database_id": tasks_db_id},
            "icon": {"type": "emoji", "emoji": icon_emoji},
            "properties": props,
        }
        res = self._post_page_safe(payload, optional_prop_keys=optional_keys)
        return res["id"]

    def query_open_tasks(self, tasks_db_id: str) -> list[dict[str, Any]]:
        tasks_out: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            payload: dict[str, Any] = {
                "filter": {
                    "and": [
                        {"property": "Status", "status": {"does_not_equal": "Erledigt"}},
                        {"property": "Status", "status": {"does_not_equal": "Archiviert"}},
                    ]
                },
                "page_size": 100,
            }
            if cursor:
                payload["start_cursor"] = cursor
            res = self._request("POST", f"/databases/{tasks_db_id}/query", payload)
            for page in res.get("results", []):
                p = page.get("properties", {})
                title_arr = p.get("Aufgabenbezeichnung", {}).get("title", [])
                title = "".join(t.get("plain_text", "") for t in title_arr).strip()
                if not title:
                    continue
                status = (p.get("Status", {}).get("status") or {}).get("name", "Nicht begonnen")
                kind = (p.get("Art", {}).get("select") or {}).get("name", "To-Do")
                prio = (p.get("Prioritaet", {}).get("select") or {}).get("name", "Mittel")
                due = (p.get("Fällig", {}).get("date") or {}).get("start")
                step_arr = p.get("Schritt", {}).get("rich_text", [])
                step = "".join(t.get("plain_text", "") for t in step_arr).strip()
                area = (p.get("Bereich", {}).get("select") or {}).get("name")
                dur_num = (p.get("Dauer (Min)", {}) or {}).get("number")
                tasks_out.append({
                    "notion_page_id": page["id"],
                    "kind": kind,
                    "title": title,
                    "priority": prio,
                    "due_date": due,
                    "next_step": step,
                    "status": status,
                    "area": area,
                    "estimated_minutes": int(dur_num) if dur_num else 30,
                })
            if not res.get("has_more"):
                break
            cursor = res.get("next_cursor")
        return tasks_out

    def query_recently_completed_tasks(self, tasks_db_id: str, limit: int = 30) -> list[dict[str, Any]]:
        """Lädt kürzlich erledigte Aufgaben (für Weekly Review)."""
        payload: dict[str, Any] = {
            "filter": {"property": "Status", "status": {"equals": "Erledigt"}},
            "sorts": [{"timestamp": "last_edited_time", "direction": "descending"}],
            "page_size": min(limit, 100),
        }
        res = self._request("POST", f"/databases/{tasks_db_id}/query", payload)
        out: list[dict[str, Any]] = []
        for page in res.get("results", []):
            p = page.get("properties", {})
            title_arr = p.get("Aufgabenbezeichnung", {}).get("title", [])
            title = "".join(t.get("plain_text", "") for t in title_arr).strip()
            if title:
                out.append({
                    "notion_page_id": page["id"],
                    "title": title,
                    "last_edited_time": page.get("last_edited_time", ""),
                })
        return out

    def mark_task_done(self, page_id: str) -> None:
        self._request("PATCH", f"/pages/{page_id}", {"properties": {"Status": {"status": {"name": "Erledigt"}}}})

    def reopen_task(self, page_id: str) -> None:
        self._request(
            "PATCH",
            f"/pages/{page_id}",
            {"archived": False, "properties": {"Status": {"status": {"name": "Nicht begonnen"}}}},
        )

    def update_task(
        self,
        page_id: str,
        due_date: str | None = None,
        priority: str | None = None,
        title: str | None = None,
    ) -> None:
        props: dict[str, Any] = {}
        if due_date and len(due_date) == 10 and due_date != "null":
            props["Fällig"] = {"date": {"start": due_date}}
        if priority and priority in ("Hoch", "Mittel", "Niedrig"):
            props["Prioritaet"] = {"select": {"name": priority}}
        if title:
            props["Aufgabenbezeichnung"] = {"title": [{"text": {"content": title[:100]}}]}
        if props:
            self._request("PATCH", f"/pages/{page_id}", {"properties": props})

    def archive_task(self, page_id: str) -> None:
        self._request("PATCH", f"/pages/{page_id}", {
            "archived": True,
            "properties": {"Status": {"status": {"name": "Archiviert"}}},
        })

    def create_document_page(
        self,
        documents_db_id: str,
        title: str,
        category: str,
        markdown_lines: list[str],
        emoji: str = "📄",
    ) -> str:
        """Erstellt eine strukturierte Seite in Notion Documents und gibt die URL zurück."""
        children: list[dict[str, Any]] = []
        for line in markdown_lines[:95]:
            s = line.strip()
            if not s:
                continue
            if s.startswith("## "):
                children.append({"object": "block", "type": "heading_2", "heading_2": {"rich_text": [{"type": "text", "text": {"content": s[3:][:1900]}}]}})
            elif s.startswith("### "):
                children.append({"object": "block", "type": "heading_3", "heading_3": {"rich_text": [{"type": "text", "text": {"content": s[4:][:1900]}}]}})
            elif s.startswith(("- ", "• ", "* ")):
                children.append({"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": s[2:][:1900]}}]}})
            else:
                children.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": s[:1900]}}]}})

        payload: dict[str, Any] = {
            "parent": {"database_id": documents_db_id},
            "icon": {"type": "emoji", "emoji": emoji},
            "properties": {
                "Name": {"title": [{"text": {"content": title[:100]}}]},
                "Category": {"select": {"name": category}},
            },
            "children": children,
        }
        res = self._post_page_safe(payload, optional_prop_keys=["Category"])
        return res.get("url", res["id"])

    def search_workspace(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        payload = {
            "query": query,
            "page_size": limit,
            "sort": {"direction": "descending", "timestamp": "last_edited_time"},
        }
        res = self._request("POST", "/search", payload)
        out: list[dict[str, Any]] = []
        for obj in res.get("results", []):
            obj_type = obj.get("object", "page")
            props = obj.get("properties", {})
            title = ""
            for pval in props.values():
                if pval.get("type") == "title":
                    title = "".join(t.get("plain_text", "") for t in pval.get("title", [])).strip()
                    break
            if not title:
                title_arr = obj.get("title", [])
                if isinstance(title_arr, list):
                    title = "".join(t.get("plain_text", "") for t in title_arr).strip()
            if not title:
                title = obj.get("url", "Unbenannt")
            out.append({
                "id": obj.get("id"),
                "type": obj_type,
                "title": title,
                "url": obj.get("url", ""),
            })
        return out
