"""
Kalender-Integration (Google Kalender + datenschutzkonformer Arbeitskalender Free/Busy + Slot-Engine).

Unterstützt:
1. Google Kalender Lesen & Schreiben via:
   - `gcalendar` CLI (wenn lokal vorhanden)
   - Google Calendar REST API mit OAuth2 Refresh-Token (für 24/7 GCP/Linux-Server)
   - Geheime iCal-URL (`GOOGLE_CALENDAR_ICS_URL`, geparst mit `icalendar` + `recurring-ical-events`)
2. Arbeitskalender (MS Teams / Outlook / Exchange) **ohne Sync vertraulicher Inhalte**:
   - `WORK_CALENDAR_FREEBUSY_ICS_URL` (z. B. Outlook "Nur Verfügbarkeit"-ICS)
   - `WORK_BUSY_SLOTS_FILE` (z. B. `work_busy_slots.json` aus Power Automate / lokalem Export)
   - Optional Microsoft Graph `getSchedule` (`MS_GRAPH_ACCESS_TOKEN` mit `Calendars.ReadBasic`)
   - Hartes Privacy-Gate: Alle Titel, Teilnehmer, Orte und Texte werden verworfen; es bleiben
     ausschließlich anonyme Zeitblöcke `(date, start_time, end_time)` für die Slot-Berechnung.
3. Deterministische Freie-Slot-Engine & Smart Time-Blocking-Vorschläge.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from core.storage import StateStore, atomic_write_json

log = logging.getLogger("integrations.calendar")

BASE_DIR = Path(__file__).resolve().parent.parent
GCAL_CACHE_PATH = BASE_DIR / "gcal_cache.json"

_OAUTH_CACHE: dict[str, Any] = {"token": "", "expires_at": 0.0}


def _normalize_hhmm(value: Any) -> str | None:
    if not value or str(value).lower() == "null":
        return None
    s = str(value).strip()
    if len(s) >= 5 and s[2] == ":" and s[:2].isdigit() and s[3:5].isdigit():
        return s[:5]
    return None


def _hhmm_to_minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":", 1)
    return int(hh) * 60 + int(mm)


def _minutes_to_hhmm(total: int) -> str:
    total = max(0, min(total, 23 * 60 + 59))
    return f"{total // 60:02d}:{total % 60:02d}"


def _berlin_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Europe/Berlin")
    except Exception:
        return None


def get_gcalendar_bin() -> str | None:
    bin_name = shutil.which("gcalendar") or shutil.which("gcalendar.cmd")
    if bin_name:
        return bin_name
    wrapper = Path(os.environ.get("USERPROFILE", "")) / ".gemini/config/plugins/gcalendar/skills/gcalendar/scripts/gcalendar.cmd"
    if wrapper.exists():
        return str(wrapper)
    return None


def get_google_oauth_access_token() -> str:
    """
    Liefert ein gültiges Google Calendar Access Token.
    Bevorzugt OAuth2 Refresh-Token (`GOOGLE_CALENDAR_CLIENT_ID`, `GOOGLE_CALENDAR_CLIENT_SECRET`,
    `GOOGLE_CALENDAR_REFRESH_TOKEN`), fällt sonst auf statisches `GOOGLE_CALENDAR_ACCESS_TOKEN` zurück.
    """
    now = time.time()
    if _OAUTH_CACHE["token"] and now < float(_OAUTH_CACHE["expires_at"]) - 60:
        return str(_OAUTH_CACHE["token"])

    client_id = os.environ.get("GOOGLE_CALENDAR_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GOOGLE_CALENDAR_CLIENT_SECRET", "").strip()
    refresh_token = os.environ.get("GOOGLE_CALENDAR_REFRESH_TOKEN", "").strip()

    if client_id and client_secret and refresh_token:
        body = parse.urlencode({
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }).encode("utf-8")
        req = request.Request(
            "https://oauth2.googleapis.com/token",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                tok = str(data.get("access_token") or "")
                exp = int(data.get("expires_in") or 3600)
                if tok:
                    _OAUTH_CACHE["token"] = tok
                    _OAUTH_CACHE["expires_at"] = now + exp
                    return tok
        except Exception as exc:
            log.warning("Google OAuth2 Token-Refresh fehlgeschlagen: %s", exc)

    return os.environ.get("GOOGLE_CALENDAR_ACCESS_TOKEN", "").strip()


# --------------------------------------------------------------------------------------
# ICS Parsing (icalendar + recurring-ical-events mit Fallback)
# --------------------------------------------------------------------------------------

def parse_ics_events(
    ics_text: str,
    start_date: dt.date,
    end_date: dt.date,
    source_label: str = "Google Kalender (ICS)",
    anonymize_for_privacy: bool = False,
) -> list[dict[str, Any]]:
    """
    Parst Termine aus ICS-Feeds inkl. Zeitzonen (`TZID`), Serienterminen (`RRULE`, `YEARLY`,
    `MONTHLY`, `WEEKLY`, `DAILY`), `EXDATE` und `RECURRENCE-ID`.
    Wenn `anonymize_for_privacy=True`, werden Titel/Orte/UIDs verworfen (für Arbeitskalender!).
    """
    tz_berlin = _berlin_tz()
    events: list[dict[str, Any]] = []

    try:
        import icalendar  # type: ignore
        import recurring_ical_events  # type: ignore

        cal = icalendar.Calendar.from_ical(ics_text)
        occurrences = recurring_ical_events.of(cal).between(
            start_date,
            end_date + dt.timedelta(days=1),
        )
        for comp in occurrences:
            status = str(comp.get("STATUS", "")).upper()
            if status == "CANCELLED":
                continue
            transp = str(comp.get("TRANSP", "")).upper()
            if anonymize_for_privacy and transp == "TRANSPARENT":
                # "Als Frei anzeigen" im Arbeitskalender blockiert keine Zeit
                continue

            dtstart = comp.get("DTSTART")
            if not dtstart:
                continue
            start_val = dtstart.dt
            dtend = comp.get("DTEND")
            end_val = dtend.dt if dtend else None

            if isinstance(start_val, dt.datetime):
                if start_val.tzinfo is None and tz_berlin:
                    start_val = start_val.replace(tzinfo=tz_berlin)
                elif tz_berlin:
                    start_val = start_val.astimezone(tz_berlin)

                if isinstance(end_val, dt.datetime):
                    if end_val.tzinfo is None and tz_berlin:
                        end_val = end_val.replace(tzinfo=tz_berlin)
                    elif tz_berlin:
                        end_val = end_val.astimezone(tz_berlin)
                else:
                    end_val = start_val + dt.timedelta(hours=1)

                ev_date = start_val.date()
                if not (start_date <= ev_date <= end_date):
                    continue
                st_str = start_val.strftime("%H:%M")
                en_str = end_val.strftime("%H:%M")
                if end_val.date() > ev_date and en_str == "00:00":
                    en_str = "23:59"
            elif isinstance(start_val, dt.date):
                # Ganztägiges Event (ggf. mehrtägig)
                ev_end_date = end_val if isinstance(end_val, dt.date) and not isinstance(end_val, dt.datetime) else (start_val + dt.timedelta(days=1))
                cur_d = max(start_val, start_date)
                last_d = min(ev_end_date - dt.timedelta(days=1), end_date)
                while cur_d <= last_d:
                    if anonymize_for_privacy:
                        events.append({
                            "title": "Arbeit: Belegt",
                            "date": cur_d.isoformat(),
                            "start_time": None,
                            "end_time": None,
                            "location": "",
                            "source": source_label,
                            "is_work_busy_only": True,
                        })
                    else:
                        events.append({
                            "id": str(comp.get("UID", "")),
                            "title": str(comp.get("SUMMARY", "Termin")),
                            "date": cur_d.isoformat(),
                            "start_time": None,
                            "end_time": None,
                            "location": str(comp.get("LOCATION", "")).strip(),
                            "link": "",
                            "source": source_label,
                            "synced_to_gcal": True,
                        })
                    cur_d += dt.timedelta(days=1)
                continue
            else:
                continue

            if anonymize_for_privacy:
                events.append({
                    "title": "Arbeit: Belegt",
                    "date": ev_date.isoformat(),
                    "start_time": st_str,
                    "end_time": en_str,
                    "location": "",
                    "source": source_label,
                    "is_work_busy_only": True,
                })
            else:
                events.append({
                    "id": str(comp.get("UID", "")),
                    "title": str(comp.get("SUMMARY", "Termin")),
                    "date": ev_date.isoformat(),
                    "start_time": st_str,
                    "end_time": en_str,
                    "location": str(comp.get("LOCATION", "")).strip(),
                    "link": "",
                    "source": source_label,
                    "synced_to_gcal": True,
                })

        events.sort(key=lambda x: (x.get("date") or "", x.get("start_time") or "00:00"))
        return events
    except ImportError:
        log.debug("icalendar/recurring_ical_events nicht installiert – nutze internen Fallback-Parser.")
    except Exception as exc:
        log.warning("icalendar Parser Hinweis (%s) – nutze Fallback-Parser.", exc)

    # Fallback-Parser (ohne externe Abhängigkeiten)
    unfolded = re.sub(r"\r?\n[ \t]", "", ics_text)

    def _parse_dt_fb(val: str) -> tuple[dt.date | None, str | None]:
        val = val.strip()
        if "T" not in val:
            try:
                return dt.date(int(val[:4]), int(val[4:6]), int(val[6:8])), None
            except Exception:
                return None, None
        is_utc = val.endswith("Z")
        clean = val.rstrip("Z")
        try:
            d_part, t_part = clean.split("T", 1)
            base = dt.datetime(
                int(d_part[:4]), int(d_part[4:6]), int(d_part[6:8]),
                int(t_part[:2]), int(t_part[2:4]),
                int(t_part[4:6]) if len(t_part) >= 6 else 0,
            )
            if is_utc and tz_berlin:
                base = base.replace(tzinfo=dt.timezone.utc).astimezone(tz_berlin)
            return base.date(), base.strftime("%H:%M")
        except Exception:
            return None, None

    in_event = False
    cur: dict[str, str] = {}
    for raw_line in unfolded.splitlines():
        line = raw_line.strip()
        if line == "BEGIN:VEVENT":
            in_event = True
            cur = {}
        elif line == "END:VEVENT" and in_event:
            in_event = False
            if cur.get("STATUS", "").upper() == "CANCELLED":
                continue
            ev_date, st_time = _parse_dt_fb(cur.get("DTSTART", ""))
            if not ev_date or not (start_date <= ev_date <= end_date):
                continue
            _, en_time = _parse_dt_fb(cur.get("DTEND", "")) if cur.get("DTEND") else (None, None)
            if anonymize_for_privacy:
                events.append({
                    "title": "Arbeit: Belegt",
                    "date": ev_date.isoformat(),
                    "start_time": st_time,
                    "end_time": en_time,
                    "location": "",
                    "source": source_label,
                    "is_work_busy_only": True,
                })
            else:
                events.append({
                    "id": cur.get("UID", ""),
                    "title": cur.get("SUMMARY", "Termin").replace("\\,", ",").replace("\\;", ";").replace("\\n", " "),
                    "date": ev_date.isoformat(),
                    "start_time": st_time,
                    "end_time": en_time,
                    "location": cur.get("LOCATION", "").replace("\\,", ",").strip(),
                    "link": "",
                    "source": source_label,
                    "synced_to_gcal": True,
                })
        elif in_event and ":" in line:
            k_part, val = line.split(":", 1)
            cur[k_part.split(";", 1)[0].upper()] = val.strip()

    events.sort(key=lambda x: (x.get("date") or "", x.get("start_time") or "00:00"))
    return events


# --------------------------------------------------------------------------------------
# ICS Export & Google Calendar CRUD
# --------------------------------------------------------------------------------------

def export_ics_event(ev: dict[str, Any], outbox_dir: Path | None = None) -> Path:
    """Schreibt eine standardkonforme .ics-Datei mit deterministischer SHA1-UID."""
    outbox = outbox_dir or (BASE_DIR / "calendar_outbox")
    outbox.mkdir(parents=True, exist_ok=True)
    title = ev.get("title") or "Termin"
    date_str = ev.get("date") or dt.date.today().isoformat()
    start_t = _normalize_hhmm(ev.get("start_time"))
    end_t = _normalize_hhmm(ev.get("end_time"))
    location = ev.get("location") or ""
    description = ev.get("description") or ""

    date_compact = date_str.replace("-", "")
    if start_t:
        st_compact = start_t.replace(":", "") + "00"
        if end_t:
            et_compact = end_t.replace(":", "") + "00"
        else:
            hour = (int(start_t[:2]) + 1) % 24
            et_compact = f"{hour:02d}{start_t[3:5]}00"
        dtstart_line = f"DTSTART:{date_compact}T{st_compact}"
        dtend_line = f"DTEND:{date_compact}T{et_compact}"
    else:
        try:
            next_day = (dt.date.fromisoformat(date_str) + dt.timedelta(days=1)).isoformat().replace("-", "")
        except ValueError:
            next_day = date_compact
        dtstart_line = f"DTSTART;VALUE=DATE:{date_compact}"
        dtend_line = f"DTEND;VALUE=DATE:{next_day}"

    digest = hashlib.sha1(f"{title}|{date_str}|{start_t or ''}".encode("utf-8")).hexdigest()[:12]
    uid = f"{date_compact}-{digest}@notion-local-assistant"
    ics_content = "\r\n".join([
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Stefan Kollmeier//Notion Local Assistant//DE",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"SUMMARY:{title}",
        dtstart_line,
        dtend_line,
        f"LOCATION:{location}",
        f"DESCRIPTION:{description}",
        "END:VEVENT",
        "END:VCALENDAR",
        "",
    ])
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in title)[:40]
    ics_path = outbox / f"{date_str}_{safe_name}.ics"
    ics_path.write_text(ics_content, encoding="utf-8")
    return ics_path


def _local_tz_offset_str(date_str: str | None = None) -> str:
    tz_berlin = _berlin_tz()
    if tz_berlin and date_str and len(date_str) == 10:
        try:
            d = dt.date.fromisoformat(date_str)
            aware = dt.datetime(d.year, d.month, d.day, 12, 0, tzinfo=tz_berlin)
            off = aware.strftime("%z")
            if len(off) == 5:
                return f"{off[:3]}:{off[3:]}"
        except Exception:
            pass
    local_offset = dt.datetime.now().astimezone().strftime("%z")
    return f"{local_offset[:3]}:{local_offset[3:]}" if len(local_offset) == 5 else "+01:00"


def create_google_calendar_event(ev: dict[str, Any]) -> str | None:
    title = ev.get("title") or "Termin"
    date_str = ev.get("date")
    if not date_str or str(date_str).lower() == "null" or len(str(date_str)) != 10:
        return None

    start_t = _normalize_hhmm(ev.get("start_time"))
    end_t = _normalize_hhmm(ev.get("end_time"))
    location = (ev.get("location") or "").strip()
    description = (ev.get("description") or "").strip()
    tz_offset = _local_tz_offset_str(date_str)

    if start_t:
        if not end_t:
            end_hour = min(int(start_t[:2]) + 1, 23)
            end_t = f"{end_hour:02d}:{start_t[3:5]}"
        start_rfc = f"{date_str}T{start_t}:00{tz_offset}"
        end_rfc = f"{date_str}T{end_t}:00{tz_offset}"
        all_day = False
    else:
        try:
            next_day = (dt.date.fromisoformat(date_str) + dt.timedelta(days=1)).isoformat()
        except ValueError:
            return None
        start_rfc = date_str
        end_rfc = next_day
        all_day = True

    gcal_bin = get_gcalendar_bin()
    if gcal_bin:
        cmd = [
            gcal_bin, "mutate", "create",
            "--summary", title, "--start", start_rfc, "--end", end_rfc,
            "--send-updates", "none",
        ]
        if all_day:
            cmd.append("--all-day")
        if description:
            cmd.extend(["--description", description])
        if location:
            cmd.extend(["--location", location])
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20, check=False)
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    if "Link:" in line or "https://www.google.com/calendar" in line:
                        return line.split("Link:", 1)[-1].strip()
                return "created-via-gcalendar"
        except Exception:
            pass

    gcal_token = get_google_oauth_access_token()
    if gcal_token:
        cal_id = parse.quote(os.environ.get("GOOGLE_CALENDAR_ID", "primary").strip())
        body: dict[str, Any] = {
            "summary": title,
            "description": description,
            "location": location,
            "start": {"date": start_rfc} if all_day else {"dateTime": start_rfc},
            "end": {"date": end_rfc} if all_day else {"dateTime": end_rfc},
        }
        req = request.Request(
            f"https://www.googleapis.com/calendar/v3/calendars/{cal_id}/events?sendUpdates=none",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {gcal_token}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=20) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                ev["gcal_event_id"] = res.get("id")
                return res.get("htmlLink") or res.get("id") or "created-via-rest"
        except Exception as exc:
            log.warning("Google Calendar REST Create Fehler: %s", exc)

    return None


def update_google_calendar_event(
    event_id: str,
    new_date: str | None = None,
    new_start_time: str | None = None,
    new_end_time: str | None = None,
    new_title: str | None = None,
) -> bool:
    if not event_id:
        return False

    tz_offset = _local_tz_offset_str(new_date)
    gcal_bin = get_gcalendar_bin()
    if gcal_bin:
        cmd = [gcal_bin, "mutate", "update", event_id]
        if new_date and new_start_time:
            start_t = _normalize_hhmm(new_start_time)
            end_t = _normalize_hhmm(new_end_time)
            if not end_t and start_t:
                end_hour = min(int(start_t[:2]) + 1, 23)
                end_t = f"{end_hour:02d}:{start_t[3:5]}"
            cmd.extend(["--start", f"{new_date}T{start_t}:00{tz_offset}", "--end", f"{new_date}T{end_t}:00{tz_offset}"])
        elif new_date and not new_start_time:
            try:
                next_day = (dt.date.fromisoformat(new_date) + dt.timedelta(days=1)).isoformat()
            except ValueError:
                next_day = new_date
            cmd.extend(["--start", f"{new_date}T00:00:00{tz_offset}", "--end", f"{next_day}T00:00:00{tz_offset}"])
        if new_title:
            cmd.extend(["--summary", new_title])
        cmd.extend(["--send-updates", "none"])
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
            if proc.returncode == 0:
                return True
        except Exception as exc:
            log.warning("gcalendar update Fehler: %s", exc)

    gcal_token = get_google_oauth_access_token()
    if gcal_token:
        cal_id = parse.quote(os.environ.get("GOOGLE_CALENDAR_ID", "primary").strip())
        patch_body: dict[str, Any] = {}
        if new_title:
            patch_body["summary"] = new_title
        if new_date and new_start_time:
            start_t = _normalize_hhmm(new_start_time)
            end_t = _normalize_hhmm(new_end_time)
            if not end_t and start_t:
                end_hour = min(int(start_t[:2]) + 1, 23)
                end_t = f"{end_hour:02d}:{start_t[3:5]}"
            patch_body["start"] = {"dateTime": f"{new_date}T{start_t}:00{tz_offset}"}
            patch_body["end"] = {"dateTime": f"{new_date}T{end_t}:00{tz_offset}"}
        elif new_date:
            try:
                next_day = (dt.date.fromisoformat(new_date) + dt.timedelta(days=1)).isoformat()
            except ValueError:
                next_day = new_date
            patch_body["start"] = {"date": new_date}
            patch_body["end"] = {"date": next_day}
        if patch_body:
            req = request.Request(
                f"https://www.googleapis.com/calendar/v3/calendars/{cal_id}/events/{parse.quote(event_id)}?sendUpdates=none",
                data=json.dumps(patch_body).encode("utf-8"),
                headers={"Authorization": f"Bearer {gcal_token}", "Content-Type": "application/json"},
                method="PATCH",
            )
            try:
                with request.urlopen(req, timeout=20) as resp:
                    return resp.status == 200
            except Exception as exc:
                log.warning("Google Calendar REST Update Fehler: %s", exc)
    return False


def delete_google_calendar_event(event_id: str) -> bool:
    if not event_id:
        return False
    gcal_bin = get_gcalendar_bin()
    if gcal_bin:
        try:
            proc = subprocess.run(
                [gcal_bin, "mutate", "delete", event_id, "--send-updates", "none"],
                capture_output=True, text=True, timeout=20, check=False,
            )
            if proc.returncode == 0:
                return True
        except Exception as exc:
            log.warning("gcalendar delete Fehler: %s", exc)

    gcal_token = get_google_oauth_access_token()
    if gcal_token:
        cal_id = parse.quote(os.environ.get("GOOGLE_CALENDAR_ID", "primary").strip())
        req = request.Request(
            f"https://www.googleapis.com/calendar/v3/calendars/{cal_id}/events/{parse.quote(event_id)}?sendUpdates=none",
            headers={"Authorization": f"Bearer {gcal_token}"},
            method="DELETE",
        )
        try:
            with request.urlopen(req, timeout=20) as resp:
                return resp.status in (200, 204)
        except Exception as exc:
            log.warning("Google Calendar REST Delete Fehler: %s", exc)
    return False


def fetch_google_calendar_events(days_ahead: int = 14) -> list[dict[str, Any]]:
    """
    Holt Termine aus Google Kalender:
    1. `gcalendar` CLI
    2. Google Calendar REST API (via OAuth2 Refresh-Token / Access-Token)
    3. `GOOGLE_CALENDAR_ICS_URL` (inkl. Serientermin-Expansion)
    4. `gcal_cache.json` Fallback
    """
    events_out: list[dict[str, Any]] = []
    today = dt.date.today()
    end_date = today + dt.timedelta(days=days_ahead)
    start_iso = f"{today.isoformat()}T00:00:00Z"
    end_iso = f"{end_date.isoformat()}T23:59:59Z"
    tz_berlin = _berlin_tz()

    # 1. gcalendar CLI
    gcal_bin = get_gcalendar_bin()
    if gcal_bin:
        cache_end_iso = f"{(today + dt.timedelta(days=max(days_ahead, 30))).isoformat()}T23:59:59Z"
        cmd = [gcal_bin, "readonly", "events", "--start", start_iso, "--end", cache_end_iso, "--max", "100", "--json"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15, check=False)
            if proc.returncode == 0 and proc.stdout.strip():
                raw_events = json.loads(proc.stdout)
                all_fetched: list[dict[str, Any]] = []
                for ev in raw_events:
                    start_obj = ev.get("start", {})
                    start_dt = start_obj.get("dateTime") or start_obj.get("date") or ""
                    end_obj = ev.get("end", {})
                    end_dt = end_obj.get("dateTime") or end_obj.get("date") or ""
                    date_part = start_dt[:10] if len(start_dt) >= 10 else ""
                    item = {
                        "id": ev.get("id"),
                        "title": ev.get("summary") or "Termin",
                        "date": date_part,
                        "start_time": start_dt[11:16] if "T" in start_dt else None,
                        "end_time": end_dt[11:16] if "T" in end_dt else None,
                        "location": (ev.get("location") or "").strip(),
                        "link": ev.get("htmlLink", ""),
                        "source": "Google Kalender",
                        "synced_to_gcal": True,
                    }
                    all_fetched.append(item)
                    if today.isoformat() <= date_part <= end_date.isoformat():
                        events_out.append(item)
                try:
                    atomic_write_json(GCAL_CACHE_PATH, {
                        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
                        "events": all_fetched,
                    })
                except Exception:
                    pass
                events_out.sort(key=lambda x: (x.get("date") or "", x.get("start_time") or "00:00"))
                return events_out
        except Exception as exc:
            log.debug("gcalendar CLI Hinweis: %s", exc)

    # 2. Google Calendar REST API (OAuth2 Refresh Token)
    gcal_token = get_google_oauth_access_token()
    if gcal_token:
        cal_id = parse.quote(os.environ.get("GOOGLE_CALENDAR_ID", "primary").strip())
        cache_end_iso = f"{(today + dt.timedelta(days=max(days_ahead, 30))).isoformat()}T23:59:59Z"
        qs = parse.urlencode({
            "timeMin": start_iso,
            "timeMax": cache_end_iso,
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": "250",
        })
        req = request.Request(
            f"https://www.googleapis.com/calendar/v3/calendars/{cal_id}/events?{qs}",
            headers={"Authorization": f"Bearer {gcal_token}"},
        )
        try:
            with request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            all_fetched = []
            for ev in data.get("items", []):
                if ev.get("status") == "cancelled":
                    continue
                s_obj = ev.get("start", {})
                e_obj = ev.get("end", {})
                s_raw = s_obj.get("dateTime") or s_obj.get("date") or ""
                e_raw = e_obj.get("dateTime") or e_obj.get("date") or ""
                if "T" in s_raw:
                    dt_s = dt.datetime.fromisoformat(s_raw.replace("Z", "+00:00"))
                    dt_e = dt.datetime.fromisoformat(e_raw.replace("Z", "+00:00")) if "T" in e_raw else dt_s + dt.timedelta(hours=1)
                    if tz_berlin:
                        dt_s = dt_s.astimezone(tz_berlin)
                        dt_e = dt_e.astimezone(tz_berlin)
                    date_part = dt_s.date().isoformat()
                    st_t = dt_s.strftime("%H:%M")
                    en_t = dt_e.strftime("%H:%M")
                else:
                    date_part = s_raw[:10]
                    st_t, en_t = None, None
                item = {
                    "id": ev.get("id"),
                    "title": ev.get("summary") or "Termin",
                    "date": date_part,
                    "start_time": st_t,
                    "end_time": en_t,
                    "location": (ev.get("location") or "").strip(),
                    "link": ev.get("htmlLink", ""),
                    "source": "Google Kalender (API)",
                    "synced_to_gcal": True,
                }
                all_fetched.append(item)
                if today.isoformat() <= date_part <= end_date.isoformat():
                    events_out.append(item)
            try:
                atomic_write_json(GCAL_CACHE_PATH, {
                    "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "events": all_fetched,
                })
            except Exception:
                pass
            events_out.sort(key=lambda x: (x.get("date") or "", x.get("start_time") or "00:00"))
            return events_out
        except Exception as exc:
            log.warning("Google Calendar REST Read Fehler: %s", exc)

    # 3. ICS URL(s)
    ics_urls_raw = os.environ.get("GOOGLE_CALENDAR_ICS_URL", "").strip()
    if ics_urls_raw:
        for single_url in re.split(r"[\r\n,;]+", ics_urls_raw):
            single_url = single_url.strip()
            if not single_url or not single_url.startswith(("http://", "https://")):
                continue
            try:
                req = request.Request(single_url, headers={"User-Agent": "NotionLocalAssistant/1.0"})
                with request.urlopen(req, timeout=15) as resp:
                    ics_text = resp.read().decode("utf-8", errors="replace")
                events_out.extend(parse_ics_events(ics_text, today, end_date))
            except Exception as exc:
                log.warning("Google Kalender ICS Fehler (%s...): %s", single_url[:40], exc)

        if events_out:
            seen = set()
            deduped = []
            for ev in events_out:
                k = (ev.get("title"), ev.get("date"), ev.get("start_time"))
                if k not in seen:
                    seen.add(k)
                    deduped.append(ev)
            events_out = deduped
            try:
                atomic_write_json(GCAL_CACHE_PATH, {
                    "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "events": events_out,
                })
            except Exception:
                pass
            events_out.sort(key=lambda x: (x.get("date") or "", x.get("start_time") or "00:00"))
            return events_out

    # 4. Lokaler Cache
    if GCAL_CACHE_PATH.exists():
        try:
            cache_data = json.loads(GCAL_CACHE_PATH.read_text(encoding="utf-8"))
            for ev in cache_data.get("events", []):
                d_part = ev.get("date") or ""
                if today.isoformat() <= d_part <= end_date.isoformat():
                    events_out.append(ev)
        except Exception:
            pass

    events_out.sort(key=lambda x: (x.get("date") or "", x.get("start_time") or "00:00"))
    return events_out


# --------------------------------------------------------------------------------------
# Datenschutzkonformer Arbeitskalender (MS Teams / Outlook Free-Busy ONLY)
# --------------------------------------------------------------------------------------

def sanitize_work_busy_slots(raw_slots: list[dict[str, Any]], source: str = "work") -> list[dict[str, Any]]:
    """
    Hartes Privacy-Gate: Entfernt ALLE Titel, Betreffs, Orte, Teilnehmer und Texte.
    Es bleiben ausschließlich anonyme Zeitblöcke (date, start_time, end_time).
    """
    clean: list[dict[str, Any]] = []
    for item in raw_slots:
        d = str(item.get("date") or "").strip()
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            # Falls ISO-DateTime übergeben wurde (z. B. aus Power Automate: 2026-11-14T09:00:00)
            start_iso = str(item.get("start") or item.get("start_iso") or "").strip()
            end_iso = str(item.get("end") or item.get("end_iso") or "").strip()
            if len(start_iso) >= 16 and "T" in start_iso:
                d = start_iso[:10]
                st = _normalize_hhmm(start_iso[11:16])
                et = _normalize_hhmm(end_iso[11:16]) if len(end_iso) >= 16 and "T" in end_iso else None
            else:
                continue
        else:
            st = _normalize_hhmm(item.get("start_time") or item.get("start"))
            et = _normalize_hhmm(item.get("end_time") or item.get("end"))

        show_as = str(item.get("show_as") or item.get("status") or "busy").lower()
        if show_as == "free":
            continue

        clean.append({
            "title": "Arbeit: Belegt",
            "date": d,
            "start_time": st,
            "end_time": et,
            "location": "",
            "source": f"Work Free/Busy ({source})",
            "is_work_busy_only": True,
        })
    return clean


def fetch_work_busy_slots(
    days_ahead: int = 14,
    store: StateStore | None = None,
) -> list[dict[str, Any]]:
    """
    Liest ausschließlich anonyme Belegt-Zeiten (Free/Busy) des Arbeitskalenders (MS Teams/Outlook).
    Speichert niemals Arbeitstitel, E-Mails oder Details und synchronisiert nichts nach Notion!

    Quellen:
    1. `WORK_CALENDAR_FREEBUSY_ICS_URL` (Outlook/M365 Free-Busy ICS)
    2. `WORK_BUSY_SLOTS_FILE` (lokale JSON-Datei mit `{date, start_time, end_time}` oder `{start, end}`)
    3. `MS_GRAPH_ACCESS_TOKEN` + `WORK_EMAIL` (`POST /me/calendar/getSchedule` -> nur busy/oof Slots)
    4. SQLite Cache (`work_busy_slots`)
    """
    today = dt.date.today()
    end_date = today + dt.timedelta(days=days_ahead)
    collected: list[dict[str, Any]] = []

    # 1. Outlook / M365 Free-Busy ICS URL
    work_ics_url = os.environ.get("WORK_CALENDAR_FREEBUSY_ICS_URL", "").strip()
    if work_ics_url.startswith(("http://", "https://")):
        try:
            req = request.Request(work_ics_url, headers={"User-Agent": "NotionLocalAssistant-FreeBusy/1.0"})
            with request.urlopen(req, timeout=15) as resp:
                ics_text = resp.read().decode("utf-8", errors="replace")
            slots = parse_ics_events(
                ics_text,
                today,
                end_date,
                source_label="Work Free/Busy (ICS)",
                anonymize_for_privacy=True,
            )
            collected.extend(slots)
            if store:
                store.replace_work_busy_slots(slots, source="ics")
        except Exception as exc:
            log.warning("Work-Free/Busy ICS Abruf fehlgeschlagen: %s", exc)

    # 2. Lokale JSON-Datei (z. B. von Power Automate / lokalem Outlook-Export auf Arbeits-PC)
    slots_file_env = os.environ.get("WORK_BUSY_SLOTS_FILE", "work_busy_slots.json").strip()
    slots_path = Path(slots_file_env) if Path(slots_file_env).is_absolute() else (BASE_DIR / slots_file_env)
    if slots_path.exists():
        try:
            raw_data = json.loads(slots_path.read_text(encoding="utf-8"))
            raw_list = raw_data.get("slots", raw_data) if isinstance(raw_data, dict) else raw_data
            if isinstance(raw_list, list):
                sanitized = [
                    s for s in sanitize_work_busy_slots(raw_list, source="file")
                    if today.isoformat() <= s["date"] <= end_date.isoformat()
                ]
                collected.extend(sanitized)
                if store:
                    store.replace_work_busy_slots(sanitized, source="file")
        except Exception as exc:
            log.warning("Work-Free/Busy Datei nicht lesbar (%s): %s", slots_path.name, exc)

    # 3. Optional Microsoft Graph getSchedule (nur Free/Busy `scheduleItems` ohne Subject)
    ms_token = os.environ.get("MS_GRAPH_ACCESS_TOKEN", "").strip()
    work_email = os.environ.get("WORK_EMAIL", "").strip()
    if ms_token and work_email:
        try:
            payload = {
                "schedules": [work_email],
                "startTime": {"dateTime": f"{today.isoformat()}T00:00:00", "timeZone": "Europe/Berlin"},
                "endTime": {"dateTime": f"{end_date.isoformat()}T23:59:59", "timeZone": "Europe/Berlin"},
                "availabilityViewInterval": 30,
            }
            req = request.Request(
                "https://graph.microsoft.com/v1.0/me/calendar/getSchedule",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": f"Bearer {ms_token}", "Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=15) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            graph_raw: list[dict[str, Any]] = []
            for sched in res.get("value", []):
                for item in sched.get("scheduleItems", []):
                    status = str(item.get("status") or "busy").lower()
                    if status == "free":
                        continue
                    s_dt = (item.get("start") or {}).get("dateTime", "")
                    e_dt = (item.get("end") or {}).get("dateTime", "")
                    if len(s_dt) >= 16 and len(e_dt) >= 16:
                        graph_raw.append({
                            "date": s_dt[:10],
                            "start_time": s_dt[11:16],
                            "end_time": e_dt[11:16],
                            "show_as": status,
                        })
            sanitized_graph = sanitize_work_busy_slots(graph_raw, source="ms_graph")
            collected.extend(sanitized_graph)
            if store:
                store.replace_work_busy_slots(sanitized_graph, source="ms_graph")
        except Exception as exc:
            log.warning("MS Graph getSchedule Hinweis: %s", exc)

    # 4. Fallback aus SQLite (falls Remote-Abruf temporär nicht aktiv)
    if not collected and store:
        collected = store.get_work_busy_slots(today.isoformat(), end_date.isoformat())

    # Deduplizieren nach Zeitfenster
    seen: set[tuple[str, str | None, str | None]] = set()
    out: list[dict[str, Any]] = []
    for s in collected:
        k = (s["date"], s.get("start_time"), s.get("end_time"))
        if k not in seen:
            seen.add(k)
            out.append(s)
    out.sort(key=lambda x: (x["date"], x.get("start_time") or "00:00"))
    return out


# --------------------------------------------------------------------------------------
# Freie-Slot-Engine & Smart Time-Blocking
# --------------------------------------------------------------------------------------

def compute_free_slots(
    events: list[dict[str, Any]],
    target_date: dt.date | str,
    work_start: str = "08:00",
    work_end: str = "20:00",
    min_slot_minutes: int = 30,
    buffer_minutes: int = 5,
    all_day_blocks: bool = False,
) -> list[dict[str, Any]]:
    """
    Berechnet alle freien Zeitfenster an `target_date` (inkl. privatem Kalender + Work Free/Busy).
    """
    date_iso = target_date.isoformat() if isinstance(target_date, dt.date) else str(target_date)
    day_start = _hhmm_to_minutes(work_start)
    day_end = _hhmm_to_minutes(work_end)

    # Wenn heute: keine Slots in der Vergangenheit vorschlagen
    now_berlin = dt.datetime.now(_berlin_tz()) if _berlin_tz() else dt.datetime.now()
    if date_iso == now_berlin.date().isoformat():
        now_min = now_berlin.hour * 60 + now_berlin.minute + 10
        # Auf nächste 15 Min aufrunden
        now_min = ((now_min + 14) // 15) * 15
        day_start = max(day_start, now_min)

    if day_start >= day_end:
        return []

    busy_intervals: list[tuple[int, int]] = []
    for ev in events:
        if ev.get("date") != date_iso:
            continue
        st = _normalize_hhmm(ev.get("start_time"))
        et = _normalize_hhmm(ev.get("end_time"))
        if not st:
            if all_day_blocks:
                return []
            continue
        s_min = _hhmm_to_minutes(st)
        e_min = _hhmm_to_minutes(et) if et else min(s_min + 60, 23 * 60 + 59)
        if e_min <= s_min:
            e_min = min(s_min + 60, 23 * 60 + 59)
        busy_intervals.append((max(day_start, s_min - buffer_minutes), min(day_end, e_min + buffer_minutes)))

    busy_intervals.sort()
    merged: list[tuple[int, int]] = []
    for s_m, e_m in busy_intervals:
        if s_m >= e_m:
            continue
        if not merged or s_m > merged[-1][1]:
            merged.append((s_m, e_m))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e_m))

    free_slots: list[dict[str, Any]] = []
    cursor = day_start
    for b_start, b_end in merged:
        if b_start - cursor >= min_slot_minutes:
            free_slots.append({
                "date": date_iso,
                "start_time": _minutes_to_hhmm(cursor),
                "end_time": _minutes_to_hhmm(b_start),
                "duration_minutes": b_start - cursor,
            })
        cursor = max(cursor, b_end)

    if day_end - cursor >= min_slot_minutes:
        free_slots.append({
            "date": date_iso,
            "start_time": _minutes_to_hhmm(cursor),
            "end_time": _minutes_to_hhmm(day_end),
            "duration_minutes": day_end - cursor,
        })

    return free_slots


def suggest_time_blocks(
    free_slots: list[dict[str, Any]],
    open_tasks: list[dict[str, Any]],
    max_suggestions: int = 3,
) -> list[dict[str, Any]]:
    """Ordnet offene Aufgaben nach Priorität & Fälligkeit passenden freien Slots zu."""
    prio_rank = {"Hoch": 0, "Mittel": 1, "Niedrig": 2}
    todos = [t for t in open_tasks if t.get("kind", "To-Do") == "To-Do" and t.get("status") != "Erledigt"]
    todos.sort(key=lambda t: (
        prio_rank.get(t.get("priority", "Mittel"), 1),
        t.get("due_date") or "9999-12-31",
    ))

    suggestions: list[dict[str, Any]] = []
    remaining_slots = [dict(s) for s in free_slots]

    for task in todos:
        if len(suggestions) >= max_suggestions:
            break
        est = int(task.get("estimated_minutes") or 30)
        est = max(15, min(est, 120))
        for idx, slot in enumerate(remaining_slots):
            if slot["duration_minutes"] >= est:
                s_min = _hhmm_to_minutes(slot["start_time"])
                block_end = _minutes_to_hhmm(s_min + est)
                suggestions.append({
                    "task_title": task.get("title", "Aufgabe"),
                    "priority": task.get("priority", "Mittel"),
                    "date": slot["date"],
                    "start_time": slot["start_time"],
                    "end_time": block_end,
                    "duration_minutes": est,
                    "notion_page_id": task.get("notion_page_id"),
                })
                # Restzeit im Slot aktualisieren
                new_start = s_min + est + 10
                orig_end = _hhmm_to_minutes(slot["end_time"])
                if orig_end - new_start >= 25:
                    remaining_slots[idx] = {
                        "date": slot["date"],
                        "start_time": _minutes_to_hhmm(new_start),
                        "end_time": slot["end_time"],
                        "duration_minutes": orig_end - new_start,
                    }
                else:
                    remaining_slots.pop(idx)
                break
    return suggestions
