# Persönlicher Notion- & Sprachnotiz-Assistent (Gemini 3.8 Flash / Ollama & Whisper)

Dieses Projekt verbindet deine mobile Sprach- und Text-Erfassung in **Notion** und per **Telegram** mit **Gemini 3.8 Flash** (oder rein lokal mit **Ollama & Whisper**), sodass deine Notizen, To-Dos und Kalendertermine präzise, fehlerfrei und ohne Spamming synchronisiert werden.

## Optimierte Hybrid-Architektur (Gemini 3.5 Flash-Lite + 3.8 Flash + Jev)

Die Architektur wurde gezielt auf minimale Gemini-Kosten, höchste Geschwindigkeit und Schutz vor Token-Spitzen optimiert:
- **Modell-Tiering mit Gemini 3.5 Flash-Lite:** Standardmäßig läuft die RSS-Zusammenfassung, Extraktion und Chat-Antworten über **`gemini-3.5-flash-lite`** mit `thinking_level: "minimal"`. Dadurch fallen keine teuren internen Reasoning-Tokens auf den Token-Preis an.
- **Gemini 3.8 Flash als Eskalationspfad:** Wird gezielt für explizit komplexe Analysen (z. B. `/analyse`, tiefes Reasoning) genutzt mit `thinking_level: "low"`.
- **Deterministisches Preprocessing vor KI-Calls (`preprocess.py`):** Entfernt HTML-Tags, Tracking-Parameter (`utm_*`, `fbclid`, `gclid`), Boilerplate-Phrasen und dedupliziert strikt über GUID, kanonische URL und Content-Hashes.
- **TypeSafe AI (Jev System One):** Schnelles, ultra-günstiges Vorfilter- und Scoring-Modell (`jev-1.13.0`, $0.042/Mtok). Bewertet RSS-Relevanz vorab (`noul`), setzt Notion-Kategorien deterministisch (`choice`) und verwirft irrelevante Artikel vor dem Gemini-Aufruf.
- **Chat-Guard & Loop-Bremse (`chat_guard.py`):** Striktes `max_iterations`-Limit (Standard: 3) und Token-Budget verhindert Endlosschleifen. Notion-Payloads werden deterministisch auf Kernfelder reduziert.
- **Datenschutz-Guardrail:** Jev läuft standardmäßig nur für öffentliche RSS-Feeds. Im Chat ist Jev standardmäßig deaktiviert (`JEV_ALLOW_CHAT_ROUTING=false`), persönliche Daten (PII) werden vor jedem externen Call maskiert.

## Deine eingerichteten Notion-Seiten & Datenbanken

- **🤖 [Persönlicher Assistent – Cockpit](https://app.notion.com/p/40928f20975581359a5dca0ea5754723):** Dein zentrales Dashboard mit verknüpften Ansichten für offene Aufgaben, KI-Vorschläge und letzte Sprachnotizen.
- **📡 [RSS-Wissensspeicher](https://app.notion.com/p/40f28f2097558173a9fbc186ba992316) / [Wissensspeicher-Datenbank](https://app.notion.com/p/c7bac0f9b2224aa6a0b42510fb987b7b):** Wachsender Wissensspeicher für kuratierte Studien, Durchbrüche, Behandlungen und Bildungsrabatte.
- **🎙️ [AI Inbox](https://app.notion.com/p/4269a0f6fa41495a96fb82417d95762d):** Hier landen deine Sprachnachrichten und kurzen Textnotizen von unterwegs.
- **✅ [Tasks](https://app.notion.com/p/f0e562e1c1324989a7d796ad5e8adc02):** Automatisch abgeleitete **To-Dos**, **Erinnerungen** (mit Fälligkeitsdatum & Priorität) sowie **proaktive KI-Vorschläge** (*„Dinge, die du tun kannst"*), direkt verknüpft mit der jeweiligen Ursprungs-Notiz.
- **🧭 [Agent Operating Guide](https://app.notion.com/p/40928f20975581fc8766eae50d15a250)** (in [Documents](https://app.notion.com/p/992d22106628493d90b0bff21d75e0fa)) & **😎 [About Stefan](https://app.notion.com/p/40928f2097558103b785dec705ce6ef7)** (in [Memories](https://app.notion.com/p/40928f20975581fbb16df5679e2942f3)).

---

## 📡 RSS-Monitor & Automatischer Wissensspeicher (`rss_monitor.py`)

Läuft automatisch **2x täglich (07:30 und 18:30 Uhr)** via Task Scheduler:
- Durchsucht ausgewählte Themen in [feeds.json](file:///c:/Projekte/Notion/feeds.json) (Google News, Fach-RSS wie Nature, Medical Xpress, medRxiv, mydealz sowie wissenschaftliche APIs wie Europe PMC).
- **Start-Themen:**
  - 🧠 **Parkinson:** Wichtige Studien (RCTs, Phasenstudien, Kohorten), Durchbrüche (Zell-/Gentherapie, Biomarker), neue Behandlungen & Zulassungen.
  - 🫁 **Long COVID:** Wichtige Studien (Pathomechanismen, RCTs, Kohorten), Biomarker, Behandlungen (Metformin, Antivirale, Immunadsorption).
  - 🔋 **ME/CFS:** Neue Studien (Mechanismen, Genetik), Behandlungen (Therapieansätze, Förderaufrufe, Versorgungsdebatte).
  - 🎓 **Rabatte Studierende/Dozierende:** Aktionen mit mindestens 50 % Rabatt oder komplett kostenlos (Software, KI-Tools, Kurse, Bildungstickets).
- **Echte Neuheiten-Prüfung:** Jeder Kandidat wird gegen den bisherigen Wissensstand in Notion abgeglichen. Redundante Meldungen werden ignoriert; Weiterentwicklungen werden erkannt und als Update markiert.
- **Ergebnisse:**
  - Jeder Fund wird mit Titel, Thema, Kategorie, Relevanz, Zusammenfassung, Link, Quelle, Datum und ggf. Rabatthöhe im **[Wissensspeicher](https://app.notion.com/p/c7bac0f9b2224aa6a0b42510fb987b7b)** abgelegt.
  - Pro Lauf mit neuen Funden wird automatisch ein übersichtlicher **RSS-Digest** in [Documents](https://app.notion.com/p/992d22106628493d90b0bff21d75e0fa) abgelegt.
  - Push-Benachrichtigung per Telegram und Windows-Toast.

### Manuelle Nutzung:
```powershell
# Alle Themen durchsuchen, bewerten und nach Notion schreiben
python rss_monitor.py run

# Nur ein bestimmtes Thema prüfen
python rss_monitor.py run --topic parkinson

# Testlauf ohne Speichern (Dry-Run)
python rss_monitor.py run --dry-run

### Themen verwalten & anpassen:
```powershell
# 1. Alle Themen anzeigen (mit Status aktiv/inaktiv & Quellen)
python rss_monitor.py topic list
python rss_monitor.py topic list -v      # Detailansicht mit allen Feed-URLs

# 2. Thema aktivieren / pausieren
python rss_monitor.py topic disable parkinson
python rss_monitor.py topic enable parkinson

# 3. Neues Thema anlegen (automatisch per KI mit Feeds, Fokus & Emoji)
python rss_monitor.py topic add "Multiple Sklerose" --auto
python rss_monitor.py topic add "Quantencomputing" --auto

# 4. Neues Thema manuell anlegen oder interaktiver Assistent
python rss_monitor.py topic add          # Interaktiver Dialog
python rss_monitor.py topic add "KI-Tools" --emoji 🤖 --focus "Neue KI-Modelle und Tools" --rss "Heise=https://www.heise.de/rss/heise-atom.xml"

# 5. Quelle zu bestehendem Thema hinzufügen
python rss_monitor.py topic add-source parkinson --name "FAZ Wissen" --url "https://www.faz.net/rss/aktuell/wissen/"

# 6. Thema entfernen
python rss_monitor.py topic remove multiple_sklerose -y
```

### Unterwegs per Telegram verwalten:

Du kannst den RSS- & Wissensspeicher entweder über deinen Haupt-Bot oder über deinen **dedizierten RSS-Bot (`@RSS8472Bot`)** steuern:

**Vorteil des dedizierten RSS-Bots (`TELEGRAM_RSSBOT_TOKEN`):**
Trennt News, Studien-Meldungen und Rabatt-Pushes komplett von deinen privaten Terminen, To-Dos und Morgen-Briefings.

Befehle im RSS-Bot:
- `/themen` – Liste aller RSS-Themen mit aktuellem Status & Quellenanzahl
- `/an <id>` – Thema aktivieren (z. B. `/an parkinson`)
- `/aus <id>` – Thema pausieren (z. B. `/aus edu_deals`)
- `/neu <Name>` – Neues Thema per KI generieren & sofort aktivieren (z. B. `/neu Alzheimer`)
- `/loeschen <id>` – Thema entfernen (z. B. `/loeschen alt_thema`)
- `/run` – Manuellen RSS-/Bluesky-Suchlauf sofort starten & Ergebnis in den Chat posten
- `/synthese` – Themenübergreifende Wochen-Synthese über alle aktiven Themen
- `/covid` – Neueste gefilterte Covid-19 Infektionszahlen von `@rv-enigma` abrufen
- `/deals` – Aktuelle Studenten- & Dozenten-Rabatte anzeigen
- `/suche <Begriff>` – Notion & lokalen Wissensspeicher (BM25) durchsuchen
- `/status` – Übersicht über aktive Themen, Feed-Health (Circuit Breaker) & nächste Läufe
- *Interaktive Buttons:* Unter neuen Artikeln direkt `👎 Irrelevant`, `📌 Task` oder `🎬 Video-Idee` klicken
- *Freitext, Sprache & PDFs:* Fragen, Sprachnachrichten oder PDF-Studien direkt im Chat senden

---

## Wie du unterwegs Sprachnachrichten, Fotos & Texte aufnimmst

### Weg 1: Telegram Haupt-Bot (Empfohlen – inkl. Buttons, Slots & Multimodalität)
Sende Sprachnotizen, Text, **Fotos** (Terminzettel/Rechnungen) oder **PDFs/Textdateien** direkt in den Telegram-Chat:
- **Multi-Tool Agent:** Versteht auch komplexe Anfragen mit mehreren Aktionen in einer Nachricht + Multi-Turn-Gedächtnis.
- **Schnellbefehle:**
  - `/morgen`, `/abend`, `/briefing` – Tages-Briefing inkl. freier Slots & Quick-Buttons (`✅ erledigen`, `📅 auf morgen`)
  - `/slots [heute|morgen|YYYY-MM-DD]` – Freie Zeitfenster (inkl. anonymer Arbeits-Blocker) & Smart Time-Blocking (`⏱️ Blocken`)
  - `/todos`, `/erledigt <Nr>`, `/undo` – Aufgaben verwalten & letzte Änderung rückgängig machen
  - `/review` – Wochen-Review (Erledigt, verschobene Engpass-Tasks, RSS-Highlights)
  - `/suche <Begriff>` – Kombinierte Suche in Notion, Notizen & RSS-Wissensspeicher (BM25)
  - `/kalender`, `/ideen`, `/mails`, `/themen`, `/status`, `/update`

### Weg 2: Direkt in der Notion-App (Empfohlen für Alltag & Diktat)
1. Füge auf deinem Smartphone-Homescreen ein **Notion-Widget** hinzu, das direkt auf deine Datenbank **[AI Inbox](https://app.notion.com/p/4269a0f6fa41495a96fb82417d95762d)** (oder das **[Cockpit](https://app.notion.com/p/40928f20975581359a5dca0ea5754723)**) zeigt.
2. Tippe auf **`+ Neu`** und nutze das **Mikrofon-Symbol deiner Smartphone-Tastatur** (iOS/Android-Diktierfunktion). Der gesprochene Text steht sofort kostenlos in der Seite, Status ist automatisch **`Neu`**.

### Weg 3: Audiodateien oder Textdateien über `inbox_drop/`
1. Lege Sprachmemos oder Textdateien in den Ordner `inbox_drop/`.
2. `assistant.py` transkribiert Audiodateien via Gemini / `faster-whisper` und synchronisiert mit Notion.

---

## Nutzung des lokalen Assistenten (`assistant.py`)

```bash
python assistant.py run               # Einmaliger Sync + Briefing
python assistant.py daemon            # 24/7 Hintergrund-Dienst (Telegram, Scheduler, Pre-Meeting-Pings)
python assistant.py briefing --mode morning --telegram
python assistant.py review --telegram # Wochen-Review erstellen & senden
python assistant.py sync-calendar     # Offene Termine mit Google Kalender abgleichen
```

Integrierter Zeitplan (Europe/Berlin):
- **07:30 & 18:30 Uhr:** RSS-Recherche
- **08:00 & 19:30 Uhr:** Morgen-Briefing & Abend-Checkliste
- **Sonntags 18:00 & 18:15 Uhr:** Weekly Review & RSS-Wochen-Synthese
- **Laufend (alle 5 Min):** Sync, Inbox & Pre-Meeting-Briefing-Pings (40 Min vor privaten Terminen)

---

## 📅 Google Kalender Integration (iCal + OAuth2 CRUD)

1. **Lesen (inkl. Zeitzonen & Serientermine via `icalendar` + `recurring-ical-events`):**
   Trage `GOOGLE_CALENDAR_ICS_URL` (Privatadresse im iCal-Format) in `.env` ein.
2. **Schreiben/Ändern/Löschen auf Headless Linux VMs (OAuth2 Refresh Token):**
   Trage optional `GOOGLE_CALENDAR_CLIENT_ID`, `GOOGLE_CALENDAR_CLIENT_SECRET` und `GOOGLE_CALENDAR_REFRESH_TOKEN` in `.env` ein. Ohne OAuth2 werden Termine weiterhin als deterministische `.ics`-Dateien erzeugt.

---

## 🔒 Datenschutz-sichere Arbeitskalender-Integration (MS Teams / Outlook: nur Frei/Belegt)

Aus Datenschutz- und Cybersicherheitsgründen werden **NIEMALS** Arbeits-E-Mails, Termintitel, Teilnehmer oder Orte mit Notion oder externen LLMs synchronisiert. Der Assistent verarbeitet vom Arbeitskalender ausschließlich anonyme Zeitfenster (`date`, `start_time`, `end_time`, `title="Arbeit: Belegt"`):
- **Option A (Outlook Free/Busy ICS):** Wenn IT-seitig erlaubt, Outlook Kalender nur als *„Nur Verfügbarkeit / Free-Busy“* teilen und die URL in `WORK_CALENDAR_FREEBUSY_ICS_URL` eintragen.
- **Option B (MS Teams / Power Automate → anonyme JSON-Datei):** Ein kleiner M365 Power Automate Flow exportiert nur `Start` + `End` (ohne Betreff/Body!) nach `work_busy_slots.json` (siehe `WORK_BUSY_SLOTS_FILE`).
- **Option C (MS Graph `getSchedule`):** Mit `MS_GRAPH_ACCESS_TOKEN` und `WORK_EMAIL` werden ausschließlich `scheduleItems` (Status `busy`/`oof`) abgefragt.

Alle drei Optionen fließen lokal über `sanitize_work_busy_slots()` in `/slots`, freie Zeitfenster und Smart Time-Blocking ein.

---

## Vollautomatischer Hintergrund-Sync ohne Antigravity

1. Erstelle unter [notion.so/profile/integrations](https://www.notion.so/profile/integrations) ein kostenloses internes Notion-Token (`ntn_...`), kopiere `.env.example` zu `.env`, trage `NOTION_TOKEN=ntn_...` ein und verbinde die Integration in Notion über das `•••`-Menü oben rechts auf `AI Inbox` und `Tasks`.
2. **Automatische Windows-Hintergrundaufgabe aktivieren:**
   ```powershell
   .\setup_scheduler.ps1
   ```

