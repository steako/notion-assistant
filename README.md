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
- `/covid` – Neueste gefilterte Covid-19 Infektionszahlen von `@rv-enigma` abrufen
- `/deals` – Aktuelle Studenten- & Dozenten-Rabatte anzeigen
- `/suche <Begriff>` – Den gesamten Notion-Wissensspeicher nach Begriffen durchsuchen
- `/status` – Übersicht über aktive Themen, gecachte Einträge und nächste Zeitplan-Läufe
- *Freitext & Sprache:* Du kannst dem Bot auch Sprachnachrichten oder freie Fragen schicken (z. B. „Gibt es neue Studien zu Long COVID?“).

---

## Wie du unterwegs Sprachnachrichten & kurze Texte aufnimmst

### Weg 1: Direkt in der Notion-App (Empfohlen für Alltag & Diktat)
1. Füge auf deinem Smartphone-Homescreen ein **Notion-Widget** hinzu, das direkt auf deine Datenbank **[AI Inbox](https://app.notion.com/p/4269a0f6fa41495a96fb82417d95762d)** (oder das **[Cockpit](https://app.notion.com/p/40928f20975581359a5dca0ea5754723)**) zeigt.
2. Tippe auf **`+ Neu`** und nutze das **Mikrofon-Symbol deiner Smartphone-Tastatur** (iOS/Android-Diktierfunktion). Der gesprochene Text steht sofort kostenlos in der Seite, Status ist automatisch **`Neu`**.

### Weg 2: Echte Sprachdateien (.m4a, .mp3, .ogg, .wav) oder Textdateien über `inbox_drop/`
1. Lege Sprachmemos oder Textdateien in den Ordner `c:\Projekte\Notion\inbox_drop\` (z. B. synchronisiert vom Handy über OneDrive, Nextcloud, Google Drive oder FolderSync).
2. `assistant.py` transkribiert Audiodateien lokal mit **`faster-whisper`** und wertet sie anschließend lokal über **Ollama** aus.

---

## Nutzung des lokalen Assistenten (`assistant.py`)

### 1. Neue Notizen auswerten & Briefing anzeigen
```powershell
python assistant.py run
```

### 2. Schnelle Notiz direkt per Kommandozeile einwerfen
```powershell
python assistant.py add --title "Idee für Vorlesung" --text "Bis Freitag Folien für Kapitel 3 ergänzen und kurzes Quiz erstellen"
```

### 3. Nur das aktuelle Tages-Briefing (inkl. Windows-Benachrichtigung) abrufen
```powershell
python assistant.py briefing --notify
```

---

## Vollautomatischer Hintergrund-Sync ohne Antigravity

1. **Optional für direkten Hintergrund-Zugriff ohne Antigravity:** Erstelle unter [notion.so/profile/integrations](https://www.notion.so/profile/integrations) ein kostenloses internes Notion-Token (`ntn_...`), kopiere `.env.example` zu `.env`, trage `NOTION_TOKEN=ntn_...` ein und verbinde die Integration in Notion über das `•••`-Menü oben rechts auf `AI Inbox` und `Tasks`.
2. **Automatische Windows-Hintergrundaufgabe aktivieren (alle 30 Min.):**
   ```powershell
   .\setup_scheduler.ps1
   ```
