# Richtet die automatischen Windows-Aufgaben (Task Scheduler) ein, die den lokalen
# Notion- & Telegram-Assistenten betreiben – inkl. Echtzeit-Telegram-Daemon!

$ProjectDir = "c:\Projekte\Notion"
$PythonExe = (Get-Command python).Source
$PythonwExe = Join-Path (Split-Path $PythonExe) "pythonw.exe"
if (-not (Test-Path $PythonwExe)) {
    $PythonwExe = $PythonExe
}

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

# Alte Duplikat-Aufgaben entfernen, damit nie zwei Prozesse gleichzeitig Telegram abfragen
Unregister-ScheduledTask -TaskName "NotionLocalAssistant" -Confirm:$false -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName "NotionLocalAssistant_Sync" -Confirm:$false -ErrorAction SilentlyContinue

# 1. Echtzeit-Telegram-Daemon (antwortet in 1-2 Sekunden auf Sprach-/Textnachrichten + prüft alle 5 Min. Notion/Mails)
$DaemonAction = New-ScheduledTaskAction `
    -Execute $PythonwExe `
    -Argument "`"$ProjectDir\assistant.py`" daemon" `
    -WorkingDirectory $ProjectDir
$DaemonTriggerWatchdog = New-ScheduledTaskTrigger `
    -Once `
    -At "07:00" `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask `
    -TaskName "NotionLocalAssistant_Daemon" `
    -Action $DaemonAction `
    -Trigger $DaemonTriggerWatchdog `
    -Settings $Settings `
    -Description "Echtzeit-Telegram-Daemon (Long-Polling, antwortet sofort in 1-2 Sek. auf Sprach- & Textnachrichten)." `
    -Force

# Autostart-Verknüpfung für sofortigen Start beim Windows-Login anlegen
$StartupDir = [Environment]::GetFolderPath("Startup")
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut((Join-Path $StartupDir "NotionLocalAssistant_Daemon.lnk"))
$Shortcut.TargetPath = $PythonwExe
$Shortcut.Arguments = "`"$ProjectDir\assistant.py`" daemon"
$Shortcut.WorkingDirectory = $ProjectDir
$Shortcut.WindowStyle = 7
$Shortcut.Save()

# 2. Morgen-Briefing um 08:00 Uhr (Push an Telegram + Windows)
$MorningAction = New-ScheduledTaskAction `
    -Execute $PythonwExe `
    -Argument "`"$ProjectDir\assistant.py`" briefing --mode morning --telegram --notify" `
    -WorkingDirectory $ProjectDir
$MorningTrigger = New-ScheduledTaskTrigger -Daily -At "08:00"
Register-ScheduledTask `
    -TaskName "NotionLocalAssistant_MorningBriefing" `
    -Action $MorningAction `
    -Trigger $MorningTrigger `
    -Settings $Settings `
    -Description "Sendet jeden Morgen um 08:00 Uhr das Tages-Briefing (Termine, To-Dos, Vorschläge) per Telegram." `
    -Force

# 3. Abend-Checkliste um 19:30 Uhr (Push an Telegram + Windows)
$EveningAction = New-ScheduledTaskAction `
    -Execute $PythonwExe `
    -Argument "`"$ProjectDir\assistant.py`" briefing --mode evening --telegram --notify" `
    -WorkingDirectory $ProjectDir
$EveningTrigger = New-ScheduledTaskTrigger -Daily -At "19:30"
Register-ScheduledTask `
    -TaskName "NotionLocalAssistant_EveningBriefing" `
    -Action $EveningAction `
    -Trigger $EveningTrigger `
    -Settings $Settings `
    -Description "Sendet jeden Abend um 19:30 Uhr die Abend-Checkliste & Rückblick per Telegram." `
    -Force

# 4. RSS-Monitor & Wissensspeicher (2x täglich: 07:30 und 18:30 Uhr)
$RSSAction = New-ScheduledTaskAction `
    -Execute $PythonwExe `
    -Argument "`"$ProjectDir\rss_monitor.py`" run" `
    -WorkingDirectory $ProjectDir

$RSSTriggerMorning = New-ScheduledTaskTrigger -Daily -At "07:30"
Register-ScheduledTask `
    -TaskName "NotionLocalAssistant_RSS_Morning" `
    -Action $RSSAction `
    -Trigger $RSSTriggerMorning `
    -Settings $Settings `
    -Description "Prüft morgens um 07:30 Uhr RSS-Feeds & Europe PMC auf neue Studien/Behandlungen/Rabatte und speichert sie im Notion-Wissensspeicher." `
    -Force

$RSSTriggerEvening = New-ScheduledTaskTrigger -Daily -At "18:30"
Register-ScheduledTask `
    -TaskName "NotionLocalAssistant_RSS_Evening" `
    -Action $RSSAction `
    -Trigger $RSSTriggerEvening `
    -Settings $Settings `
    -Description "Prüft abends um 18:30 Uhr RSS-Feeds & Europe PMC auf neue Studien/Behandlungen/Rabatte und speichert sie im Notion-Wissensspeicher." `
    -Force

# Daemon sofort im Hintergrund starten
Start-ScheduledTask -TaskName "NotionLocalAssistant_Daemon"
Write-Host "Fertig! Echtzeit-Daemon, Morgen-Briefing (08:00), Abend-Checkliste (19:30) und RSS-Monitor (07:30 & 18:30) laufen." -ForegroundColor Green
