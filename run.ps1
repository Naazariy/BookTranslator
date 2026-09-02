param (
    [Parameter(Mandatory=$true, HelpMessage="Шлях до вхідного файлу")]
    [string]$InputFile,
    
    [Parameter(Mandatory=$true, HelpMessage="Шлях до вихідного файлу (перекладу)")]
    [string]$OutputFile
)

# Створення та активація віртуального середовища, якщо його немає
if (-Not (Test-Path ".venv")) {
    Write-Host "Віртуальне середовище не знайдено. Створюємо нове..." -ForegroundColor Cyan
    python -m venv .venv
    Write-Host "Встановлюємо залежності..." -ForegroundColor Cyan
    .venv\Scripts\python.exe -m pip install -q -r requirements.txt
    
    # Так як requirements.txt ще немає, згенеруємо його з pyproject.toml
    .venv\Scripts\python.exe -m pip install -q .
}

Write-Host "Запуск BookTranslator..." -ForegroundColor Green
.venv\Scripts\python.exe -m src.launcher.cli --file "$InputFile" --out "$OutputFile"
