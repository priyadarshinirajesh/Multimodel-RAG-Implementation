@echo off
echo Stopping previous Ollama servers...
taskkill /IM "ollama.exe" /F >nul 2>&1
taskkill /IM "ollama app.exe" /F >nul 2>&1

echo Starting Ollama server...
start "" ollama serve
timeout /t 3