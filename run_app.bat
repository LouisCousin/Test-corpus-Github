@echo off
echo 🚀 Lancement du Générateur d'Ouvrage Assisté par IA...

REM Définir le répertoire du script
set SCRIPT_DIR=%~dp0

REM Ajouter le dossier src au PYTHONPATH
set PYTHONPATH=%SCRIPT_DIR%src;%PYTHONPATH%

REM Lancer l'application Python
python "%SCRIPT_DIR%run_app.py"

pause