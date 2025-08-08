#!/usr/bin/env python3
"""
Script de lancement pour l'application Streamlit.
Ce script configure le PYTHONPATH et lance l'application.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Ajouter le dossier src au PYTHONPATH
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Définir la variable d'environnement PYTHONPATH pour Streamlit
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = str(src_path)
    
    # Lancer Streamlit
    app_path = src_path / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    
    print(f"🚀 Lancement de l'application Streamlit...")
    print(f"📁 PYTHONPATH: {env['PYTHONPATH']}")
    print(f"⚡ Commande: {' '.join(cmd)}")
    print(f"🌐 L'application sera disponible sur http://localhost:8501")
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n👋 Arrêt de l'application.")
    except Exception as e:
        print(f"❌ Erreur lors du lancement: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())