
# Générateur d'Ouvrage Assisté par IA

Application Streamlit pour la génération d'ouvrages à partir d'un plan DOCX et d'un corpus enrichi (Excel analysé).

## Lancer
```bash
pip install -r requirements.txt
streamlit run src/app.py
```


---

## Chemins par défaut & .env

Dans `config/user.yaml`, renseignez :
```yaml
default_paths:
  plan_docx: 'D:\Vibe\HITL\Rédacteur\Input rédacteur\20250518 HTL.docx'
  excel_corpus: 'D:\Vibe\HITL\Rédacteur\Input rédacteur\plan_Analyse_HITL - Copie_ANALYZED.xlsx'
  keywords_json: 'D:\Vibe\HITL\Rédacteur\Input rédacteur\keywords_mapping.json'
  env_dir: 'D:\Vibe\HITL\Rédacteur\Input rédacteur'
```

Créez un fichier `.env` **dans `env_dir`** (voir `.env.template`) :
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
```
Puis dans l'app (Accueil), cliquez sur **"Charger depuis les chemins par défaut"**.
