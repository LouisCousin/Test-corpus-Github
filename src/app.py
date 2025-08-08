
from __future__ import annotations
import os, json
from dotenv import load_dotenv
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

from config_manager import (
    __version__, get_config,
    AVAILABLE_OPENAI_MODELS, AVAILABLE_ANTHROPIC_MODELS, MODEL_ALIASES
)
from core.utils import (
    parse_docx_plan, call_openai, call_anthropic, generate_styled_docx,
    extract_used_references_apa, generate_bibliography, truncate_to_tokens,
    export_markdown, export_docx
)
from core.corpus_manager import CorpusManager
from core.prompt_builder import PromptBuilder

st.set_page_config(page_title="Générateur d'Ouvrage Assisté par IA", layout="wide")
st.markdown(f"### Générateur d'Ouvrage Assisté par IA — version {__version__}")

# Fonction principale pour les générations avec indicateurs d'activité
def run_generation(mode: str, prompt: str, provider: str, model: str, params: dict, styles: dict, base_name: str):
    """Exécute une génération avec indicateurs visuels et exports automatiques."""
    with st.status("Initialisation…", expanded=True) as status:
        prog = st.progress(0)
        
        # Étape 1: Préparation du prompt
        status.update(label="Préparation du prompt", state="running")
        truncated = truncate_to_tokens(prompt, params["max_input_tokens"], model=model)
        prog.progress(20)
        
        # Étape 2: Appel LLM
        status.update(label=f"Appel {provider}…", state="running")
        try:
            if provider == "OpenAI":
                text = call_openai(
                    model, truncated,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    max_output_tokens=params["max_output_tokens"],
                    api_key=st.session_state.openai_key
                )
            else:
                text = call_anthropic(
                    model, truncated,
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    max_output_tokens=params["max_output_tokens"],
                    api_key=st.session_state.anthropic_key
                )
        except Exception as e:
            status.update(label=f"Erreur {provider}: {str(e)}", state="error")
            st.error(f"Erreur API ({provider}): {e}")
            return None, None, None
        prog.progress(70)
        
        # Étape 3: Exports
        status.update(label="Exports en cours…", state="running")
        export_dir = st.session_state.get("export_dir", "output")
        os.makedirs(export_dir, exist_ok=True)
        
        md_path = export_markdown(text, base_name=base_name, mode=mode, export_dir=export_dir)
        docx_path = export_docx(text, base_name=base_name, mode=mode, export_dir=export_dir, styles=styles)
        prog.progress(100)
        
        status.update(label="Terminé ✅", state="complete")
    
    # Toast de confirmation
    st.toast(f"{mode.capitalize()} exporté :\n• {md_path}\n• {docx_path}")
    return text, md_path, docx_path

cfg = get_config()
ss = st.session_state
ss.setdefault("plan_items", [])
ss.setdefault("selected_sections", [])
ss.setdefault("mode", "Manuel")
ss.setdefault("drafts", {})
ss.setdefault("finals", {})
ss.setdefault("sections_processed", [])
ss.setdefault("openai_key", os.getenv("OPENAI_API_KEY", ""))
ss.setdefault("anthropic_key", os.getenv("ANTHROPIC_API_KEY", ""))
ss.setdefault("drafter_provider", "OpenAI")
ss.setdefault("refiner_provider", "Anthropic")
ss.setdefault("drafter_model", "gpt-4.1")
ss.setdefault("refiner_model", "claude-3.5-sonnet-20240620")
ss.setdefault("prefix", "Chapitre_{code}")
ss.setdefault("excel_path", None)
ss.setdefault("kmap_path", None)
ss.setdefault("cm", None)
ss.setdefault("gpt_template", cfg.gpt_prompt_template)
ss.setdefault("claude_template", cfg.claude_prompt_template)
ss.setdefault("styles", cfg.styles)
ss.setdefault("cm_overrides", {})
# Persistance des paramètres de corpus
ss.setdefault("min_relevance_score", cfg.min_relevance_score)
ss.setdefault("max_citations_per_section", cfg.max_citations_per_section)
ss.setdefault("include_secondary_matches", cfg.include_secondary_matches)
ss.setdefault("confidence_threshold", cfg.confidence_threshold)
# Nouveaux paramètres pour les hyperparamètres et export
ss.setdefault("export_dir", cfg.export_dir)
ss.setdefault("draft_params", cfg.draft_params)
ss.setdefault("final_params", cfg.final_params)

# --- Auto-rehydrate uploads across pages (so nothing "disparait") ---
from pathlib import Path as _Path
def _auto_rehydrate():
    # (1) Recréer le CorpusManager si on a déjà le chemin Excel
    if ss.get("cm") is None and ss.get("excel_path"):
        p = _Path(ss.excel_path)
        if p.exists():
            try:
                ss.cm = CorpusManager(ss.excel_path, ss.get("kmap_path"), ss.get("cm_overrides"))
            except Exception:
                pass

    # (2) Si on n’a pas de chemin Excel, tenter data/input/*.xlsx|*.csv
    if not ss.get("excel_path"):
        for ext in ("*.xlsx", "*.csv"):
            files = list(_Path("data/input").glob(ext))
            if files:
                ss.excel_path = str(files[0])
                try:
                    ss.cm = CorpusManager(ss.excel_path, ss.get("kmap_path"), ss.get("cm_overrides"))
                except Exception:
                    pass
                break

    # (3) Restaurer le plan si absent : cache/plan_upload.docx, sinon 1er .docx de data/input/
    if not ss.get("plan_items"):
        cache_plan = _Path("cache/plan_upload.docx")
        candidate = cache_plan if cache_plan.exists() else (list(_Path("data/input").glob("*.docx"))[:1] or [None])[0]
        if candidate:
            try:
                ss.plan_items = parse_docx_plan(str(candidate))
            except Exception:
                ss.plan_items = []

_auto_rehydrate()

# -- Inject defaults from config (paths + .env) --
def _apply_defaults_from_config():
    ss.defaults_loaded = False
    ss.env_loaded_from = None
    dp = cfg.default_paths
    # .env
    env_dir = dp.get("env_dir") or ""
    if env_dir and os.path.isdir(env_dir):
        env_path = os.path.join(env_dir, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            ss.env_loaded_from = env_path
            # refresh session keys if empty or outdated
            ss.openai_key = os.getenv("OPENAI_API_KEY", ss.get("openai_key", ""))
            ss.anthropic_key = os.getenv("ANTHROPIC_API_KEY", ss.get("anthropic_key", ""))

    # Plan
    if not ss.get("plan_items") and dp.get("plan_docx") and os.path.exists(dp["plan_docx"]):
        try:
            ss.plan_items = parse_docx_plan(dp["plan_docx"])
        except Exception:
            pass
    # Excel
    if not ss.get("excel_path") and dp.get("excel_corpus") and os.path.exists(dp["excel_corpus"]):
        ss.excel_path = dp["excel_corpus"]
        try:
            ss.cm = CorpusManager(ss.excel_path, ss.get("kmap_path"), ss.get("cm_overrides"))
        except Exception:
            pass
    # Keywords mapping
    if not ss.get("kmap_path") and dp.get("keywords_json") and os.path.exists(dp["keywords_json"]):
        ss.kmap_path = dp["keywords_json"]
    # Mark defaults loaded (we'll validate on UI)
    ss.defaults_loaded = True

_apply_defaults_from_config()

# Sidebar: NAV only
PAGES = [
    "Accueil",
    "Validation & Qualité",
    "Configuration du corpus",
    "Configuration IA",
    "Traitement",
    "Prompts",
    "Styles DOCX",
    "Nommage des fichiers",
    "Analyse du corpus",
    "Prévisualisation",
    "Brouillon (IA 1)",
    "Version Finale (IA 2)",
    "Document Complet",
]
with st.sidebar:
    page = st.radio("Navigation", PAGES, index=0)
    
    st.markdown("---")
    st.subheader("Paramètres des générations")
    
    # Fonction helper pour les blocs de paramètres
    def _params_block(label, key_prefix, defaults):
        with st.expander(label, expanded=False):
            t = st.slider("Température", 0.0, 2.0, float(defaults["temperature"]), 0.05, key=f"{key_prefix}_t")
            p = st.slider("Top-p", 0.0, 1.0, float(defaults["top_p"]), 0.01, key=f"{key_prefix}_p")
            mi = st.number_input("Max tokens Input", min_value=256, max_value=32768, value=int(defaults["max_input_tokens"]), step=128, key=f"{key_prefix}_mi")
            mo = st.number_input("Max tokens Output", min_value=16, max_value=8192, value=int(defaults["max_output_tokens"]), step=16, key=f"{key_prefix}_mo")
            return {"temperature": t, "top_p": p, "max_input_tokens": mi, "max_output_tokens": mo}
    
    # Paramètres par traitement
    ss["draft_params"] = _params_block("Paramètres Brouillon (IA 1)", "draft", cfg.draft_params)
    ss["final_params"] = _params_block("Paramètres Version finale (IA 2)", "final", cfg.final_params)
    
    # Répertoire d'export
    ss["export_dir"] = st.text_input("Dossier d'export", value=ss.get("export_dir", cfg.export_dir), key="export_dir_input")

# Pages
if page == "Accueil":
    st.subheader("Chargement des fichiers")
    st.caption("Vous pouvez aussi définir des chemins par défaut dans config/user.yaml (plan_docx, excel_corpus, keywords_json, env_dir).")
    if st.button("Charger depuis les chemins par défaut"):
        # Re-appliquer les defaults (utile si vous venez de modifier user.yaml)
        _apply_defaults_from_config()
        if ss.excel_path:
            ss.cm = CorpusManager(ss.excel_path, ss.kmap_path, ss.get("cm_overrides"))
        _rr = getattr(st, 'rerun', None) or getattr(st, 'experimental_rerun', None)
        
        if _rr:
            _rr()

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        plan_docx = st.file_uploader("Plan (.docx)", type=["docx"], key="plan_docx_main")
    with col2:
        excel_file = st.file_uploader("Corpus enrichi (_ANALYZED.xlsx/CSV)", type=["xlsx", "csv"], key="excel_file_main")
    with col3:
        keywords_json = st.file_uploader("keywords_mapping.json (optionnel)", type=["json"], key="kjson_main")

    if plan_docx is not None:
        bytes_data = plan_docx.read()
        os.makedirs("cache", exist_ok=True)
        tmp_path = os.path.join("cache", "plan_upload.docx")
        with open(tmp_path, "wb") as f:
            f.write(bytes_data)
        ss.plan_items = parse_docx_plan(tmp_path)
        st.success(f"Plan chargé : {len(ss.plan_items)} sections.")

    if excel_file is not None:
        ep = os.path.join("cache", "corpus_upload.xlsx" if excel_file.name.endswith(".xlsx") else "corpus_upload.csv")
        with open(ep, "wb") as f:
            f.write(excel_file.read())
        ss.excel_path = ep

    if keywords_json is not None:
        kp = os.path.join("cache", "keywords_mapping.json")
        with open(kp, "wb") as f:
            f.write(keywords_json.read())
        ss.kmap_path = kp

    if ss.excel_path:
        ss.cm = CorpusManager(ss.excel_path, ss.kmap_path, ss.get("cm_overrides"))

    if ss.plan_items:
        codes = [it["code"] for it in ss.plan_items if it.get("code")]
        st.write("Sections détectées (aperçu) :")
        st.code(", ".join(codes[:50]) + (" ..." if len(codes) > 50 else ""))

    # --- Status panel ---
    st.markdown('---')
    st.markdown('#### Statut du chargement par défaut')
    # Validate plan
    plan_ok = bool(ss.get('plan_items'))
    excel_ok = bool(ss.get('excel_path')) and bool(ss.get('cm')) and hasattr(ss.cm, 'df') and len(getattr(ss.cm, 'df', []))>0
    json_ok = bool(ss.get('kmap_path')) and os.path.exists(ss.get('kmap_path')) if ss.get('kmap_path') else False
    env_ok = bool(ss.get('env_loaded_from')) or bool(os.getenv('OPENAI_API_KEY')) or bool(os.getenv('ANTHROPIC_API_KEY'))
    st.write(f"Plan: {'✅' if plan_ok else '❌'} | Sections: {len(ss.get('plan_items') or [])}")
    st.write(f"Corpus (_ANALYZED.xlsx/CSV): {'✅' if excel_ok else '❌'} | Lignes: {len(getattr(ss.cm, 'df', [])) if ss.get('cm') else 0}")
    st.write(f"keywords_mapping.json: {'✅' if json_ok else '⚠️ Non fourni'}")
    # Mask API keys but show detection
    ok_oa = '✅' if ss.get('openai_key') else '❌'
    ok_an = '✅' if ss.get('anthropic_key') else '❌'
    st.write(f"Clé OpenAI: {ok_oa} | Clé Anthropic: {ok_an}")
    if ss.get('env_loaded_from'):
        st.caption(f".env chargé depuis: {ss.env_loaded_from}")


elif page == "Validation & Qualité":
    st.subheader("Validation et diagnostic")
    st.caption("Si la couverture est vide, vérifiez le mapping des colonnes ci-dessous.")
    if ss.cm is None:
        st.info("Chargez le corpus et/ou le plan dans Accueil.")
    else:
        st.code(f"sections={ss.cm.col_sections}, score={ss.cm.col_score}, keywords={ss.cm.col_keywords}, type={ss.cm.col_match_type}, confiance={ss.cm.col_confidence}, texte={ss.cm.col_text}")
        if ss.plan_items:
            codes = [it["code"] for it in ss.plan_items if it.get("code")]
            cov = ss.cm.get_section_coverage(
                codes,
                min_score=ss.min_relevance_score,
                max_items=ss.max_citations_per_section,
                include_secondary=ss.include_secondary_matches,
                confidence_threshold=ss.confidence_threshold,
            )
            st.dataframe(cov, use_container_width=True)
            with st.expander("Mapping manuel des colonnes"): 
                cols = list(ss.cm.df.columns) if ss.cm is not None else []
                c1,c2,c3 = st.columns(3)
                with c1:
                    sec = st.selectbox("Colonne Sections (codes: 1.1|1.2|...)", options=cols, index=cols.index(ss.cm.col_sections) if cols else 0 if cols else None)
                    sc = st.selectbox("Colonne Score", options=cols, index=cols.index(ss.cm.col_score) if cols else 0 if cols else None)
                with c2:
                    kw = st.selectbox("Colonne Mots-clés (optionnel)", options=cols, index=cols.index(ss.cm.col_keywords) if cols else 0 if cols else None)
                    mt = st.selectbox("Colonne Type (primary/secondary)", options=cols, index=cols.index(ss.cm.col_match_type) if cols else 0 if cols else None)
                with c3:
                    cf = st.selectbox("Colonne Confiance (%)", options=cols, index=cols.index(ss.cm.col_confidence) if cols else 0 if cols else None)
                    tx = st.selectbox("Colonne Texte (contenu à citer)", options=cols, index=cols.index(ss.cm.col_text) if cols else 0 if cols else None)
                if st.button("Appliquer le mapping"):
                    ss.cm_overrides = {"sections": sec, "score": sc, "keywords": kw, "type": mt, "confidence": cf, "text": tx}
                    ss.cm = CorpusManager(ss.excel_path, ss.kmap_path, ss.cm_overrides)
                    st.success("Nouveau mapping appliqué.")
                st.caption("Aperçu de la feuille utilisée :")
                st.dataframe(ss.cm.df.head(8), use_container_width=True)
            if ss.cm is not None and getattr(ss.cm, "df", None) is not None:
                st.caption("Aperçu (5 premières lignes de la feuille chargée):")
                st.dataframe(ss.cm.df.head(5), use_container_width=True)
        else:
            st.info("Aucune section détectée dans le plan.")

elif page == "Configuration du corpus":
    st.subheader("Paramètres de filtrage du corpus")
    ss.min_relevance_score = st.slider("Score minimum de pertinence", 0, 10, ss.min_relevance_score)
    ss.max_citations_per_section = st.number_input("Nombre max de citations par section", 1, 200, ss.max_citations_per_section)
    ss.include_secondary_matches = st.checkbox("Inclure les matchs secondaires", ss.include_secondary_matches)
    ss.confidence_threshold = st.slider("Seuil de confiance (%)", 0, 100, ss.confidence_threshold)

elif page == "Configuration IA":
    st.subheader("Clés API & Modèles (utilisables partout)")
    ss.openai_key = st.text_input("OpenAI API Key", type="password", value=ss.openai_key)
    ss.anthropic_key = st.text_input("Anthropic API Key", type="password", value=ss.anthropic_key)

    colA, colB = st.columns(2)
    with colA:
        ss.drafter_provider = st.selectbox("Étape 1 — Fournisseur", ["OpenAI", "Anthropic"], index=0 if ss.drafter_provider=="OpenAI" else 1)
        ss.drafter_model = st.selectbox("Modèle (Étape 1)",
            AVAILABLE_OPENAI_MODELS if ss.drafter_provider=="OpenAI" else AVAILABLE_ANTHROPIC_MODELS,
            index=0)
    with colB:
        ss.refiner_provider = st.selectbox("Étape 2 — Fournisseur", ["OpenAI", "Anthropic"], index=0 if ss.refiner_provider=="OpenAI" else 1)
        ss.refiner_model = st.selectbox("Modèle (Étape 2)",
            AVAILABLE_OPENAI_MODELS if ss.refiner_provider=="OpenAI" else AVAILABLE_ANTHROPIC_MODELS,
            index=0)

elif page == "Traitement":
    st.subheader("Paramètres d'exécution")
    ss.mode = st.radio("Mode de travail", ["Manuel", "Automatique", "Semi-automatique"], index=["Manuel","Automatique","Semi-automatique"].index(ss.mode))
    codes = [it["code"] for it in ss.plan_items if it.get("code")] if ss.plan_items else []
    if ss.mode == "Manuel":
        sel = st.selectbox("Sélectionnez une section", codes) if codes else None
        ss.selected_sections = [sel] if sel else []
    else:
        ss.selected_sections = st.multiselect("Sections à traiter", codes, default=ss.selected_sections)

    start = st.button("Lancer le traitement")
    if start:
        if not ss.cm:
            st.error("Veuillez charger un fichier Excel du corpus enrichi (Accueil).")
        elif not ss.selected_sections:
            st.warning("Sélectionnez au moins une section.")
        else:
            pb = PromptBuilder(ss.gpt_template, ss.claude_template)
            for code in ss.selected_sections:
                section_plan = next((it["title"] for it in (ss.plan_items or []) if it["code"] == code), code)
                df_corpus = ss.cm.extract_corpus_for_section(
                    code,
                    min_score=ss.min_relevance_score,
                    max_items=ss.max_citations_per_section,
                    include_secondary=ss.include_secondary_matches,
                    confidence_threshold=ss.confidence_threshold,
                )
                stats = {"corpus_count": len(df_corpus), "avg_score": float(df_corpus[ss.cm.col_score].mean()) if len(df_corpus) else 0}
                kws_map = []
                try:
                    if ss.cm.kmap and code in ss.cm.kmap:
                        kw = ss.cm.kmap[code].get("keywords", {})
                        kws_map = sorted(set((kw.get("primary") or []) + (kw.get("secondary") or [])))
                except Exception:
                    pass
                draft_prompt = pb.build_draft_prompt(
                    section_title=code, section_plan=section_plan,
                    corpus_entries=[{"Texte": v} for v in df_corpus[ss.cm.col_text].astype(str).tolist()],
                    keywords=kws_map, stats=stats,
                )
                
                # Utiliser la nouvelle fonction run_generation pour le brouillon
                result = run_generation(
                    mode="brouillon",
                    prompt=draft_prompt,
                    provider=ss.drafter_provider,
                    model=ss.drafter_model,
                    params=ss["draft_params"],
                    styles=ss.styles,
                    base_name=f"{code}_{section_plan}"
                )
                
                if result[0] is not None:  # Si pas d'erreur
                    draft_md, md_path, docx_path = result
                    ss.drafts[code] = draft_md
                else:
                    st.error(f"Erreur lors de la génération du brouillon pour {code}")
                    continue
            st.success("Étape 1 terminée. Allez à 'Brouillon (IA 1)' pour raffiner.")

elif page == "Prompts":
    st.subheader("Templates de prompts")
    ss.gpt_template = st.text_area("Template Étape 1 (IA 1)", value=ss.gpt_template, height=300)
    ss.claude_template = st.text_area("Template Étape 2 (IA 2)", value=ss.claude_template, height=200)

elif page == "Styles DOCX":
    st.subheader("Styles DOCX")
    styles = ss.styles
    styles["font_family"] = st.text_input("Police", styles.get("font_family", "Aptos"))
    styles["body_font_size"] = st.number_input("Taille corps", 8, 18, int(styles.get("body_font_size", 11)))
    styles["h1_font_size"] = st.number_input("Taille Titre 1", 10, 36, int(styles.get("h1_font_size", 14)))
    styles["h2_font_size"] = st.number_input("Taille Titre 2", 10, 36, int(styles.get("h2_font_size", 12)))
    c1,c2,c3,c4 = st.columns(4)
    with c1: styles["margins_cm"]["top"] = st.number_input("Marge haute (cm)", 0.0, 5.0, float(styles["margins_cm"]["top"]))
    with c2: styles["margins_cm"]["bottom"] = st.number_input("Marge basse (cm)", 0.0, 5.0, float(styles["margins_cm"]["bottom"]))
    with c3: styles["margins_cm"]["left"] = st.number_input("Marge gauche (cm)", 0.0, 5.0, float(styles["margins_cm"]["left"]))
    with c4: styles["margins_cm"]["right"] = st.number_input("Marge droite (cm)", 0.0, 5.0, float(styles["margins_cm"]["right"]))
    ss.styles = styles

elif page == "Nommage des fichiers":
    st.subheader("Nommage des fichiers")
    ss.prefix = st.text_input("Préfixe (ex: Chapitre_{code})", ss.prefix or "Chapitre_{code}")
    st.caption(f"Aperçu: output/{ss.prefix.format(code='1.1')}/{ss.prefix.format(code='1.1')}.md")

elif page == "Analyse du corpus":
    st.subheader("Couverture du corpus par section")
    if ss.cm and ss.plan_items:
        codes = [it["code"] for it in ss.plan_items if it.get("code")]
        selected = st.multiselect("Sections à analyser", codes, default=ss.selected_sections or codes[:10])
        if selected:
            cov = ss.cm.get_section_coverage(
                selected,
                min_score=ss.min_relevance_score,
                max_items=ss.max_citations_per_section,
                include_secondary=ss.include_secondary_matches,
                confidence_threshold=ss.confidence_threshold,
            )
            st.dataframe(cov, use_container_width=True)
            if ss.cm is not None and getattr(ss.cm, "df", None) is not None:
                st.caption("Aperçu (5 premières lignes de la feuille chargée):")
                st.dataframe(ss.cm.df.head(5), use_container_width=True)
    else:
        st.info("Chargez un plan + corpus dans Accueil.")

elif page == "Prévisualisation":
    st.subheader("Prévisualisation du corpus filtré")
    if ss.cm and ss.selected_sections:
        code = ss.selected_sections[0]
        df_corpus = ss.cm.extract_corpus_for_section(
            code,
            min_score=ss.min_relevance_score,
            max_items=ss.max_citations_per_section,
            include_secondary=ss.include_secondary_matches,
            confidence_threshold=ss.confidence_threshold,
        )
        st.write(f"Sous-partie: {code} — {len(df_corpus)} éléments")
        st.dataframe(df_corpus.head(100), use_container_width=True)
    else:
        st.info("Sélectionnez au moins une section dans 'Traitement'.")

elif page == "Brouillon (IA 1)":
    st.subheader("Brouillon (IA 1)")
    if ss.drafts:
        for code, text_md in ss.drafts.items():
            st.markdown(f"#### Section {code}")
            st.markdown(text_md)
            
            # Boutons de téléchargement pour le brouillon existant
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.download_button(
                    "Télécharger .md", 
                    data=text_md.encode("utf-8"), 
                    file_name=f"brouillon_{code}.md", 
                    mime="text/markdown",
                    key=f"dl_md_{code}"
                )
            with col2:
                # Export temporaire en docx pour téléchargement
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                    generate_styled_docx(text_md, tmp.name, ss.styles)
                    with open(tmp.name, "rb") as f:
                        docx_data = f.read()
                    os.unlink(tmp.name)
                st.download_button(
                    "Télécharger .docx", 
                    data=docx_data, 
                    file_name=f"brouillon_{code}.docx", 
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"dl_docx_{code}"
                )
            with col3:
                if st.button(f"Lancer le raffinage (IA 2) — {code}", key=f"refine_{code}"):
                    pb = PromptBuilder(ss.gpt_template, ss.claude_template)
                    refine_prompt = pb.build_refine_prompt(text_md)
                    
                    # Utiliser la nouvelle fonction run_generation
                    section_title = next((it["title"] for it in (ss.plan_items or []) if it["code"] == code), code)
                    result = run_generation(
                        mode="finale",
                        prompt=refine_prompt,
                        provider=ss.refiner_provider,
                        model=ss.refiner_model,
                        params=ss["final_params"],
                        styles=ss.styles,
                        base_name=f"{code}_{section_title}"
                    )
                    
                    if result[0] is not None:  # Si pas d'erreur
                        refined, md_path, docx_path = result
                        ss.finals[code] = refined
                        
                        # Boutons de téléchargement immédiat
                        st.markdown("**Version finale exportée :**")
                        dl_col1, dl_col2 = st.columns(2)
                        with dl_col1:
                            st.download_button(
                                "Télécharger .md", 
                                data=refined.encode("utf-8"), 
                                file_name=os.path.basename(md_path), 
                                mime="text/markdown",
                                key=f"final_dl_md_{code}"
                            )
                        with dl_col2:
                            with open(docx_path, "rb") as f:
                                st.download_button(
                                    "Télécharger .docx", 
                                    data=f.read(), 
                                    file_name=os.path.basename(docx_path), 
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    key=f"final_dl_docx_{code}"
                                )
    else:
        st.info("Aucun brouillon généré pour le moment.")

elif page == "Version Finale (IA 2)":
    st.subheader("Version Finale (IA 2)")
    if ss.finals:
        for code, text_md in ss.finals.items():
            st.markdown(f"#### Section {code}")
            st.markdown(text_md)
    else:
        st.info("Pas encore de versions finales.")

elif page == "Document Complet":
    st.subheader("Document Complet")
    if ss.finals:
        all_codes = list(ss.finals.keys())
        all_text = "\n\n".join([ss.finals[c] for c in all_codes])
        used_refs = extract_used_references_apa(all_text)
        biblio_md = ""
        if ss.excel_path:
            biblio_md = "\n\n## Bibliographie\n" + generate_bibliography(used_refs, ss.excel_path)
        final_md = all_text + biblio_md

        os.makedirs("output", exist_ok=True)
        base_name = "Document_Complet"
        docx_path = os.path.join("output", f"{base_name}.docx")
        generate_styled_docx(final_md, docx_path, ss.styles)
        with open(docx_path, "rb") as f:
            docx_bytes = f.read()

        st.download_button("Télécharger .md", data=final_md, file_name=f"{base_name}.md", mime="text/markdown")
        st.download_button("Télécharger .docx", data=docx_bytes, file_name=f"{base_name}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    else:
        st.info("Générez des versions finales pour composer le document complet.")
