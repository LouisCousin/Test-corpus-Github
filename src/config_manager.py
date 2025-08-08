
from __future__ import annotations
import os, copy, yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List

__version__ = "2.0.0"

# Catalogues modèles
AVAILABLE_DRAFTER_MODELS = ["GPT-4.1", "GPT-4.1 mini", "GPT-4.1 nano"]
AVAILABLE_REFINER_MODELS = ["Claude 4 Sonnet"]

# Fournisseurs détaillés
AVAILABLE_OPENAI_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o-mini"]
AVAILABLE_ANTHROPIC_MODELS = ["claude-3.5-sonnet-20240620", "claude-3-opus-20240229"]

MODEL_ALIASES = {
    "GPT-4.1": "gpt-4.1",
    "GPT-4.1 mini": "gpt-4.1-mini",
    "GPT-4.1 nano": "gpt-4.1-nano",
    "Claude 4 Sonnet": "claude-3.5-sonnet-20240620",
}

# Defaults
DEFAULT_DRAFTER_MODEL = "GPT-4.1"
DEFAULT_REFINER_MODEL = "Claude 4 Sonnet"
API_RETRY_DELAYS = [60, 300]

DEFAULT_STYLES = {

    "font_family": "Aptos",
    "body_font_size": 11,
    "h1_font_size": 14,
    "h2_font_size": 12,
    "margins_cm": {"top": 2.54, "bottom": 2.54, "left": 2.54, "right": 2.54},
}

# Chemins par défaut (peuvent être surchargés dans config/user.yaml)
DEFAULT_PATHS = {
    "plan_docx": "",
    "excel_corpus": "",
    "keywords_json": "",
    "env_dir": "",
}

# Corpus filtering
MIN_RELEVANCE_SCORE = 7
MAX_CITATIONS_PER_SECTION = 30
INCLUDE_SECONDARY_MATCHES = True
CONFIDENCE_THRESHOLD = 60

DEFAULT_GPT_PROMPT_TEMPLATE = r"""
# Rédaction de la sous-partie {section_title}
## Contexte et objectifs
{section_plan}
## Consignes de rédaction
- Utilise un maximum d'analyses et citations du corpus fourni.
- Intègre chaque citation entre guillemets et au format APA : (Auteur, Année).
- Structure en markdown.
- Termine par un résumé flash de 200-500 tokens.
## Statistiques du corpus
- Nombre d'éléments: {corpus_count}
- Score moyen: {avg_score}
- Mots-clés détectés: {keywords_found}
## Corpus à utiliser
{corpus}
## Résumés flash précédents
{previous_summaries}
"""

DEFAULT_CLAUDE_PROMPT_TEMPLATE = r"""
Tu es un éditeur de style. Réécris et condense le texte fourni en assurant cohérence,
clarté et fluidité, en gardant les citations APA. Harmonise le ton et la terminologie.
"""

@dataclass
class AppConfig:
    drafter_model: str = DEFAULT_DRAFTER_MODEL
    refiner_model: str = DEFAULT_REFINER_MODEL
    api_retry_delays: List[int] = field(default_factory=lambda: API_RETRY_DELAYS.copy())
    styles: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_STYLES))
    default_paths: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_PATHS))

    min_relevance_score: int = MIN_RELEVANCE_SCORE
    max_citations_per_section: int = MAX_CITATIONS_PER_SECTION
    include_secondary_matches: bool = INCLUDE_SECONDARY_MATCHES
    confidence_threshold: int = CONFIDENCE_THRESHOLD

    gpt_prompt_template: str = DEFAULT_GPT_PROMPT_TEMPLATE
    claude_prompt_template: str = DEFAULT_CLAUDE_PROMPT_TEMPLATE
    
    # Paramètres d'export et hyperparamètres par traitement
    export_dir: str = "output"
    draft_params: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.9,
        "top_p": 0.95,
        "max_input_tokens": 4000,
        "max_output_tokens": 800
    })
    final_params: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_input_tokens": 8000,
        "max_output_tokens": 1600
    })

def _deep_update(base: dict, updates: dict) -> dict:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base

def load_user_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_prompts_yaml(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    res = {}
    if "drafter" in data: res["gpt_prompt_template"] = data["drafter"]
    if "refiner" in data: res["claude_prompt_template"] = data["refiner"]
    return res

def get_config(user_yaml_path: str = "config/user.yaml",
               prompts_yaml_path: str = "config/prompts.yaml") -> AppConfig:
    base = AppConfig().__dict__
    overrides = load_user_yaml(user_yaml_path)
    prompt_overrides = load_prompts_yaml(prompts_yaml_path)
    merged = _deep_update(base.copy(), overrides)
    merged = _deep_update(merged, prompt_overrides)
    return AppConfig(**merged)
