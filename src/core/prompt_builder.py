
from typing import List, Dict, Any
from config_manager import DEFAULT_GPT_PROMPT_TEMPLATE

class PromptBuilder:
    def __init__(self, draft_template: str = DEFAULT_GPT_PROMPT_TEMPLATE, refine_template: str = "Réécris ce texte en le condensant et en unifiant le style."):
        self.draft_template = draft_template
        self.refine_template = refine_template

    def build_draft_prompt(self, section_title: str, section_plan: str, corpus_entries: List[Dict[str, Any]],
                           keywords: List[str] = None, previous_summaries: str = "", stats: Dict[str, Any] = None) -> str:
        corpus_text = "\\n".join([f"- {e.get('Texte') or e.get('text') or ''}" for e in corpus_entries])
        kw = ", ".join(keywords or [])
        stats = stats or {}
        tpl = self.draft_template
        tpl = tpl.replace("{section_title}", section_title)
        tpl = tpl.replace("{section_plan}", section_plan or section_title)
        tpl = tpl.replace("{corpus}", corpus_text)
        tpl = tpl.replace("{keywords_found}", kw)
        tpl = tpl.replace("{corpus_count}", str(stats.get("corpus_count", len(corpus_entries))))
        tpl = tpl.replace("{avg_score}", str(stats.get("avg_score", "")))
        tpl = tpl.replace("{previous_summaries}", previous_summaries or "")
        return tpl

    def build_refine_prompt(self, draft_markdown: str) -> str:
        return f"{self.refine_template}\\n\\n---\\n\\nTexte:\\n{draft_markdown}"
