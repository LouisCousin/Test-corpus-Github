
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

from config_manager import (
    MIN_RELEVANCE_SCORE, MAX_CITATIONS_PER_SECTION,
    INCLUDE_SECONDARY_MATCHES, CONFIDENCE_THRESHOLD
)

POSSIBLE_TEXT_COLS = ["Texte", "Extrait", "Citation", "Content", "Text"]

def _detect_text_col(df: pd.DataFrame) -> str:
    for c in POSSIBLE_TEXT_COLS:
        if c in df.columns:
            return c
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            return c
    return df.columns[0]

def _col(df: pd.DataFrame, candidates: List[str], fallback_index: Optional[int] = None) -> str:
    low = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        n = name.lower()
        if n in low:
            return low[n]
        for k, original in low.items():
            if n in k:
                return original
    if fallback_index is not None and 0 <= fallback_index < df.shape[1]:
        return df.columns[fallback_index]
    return df.columns[0]

class CorpusManager:
    def __init__(self, excel_path: str, keywords_json_path: Optional[str] = None, column_overrides: Optional[dict] = None):
        self.path = excel_path
        self.kmap_path = keywords_json_path
        self.df = self._load_excel(excel_path)
        self.kmap = self._load_keywords(keywords_json_path) if keywords_json_path else {}
        self.overrides = column_overrides or {}

        cand_sections = ["sections", "sections matchées", "sections associees", "section_code", "sous-partie", "sous-parties", "sections_matchées", "sections_matchees"]
        cand_score = ["score", "score de pertinence", "pertinence", "relevance", "m", "score (0-10)", "score_pertinence", "score-pertinence"]
        cand_keywords = ["mots_clés_trouvés", "mots-clés trouvés", "keywords_found", "mots-clés", "mots cles", "keywords", "n"]
        cand_type = ["type", "match", "primary", "secondary", "o", "type de match", "type_match"]
        cand_conf = ["confiance", "confidence", "p", "niveau de confiance", "confiance_pct", "confiance (%)"]

        self.col_sections   = self.overrides.get("sections")   or _col(self.df, cand_sections, 11 if self.df.shape[1] > 11 else None)
        self.col_score      = self.overrides.get("score")      or _col(self.df, cand_score, 12 if self.df.shape[1] > 12 else None)
        self.col_keywords   = self.overrides.get("keywords")   or _col(self.df, cand_keywords, 13 if self.df.shape[1] > 13 else None)
        self.col_match_type = self.overrides.get("type")       or _col(self.df, cand_type, 14 if self.df.shape[1] > 14 else None)
        self.col_confidence = self.overrides.get("confidence") or _col(self.df, cand_conf, 15 if self.df.shape[1] > 15 else None)
        self.col_text       = self.overrides.get("text")       or _detect_text_col(self.df)

    def _load_excel(self, path: str) -> pd.DataFrame:
        p = Path(path)
        suf = p.suffix.lower()
        if suf in [".xlsx", ".xlsm", ".xls"]:
            try:
                xls = pd.ExcelFile(p, engine="openpyxl")
            except Exception:
                xls = pd.ExcelFile(p)
            pick = None
            for name in xls.sheet_names:
                if "extrait" in name.lower():
                    pick = name
                    break
            if pick is None:
                pick = xls.sheet_names[0]
            return xls.parse(pick)
        elif suf == ".csv":
            return pd.read_csv(p)
        else:
            raise ValueError(f"Unsupported file type: {p.suffix}")

    def _load_keywords(self, path: str) -> Dict[str, Any]:
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _sections_list(self, cell: Any) -> List[str]:
        if pd.isna(cell):
            return []
        parts = re.split(r"[|,;\n]+", str(cell))
        return [p.strip() for p in parts if p.strip()]
    
    def _convert_arabic_to_roman_section(self, section_code: str) -> List[str]:
        """Convertit une section arabe (1.2.3) vers les formats romains possibles"""
        if not section_code:
            return [section_code]
        
        # Mapper les chiffres arabes vers romains pour les parties principales
        roman_map = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV', '5': 'V'}
        
        # Si c'est déjà en format romain, on garde tel quel
        if any(roman in section_code for roman in ['I', 'II', 'III', 'IV', 'V']):
            return [section_code]
        
        variations = [section_code]  # Garde l'original
        
        parts = section_code.split('.')
        if len(parts) >= 1 and parts[0] in roman_map:
            # Convertir la partie principale (ex: 1.2.3 -> I.2.3)
            roman_version = roman_map[parts[0]] + '.' + '.'.join(parts[1:]) if len(parts) > 1 else roman_map[parts[0]]
            variations.append(roman_version)
            
            # Version raccourcie si plusieurs niveaux (ex: 1.2.3 -> I.2)
            if len(parts) >= 3:
                short_version = roman_map[parts[0]] + '.' + parts[1]
                variations.append(short_version)
        
        return list(set(variations))  # Supprimer doublons

    def extract_corpus_for_section(
        self,
        section_code: str,
        min_score: int = MIN_RELEVANCE_SCORE,
        max_items: int = MAX_CITATIONS_PER_SECTION,
        include_secondary: bool = INCLUDE_SECONDARY_MATCHES,
        confidence_threshold: int = CONFIDENCE_THRESHOLD,
    ) -> pd.DataFrame:
        df = self.df.copy()

        def _match_codes(cell: Any) -> bool:
            codes = self._sections_list(cell)
            # Essayer toutes les variations du code de section
            for variation in self._convert_arabic_to_roman_section(section_code):
                if variation in codes:
                    return True
                prefix = f"{variation}."
                if any(str(c).startswith(prefix) for c in codes):
                    return True
            return False

        mask_section = df[self.col_sections].apply(_match_codes)
        df = df[mask_section]

        if not include_secondary:
            df = df[df[self.col_match_type].astype(str).str.lower().str.contains("primary")]

        df = df[df[self.col_score].fillna(0).astype(float) >= float(min_score)]
        df = df[df[self.col_confidence].fillna(0).astype(float) >= float(confidence_threshold)]

        df = df.sort_values(by=self.col_score, ascending=False)

        if max_items:
            df = df.head(int(max_items)).copy()

        if self.kmap and section_code in self.kmap:
            kw = self.kmap[section_code].get("keywords", {})
            df["__keywords_map__"] = ", ".join(sorted(set((kw.get("primary") or []) + (kw.get("secondary") or []))))
        else:
            df["__keywords_map__"] = ""

        if self.col_text not in df.columns:
            df[self.col_text] = df.apply(lambda r: " | ".join([str(v) for v in r.values if isinstance(v, (str,int,float))][:3]), axis=1)

        return df

    def get_section_coverage(self, section_codes: List[str], **kwargs) -> pd.DataFrame:
        rows = []
        for code in section_codes:
            d = self.extract_corpus_for_section(code, **kwargs)
            count = int(len(d))
            status = "vert" if count >= 10 else ("orange" if count >= 3 else "rouge")
            rows.append({
                "section": code,
                "citations": count,
                "statut": status,
                "score_moyen": round(float(d[self.col_score].mean()), 2) if count else 0
            })
        return pd.DataFrame(rows)

    def format_corpus_for_prompt(self, df: pd.DataFrame) -> str:
        lines = []
        for _, row in df.iterrows():
            sc = row.get(self.col_score, "")
            mt = row.get(self.col_match_type, "")
            conf = row.get(self.col_confidence, "")
            txt = str(row.get(self.col_text, ""))
            lines.append(f"- ({sc}/10; {mt}; {conf}%) {txt}")
        return "\n".join(lines)
