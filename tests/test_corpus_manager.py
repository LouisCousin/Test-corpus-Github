
import pandas as pd, tempfile
from src.core.corpus_manager import CorpusManager
from pathlib import Path

def test_extract(tmp_path: Path):
    p = tmp_path / "mini.csv"
    p.write_text("Sections,Score,Texte,MatchType,Confidence\n1.1|1.2,9,Hello,primary,80\n2,3,World,secondary,50\n", encoding="utf-8")
    cm = CorpusManager(str(p))
    df = cm.extract_corpus_for_section("1.1", min_score=5, include_secondary=False, confidence_threshold=60)
    assert len(df) == 1 and df.iloc[0]["Texte"] == "Hello"
