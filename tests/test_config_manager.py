
from src.config_manager import get_config
def test_config_load_defaults():
    cfg = get_config()
    assert cfg.styles["font_family"]
