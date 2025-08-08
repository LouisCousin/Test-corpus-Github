import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.core.utils import (
    truncate_to_tokens, export_markdown, export_docx, 
    call_openai, call_anthropic, _generate_filename
)
from src.config_manager import get_config


class TestTokenTruncation:
    def test_truncate_to_tokens_with_tiktoken(self):
        """Test la troncature avec tiktoken si disponible."""
        text = "Ceci est un texte de test assez long pour être tronqué par la fonction de troncature des tokens."
        max_tokens = 10
        
        # Test avec un texte plus long que la limite
        result = truncate_to_tokens(text, max_tokens)
        assert len(result) <= len(text)
        assert isinstance(result, str)
    
    def test_truncate_to_tokens_fallback(self):
        """Test la troncature avec fallback heuristique."""
        with patch('src.core.utils.tiktoken', side_effect=ImportError):
            text = "A" * 1000  # 1000 caractères
            max_tokens = 100   # 100 tokens ≈ 400 caractères
            
            result = truncate_to_tokens(text, max_tokens)
            assert len(result) <= 400  # max_tokens * 4
    
    def test_truncate_to_tokens_no_truncation_needed(self):
        """Test quand aucune troncature n'est nécessaire."""
        text = "Court texte"
        max_tokens = 1000
        
        result = truncate_to_tokens(text, max_tokens)
        assert result == text


class TestExportFunctions:
    def test_generate_filename(self):
        """Test la génération de noms de fichiers."""
        filename = _generate_filename("Test Section 1.1", "brouillon")
        
        # Vérifier le format : YYYYMMDD-HHMMSS_mode_slug
        parts = filename.split("_")
        assert len(parts) >= 3
        assert parts[1] == "brouillon"
        assert "Test" in filename or "test" in filename.lower()
    
    def test_export_markdown(self):
        """Test l'export markdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text = "# Test Markdown\n\nCeci est un test."
            base_name = "test_section"
            mode = "brouillon"
            
            result_path = export_markdown(text, base_name, mode, tmpdir)
            
            assert os.path.exists(result_path)
            assert result_path.endswith(".md")
            
            # Vérifier le contenu
            with open(result_path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert content == text
    
    def test_export_docx(self):
        """Test l'export DOCX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            text = "# Test DOCX\n\nCeci est un test."
            base_name = "test_section"
            mode = "finale"
            styles = {"font_family": "Arial", "body_font_size": 11}
            
            result_path = export_docx(text, base_name, mode, tmpdir, styles)
            
            assert os.path.exists(result_path)
            assert result_path.endswith(".docx")


class TestLLMCalls:
    @patch('src.core.utils.OpenAI')
    def test_call_openai_with_parameters(self, mock_openai_class):
        """Test l'appel OpenAI avec les nouveaux paramètres."""
        # Mock de la réponse
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = call_openai(
            model_name="gpt-4",
            prompt="Test prompt",
            api_key="test_key",
            temperature=0.8,
            top_p=0.9,
            max_output_tokens=500
        )
        
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.8,
            top_p=0.9,
            max_tokens=500
        )
    
    @patch('src.core.utils.anthropic')
    def test_call_anthropic_with_parameters(self, mock_anthropic_module):
        """Test l'appel Anthropic avec les nouveaux paramètres."""
        # Mock de la réponse
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = "Test response"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_module.Anthropic.return_value = mock_client
        
        result = call_anthropic(
            model_name="claude-3-sonnet",
            prompt="Test prompt",
            api_key="test_key",
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=1000
        )
        
        assert result == "Test response"
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-sonnet",
            max_tokens=1000,
            temperature=0.7,
            top_p=0.95,
            messages=[{"role": "user", "content": "Test prompt"}]
        )


class TestConfigurationParameters:
    def test_draft_params_defaults(self):
        """Test les paramètres par défaut pour le brouillon."""
        cfg = get_config()
        
        assert "draft_params" in cfg.__dict__
        draft_params = cfg.draft_params
        
        assert "temperature" in draft_params
        assert "top_p" in draft_params
        assert "max_input_tokens" in draft_params
        assert "max_output_tokens" in draft_params
        
        # Vérifier les valeurs par défaut attendues
        assert draft_params["temperature"] == 0.9
        assert draft_params["top_p"] == 0.95
        assert draft_params["max_input_tokens"] == 4000
        assert draft_params["max_output_tokens"] == 800
    
    def test_final_params_defaults(self):
        """Test les paramètres par défaut pour la version finale."""
        cfg = get_config()
        
        assert "final_params" in cfg.__dict__
        final_params = cfg.final_params
        
        assert "temperature" in final_params
        assert "top_p" in final_params
        assert "max_input_tokens" in final_params
        assert "max_output_tokens" in final_params
        
        # Vérifier les valeurs par défaut attendues
        assert final_params["temperature"] == 0.7
        assert final_params["top_p"] == 0.9
        assert final_params["max_input_tokens"] == 8000
        assert final_params["max_output_tokens"] == 1600
    
    def test_export_dir_default(self):
        """Test le répertoire d'export par défaut."""
        cfg = get_config()
        
        assert hasattr(cfg, "export_dir")
        assert cfg.export_dir == "output"


class TestIntegration:
    def test_complete_workflow_mock(self):
        """Test d'intégration du workflow complet (avec mocks)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simuler un appel complet avec tous les nouveaux paramètres
            text_input = "# Test\n\nCeci est un test d'intégration."
            
            # Test de troncature
            truncated = truncate_to_tokens(text_input, 1000)
            assert truncated == text_input  # Pas de troncature nécessaire
            
            # Test d'export markdown
            md_path = export_markdown(truncated, "test_integration", "test", tmpdir)
            assert os.path.exists(md_path)
            
            # Test d'export docx
            docx_path = export_docx(truncated, "test_integration", "test", tmpdir)
            assert os.path.exists(docx_path)
            
            # Vérifier que les fichiers ont été créés avec les bons noms
            assert "test_integration" in os.path.basename(md_path)
            assert "test_integration" in os.path.basename(docx_path)