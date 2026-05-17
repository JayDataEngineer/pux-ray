"""Tests for MOSS family handler — Wan2GP contract compliance."""
import sys
import pytest

sys.path.insert(0, "opt/wan2gp")

from models.moss.moss_handler import (
    family_handler,
    VARIANTS,
    _get_moss_model_def,
    _Pipeline,
)


class TestHandlerContract:
    """Verify Wan2GP family_handler static method contract."""

    def test_supported_types(self):
        types = family_handler.query_supported_types()
        assert types == ["moss-soundeffect", "moss-tts", "moss-ttsd", "moss-voicegenerator"]

    def test_family_maps(self):
        maps, maps2 = family_handler.query_family_maps()
        assert maps == {}
        assert maps2 == {}

    def test_model_family(self):
        assert family_handler.query_model_family() == "moss"

    def test_family_infos(self):
        infos = family_handler.query_family_infos()
        assert "moss" in infos
        assert infos["moss"] == (303, "MOSS Audio")

    def test_model_def_all_variants(self):
        for variant in VARIANTS:
            md = family_handler.query_model_def(variant, {})
            assert md["audio_only"] is True
            assert md["image_outputs"] is False
            assert "duration_slider" in md

    def test_model_def_soundeffect(self):
        md = _get_moss_model_def("moss-soundeffect")
        assert md["profiles_dir"] == ["moss-soundeffect"]

    def test_model_def_tts_has_voice_cloning(self):
        md = _get_moss_model_def("moss-tts")
        assert md["any_audio_prompt"] is True
        assert "alt_prompt" in md
        assert md["alt_prompt"]["label"] == "Instruction (optional emotion/style)"

    def test_model_def_voicegenerator_has_instruction(self):
        md = _get_moss_model_def("moss-voicegenerator")
        assert "alt_prompt" in md
        assert md["alt_prompt"]["label"] == "Voice instruction"


class TestDefaults:
    """Per-variant default settings."""

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_defaults_have_prompt(self, variant):
        defaults = {}
        family_handler.update_default_settings(variant, {}, defaults)
        assert "prompt" in defaults
        assert len(defaults["prompt"]) > 0

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_defaults_have_common_fields(self, variant):
        defaults = {}
        family_handler.update_default_settings(variant, {}, defaults)
        assert "temperature" in defaults
        assert "top_k" in defaults
        assert "duration_seconds" in defaults


class TestValidation:
    """Prompt validation."""

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_empty_prompt_rejected(self, variant):
        result = family_handler.validate_generative_prompt(variant, {}, {}, "")
        assert result is not None

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_valid_prompt_accepted(self, variant):
        result = family_handler.validate_generative_prompt(
            variant, {}, {}, "some text"
        )
        assert result is None


class TestPipelineRouting:
    """Verify _Pipeline._build_conversation routes to correct UserMessage fields."""

    def _make_pipeline(self, model_type):
        """Create a _Pipeline with a mock processor that records calls."""
        class MockProcessor:
            def __init__(self):
                self.last_msg_kwargs = None

            def build_user_message(self, **kwargs):
                self.last_msg_kwargs = kwargs
                return kwargs

        proc = MockProcessor()
        return _Pipeline(model=None, processor=proc, audio_tokenizer=None, model_type=model_type), proc

    def test_soundeffect_uses_ambient_sound(self):
        pipe, proc = self._make_pipeline("moss-soundeffect")
        pipe._build_conversation(input_prompt="thunder")
        assert proc.last_msg_kwargs["ambient_sound"] == "thunder"

    def test_tts_uses_text_field(self):
        pipe, proc = self._make_pipeline("moss-tts")
        pipe._build_conversation(input_prompt="Hello world")
        assert proc.last_msg_kwargs["text"] == "Hello world"

    def test_tts_with_instruction(self):
        pipe, proc = self._make_pipeline("moss-tts")
        pipe._build_conversation(input_prompt="Hello", instruction="warm and friendly")
        assert proc.last_msg_kwargs["text"] == "Hello"
        assert proc.last_msg_kwargs["instruction"] == "warm and friendly"

    def test_tts_with_reference(self):
        pipe, proc = self._make_pipeline("moss-tts")
        pipe._build_conversation(input_prompt="Hello", reference="ref.wav")
        assert proc.last_msg_kwargs["reference"] == ["ref.wav"]

    def test_ttsd_uses_text_field(self):
        pipe, proc = self._make_pipeline("moss-ttsd")
        pipe._build_conversation(input_prompt="How are you?")
        assert proc.last_msg_kwargs["text"] == "How are you?"

    def test_voicegenerator_uses_instruction(self):
        pipe, proc = self._make_pipeline("moss-voicegenerator")
        pipe._build_conversation(input_prompt="warm female voice")
        assert proc.last_msg_kwargs["instruction"] == "warm female voice"
        assert "text" not in proc.last_msg_kwargs

    def test_soundeffect_with_tokens(self):
        pipe, proc = self._make_pipeline("moss-soundeffect")
        pipe._build_conversation(input_prompt="rain", tokens=512)
        assert proc.last_msg_kwargs["tokens"] == 512

    def test_language_routing(self):
        pipe, proc = self._make_pipeline("moss-tts")
        pipe._build_conversation(input_prompt="Bonjour", language="fr")
        assert proc.last_msg_kwargs["language"] == "fr"
