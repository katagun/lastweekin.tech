"""Tests for configuration loading, environment expansion and validation."""

import pytest

from lastweekintech.config import Config, ConfigError, expand_env, get_config

MINIMAL = """
feeds:
  - name: Example
    url: https://example.com/rss
summarizer:
  model_name: vendor/model
"""


def write_config(tmp_path, body):
    path = tmp_path / "config.yaml"
    path.write_text(body)
    return path


class TestExpandEnv:
    def test_replaces_a_reference_with_the_environment_value(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "secret-value")
        assert expand_env("${HF_TOKEN}") == "secret-value"

    def test_resolves_an_unset_reference_to_none(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert expand_env("${HF_TOKEN}") is None

    def test_leaves_ordinary_strings_alone(self):
        assert expand_env("vendor/model") == "vendor/model"

    def test_expands_nested_structures(self, monkeypatch):
        monkeypatch.setenv("TOKEN", "abc")
        assert expand_env({"a": ["${TOKEN}", 1]}) == {"a": ["abc", 1]}


class TestFromYaml:
    def test_loads_a_minimal_configuration(self, tmp_path):
        config = Config.from_yaml(write_config(tmp_path, MINIMAL))
        assert config.feeds[0].name == "Example"
        assert config.summarizer.model_name == "vendor/model"

    def test_applies_defaults_for_omitted_sections(self, tmp_path):
        config = Config.from_yaml(write_config(tmp_path, MINIMAL))
        assert config.digest.story_count == 7
        assert config.window_days == 7
        assert config.hn.min_points == 50

    def test_expands_secrets_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf-secret")
        body = MINIMAL + '  hf_token: "${HF_TOKEN}"\n'
        assert Config.from_yaml(write_config(tmp_path, body)).summarizer.hf_token == "hf-secret"

    def test_an_unset_secret_disables_the_feature_rather_than_leaking_the_placeholder(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        body = MINIMAL + '  hf_token: "${HF_TOKEN}"\n'
        assert Config.from_yaml(write_config(tmp_path, body)).summarizer.hf_token is None

    def test_reports_a_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            Config.from_yaml(tmp_path / "absent.yaml")

    def test_reports_malformed_yaml(self, tmp_path):
        with pytest.raises(ConfigError):
            Config.from_yaml(write_config(tmp_path, "feeds: [unclosed"))

    def test_reports_an_unknown_key(self, tmp_path):
        with pytest.raises(ConfigError, match="Invalid configuration"):
            Config.from_yaml(write_config(tmp_path, MINIMAL + "\nnonsense_key: 1\n"))

    def test_rejects_a_configuration_with_no_feeds(self, tmp_path):
        body = "feeds: []\nsummarizer:\n  model_name: vendor/model\n"
        with pytest.raises(ConfigError, match="no feeds"):
            Config.from_yaml(write_config(tmp_path, body))

    def test_rejects_an_ai_floor_above_the_story_count(self, tmp_path):
        body = MINIMAL + "\ndigest:\n  story_count: 5\n  min_ai_stories: 6\n"
        with pytest.raises(ConfigError, match="min_ai_stories"):
            Config.from_yaml(write_config(tmp_path, body))

    def test_applies_ai_daily_defaults_when_the_section_is_omitted(self, tmp_path):
        config = Config.from_yaml(write_config(tmp_path, MINIMAL))
        assert config.ai_daily.enabled is True
        assert config.ai_daily.window_days == 1
        assert config.ai_daily.story_count == 5
        assert config.ai_daily.min_story_count == 4
        assert config.ai_daily.briefing.model_name == "anthropic/claude-sonnet-5"

    def test_rejects_an_unknown_ai_daily_key(self, tmp_path):
        body = MINIMAL + "\nai_daily:\n  nonsense: 1\n"
        with pytest.raises(ConfigError, match="Invalid configuration"):
            Config.from_yaml(write_config(tmp_path, body))


class TestAiDaily:
    def test_overrides_nested_hn_weights_and_briefing_settings(self, tmp_path):
        body = MINIMAL + (
            "\nai_daily:\n"
            "  story_count: 6\n"
            "  min_story_count: 5\n"
            "  hn:\n"
            "    min_points: 5\n"
            "  weights:\n"
            "    hn: 10\n"
            "  briefing:\n"
            "    model_name: vendor/other-model\n"
            "    max_tokens: 1234\n"
        )
        config = Config.from_yaml(write_config(tmp_path, body))
        assert config.ai_daily.story_count == 6
        assert config.ai_daily.min_story_count == 5
        assert config.ai_daily.hn.min_points == 5
        # Untouched nested defaults still apply.
        assert config.ai_daily.hn.points_cap == 300
        assert config.ai_daily.weights.hn == 10
        assert config.ai_daily.briefing.model_name == "vendor/other-model"
        assert config.ai_daily.briefing.max_tokens == 1234
        assert config.ai_daily.briefing.temperature == 0.3

    def test_rejects_a_min_story_count_above_story_count(self, tmp_path):
        body = MINIMAL + "\nai_daily:\n  story_count: 3\n  min_story_count: 4\n"
        with pytest.raises(ConfigError, match="min_story_count"):
            Config.from_yaml(write_config(tmp_path, body))

    def test_rejects_a_candidate_pool_smaller_than_story_count(self, tmp_path):
        body = MINIMAL + "\nai_daily:\n  story_count: 5\n  candidate_pool: 3\n"
        with pytest.raises(ConfigError, match="candidate_pool"):
            Config.from_yaml(write_config(tmp_path, body))

    def test_rejects_an_empty_briefing_model_name(self, tmp_path):
        body = MINIMAL + '\nai_daily:\n  briefing:\n    model_name: ""\n'
        with pytest.raises(ConfigError, match="briefing.model_name"):
            Config.from_yaml(write_config(tmp_path, body))


class TestShippedConfig:
    def test_the_packaged_configuration_is_valid(self):
        config = get_config()
        assert config.feeds
        assert config.digest.min_ai_stories <= config.digest.story_count

    def test_the_packaged_configuration_meets_the_spec_ratio(self):
        """The spec calls for at least half of the seven stories to be AI."""
        config = get_config()
        assert config.digest.min_ai_stories >= config.digest.story_count // 2
