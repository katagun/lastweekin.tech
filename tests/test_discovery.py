"""Tests for the Perplexity consensus check."""

import json

import pytest
from conftest import NOW, make_article, make_story

from lastweekintech import discovery
from lastweekintech.config import PerplexitySettings
from lastweekintech.discovery import ConsensusStory


@pytest.fixture
def px_config(config):
    config.perplexity = PerplexitySettings(api_key="test-key")
    return config


def canned(entries):
    payload = json.dumps(entries)

    def search(model, prompt):
        return payload

    return search


class TestFetchConsensus:
    def test_returns_parsed_stories(self, px_config):
        entries = [{"headline": "EU fines a chipmaker", "why": "Precedent.", "urls": ["https://x"]}]
        stories = discovery.fetch_consensus(px_config, now=NOW, search=canned(entries))
        assert [s.headline for s in stories] == ["EU fines a chipmaker"]
        assert stories[0].urls == ["https://x"]

    def test_disabled_stage_returns_nothing(self, px_config):
        px_config.perplexity.enabled = False
        assert discovery.fetch_consensus(px_config, now=NOW, search=canned([{}])) == []

    def test_missing_key_skips_rather_than_fails(self, px_config):
        px_config.perplexity.api_key = None
        assert discovery.fetch_consensus(px_config, now=NOW) == []

    def test_a_failing_search_costs_the_signal_not_the_run(self, px_config):
        def search(model, prompt):
            raise RuntimeError("api down")

        assert discovery.fetch_consensus(px_config, now=NOW, search=search) == []

    def test_caps_the_list_at_the_configured_count(self, px_config):
        px_config.perplexity.story_count = 2
        entries = [{"headline": f"story {i}"} for i in range(10)]
        assert len(discovery.fetch_consensus(px_config, now=NOW, search=canned(entries))) == 2

    def test_the_prompt_names_the_window(self, px_config):
        seen = {}

        def search(model, prompt):
            seen["prompt"] = prompt
            return "[]"

        discovery.fetch_consensus(px_config, now=NOW, search=search)
        assert "2026-08-03" in seen["prompt"] and "2026-08-10" in seen["prompt"]

    def test_defaults_to_the_weekly_prompt(self, px_config):
        seen = {}

        def search(model, prompt):
            seen["prompt"] = prompt
            return "[]"

        discovery.fetch_consensus(px_config, now=NOW, search=search)
        assert "weekly technology digest" in seen["prompt"]

    def test_accepts_a_custom_prompt_template(self, px_config):
        seen = {}

        def search(model, prompt):
            seen["prompt"] = prompt
            return "[]"

        discovery.fetch_consensus(
            px_config, now=NOW, search=search, prompt_template=discovery.AI_DAILY_PROMPT_TEMPLATE
        )
        assert "AI news editor of a daily AI-only briefing" in seen["prompt"]
        assert "2026-08-10" in seen["prompt"]


class TestParseConsensus:
    def test_tolerates_prose_and_code_fences_around_the_array(self):
        answer = 'Here you go:\n```json\n[{"headline": "A story"}]\n```\nHope that helps!'
        assert [s.headline for s in discovery.parse_consensus(answer)] == ["A story"]

    def test_unparseable_answers_yield_nothing(self):
        assert discovery.parse_consensus("no json here") == []
        assert discovery.parse_consensus("[not, valid") == []
        assert discovery.parse_consensus("") == []

    def test_malformed_elements_are_dropped_not_fatal(self):
        answer = json.dumps([{"headline": "Good"}, "junk", {"why": "no headline"}, None])
        assert [s.headline for s in discovery.parse_consensus(answer)] == ["Good"]


class TestApplyConsensusBoost:
    def test_a_url_match_is_decisive(self):
        story = make_story(
            title="Totally different words",
            score=1.0,
            articles=[make_article(url="https://ars.example/chips?utm_source=rss")],
        )
        entry = ConsensusStory(headline="Unrelated headline", urls=["https://ars.example/chips"])
        missed = discovery.apply_consensus_boost([story], [entry], weight=2.0)
        assert story.consensus and story.score == 3.0
        assert missed == []

    def test_a_paraphrased_headline_matches_on_shared_substance(self):
        story = make_story(title="Nvidia unveils its Rubin GPU architecture at GTC", score=1.0)
        entry = ConsensusStory(headline="Nvidia announces new Rubin GPU architecture")
        missed = discovery.apply_consensus_boost([story], [entry], weight=2.0)
        assert story.consensus and story.score == 3.0
        assert missed == []

    def test_thin_word_overlap_does_not_match(self):
        story = make_story(title="Apple updates its App Store rules", score=1.0)
        entry = ConsensusStory(headline="Apple updates its Mac lineup with new chips")
        missed = discovery.apply_consensus_boost([story], [entry], weight=2.0)
        assert not story.consensus and story.score == 1.0
        assert missed == [entry]

    def test_each_consensus_story_boosts_at_most_one_candidate(self):
        first = make_story(title="Nvidia unveils Rubin GPU architecture", score=5.0)
        second = make_story(title="Nvidia's Rubin GPU architecture unveiled", score=1.0)
        entry = ConsensusStory(headline="Nvidia announces Rubin GPU architecture")
        discovery.apply_consensus_boost([first, second], [entry], weight=2.0)
        assert (first.consensus, second.consensus).count(True) == 1

    def test_unmatched_stories_are_returned_for_the_metrics(self):
        story = make_story(title="A quiet week in databases", score=1.0)
        entries = [ConsensusStory(headline="SpaceX Starship reaches orbit")]
        missed = discovery.apply_consensus_boost([story], entries, weight=2.0)
        assert [m.headline for m in missed] == ["SpaceX Starship reaches orbit"]


class TestStoriesFromMissed:
    def test_builds_a_single_article_story_per_entry(self):
        missed = [
            ConsensusStory(
                headline="A story the feeds never saw",
                urls=["https://www.example.com/a-story"],
            )
        ]
        stories = discovery.stories_from_missed(missed, weight=2.0)
        assert len(stories) == 1
        story = stories[0]
        assert story.title == "A story the feeds never saw"
        assert story.score == 2.0
        assert story.consensus is True
        assert len(story.articles) == 1
        assert story.articles[0].url == "https://www.example.com/a-story"
        assert story.articles[0].title == "A story the feeds never saw"

    def test_derives_the_source_from_the_url_domain(self):
        missed = [ConsensusStory(headline="x", urls=["https://www.theverge.com/a/b"])]
        stories = discovery.stories_from_missed(missed, weight=2.0)
        assert stories[0].articles[0].source == "theverge.com"

    def test_an_entry_with_no_urls_yields_no_candidate(self):
        missed = [ConsensusStory(headline="Nothing to fetch", urls=[])]
        assert discovery.stories_from_missed(missed, weight=2.0) == []

    def test_uses_only_the_first_url_when_several_are_given(self):
        missed = [ConsensusStory(headline="x", urls=["https://a.example/1", "https://b.example/2"])]
        stories = discovery.stories_from_missed(missed, weight=2.0)
        assert len(stories) == 1
        assert stories[0].articles[0].url == "https://a.example/1"
