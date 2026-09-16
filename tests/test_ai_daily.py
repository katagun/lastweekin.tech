"""End-to-end tests for the AI Daily pipeline, with the network stubbed out."""

import json
from datetime import timedelta
from pathlib import Path

from conftest import NOW, make_article
from test_build_digest import entry, feeds_for

from lastweekintech import ai_daily
from lastweekintech.briefer import BriefPick, BriefVerdict
from lastweekintech.config import Feed


class FakeBriefer:
    def __init__(self, verdict):
        self.verdict = verdict
        self.last_model = "fake/briefer"
        self.last_completion_tokens = None
        self.last_reasoning_tokens = None
        self.max_tokens = 0
        self.seen_recent_titles = None

    def brief(self, candidates, count, min_count, max_per_source, recent_titles):
        self.seen_recent_titles = recent_titles
        self.seen_candidates = [s.title for s in candidates]
        self.seen_candidates_full = candidates
        return self.verdict


def small_config(config):
    config.ai_daily.story_count = 2
    config.ai_daily.min_story_count = 1
    config.ai_daily.candidate_pool = 10
    config.feeds = [Feed(name="Ars Technica", url="ars"), Feed(name="WIRED", url="wired")]
    return config


def brief_pick(n, theme="Capabilities"):
    return BriefPick(
        n=n,
        theme=theme,
        what_happened=f"Thing {n} happened.",
        why_it_matters=f"It matters because {n}.",
        watch_next=f"Watch for {n} next.",
    )


def unique_entries(count, prefix="https://ars.example"):
    subjects = [
        "OpenAI ships a new reasoning model",
        "Anthropic publishes a safety paper",
        "EU regulators open an AI antitrust probe",
        "A startup open-sources a 70B model",
        "Nvidia announces its next AI chip",
        "A major AI funding round closes",
    ]
    return [
        entry(f"{subjects[i % len(subjects)]} ({i})", f"{prefix}/{i}", age_hours=1 + i)
        for i in range(count)
    ]


def run(config, briefer, **overrides):
    kwargs = {
        "now": NOW,
        "parse": feeds_for({}),
        "download": lambda url: f"body for {url}",
        "hn_fetch": lambda url, params: {"hits": []},
        "delay": 0,
    }
    return ai_daily.build_ai_daily(config, briefer, **(kwargs | overrides))


class TestBuildAiDaily:
    def test_produces_the_briefer_verdicts_as_ai_briefs(self, config):
        parse = feeds_for({"ars": unique_entries(5), "wired": []})
        briefer = FakeBriefer(BriefVerdict(picks=[brief_pick(2), brief_pick(1)], intro="A day."))
        digest = run(small_config(config), briefer, parse=parse)
        assert digest.intro == "A day."
        assert len(digest.briefs) == 2
        assert digest.briefs[0].what_happened == "Thing 2 happened."
        assert digest.briefs[0].theme == "Capabilities"

    def test_a_failed_briefer_yields_an_empty_digest(self, config):
        parse = feeds_for({"ars": unique_entries(5), "wired": []})
        digest = run(small_config(config), FakeBriefer(None), parse=parse)
        assert digest.briefs == []
        assert digest.intro is None

    def test_a_perplexity_consensus_match_boosts_a_story_into_the_candidate_pool(self, config):
        # Three "loud" recent stories exactly fill a 3-slot candidate pool on
        # ranking alone; a fourth, older story only a consensus boost can lift
        # in — this is the fix for the live failure where a 14-feed, 24-hour
        # pool recalled only what those feeds happened to publish, missing AI
        # stories the wider web already carried.
        test_config = small_config(config)
        test_config.ai_daily.candidate_pool = 3
        test_config.ai_daily.weights.consensus = 10
        test_config.perplexity.api_key = "test-key"

        loud = unique_entries(3)
        quiet = entry("A quiet AI policy story", "https://ars.example/quiet", age_hours=23)
        parse = feeds_for({"ars": [*loud, quiet], "wired": []})

        def search(model, prompt):
            return json.dumps([
                {"headline": "A quiet AI policy story", "urls": ["https://ars.example/quiet"]}
            ])

        briefer = FakeBriefer(BriefVerdict(picks=[]))
        run(test_config, briefer, parse=parse, search=search)
        assert "A quiet AI policy story" in briefer.seen_candidates

    def test_a_consensus_story_the_pool_never_fetched_becomes_a_real_candidate(self, config):
        # The corroboration boost above only helps a story already in the
        # pool. Most of what a live search finds is not there at all — this
        # is the fix for that: a consensus entry with no match in our own
        # fetch gets ingested as its own candidate, body extracted like any
        # other, so the briefer can actually pick it rather than the miss
        # only being logged.
        test_config = small_config(config)
        parse = feeds_for({"ars": unique_entries(3), "wired": []})

        def search(model, prompt):
            return json.dumps([
                {
                    "headline": "A story no feed carried",
                    "urls": ["https://exclusive.example/story"],
                }
            ])

        briefer = FakeBriefer(BriefVerdict(picks=[]))
        run(test_config, briefer, parse=parse, search=search)
        assert "A story no feed carried" in briefer.seen_candidates

    def test_a_consensus_story_with_no_extractable_body_is_not_picked(self, config):
        # A candidate that never gets a real article body still has to fail
        # Briefer's own no-article-text check — ingesting a headline is not a
        # way around "no article, no publish."
        test_config = small_config(config)
        parse = feeds_for({"ars": unique_entries(3), "wired": []})

        def search(model, prompt):
            return json.dumps([
                {"headline": "An unreachable story", "urls": ["https://dead.example/story"]}
            ])

        briefer = FakeBriefer(BriefVerdict(picks=[]))
        run(
            test_config,
            briefer,
            parse=parse,
            search=search,
            download=lambda url: "" if "dead.example" in url else f"body for {url}",
        )
        assert "An unreachable story" in briefer.seen_candidates
        # It was handed to the briefer as a candidate, but with no body —
        # confirmed via the same field Briefer._has_article_text checks.
        ingested = next(
            c for c in briefer.seen_candidates_full if c.title == "An unreachable story"
        )
        assert not any(a.content for a in ingested.articles)

    def test_no_perplexity_key_does_not_fail_the_run(self, config):
        # discovery.fetch_consensus already degrades gracefully when
        # unconfigured; this just confirms build_ai_daily doesn't need a
        # search function or an API key to complete.
        parse = feeds_for({"ars": unique_entries(3), "wired": []})
        digest = run(
            small_config(config), FakeBriefer(BriefVerdict(picks=[brief_pick(1)])), parse=parse
        )
        assert len(digest.briefs) == 1

    def test_uses_the_ai_daily_window_and_hn_settings_not_the_weekly_ones(self, config):
        # A story published 3 days ago is outside ai_daily's 1-day window but
        # would be inside the weekly config's 7-day window. An empty verdict
        # is used deliberately: with candidates correctly filtered down to
        # none, a non-empty canned verdict would make select_ai_daily_edition
        # index into an empty candidate list.
        config = small_config(config)
        config.window_days = 7
        parse = feeds_for({
            "ars": [entry("An old AI story", "https://ars.example/old", age_hours=72)],
            "wired": [],
        })
        briefer = FakeBriefer(BriefVerdict(picks=[]))
        run(config, briefer, parse=parse)
        assert briefer.seen_candidates == []

    def test_passes_recent_archive_titles_to_the_briefer(self, config):
        parse = feeds_for({"ars": unique_entries(3), "wired": []})
        briefer = FakeBriefer(BriefVerdict(picks=[brief_pick(1)]))
        editions = [
            {
                "date": (NOW - timedelta(days=1)).strftime("%Y-%m-%d"),
                "generated_at": NOW.isoformat(),
                "stories": [{"title": "Yesterday's AI story", "url": "https://x/1"}],
            }
        ]
        run(small_config(config), briefer, parse=parse, editions=editions)
        assert "Yesterday's AI story" in briefer.seen_recent_titles

    def test_drops_a_story_already_published_in_the_ai_daily_archive(self, config):
        # story_count=1 (rather than small_config's default 2) means one
        # fresh story already meets keep_at_least, so drop_recently_published
        # has no reason to restore the repeat to avoid starving the edition —
        # otherwise this test would pass for the wrong reason.
        test_config = small_config(config)
        test_config.ai_daily.story_count = 1
        test_config.ai_daily.min_story_count = 1

        parse = feeds_for({
            "ars": [
                entry("A repeat AI story", "https://ars.example/repeat"),
                entry("A fresh AI story", "https://ars.example/fresh"),
            ],
            "wired": [],
        })
        editions = [
            {
                "date": NOW.strftime("%Y-%m-%d"),
                "generated_at": NOW.isoformat(),
                "stories": [{"title": "A repeat AI story", "url": "https://ars.example/repeat"}],
            }
        ]
        briefer = FakeBriefer(BriefVerdict(picks=[]))
        run(test_config, briefer, parse=parse, editions=editions)
        assert "A repeat AI story" not in briefer.seen_candidates
        assert "A fresh AI story" in briefer.seen_candidates


class TestAiEditionStorage:
    def brief(self, n=1):
        from lastweekintech.domain import AiBrief

        return AiBrief(
            title=f"Story {n}",
            articles=[make_article(title=f"Story {n}", url=f"https://example.com/{n}")],
            theme="Capabilities",
            what_happened="It happened.",
            why_it_matters="It matters.",
            watch_next="Watch this.",
            score=10 - n,
        )

    def test_build_ai_edition_shapes_the_payload(self):
        edition = ai_daily.build_ai_edition(
            [self.brief(1), self.brief(2)], date="2026-09-15", now=NOW
        )
        assert edition["date"] == "2026-09-15"
        assert len(edition["stories"]) == 2
        assert edition["stories"][0]["rank"] == 1
        assert edition["stories"][0]["theme"] == "Capabilities"
        assert edition["stories"][0]["what_happened"] == "It happened."
        assert edition["stories"][0]["why_it_matters"] == "It matters."
        assert edition["stories"][0]["watch_next"] == "Watch this."
        # A flattened field for the syndication helpers this reuses.
        assert edition["stories"][0]["summary"] == "It happened."

    def test_save_and_list_round_trip(self, tmp_path):
        edition = ai_daily.build_ai_edition([self.brief()], date="2026-09-15", now=NOW)
        ai_daily.save_ai_edition(edition, tmp_path)
        assert (tmp_path / "ai" / "latest.json").exists()
        assert (tmp_path / "ai" / "archive" / "2026-09-15.json").exists()

        editions = ai_daily.list_ai_editions(tmp_path)
        assert len(editions) == 1
        assert editions[0]["date"] == "2026-09-15"

    def test_list_editions_is_empty_when_nothing_published_yet(self, tmp_path):
        assert ai_daily.list_ai_editions(tmp_path) == []

    def test_list_editions_sorts_newest_first(self, tmp_path):
        for date in ["2026-09-13", "2026-09-15", "2026-09-14"]:
            ai_daily.save_ai_edition(
                ai_daily.build_ai_edition([self.brief()], date=date, now=NOW), tmp_path
            )
        editions = ai_daily.list_ai_editions(tmp_path)
        assert [e["date"] for e in editions] == ["2026-09-15", "2026-09-14", "2026-09-13"]


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "src" / "lastweekintech" / "templates"
STATIC_DIR = Path(__file__).resolve().parents[1] / "src" / "lastweekintech" / "static"


def ai_edition(date="2026-09-15", title="OpenAI ships a new model", theme="Capabilities"):
    from lastweekintech.domain import AiBrief

    brief = AiBrief(
        title=title,
        articles=[make_article(title=title, source="Ars Technica")],
        theme=theme,
        what_happened="A factual account of what happened.",
        why_it_matters="A paragraph on why this matters and what it implies.",
        watch_next="What to watch for next.",
    )
    return ai_daily.build_ai_edition([brief], date=date, now=NOW)


def generate(tmp_path, editions):
    return ai_daily.generate_ai_site(
        editions, output_dir=tmp_path, template_dir=TEMPLATE_DIR, static_dir=STATIC_DIR
    )


class TestGenerateAiSite:
    def test_writes_the_latest_edition_under_ai(self, tmp_path):
        generate(tmp_path, [ai_edition()])
        page = (tmp_path / "ai" / "index.html").read_text()
        assert "OpenAI ships a new model" in page
        assert "A paragraph on why this matters" in page
        assert "What to watch for next" in page

    def test_writes_a_page_per_archived_edition(self, tmp_path):
        generate(tmp_path, [ai_edition(date="2026-09-15"), ai_edition(date="2026-09-14")])
        assert (tmp_path / "ai" / "archive" / "2026-09-14.html").exists()
        assert (tmp_path / "ai" / "archive" / "2026-09-15.html").exists()

    def test_writes_its_own_feed_and_sitemap(self, tmp_path):
        generate(tmp_path, [ai_edition()])
        assert (tmp_path / "ai" / "feed.xml").exists()
        assert (tmp_path / "ai" / "sitemap.xml").exists()
        assert "AI Daily" in (tmp_path / "ai" / "feed.xml").read_text()

    def test_copies_static_assets_under_ai_too(self, tmp_path):
        generate(tmp_path, [ai_edition()])
        assert (tmp_path / "ai" / "style.css").exists()

    def test_archive_page_links_back_to_the_ai_index_and_weekly_digest(self, tmp_path):
        generate(tmp_path, [ai_edition(date="2026-09-15"), ai_edition(date="2026-09-14")])
        page = (tmp_path / "ai" / "archive" / "2026-09-14.html").read_text()
        assert 'href="../index.html"' in page
        assert 'href="../../index.html"' in page

    def test_no_editions_writes_nothing(self, tmp_path):
        assert ai_daily.generate_ai_site([], output_dir=tmp_path) == []
        assert not (tmp_path / "ai").exists()
