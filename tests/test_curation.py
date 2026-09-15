"""Tests for deduplication, categorization, scoring and selection."""

from datetime import timedelta

from conftest import NOW, make_article, make_story

from lastweekintech import pipeline
from lastweekintech.text import CATEGORIES


class TestDedupeArticles:
    def test_collapses_urls_that_differ_only_by_tracking_parameters(self):
        articles = [
            make_article(url="https://example.com/story"),
            make_article(url="https://example.com/story?utm_source=rss"),
        ]
        assert len(pipeline.dedupe_articles(articles)) == 1

    def test_keeps_genuinely_different_urls(self):
        articles = [
            make_article(url="https://example.com/a"),
            make_article(url="https://example.com/b"),
        ]
        assert len(pipeline.dedupe_articles(articles)) == 2

    def test_keeps_the_highest_hn_points_across_duplicates(self):
        articles = [
            make_article(url="https://example.com/story", hn_points=None),
            make_article(url="https://example.com/story", hn_points=320),
        ]
        assert pipeline.dedupe_articles(articles)[0].hn_points == 320

    def test_prefers_the_duplicate_that_has_content(self):
        articles = [
            make_article(url="https://example.com/story", content=None),
            make_article(url="https://example.com/story", content="the real body"),
        ]
        assert pipeline.dedupe_articles(articles)[0].content == "the real body"

    def test_drops_articles_without_a_url(self):
        assert pipeline.dedupe_articles([make_article(url="")]) == []


class TestCategorizeStories:
    def test_labels_a_story_with_ai_in_the_title(self):
        stories = pipeline.categorize_stories([make_story(title="OpenAI launches a new model")])
        assert stories[0].category == "AI"

    def test_does_not_label_substring_matches_as_ai(self):
        """ "ai" inside "Britain", "train", "Aids" and "email" once made AI stories."""
        titles = [
            "London still dominates Britain's datacenter map",
            "Smart glasses are only smart if we train them to be good",
            "Yeasound RIC800 Hearing Aids Review: Good Audio",
            "Microsoft accidentally kills epic Outlook email threads",
            "C programmers commit fresh crimes against readability",
        ]
        stories = pipeline.categorize_stories([
            make_story(title=t, articles=[make_article(title=t, content=None)]) for t in titles
        ])
        assert all(s.category != "AI" for s in stories)

    def test_labels_a_story_whose_body_is_substantially_about_ai(self):
        body = "The chatbot uses a large language model. The AI system was trained on data."
        title = "A quiet launch on a Tuesday afternoon"
        story = make_story(title=title, articles=[make_article(title=title, content=body)])
        assert pipeline.categorize_stories([story])[0].category == "AI"

    def test_ignores_a_single_passing_mention_in_the_body(self):
        body = "A long article about office furniture that mentions AI once in passing."
        title = "A quiet launch on a Tuesday afternoon"
        story = make_story(title=title, articles=[make_article(title=title, content=body)])
        assert pipeline.categorize_stories([story])[0].category != "AI"

    def test_handles_articles_without_content(self):
        story = make_story(title="Plain story", articles=[make_article(content=None)])
        assert pipeline.categorize_stories([story])[0].category == "General Tech"

    def test_labels_the_expanded_taxonomy(self):
        titles = {
            "Ransomware crew leaks 1.6M customer records": "Security",
            "EU regulators open an antitrust probe into app stores": "Policy",
            "Debian votes to replace systemd in the next release": "Open Source",
            "TSMC starts risk production on its 2nm node": "Hardware",
            "Adobe's CEO steps down after a quarter of falling revenue": "Business",
            "Whale migrations are being disrupted by climate change": "General Tech",
        }
        stories = pipeline.categorize_stories([
            make_story(title=t, articles=[make_article(title=t, content=None)]) for t in titles
        ])
        assert {s.title: s.category for s in stories} == titles

    def test_only_ever_emits_agreed_category_strings(self):
        """The front end styles a chip per category, so the set is a contract."""
        stories = pipeline.categorize_stories([
            make_story(title=t, articles=[make_article(title=t, content=None)])
            for t in ("An AI model", "A ransomware breach", "A quiet afternoon")
        ])
        assert all(s.category in CATEGORIES for s in stories)

    def test_precedence_puts_an_ai_security_story_in_ai(self):
        """Deterministic precedence: AI is the category the digest floor counts."""
        title = "Researchers exploit a zero-day in Claude's agent sandbox"
        story = make_story(title=title, articles=[make_article(title=title, content=None)])
        assert pipeline.categorize_stories([story])[0].category == "AI"

    def test_a_headline_beats_the_body_even_for_a_lower_precedence_category(self):
        """A title mention is decisive; a body only speaks when no title does."""
        story = make_story(
            title="Ransomware crew hits a hospital chain",
            articles=[
                make_article(
                    title="Ransomware crew hits a hospital chain",
                    content="The AI model was trained by an AI lab on AI data.",
                )
            ],
        )
        assert pipeline.categorize_stories([story])[0].category == "Security"

    def test_a_body_categorizes_a_story_whose_headline_says_nothing(self):
        body = "The malware spread fast. Ransomware operators used the malware to encrypt disks."
        story = make_story(
            title="A quiet Tuesday at a logistics firm",
            articles=[make_article(title="A quiet Tuesday at a logistics firm", content=body)],
        )
        assert pipeline.categorize_stories([story])[0].category == "Security"


class TestScoreStories:
    def test_multi_source_coverage_outranks_single_source(self, config):
        single = make_story(articles=[make_article(source="A")])
        multi = make_story(
            articles=[
                make_article(source="A", url="https://a.example/1"),
                make_article(source="B", url="https://b.example/1"),
            ]
        )
        pipeline.score_stories([single, multi], config, now=NOW)
        assert multi.score > single.score

    def test_a_lone_single_source_story_gets_no_coverage_credit(self, config):
        story = make_story(articles=[make_article(source="A", age_hours=0)])
        pipeline.score_stories([story], config, now=NOW)
        # Recency only: the source component must not hand out a flat baseline.
        assert story.score == config.weights.rec

    def test_hacker_news_points_raise_the_score(self, config):
        quiet = make_story(articles=[make_article(hn_points=None)])
        loud = make_story(articles=[make_article(hn_points=400)])
        pipeline.score_stories([quiet, loud], config, now=NOW)
        assert loud.score > quiet.score

    def test_hacker_news_contribution_is_capped(self, config):
        big = make_story(articles=[make_article(hn_points=500)])
        huge = make_story(articles=[make_article(hn_points=50_000)])
        pipeline.score_stories([big, huge], config, now=NOW)
        assert big.score == huge.score

    def test_recency_never_goes_negative_for_stale_articles(self, config):
        stale = make_story(articles=[make_article(age_hours=24 * 30)])
        pipeline.score_stories([stale], config, now=NOW)
        assert stale.score == 0.0

    def test_returns_stories_sorted_by_score(self, config):
        low = make_story(title="low", articles=[make_article(age_hours=24 * 6)])
        high = make_story(title="high", articles=[make_article(hn_points=500)])
        assert [s.title for s in pipeline.score_stories([low, high], config, now=NOW)] == [
            "high",
            "low",
        ]

    def test_tolerates_articles_without_a_publication_date(self, config):
        article = make_article()
        article.published_at = None
        story = make_story(articles=[article])
        pipeline.score_stories([story], config, now=NOW)
        assert story.score >= 0.0


def edition(week, *stories):
    """An archived edition payload holding ``(title, url)`` pairs."""
    return {
        "week": week,
        "stories": [{"title": t, "url": u} for t, u in stories],
    }


class TestDropRecentlyPublished:
    """Hacker News traction persists for days, so a hot story can span two weeks.

    The old recency-driven ranking never repeated across 44 archived editions,
    so there is no historical precedent to lean on; this is reasoned from how
    the new ranking behaves.
    """

    def drop(self, stories, editions, keep_at_least=1, lookback_weeks=3):
        return pipeline.drop_recently_published(
            stories,
            editions,
            now=NOW,
            lookback=timedelta(weeks=lookback_weeks),
            keep_at_least=keep_at_least,
        )

    def test_drops_a_story_already_published_at_the_same_url(self):
        stories = [
            make_story(title="Fresh news", articles=[make_article(url="https://a.example/new")]),
            make_story(title="Old news", articles=[make_article(url="https://a.example/old")]),
        ]
        kept = self.drop(stories, [edition("2026-08-03", ("Old news", "https://a.example/old"))])
        assert [s.title for s in kept] == ["Fresh news"]

    def test_normalizes_the_url_before_comparing(self):
        stories = [
            make_story(title="A", articles=[make_article(url="https://a.example/new")]),
            make_story(
                title="Repeat",
                articles=[make_article(url="https://www.a.example/old/?utm_source=rss")],
            ),
        ]
        kept = self.drop(stories, [edition("2026-08-03", ("Old", "https://a.example/old"))])
        assert [s.title for s in kept] == ["A"]

    def test_drops_a_repeat_that_arrived_via_a_different_url(self):
        """The same event often reaches us through a different outlet's link."""
        stories = [
            make_story(title="Unrelated kernel change", articles=[make_article(url="https://x/1")]),
            make_story(
                title="Anthropic to add invisible watermarks to Claude-generated text",
                articles=[make_article(url="https://b.example/other")],
            ),
        ]
        kept = self.drop(
            stories,
            [
                edition(
                    "2026-08-03",
                    (
                        "Anthropic to Add Invisible Watermarks to Claude-Generated Text",
                        "https://a.example/1",
                    ),
                )
            ],
        )
        assert [s.title for s in kept] == ["Unrelated kernel change"]

    def test_keeps_a_merely_similar_headline_about_a_different_event(self):
        stories = [
            make_story(
                title="Apple releases iOS 27 with a redesigned lock screen",
                articles=[make_article(url="https://a.example/1")],
            )
        ]
        kept = self.drop(
            stories,
            [edition("2026-08-03", ("Google releases Android 18 beta", "https://g.example/1"))],
        )
        assert len(kept) == 1

    def test_ignores_editions_older_than_the_lookback(self):
        stories = [make_story(title="X", articles=[make_article(url="https://a.example/old")])]
        old = [edition("2026-01-05", ("X", "https://a.example/old"))]
        assert len(self.drop(stories, old, lookback_weeks=3)) == 1

    def test_a_longer_lookback_reaches_further_back(self):
        stories = [make_story(title="X", articles=[make_article(url="https://a.example/old")])]
        old = [edition("2026-07-06", ("X", "https://a.example/old"))]
        assert self.drop(stories, old, keep_at_least=0, lookback_weeks=1) == stories
        assert self.drop(stories, old, keep_at_least=0, lookback_weeks=8) == []

    def test_a_zero_lookback_disables_the_filter(self):
        stories = [make_story(title="X", articles=[make_article(url="https://a.example/old")])]
        recent = [edition("2026-08-03", ("X", "https://a.example/old"))]
        assert self.drop(stories, recent, lookback_weeks=0) == stories

    def test_never_starves_the_edition_below_the_story_count(self):
        """Refusing every repeat must not ship a three-story digest."""
        stories = [
            make_story(title=f"s{i}", score=100 - i, articles=[make_article(url=f"https://a/{i}")])
            for i in range(5)
        ]
        published = edition("2026-08-03", *[(f"s{i}", f"https://a/{i}") for i in range(5)])
        kept = self.drop(stories, [published], keep_at_least=3)
        assert len(kept) == 3
        # The restored ones are the highest-ranked repeats, in ranking order.
        assert [s.title for s in kept] == ["s0", "s1", "s2"]

    def test_restores_repeats_only_after_every_fresh_story(self):
        stories = [
            make_story(title="repeat", score=100, articles=[make_article(url="https://a/old")]),
            make_story(title="fresh", score=1, articles=[make_article(url="https://a/new")]),
        ]
        published = [edition("2026-08-03", ("repeat", "https://a/old"))]
        kept = self.drop(stories, published, keep_at_least=2)
        assert {s.title for s in kept} == {"fresh", "repeat"}

    def test_survives_a_missing_archive(self):
        stories = [make_story(title="X", articles=[make_article()])]
        assert self.drop(stories, []) == stories

    def test_survives_malformed_archive_entries(self):
        """An unreadable archive must degrade to "publish everything", not crash."""
        junk = [
            {"week": "2026-08-03"},
            {"week": "2026-08-03", "stories": "not a list"},
            {"week": None, "stories": [{"title": "X"}]},
            {"week": "not-a-date", "stories": [{"url": "https://a.example/old"}]},
            {"week": "2026-08-03", "stories": [{"nothing": "useful"}, None]},
        ]
        stories = [make_story(title="X", articles=[make_article(url="https://a.example/old")])]
        assert self.drop(stories, junk) == stories


class TestSelectTopStories:
    def test_returns_the_requested_number_of_stories(self):
        stories = [make_story(title=f"s{i}", score=100 - i) for i in range(20)]
        assert len(pipeline.select_top_stories(stories, count=7, min_ai=4)) == 7

    def test_promotes_ai_stories_to_meet_the_floor(self):
        stories = [make_story(title=f"gen{i}", score=100 - i) for i in range(7)]
        stories += [make_story(title=f"ai{i}", category="AI", score=50 - i) for i in range(4)]
        selected = pipeline.select_top_stories(stories, count=7, min_ai=4)
        assert sum(s.category == "AI" for s in selected) == 4

    def test_replaces_the_lowest_scoring_general_stories_when_promoting(self):
        stories = [make_story(title=f"gen{i}", score=100 - i) for i in range(7)]
        stories += [make_story(title="ai0", category="AI", score=50)]
        selected = pipeline.select_top_stories(stories, count=7, min_ai=1)
        titles = [s.title for s in selected]
        assert "ai0" in titles and "gen6" not in titles

    def test_does_not_cap_ai_stories_at_the_floor(self):
        stories = [make_story(title=f"ai{i}", category="AI", score=100 - i) for i in range(7)]
        stories += [make_story(title="gen0", score=10)]
        selected = pipeline.select_top_stories(stories, count=7, min_ai=4)
        assert sum(s.category == "AI" for s in selected) == 7

    def test_still_fills_every_slot_when_ai_stories_are_scarce(self):
        stories = [make_story(title=f"gen{i}", score=100 - i) for i in range(10)]
        stories += [make_story(title="ai0", category="AI", score=1)]
        selected = pipeline.select_top_stories(stories, count=7, min_ai=4)
        assert len(selected) == 7
        assert sum(s.category == "AI" for s in selected) == 1

    def test_output_is_ordered_by_score(self):
        stories = [make_story(title=f"gen{i}", score=100 - i) for i in range(7)]
        stories += [make_story(title="ai0", category="AI", score=50)]
        selected = pipeline.select_top_stories(stories, count=7, min_ai=1)
        assert [s.score for s in selected] == sorted((s.score for s in selected), reverse=True)

    def test_returns_everything_available_when_short_of_count(self):
        stories = [make_story(title="only", score=5)]
        assert len(pipeline.select_top_stories(stories, count=7, min_ai=4)) == 1


def story_from(source: str, title: str, score: float, **kwargs) -> "pipeline.Story":
    article = make_article(title=title, url=f"https://{source}.example/{title}", source=source)
    story = make_story(title=title, score=score, articles=[article], **kwargs)
    return story


class TestSourceCap:
    """One outlet may not own the edition: WIRED once supplied 40% of every
    story ever published, and the first HN-scored edition was seven-for-seven
    Hacker News."""

    def test_caps_stories_per_source(self):
        stories = [story_from("A", f"a{i}", 100 - i) for i in range(7)]
        stories += [story_from("B", f"b{i}", 50 - i) for i in range(3)]
        selected = pipeline.select_top_stories(stories, count=7, min_ai=0, max_per_source=2)
        by_source = {s.articles[0].source for s in selected}
        assert by_source == {"A", "B"}
        counts = [s.articles[0].source for s in selected]
        # Both outlets are capped at 2; the three remaining slots backfill
        # best-first from the deferred stories, which are all A's.
        assert counts.count("B") == 2 and counts.count("A") == 5

    def test_backfills_over_the_cap_rather_than_shorten_the_edition(self):
        stories = [story_from("A", f"a{i}", 100 - i) for i in range(7)]
        stories += [story_from("B", f"b{i}", 50 - i) for i in range(3)]
        selected = pipeline.select_top_stories(stories, count=7, min_ai=0, max_per_source=2)
        assert len(selected) == 7  # 2 A + 3 B leaves 5; the cap must yield, not starve

    def test_cap_of_zero_disables_the_limit(self):
        stories = [story_from("A", f"a{i}", 100 - i) for i in range(7)]
        selected = pipeline.select_top_stories(stories, count=7, min_ai=0, max_per_source=0)
        assert len(selected) == 7

    def test_capped_selection_is_still_ordered_by_score(self):
        stories = [story_from("A", f"a{i}", 100 - i) for i in range(7)]
        stories += [story_from("B", f"b{i}", 50 - i) for i in range(5)]
        selected = pipeline.select_top_stories(stories, count=7, min_ai=0, max_per_source=2)
        assert [s.score for s in selected] == sorted((s.score for s in selected), reverse=True)


class TestBodyPreference:
    """A story whose body could not be extracted cannot be summarized, and one
    too many of those fails the publish gate — which is exactly how the
    2026-08-24 run died. Selection prefers stories it can actually ship."""

    def test_prefers_stories_with_extracted_bodies(self):
        bodyless = make_story(
            title="paywalled", score=100, articles=[make_article(title="paywalled", content=None)]
        )
        stories = [bodyless] + [story_from("A", f"a{i}", 50 - i) for i in range(7)]
        selected = pipeline.select_top_stories(stories, count=7, min_ai=0)
        assert "paywalled" not in {s.title for s in selected}

    def test_bodyless_stories_backfill_a_short_week(self):
        bodyless = make_story(
            title="paywalled", score=100, articles=[make_article(title="paywalled", content=None)]
        )
        stories = [bodyless] + [story_from("A", f"a{i}", 50 - i) for i in range(3)]
        selected = pipeline.select_top_stories(stories, count=7, min_ai=0)
        assert len(selected) == 4
        assert "paywalled" in {s.title for s in selected}


class TestAggregatorSources:
    def test_representative_article_prefers_the_original_outlet(self):
        aggregator = make_article(
            title="Big story",
            url="https://techmeme.example/1",
            source="Techmeme",
            content="x" * 900,
        )
        aggregator.aggregator = True
        outlet = make_article(
            title="Big story", url="https://ars.example/1", source="Ars Technica", content="x" * 100
        )
        story = make_story(title="Big story", articles=[aggregator, outlet])
        edition = pipeline.build_edition([story], week="2026-08-10")
        assert edition["stories"][0]["source"] == "Ars Technica"
        assert edition["stories"][0]["url"] == "https://ars.example/1"

    def test_aggregator_still_counts_toward_breadth(self):
        aggregator = make_article(title="Big story", url="https://tm.example/1", source="Techmeme")
        aggregator.aggregator = True
        outlet = make_article(title="Big story", url="https://ars.example/1", source="Ars Technica")
        story = make_story(title="Big story", articles=[aggregator, outlet])
        edition = pipeline.build_edition([story], week="2026-08-10")
        assert edition["stories"][0]["source_count"] == 2
