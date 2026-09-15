"""Tests for feed, sitemap and robots.txt generation."""

import xml.etree.ElementTree as ET

import pytest

from lastweekintech import syndication

ATOM = "{http://www.w3.org/2005/Atom}"
SITEMAP = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

SITE = "https://lastweekin.tech"


def story(rank=1, title="Anthropic ships something", summary="A real summary here."):
    return {
        "rank": rank,
        "title": title,
        "source": "WIRED",
        "sources": ["WIRED"],
        "source_count": 1,
        "url": f"https://example.com/{rank}",
        "category": "AI",
        "hn_points": 120,
        "score": 4.2,
        "summary": summary,
    }


def edition(week="2026-08-10", generated_at="2026-08-10T09:32:55Z", stories=None):
    return {
        "week": week,
        "generated_at": generated_at,
        "stories": stories if stories is not None else [story()],
    }


def parse(path):
    """Parse a written file, which fails loudly if the output is not well-formed XML."""
    return ET.parse(path).getroot()  # noqa: S314 - the input is what this module just wrote


class TestWriteFeed:
    def test_writes_a_well_formed_atom_feed(self, tmp_path):
        path = syndication.write_feed([edition()], tmp_path, SITE)
        assert path == tmp_path / "feed.xml"
        root = parse(path)
        assert root.tag == f"{ATOM}feed"

    def test_feed_identifies_itself_with_absolute_urls(self, tmp_path):
        root = parse(syndication.write_feed([edition()], tmp_path, SITE))
        assert root.findtext(f"{ATOM}id") == f"{SITE}/"
        links = {link.get("rel"): link.get("href") for link in root.findall(f"{ATOM}link")}
        assert links["self"] == f"{SITE}/feed.xml"
        assert links["alternate"] == f"{SITE}/"

    def test_one_entry_per_edition_not_per_story(self, tmp_path):
        stories = [story(rank=i, title=f"Headline number {i}") for i in range(1, 8)]
        root = parse(syndication.write_feed([edition(stories=stories)], tmp_path, SITE))
        assert len(root.findall(f"{ATOM}entry")) == 1

    def test_entry_carries_id_title_updated_and_link(self, tmp_path):
        root = parse(syndication.write_feed([edition()], tmp_path, SITE))
        entry = root.find(f"{ATOM}entry")
        assert entry.findtext(f"{ATOM}id") == "tag:lastweekin.tech,2026-08-10:edition/2026-08-10"
        assert "2026-08-10" in entry.findtext(f"{ATOM}title")
        assert entry.findtext(f"{ATOM}updated") == "2026-08-10T09:32:55Z"
        link = entry.find(f"{ATOM}link")
        assert link.get("href") == f"{SITE}/archive/2026-08-10.html"

    def test_entry_content_lists_every_story_with_link_source_and_summary(self, tmp_path):
        stories = [story(rank=i, title=f"Headline number {i}") for i in range(1, 8)]
        root = parse(syndication.write_feed([edition(stories=stories)], tmp_path, SITE))
        content = root.find(f"{ATOM}entry/{ATOM}content")
        assert content.get("type") == "html"
        # type="html" carries the fragment as a text node, so parsing hands back markup.
        html = content.text
        assert html.count("<li>") == 7
        assert "<ol" in html
        assert 'href="https://example.com/3"' in html
        assert "Headline number 3" in html
        assert "WIRED" in html
        assert "A real summary here." in html

    def test_newest_edition_first(self, tmp_path):
        editions = [edition(week="2026-07-27"), edition(week="2026-08-10")]
        root = parse(syndication.write_feed(editions, tmp_path, SITE))
        weeks = [e.findtext(f"{ATOM}title") for e in root.findall(f"{ATOM}entry")]
        assert "2026-08-10" in weeks[0]
        assert "2026-07-27" in weeks[1]

    def test_caps_the_feed_at_limit_entries(self, tmp_path):
        editions = [edition(week=f"2026-0{m}-01") for m in range(1, 8)]
        root = parse(syndication.write_feed(editions, tmp_path, SITE, limit=3))
        assert len(root.findall(f"{ATOM}entry")) == 3

    def test_feed_updated_tracks_the_newest_edition(self, tmp_path):
        editions = [
            edition(week="2026-07-27", generated_at="2026-07-27T11:34:14Z"),
            edition(week="2026-08-10", generated_at="2026-08-10T09:32:55Z"),
        ]
        root = parse(syndication.write_feed(editions, tmp_path, SITE))
        assert root.findtext(f"{ATOM}updated") == "2026-08-10T09:32:55Z"

    @pytest.mark.parametrize("site", [SITE, f"{SITE}/", f"{SITE}///"])
    def test_tolerates_a_trailing_slash_on_the_site_url(self, tmp_path, site):
        root = parse(syndication.write_feed([edition()], tmp_path, site))
        assert root.findtext(f"{ATOM}id") == f"{SITE}/"
        assert root.find(f"{ATOM}entry/{ATOM}link").get("href") == f"{SITE}/archive/2026-08-10.html"

    def test_normalizes_offset_timestamps_to_rfc3339_utc(self, tmp_path):
        root = parse(
            syndication.write_feed(
                [edition(generated_at="2026-08-10T11:32:55+02:00")], tmp_path, SITE
            )
        )
        assert root.findtext(f"{ATOM}entry/{ATOM}updated") == "2026-08-10T09:32:55Z"

    def test_falls_back_to_the_week_when_the_timestamp_is_unusable(self, tmp_path):
        root = parse(syndication.write_feed([edition(generated_at="not a date")], tmp_path, SITE))
        assert root.findtext(f"{ATOM}entry/{ATOM}updated") == "2026-08-10T00:00:00Z"

    def test_writes_a_valid_feed_without_editions(self, tmp_path):
        path = syndication.write_feed([], tmp_path, SITE)
        root = parse(path)
        assert root.findall(f"{ATOM}entry") == []
        assert root.findtext(f"{ATOM}updated")

    def test_accepts_a_custom_title_and_subtitle(self, tmp_path):
        root = parse(
            syndication.write_feed(
                [edition()], tmp_path, SITE, title="Custom Title", subtitle="Custom subtitle"
            )
        )
        assert root.findtext(f"{ATOM}title") == "Custom Title"
        assert root.findtext(f"{ATOM}subtitle") == "Custom subtitle"

    def test_defaults_reproduce_the_weekly_title_and_subtitle(self, tmp_path):
        root = parse(syndication.write_feed([edition()], tmp_path, SITE))
        assert root.findtext(f"{ATOM}title") == syndication.SITE_TITLE
        assert root.findtext(f"{ATOM}subtitle") == syndication.SITE_SUBTITLE

    def test_accepts_a_custom_entry_title(self, tmp_path):
        root = parse(
            syndication.write_feed(
                [edition()], tmp_path, SITE, entry_title=lambda week: f"AI Daily — {week}"
            )
        )
        entry_title = root.find(f"{ATOM}entry/{ATOM}title")
        assert entry_title.text == "AI Daily — 2026-08-10"

    def test_default_entry_title_reproduces_the_weekly_wording(self, tmp_path):
        root = parse(syndication.write_feed([edition()], tmp_path, SITE))
        entry_title = root.find(f"{ATOM}entry/{ATOM}title")
        assert entry_title.text == "Week ending 2026-08-10"

    def test_default_entry_kind_reproduces_the_prior_entry_id_format(self, tmp_path):
        """No ``entry_kind`` argument must keep producing exactly the old id.

        Existing subscribers key on this id; if it changed, 20 archived editions
        would reappear in every reader as "new".
        """
        root = parse(syndication.write_feed([edition()], tmp_path, SITE))
        entry = root.find(f"{ATOM}entry")
        assert entry.findtext(f"{ATOM}id") == "tag:lastweekin.tech,2026-08-10:edition/2026-08-10"

    def test_a_custom_entry_kind_produces_a_distinguishable_id(self, tmp_path):
        root = parse(syndication.write_feed([edition()], tmp_path, SITE, entry_kind="ai-daily"))
        entry = root.find(f"{ATOM}entry")
        assert entry.findtext(f"{ATOM}id") == "tag:lastweekin.tech,2026-08-10:ai-daily/2026-08-10"

    def test_weekly_and_ai_daily_style_feeds_do_not_collide_on_the_same_date(self, tmp_path):
        """Regression test for the entry-id collision bug.

        AI Daily's feed is written with an ``/ai``-suffixed ``site_url``, which
        shares the same netloc as the weekly feed's ``site_url``. Before
        ``entry_kind`` existed, ``_entry_id`` only used the netloc, so both
        feeds emitted byte-identical ``<id>`` values for the same date.
        """
        weekly_root = parse(syndication.write_feed([edition()], tmp_path / "weekly", SITE))
        ai_daily_root = parse(
            syndication.write_feed(
                [edition()],
                tmp_path / "ai",
                f"{SITE}/ai",
                entry_title=lambda date: f"AI Daily — {date}",
                entry_kind="ai-daily",
            )
        )
        weekly_id = weekly_root.find(f"{ATOM}entry").findtext(f"{ATOM}id")
        ai_daily_id = ai_daily_root.find(f"{ATOM}entry").findtext(f"{ATOM}id")
        assert weekly_id != ai_daily_id


class TestEscaping:
    def test_escapes_markup_in_a_story_title(self, tmp_path):
        evil = "<script>alert('x')</script>"
        path = syndication.write_feed([edition(stories=[story(title=evil)])], tmp_path, SITE)
        raw = path.read_text(encoding="utf-8")
        assert "<script>" not in raw
        # Still well-formed, and the fragment a reader unwraps carries no live tag.
        html = parse(path).find(f"{ATOM}entry/{ATOM}content").text
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert evil not in html

    def test_escapes_bare_ampersands_in_titles_and_urls(self, tmp_path):
        stories = [story(title="Rock & roll <b>hardware</b>")]
        stories[0]["url"] = "https://example.com/a?x=1&y=2"
        path = syndication.write_feed([edition(stories=stories)], tmp_path, SITE)
        assert "&amp;" in path.read_text(encoding="utf-8")
        html = parse(path).find(f"{ATOM}entry/{ATOM}content").text  # parse raises on a bare "&"
        assert "Rock &amp; roll" in html
        assert "<b>hardware</b>" not in html
        assert "x=1&amp;y=2" in html

    def test_escapes_quotes_so_an_href_cannot_be_broken_out_of(self, tmp_path):
        stories = [story()]
        stories[0]["url"] = 'https://example.com/a" onmouseover="alert(1)'
        path = syndication.write_feed([edition(stories=stories)], tmp_path, SITE)
        html = parse(path).find(f"{ATOM}entry/{ATOM}content").text
        assert 'onmouseover="alert(1)"' not in html
        assert "&quot;" in html

    def test_strips_characters_that_xml_cannot_represent(self, tmp_path):
        path = syndication.write_feed(
            [edition(stories=[story(title="Bell \x07 and formfeed \x0c")])], tmp_path, SITE
        )
        html = parse(path).find(f"{ATOM}entry/{ATOM}content").text
        assert "\x07" not in html
        assert "\x0c" not in html


class TestWriteSitemap:
    def test_lists_the_index_and_every_archive_page(self, tmp_path):
        editions = [edition(week="2026-08-10"), edition(week="2026-07-27")]
        path = syndication.write_sitemap(editions, tmp_path, SITE)
        assert path == tmp_path / "sitemap.xml"
        root = parse(path)
        locs = [u.findtext(f"{SITEMAP}loc") for u in root.findall(f"{SITEMAP}url")]
        assert locs == [
            f"{SITE}/",
            f"{SITE}/archive/2026-08-10.html",
            f"{SITE}/archive/2026-07-27.html",
        ]

    def test_each_url_carries_a_lastmod(self, tmp_path):
        root = parse(syndication.write_sitemap([edition()], tmp_path, SITE))
        for url in root.findall(f"{SITEMAP}url"):
            assert url.findtext(f"{SITEMAP}lastmod") == "2026-08-10T09:32:55Z"

    def test_index_lastmod_follows_the_newest_edition(self, tmp_path):
        editions = [
            edition(week="2026-07-27", generated_at="2026-07-27T11:34:14Z"),
            edition(week="2026-08-10", generated_at="2026-08-10T09:32:55Z"),
        ]
        root = parse(syndication.write_sitemap(editions, tmp_path, SITE))
        assert root.find(f"{SITEMAP}url/{SITEMAP}lastmod").text == "2026-08-10T09:32:55Z"

    def test_still_lists_the_index_without_editions(self, tmp_path):
        root = parse(syndication.write_sitemap([], tmp_path, SITE))
        assert [u.findtext(f"{SITEMAP}loc") for u in root.findall(f"{SITEMAP}url")] == [f"{SITE}/"]


class TestWriteRobots:
    def test_allows_crawling_and_points_at_the_sitemap(self, tmp_path):
        path = syndication.write_robots(tmp_path, f"{SITE}/")
        assert path == tmp_path / "robots.txt"
        text = path.read_text(encoding="utf-8")
        assert "User-agent: *" in text
        assert "Allow: /" in text
        assert f"Sitemap: {SITE}/sitemap.xml" in text
        assert "Disallow: /" not in text


class TestWriteSyndication:
    def test_writes_all_three_files(self, tmp_path):
        written = syndication.write_syndication([edition()], tmp_path, SITE)
        assert set(written) == {
            tmp_path / "feed.xml",
            tmp_path / "sitemap.xml",
            tmp_path / "robots.txt",
        }
        assert all(p.exists() for p in written)

    def test_creates_the_output_directory_if_missing(self, tmp_path):
        target = tmp_path / "site" / "nested"
        written = syndication.write_syndication([edition()], target, SITE)
        assert all(p.exists() for p in written)

    def test_survives_an_empty_archive(self, tmp_path):
        written = syndication.write_syndication([], tmp_path, SITE)
        assert len(written) == 3
        for path in written:
            if path.suffix == ".xml":
                parse(path)
