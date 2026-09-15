"""Syndication artifacts for the published site: Atom feed, sitemap and robots.txt.

Stdlib only, on purpose: this runs in the same unattended GitHub Actions job as the
rest of the pipeline, and a feed generator is not worth a dependency.

Everything here is pure and offline — it turns already-published edition dicts into
files. The one rule that matters is escaping: story titles, summaries and URLs come
from third-party RSS feeds, so every interpolated value is escaped before it reaches
the document. XML is unforgiving in a way HTML is not — a single bare ``&`` in a
headline makes the whole feed unparsable and every subscriber silently loses the
site.
"""

import html
import logging
import re

# Used only to *build* documents, never to parse anything received from the
# network, so the XML-attack surface bandit warns about does not exist here.
import xml.etree.ElementTree as ET  # nosec B405
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ATOM_NS = "http://www.w3.org/2005/Atom"
SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"

FEED_FILENAME = "feed.xml"
SITEMAP_FILENAME = "sitemap.xml"
ROBOTS_FILENAME = "robots.txt"

SITE_TITLE = "LastWeekIn.Tech"
SITE_SUBTITLE = "The seven tech and AI stories that mattered last week."
SITE_AUTHOR = "LastWeekIn.Tech"

DEFAULT_FEED_LIMIT = 20

# Atom makes feed/updated mandatory, so a feed with no editions still needs a value.
# The epoch is deterministic (a clock read here would churn the file every run) and
# reads unambiguously as "nothing has been published yet".
_EPOCH = "1970-01-01T00:00:00Z"

# Characters XML 1.0 cannot represent at all, escaped or not. Feed-supplied text
# occasionally carries stray control bytes, and they would poison the document.
_INVALID_XML = re.compile(r"[^\x09\x0a\x0d\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def write_feed(
    editions: list[dict[str, Any]],
    output_dir: Path,
    site_url: str,
    limit: int = DEFAULT_FEED_LIMIT,
    title: str = SITE_TITLE,
    subtitle: str = SITE_SUBTITLE,
    entry_title: Callable[[str], str] = lambda week: f"Week ending {week}",
) -> Path:
    """Write an Atom 1.0 feed with one entry per edition, newest first.

    The entry is the edition rather than the story: this is a weekly digest, and a
    reader subscribes to the week. Seven entries a week would also make every
    edition look like seven unrelated updates in a reader's timeline.

    ``title``/``subtitle``/``entry_title`` default to the weekly digest's own
    wording, so every existing call site is unaffected; a second feed for a
    different section of the site (see ``ai_daily.generate_ai_site``) passes
    its own.
    """
    base = _site_base(site_url)
    ordered = _newest_first(editions)[: max(limit, 0)]

    root = ET.Element("feed", {"xmlns": ATOM_NS})
    _text(root, "title", title)
    _text(root, "subtitle", subtitle)
    _text(root, "id", f"{base}/")
    _text(root, "updated", _updated(ordered[0]) if ordered else _EPOCH)
    ET.SubElement(
        root, "link", {"rel": "self", "type": "application/atom+xml", "href": _feed_url(base)}
    )
    ET.SubElement(root, "link", {"rel": "alternate", "type": "text/html", "href": f"{base}/"})
    _text(ET.SubElement(root, "author"), "name", SITE_AUTHOR)

    for edition in ordered:
        _append_entry(root, edition, base, entry_title)

    return _write_xml(root, output_dir / FEED_FILENAME)


def write_sitemap(editions: list[dict[str, Any]], output_dir: Path, site_url: str) -> Path:
    """Write a sitemap covering the index page and every archived edition page."""
    base = _site_base(site_url)
    ordered = _newest_first(editions)

    root = ET.Element("urlset", {"xmlns": SITEMAP_NS})
    # The index always shows the newest edition, so that is its last modification.
    _append_url(root, f"{base}/", _updated(ordered[0]) if ordered else None)
    for edition in ordered:
        _append_url(root, _page_url(base, edition), _updated(edition))

    return _write_xml(root, output_dir / SITEMAP_FILENAME)


def write_robots(output_dir: Path, site_url: str) -> Path:
    """Write a robots.txt that welcomes crawlers and advertises the sitemap."""
    base = _site_base(site_url)
    body = "\n".join([
        "User-agent: *",
        "Allow: /",
        "",
        f"Sitemap: {base}/{SITEMAP_FILENAME}",
        "",
    ])
    path = output_dir / ROBOTS_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def write_syndication(
    editions: list[dict[str, Any]],
    output_dir: Path,
    site_url: str,
    limit: int = DEFAULT_FEED_LIMIT,
) -> list[Path]:
    """Write every syndication artifact; the single call the pipeline needs."""
    written = [
        write_feed(editions, output_dir, site_url, limit=limit),
        write_sitemap(editions, output_dir, site_url),
        write_robots(output_dir, site_url),
    ]
    logging.info(f"Wrote {len(written)} syndication files in {output_dir}")
    return written


def _append_entry(
    root: ET.Element, edition: dict[str, Any], base: str, entry_title: Callable[[str], str]
) -> None:
    week = _week(edition)
    page = _page_url(base, edition)

    entry = ET.SubElement(root, "entry")
    _text(entry, "title", entry_title(week))
    _text(entry, "id", _entry_id(base, week))
    _text(entry, "updated", _updated(edition))
    ET.SubElement(entry, "link", {"rel": "alternate", "type": "text/html", "href": page})
    # type="html" means the value is an escaped HTML fragment, which is what a
    # text node gives us for free: the serializer escapes it on the way out.
    _text(entry, "content", _stories_html(edition), attrib={"type": "html"})


def _append_url(root: ET.Element, loc: str, lastmod: str | None) -> None:
    url = ET.SubElement(root, "url")
    _text(url, "loc", loc)
    if lastmod:
        _text(url, "lastmod", lastmod)


def _stories_html(edition: dict[str, Any]) -> str:
    """Render the week's stories as an ordered list of escaped HTML."""
    items = []
    for story in _ranked(edition):
        title = _esc(story.get("title")) or "Untitled"
        url = _esc(story.get("url"))
        source = _esc(story.get("source"))
        summary = _esc(story.get("summary"))

        headline = f'<a href="{url}">{title}</a>' if url else title
        if source:
            headline += f" <em>&mdash; {source}</em>"
        body = f"<p>{headline}</p>"
        if summary:
            body += f"<p>{summary}</p>"
        items.append(f"<li>{body}</li>")

    return "<ol>" + "".join(items) + "</ol>"


def _ranked(edition: dict[str, Any]) -> list[dict[str, Any]]:
    stories = edition.get("stories") or []
    return sorted(stories, key=lambda s: _rank(s))


def _rank(story: dict[str, Any]) -> int:
    rank = story.get("rank")
    return rank if isinstance(rank, int) else 0


def _esc(value: Any) -> str:
    """Escape a feed-supplied value for use in HTML text or an attribute value."""
    if value is None:
        return ""
    return html.escape(_INVALID_XML.sub("", str(value)), quote=True)


def _text(parent: ET.Element, tag: str, value: str, attrib: dict[str, str] | None = None) -> None:
    element = ET.SubElement(parent, tag, attrib or {})
    element.text = value


def _site_base(site_url: str) -> str:
    """Normalise the configured site URL to an origin with no trailing slash."""
    return site_url.strip().rstrip("/")


def _feed_url(base: str) -> str:
    return f"{base}/{FEED_FILENAME}"


def _page_url(base: str, edition: dict[str, Any]) -> str:
    return f"{base}/archive/{_week(edition)}.html"


def _entry_id(base: str, week: str) -> str:
    """Build a tag: URI for the edition.

    A tag: URI stays valid if the site ever moves path or protocol, which a page URL
    would not; feed readers key on the id and would show every entry again.
    """
    authority = urlsplit(base).netloc or base
    return f"tag:{authority},{week}:edition/{week}"


def _week(edition: dict[str, Any]) -> str:
    return str(edition.get("week") or "")


def _newest_first(editions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(editions, key=_week, reverse=True)


def _updated(edition: dict[str, Any]) -> str:
    """Return the edition's timestamp as RFC 3339 UTC, which Atom requires.

    Archived editions carry a ``Z`` suffix while a fresh ``build_edition`` writes
    ``+00:00``; both parse, and anything unusable falls back to the week itself so a
    malformed archive entry cannot break the whole feed.
    """
    raw = str(edition.get("generated_at") or "")
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return f"{_week(edition) or '1970-01-01'}T00:00:00Z"

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_xml(root: ET.Element, path: Path) -> Path:
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path
