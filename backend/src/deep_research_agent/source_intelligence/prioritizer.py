from __future__ import annotations

import re
from urllib.parse import urlsplit

from .contracts import LinkCandidate, PrioritizedLink

NOISE_PATH_PARTS = {
    "account",
    "advertise",
    "auth",
    "billing",
    "careers",
    "cart",
    "checkout",
    "cookie",
    "jobs",
    "legal",
    "login",
    "logout",
    "privacy",
    "pricing",
    "register",
    "signin",
    "signup",
    "subscribe",
    "terms",
}
NOISE_HOSTS = {
    "facebook.com",
    "github.com",
    "instagram.com",
    "linkedin.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "youtube.com",
}
DOC_HINTS = {"docs", "documentation", "guide", "guides", "reference", "manual", "api", "learn"}
CITATION_HINTS = {
    "citation",
    "citations",
    "paper",
    "papers",
    "research",
    "study",
    "source",
    "sources",
}
USEFUL_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


def _tokens(value: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{3,}", (value or "").lower()) if len(t) >= 3}


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def _path_tokens(url: str) -> set[str]:
    return _tokens((urlsplit(url).path or "").replace("-", " ").replace("_", " "))


def _has_useful_extension(path: str) -> bool:
    path_l = path.lower()
    return any(path_l.endswith(ext) for ext in USEFUL_EXTENSIONS)


def noise_reason(candidate: LinkCandidate) -> str | None:
    parts = urlsplit(candidate.normalized_url)
    host = _host(candidate.normalized_url)
    if host in NOISE_HOSTS:
        return "noise_social_link"

    path_parts = {p.lower() for p in re.split(r"[/._-]+", parts.path or "") if p}
    anchor_tokens = _tokens(candidate.anchor_text)
    noise_overlap = (path_parts | anchor_tokens) & NOISE_PATH_PARTS
    if noise_overlap:
        return "noise_" + sorted(noise_overlap)[0]

    if parts.query and len(parts.query) > 180:
        return "noise_long_query"

    return None


def prioritize_link(candidate: LinkCandidate, *, root_url: str, question: str) -> PrioritizedLink:
    skip_reason = noise_reason(candidate)
    if skip_reason:
        return PrioritizedLink(
            candidate=candidate,
            score=-100.0,
            reasons=(),
            skip_reason=skip_reason,
        )

    score = 0.0
    reasons: list[str] = []
    root_host = _host(root_url)
    link_host = _host(candidate.normalized_url)

    if link_host == root_host:
        score += 20
        reasons.append("same_domain")
    elif root_host and (link_host.endswith("." + root_host) or root_host.endswith("." + link_host)):
        score += 12
        reasons.append("related_domain")
    else:
        score -= 8
        reasons.append("external_domain")

    question_tokens = _tokens(question)
    path_overlap = question_tokens & _path_tokens(candidate.normalized_url)
    anchor_overlap = question_tokens & _tokens(candidate.anchor_text)
    if path_overlap:
        score += min(18, len(path_overlap) * 4)
        reasons.append("path_relevance")
    if anchor_overlap:
        score += min(20, len(anchor_overlap) * 5)
        reasons.append("anchor_relevance")

    path = urlsplit(candidate.normalized_url).path or ""
    path_tokens = _path_tokens(candidate.normalized_url)
    if path_tokens & DOC_HINTS:
        score += 15
        reasons.append("documentation_hint")
    if path_tokens & CITATION_HINTS:
        score += 12
        reasons.append("citation_hint")
    if _has_useful_extension(path):
        score += 12
        reasons.append("useful_file_type")
    elif path and "." not in path.rsplit("/", 1)[-1]:
        score += 3
        reasons.append("content_page")

    return PrioritizedLink(candidate=candidate, score=score, reasons=tuple(reasons))


def prioritize_links(
    candidates: list[LinkCandidate],
    *,
    root_url: str,
    question: str,
) -> list[PrioritizedLink]:
    prioritized = [prioritize_link(c, root_url=root_url, question=question) for c in candidates]
    return sorted(
        prioritized,
        key=lambda item: (
            item.should_skip,
            -item.score,
            _host(item.candidate.normalized_url) != _host(root_url),
            item.candidate.normalized_url,
        ),
    )
