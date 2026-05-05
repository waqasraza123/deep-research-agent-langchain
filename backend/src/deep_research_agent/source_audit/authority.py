from __future__ import annotations

from ._heuristics import clamp, host_domain, url_path
from .contracts import SourceAuthorityScore

GOV_SUFFIXES = (".gov", ".gov.uk", ".europa.eu", ".mil")
ACADEMIC_SUFFIXES = (".edu", ".ac.uk", ".edu.au")
STANDARDS_HOSTS = (
    "ietf.org",
    "rfc-editor.org",
    "w3.org",
    "iso.org",
    "ieee.org",
    "nist.gov",
    "oasis-open.org",
    "ecma-international.org",
)
ACADEMIC_HOSTS = ("arxiv.org", "doi.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov")
REPO_HOSTS = ("github.com", "gitlab.com")
WEAK_HOST_TERMS = ("medium.com", "substack.com", "blogspot.", "wordpress.com")


def score_authority(
    *,
    url: str,
    title: str | None,
    text: str,
    source_type: str = "unknown",
) -> SourceAuthorityScore:
    domain = host_domain(url)
    path = url_path(url)
    title_l = (title or "").lower()
    text_l = (text or "")[:60_000].lower()
    signals: list[str] = []
    weaknesses: list[str] = []
    score = 0.35
    role = "unknown"

    if domain.endswith(GOV_SUFFIXES) or domain == "europa.eu" or domain.endswith(".europa.eu"):
        score += 0.35
        role = "primary"
        signals.append("Government or regulatory domain.")
    if domain.endswith(ACADEMIC_SUFFIXES) or domain in ACADEMIC_HOSTS:
        score += 0.28
        role = "primary"
        signals.append("Academic or scholarly domain.")
    if any(domain == host or domain.endswith("." + host) for host in STANDARDS_HOSTS):
        score += 0.38
        role = "primary"
        signals.append("Standards body domain.")
    if any(domain == host for host in REPO_HOSTS) and any(
        part in path for part in ("/releases", "/tags", "/blob/", "/tree/")
    ):
        score += 0.23
        role = "primary"
        signals.append("Repository, release, or source artifact.")
    if any(part in path for part in ("/docs", "/documentation", "/reference", "/api/")):
        score += 0.2
        if role == "unknown":
            role = "primary"
        signals.append("Documentation or reference path.")
    if any(part in path for part in ("/legal", "/regulations", "/rules", "/policy")):
        score += 0.18
        role = "primary"
        signals.append("Legal, policy, or rules path.")
    if any(term in text_l for term in ("abstract", "methodology", "references", "doi:")):
        score += 0.12
        signals.append("Scholarly/report structure detected.")
    if any(term in text_l for term in ("press release", "announces", "company blog")):
        score += 0.05
        signals.append("Company announcement or official blog signal.")
    if source_type in {"pdf", "csv", "docx"}:
        score += 0.08
        signals.append("Document or data file source type.")

    if any(term in domain for term in WEAK_HOST_TERMS):
        score -= 0.15
        role = "secondary" if role == "unknown" else role
        weaknesses.append("General publishing platform rather than an institutional source.")
    if any(part in path for part in ("/blog", "/tutorial", "/opinion", "/review")):
        score -= 0.08
        if role == "unknown":
            role = "secondary"
        weaknesses.append("Blog, tutorial, opinion, or review path.")
    if any(term in title_l for term in ("best ", "top ", "ultimate", "review", "vs ")):
        score -= 0.09
        weaknesses.append("Title suggests comparative or promotional secondary content.")
    if any(term in path for term in ("/tag/", "/category/", "/author/")):
        score -= 0.12
        role = "weak" if role == "unknown" else role
        weaknesses.append("Archive, tag, category, or author index path.")

    if role == "unknown":
        role = "secondary" if score >= 0.45 else "weak"

    reasons = signals[:]
    reasons.extend(weaknesses)
    if not reasons:
        reasons.append("No strong authority indicators were detected.")

    return SourceAuthorityScore(
        score=round(clamp(score), 3),
        source_role=role,  # type: ignore[arg-type]
        authority_signals=signals,
        weakness_signals=weaknesses,
        reasons=reasons,
    )
