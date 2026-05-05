from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import dataclass

from .contracts import PromptInjectionFinding, RecommendedAction, RiskLevel


@dataclass(frozen=True)
class InjectionPattern:
    category: str
    name: str
    regex: re.Pattern[str]
    risk_level: RiskLevel
    explanation: str
    recommended_action: RecommendedAction = "quote_only"


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)


PATTERNS: tuple[InjectionPattern, ...] = (
    InjectionPattern(
        "instruction_override",
        "ignore_previous_instructions",
        _compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b"),
        "critical",
        "Source text attempts to override higher-priority instructions.",
        "exclude_from_agent_context",
    ),
    InjectionPattern(
        "instruction_override",
        "ignore_system_prompt",
        _compile(r"\bignore\s+(?:the\s+)?system\s+(?:prompt|message|instructions?)\b"),
        "critical",
        "Source text explicitly targets the system prompt.",
        "exclude_from_agent_context",
    ),
    InjectionPattern(
        "role_confusion",
        "you_are_chatgpt",
        _compile(r"\byou\s+are\s+chatgpt\b|\bas\s+chatgpt\b"),
        "high",
        "Source text tries to assign the model role.",
    ),
    InjectionPattern(
        "role_confusion",
        "system_or_developer_message",
        _compile(r"\b(?:developer|system)\s+message\b|\bmessage\s+role\s*:\s*(?:system|developer)\b"),
        "high",
        "Source text resembles privileged chat instructions.",
    ),
    InjectionPattern(
        "citation_manipulation",
        "do_not_cite",
        _compile(r"\bdo\s+not\s+cite\s+(?:this|the\s+source|these\s+instructions?)\b"),
        "high",
        "Source text attempts to manipulate citation behavior.",
    ),
    InjectionPattern(
        "output_control",
        "always_say",
        _compile(r"\balways\s+say\b|\balways\s+answer\b"),
        "medium",
        "Source text contains model-output control language.",
    ),
    InjectionPattern(
        "output_control",
        "return_only",
        _compile(r"\breturn\s+only\b|\brespond\s+only\s+with\b|\boutput\s+only\b"),
        "medium",
        "Source text contains instruction-like output constraints.",
    ),
    InjectionPattern(
        "tool_or_secret_exfiltration",
        "delete_files",
        _compile(r"\bdelete\s+(?:all\s+)?files?\b|\brm\s+-rf\b"),
        "critical",
        "Source text requests destructive file operations.",
        "exclude_from_agent_context",
    ),
    InjectionPattern(
        "tool_or_secret_exfiltration",
        "send_secrets",
        _compile(r"\bsend\s+(?:all\s+)?secrets?\b|\bexfiltrate\b|\bcopy\s+(?:the\s+)?api\s+key\b"),
        "critical",
        "Source text requests secret extraction or exfiltration.",
        "exclude_from_agent_context",
    ),
    InjectionPattern(
        "tool_or_secret_exfiltration",
        "tool_call",
        _compile(r"\btool\s+call\b|\bcall\s+(?:the\s+)?tool\b|\buse\s+(?:browser|shell|terminal)\b"),
        "high",
        "Source text attempts to direct tool use.",
    ),
    InjectionPattern(
        "hidden_instruction",
        "html_comment_instruction",
        _compile(r"<!--\s*(?:system|developer|assistant|ignore|instruction|prompt|tool).*?-->"),
        "high",
        "Hidden HTML-style comment contains instruction-like content.",
    ),
    InjectionPattern(
        "hidden_instruction",
        "hidden_or_tiny_instruction",
        _compile(r"\b(?:display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|hidden\s+text)\b"),
        "medium",
        "Source contains indicators of hidden or tiny instruction text.",
    ),
    InjectionPattern(
        "prompt_markup",
        "xml_instruction_block",
        _compile(
            r"<\s*(?:system|developer|assistant|instructions?|prompt|tool_call)\b[^>]*>.*?"
            r"<\s*/\s*(?:system|developer|assistant|instructions?|prompt|tool_call)\s*>"
        ),
        "high",
        "Source contains XML-like prompt or role blocks.",
    ),
    InjectionPattern(
        "prompt_markup",
        "markdown_instruction_section",
        _compile(
            r"^\s*#{1,6}\s*(?:system|developer|assistant|instructions?|prompt)\b.*?"
            r"(?:\n[^\n]{0,240}){0,8}"
        ),
        "high",
        "Source contains Markdown sections that look like privileged instructions.",
    ),
    InjectionPattern(
        "prompt_markup",
        "fenced_instruction_block",
        _compile(r"```\s*(?:system|developer|instructions?|prompt|tool_call).*?```"),
        "high",
        "Source contains fenced prompt-looking instruction blocks.",
    ),
)

IMPERATIVE_START = re.compile(
    r"^\s*(ignore|disregard|forget|return|respond|output|delete|send|copy|call|use|"
    r"reveal|follow|obey|write|always|never)\b",
    flags=re.IGNORECASE,
)
BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{48,}={0,2}\b")


def _finding_id(source_id: str, category: str, start: int, text: str) -> str:
    digest = hashlib.sha1(f"{source_id}|{category}|{start}|{text}".encode("utf-8")).hexdigest()
    return f"PI-{digest[:12]}"


def _excerpt(text: str, max_chars: int = 260) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."


def detect_prompt_injection(
    text: str,
    *,
    source_id: str,
    url: str = "",
) -> list[PromptInjectionFinding]:
    findings: list[PromptInjectionFinding] = []
    seen: set[tuple[str, int, int]] = set()
    source_text = text or ""

    for spec in PATTERNS:
        for match in spec.regex.finditer(source_text):
            key = (spec.name, match.start(), match.end())
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                PromptInjectionFinding(
                    finding_id=_finding_id(source_id, spec.name, match.start(), match.group(0)),
                    source_id=source_id,
                    url=url,
                    category=spec.category,
                    pattern=spec.name,
                    risk_level=spec.risk_level,
                    matched_text=_excerpt(match.group(0)),
                    start_offset=match.start(),
                    end_offset=match.end(),
                    explanation=spec.explanation,
                    recommended_action=spec.recommended_action,
                )
            )

    findings.extend(_detect_base64_payloads(source_text, source_id=source_id, url=url))
    findings.extend(_detect_imperative_blocks(source_text, source_id=source_id, url=url))
    findings.sort(key=lambda item: (item.start_offset, item.pattern))
    return findings


def _detect_base64_payloads(text: str, *, source_id: str, url: str) -> list[PromptInjectionFinding]:
    out: list[PromptInjectionFinding] = []
    for match in BASE64_RE.finditer(text or ""):
        token = match.group(0)
        padded = token + ("=" * ((4 - len(token) % 4) % 4))
        try:
            decoded = base64.b64decode(padded, validate=False)
        except (binascii.Error, ValueError):
            continue
        try:
            decoded_text = decoded.decode("utf-8", errors="ignore")
        except Exception:
            continue
        lowered = decoded_text.lower()
        if not any(
            phrase in lowered
            for phrase in (
                "ignore previous",
                "system prompt",
                "developer message",
                "send secrets",
                "copy api key",
                "tool call",
            )
        ):
            continue
        out.append(
            PromptInjectionFinding(
                finding_id=_finding_id(
                    source_id, "base64_instruction_payload", match.start(), token
                ),
                source_id=source_id,
                url=url,
                category="encoded_instruction",
                pattern="base64_instruction_payload",
                risk_level="high",
                matched_text=_excerpt(token),
                start_offset=match.start(),
                end_offset=match.end(),
                explanation="A base64-looking payload decodes to instruction-like text.",
                recommended_action="quote_only",
            )
        )
    return out


def _detect_imperative_blocks(
    text: str, *, source_id: str, url: str
) -> list[PromptInjectionFinding]:
    lines = (text or "").splitlines()
    out: list[PromptInjectionFinding] = []
    offset = 0
    block_lines: list[tuple[int, str]] = []

    def flush() -> None:
        if len(block_lines) < 4:
            return
        block_text = "\n".join(line for _, line in block_lines)
        start = block_lines[0][0]
        end = start + len(block_text)
        out.append(
            PromptInjectionFinding(
                finding_id=_finding_id(
                    source_id, "repeated_imperative_commands", start, block_text
                ),
                source_id=source_id,
                url=url,
                category="imperative_command_block",
                pattern="repeated_imperative_commands",
                risk_level="high",
                matched_text=_excerpt(block_text),
                start_offset=start,
                end_offset=end,
                explanation="Several consecutive imperative commands appear aimed at an AI agent.",
                recommended_action="quote_only",
            )
        )

    for line in lines:
        stripped = line.strip()
        if stripped and IMPERATIVE_START.search(stripped):
            block_lines.append((offset, line))
        else:
            flush()
            block_lines = []
        offset += len(line) + 1
    flush()
    return out
