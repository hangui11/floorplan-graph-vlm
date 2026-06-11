"""
Robust JSON extraction from VLM text output.

VLMs often wrap JSON in markdown code blocks, add explanatory text,
or produce slightly malformed JSON. This module handles all of that.
"""
import json
import re
import logging

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict | None:
    """
    Extract a JSON object from VLM output text.

    Tries multiple strategies:
    1. Direct parse (text is already valid JSON)
    2. Extract from markdown code block (```json ... ```)
    3. Find first { ... } or [ ... ] block via brace matching
    4. Fix common issues (trailing commas, single quotes)

    Returns the parsed dict/list, or None if all strategies fail.
    """
    text = text.strip()

    # Strategy 1: direct parse
    result = _try_parse(text)
    if result is not None:
        return result

    # Strategy 2: markdown code block
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block:
        result = _try_parse(code_block.group(1).strip())
        if result is not None:
            return result

    # Strategy 3: find outermost { ... } via brace matching
    result = _extract_braced(text)
    if result is not None:
        return result

    # Strategy 4: fix common issues and retry
    result = _try_parse_with_fixes(text)
    if result is not None:
        return result

    logger.warning("Failed to extract JSON from VLM output:\n%s", text[:500])
    return None


def _try_parse(text: str) -> dict | list | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_braced(text: str) -> dict | list | None:
    """Find the outermost JSON object or array via brace/bracket matching."""
    for open_char, close_char in [('{', '}'), ('[', ']')]:
        start = text.find(open_char)
        if start == -1:
            continue

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    result = _try_parse(candidate)
                    if result is not None:
                        return result
                    # Try with fixes
                    result = _try_parse_with_fixes(candidate)
                    if result is not None:
                        return result
                    break
    return None


def _try_parse_with_fixes(text: str) -> dict | list | None:
    """Apply common fixes to malformed JSON and try parsing."""
    fixed = text

    # Remove trailing commas before } or ]
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

    # Replace single quotes with double quotes (naive but often works)
    # Only if no double quotes are present in string values
    if '"' not in fixed and "'" in fixed:
        fixed = fixed.replace("'", '"')

    # Fix unquoted keys: {key: "value"} -> {"key": "value"}
    fixed = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r' "\1":', fixed)

    # Replace None/True/False python literals with JSON equivalents
    fixed = re.sub(r'\bNone\b', 'null', fixed)
    fixed = re.sub(r'\bTrue\b', 'true', fixed)
    fixed = re.sub(r'\bFalse\b', 'false', fixed)

    return _try_parse(fixed)
