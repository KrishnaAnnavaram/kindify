"""Chat messages for an instruct model.

The comment is never cut away: when the prompt is too long, the few-shot examples go first.
If the comment alone is too long, `CommentTooLong` is raised and the service does not rewrite.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You rewrite online comments so that they are polite and respectful. "
    "Keep the meaning, the facts and the disagreement of the comment. "
    "Remove insults, threats and profanity. "
    "Reply with the rewritten comment only, with no explanation."
)

EXAMPLES = (
    ("You are an idiot and this edit is garbage.", "I disagree with this edit, and I do not think it improves the page."),
    ("Shut up, nobody asked for your opinion.", "Please let others finish. I see this point in a different way."),
    ("This article is pathetic trash.", "This article needs a lot of work."),
)


class CommentTooLong(ValueError):
    """The comment does not fit in the prompt budget."""


def _size(messages) -> int:
    return sum(len(m["content"]) for m in messages)


def build_messages(comment: str, examples=EXAMPLES, max_chars: int = 4000) -> list[dict]:
    """System prompt, as many few-shot turns as fit, then the comment as the last user turn."""
    comment = str(comment).strip()
    if not comment:
        raise ValueError("empty comment")
    base = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": comment}]
    if _size(base) > max_chars:
        raise CommentTooLong(f"the comment has {len(comment)} characters, the budget is {max_chars}")
    shots: list[dict] = []
    for src, dst in examples:
        pair = [{"role": "user", "content": src}, {"role": "assistant", "content": dst}]
        if _size(base) + _size(shots) + _size(pair) > max_chars:
            break
        shots += pair
    return [base[0], *shots, base[1]]
