"""Polite rewrites: prompts, rewriters and guards."""

from kindify.rewrite.guards import GuardResult, check_rewrite, echoes_prompt
from kindify.rewrite.prompts import SYSTEM_PROMPT, CommentTooLong, build_messages
from kindify.rewrite.rewriters import HFChatRewriter, OpenAICompatibleRewriter, RewriteError, RuleRewriter

__all__ = [
    "SYSTEM_PROMPT",
    "CommentTooLong",
    "GuardResult",
    "HFChatRewriter",
    "OpenAICompatibleRewriter",
    "RewriteError",
    "RuleRewriter",
    "build_messages",
    "check_rewrite",
    "echoes_prompt",
]
