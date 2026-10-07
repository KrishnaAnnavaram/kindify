"""Rewriters behind one interface: `rewrite(comment) -> str`.

* `RuleRewriter`: deterministic word and pattern rules. Offline, used for the demo, the tests
  and as the fallback.
* `OpenAICompatibleRewriter`: any `/chat/completions` server (for example Ollama or vLLM). The
  reply holds only the new message, so the prompt cannot come back in it.
* `HFChatRewriter`: a local instruct model (extra `transformers`). It uses the chat template and
  decodes only the new tokens.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

from kindify.rewrite.prompts import build_messages


class RewriteError(RuntimeError):
    """The rewriter failed or gave an empty text."""


RULES: tuple[tuple[str, str], ...] = (
    (r"\b(?:you(?:'re| are)|ur)\s+(?:a|an|such an?|the)?\s*(?:total\s+|complete\s+)?"
     r"(?:idiot|moron|clown|fool|loser|jerk)s?\b", "I disagree with you"),
    (r"^\s*\w+ or not,\s*", ""),
    (r"^\s*listen,\s*(?:you\s+)?(?:idiot|moron|clown|fool|loser|jerk)s?,\s*", ""),
    (r"\bshut up\b", "please let me finish"),
    (r"\brevert it again and you will regret it\b", "please discuss it on the talk page first"),
    (r"\bnobody (?:wants|asked for) your\b", "I do not agree with your"),
    (r"\bstop touching\b", "please discuss changes to"),
    (r"\b(?:typical|stupid|dumb)\s+\w+\s+(?:idiot|moron|clown|fool|loser)\b", "I disagree with this"),
    (r",?\s*(?:you\s+)?(?:idiot|moron|clown|fool|loser|jerk)s?\b", ""),
    (r"\byou will regret it\b", "I will ask for a review"),
    (r"\b(?:damn|crap|hell)\b", ""),
    (r"\b(?:garbage|trash|pathetic)\b", "weak"),
    (r"\b(?:stupid|dumb)\b", "unclear"),
    (r"\bwhat an? weak\b", "this is a weak"),
    (r"\bweak,?\s+weak\b", "weak"),
)


def _tidy(text: str) -> str:
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"!+", ".", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,")
    text = re.sub(r"(^|[.?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    if text and text[-1] not in ".?":
        text += "."
    return text


class RuleRewriter:
    name = "rules"

    def rewrite(self, comment: str) -> str:
        text = str(comment)
        if text.isupper():
            text = text.lower()
        for pattern, repl in RULES:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        out = _tidy(text)
        if not out.strip(" ."):
            raise RewriteError("nothing is left after the rules")
        return out


class OpenAICompatibleRewriter:
    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 20.0,
                 temperature: float = 0.2, max_chars: int = 4000):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.temperature = temperature
        self.max_chars = max_chars

    def _post(self, url: str, payload: dict) -> dict:  # pragma: no cover - network
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def rewrite(self, comment: str) -> str:
        payload = {"model": self.model, "messages": build_messages(comment, max_chars=self.max_chars),
                   "temperature": self.temperature}
        try:
            data = self._post(f"{self.base_url}/chat/completions", payload)
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError, OSError) as exc:
            raise RewriteError(f"LLM call failed: {exc}") from exc
        text = str(text).strip().strip('"').strip()
        if not text:
            raise RewriteError("the LLM gave an empty text")
        return text


def decode_new_tokens(tokenizer, output_ids, input_length: int) -> str:
    """Decode only the tokens after the prompt. The prompt text can never be in the result."""
    return tokenizer.decode(list(output_ids)[input_length:], skip_special_tokens=True).strip()


class HFChatRewriter:
    name = "hf"

    def __init__(self, model=None, tokenizer=None, model_name: str | None = None, max_new_tokens: int = 120,
                 max_chars: int = 4000):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_chars = max_chars

    def _load(self):  # pragma: no cover - needs a download
        if self.model is None:
            try:
                import transformers
            except ImportError as exc:
                raise RewriteError("install the extra: pip install 'kindify[transformers]'") from exc
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name)

    def rewrite(self, comment: str) -> str:
        self._load()
        import torch

        messages = build_messages(comment, max_chars=self.max_chars)
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id or 0)
        text = decode_new_tokens(self.tokenizer, out[0].tolist(), ids.shape[1])
        if not text:
            raise RewriteError("the model gave an empty text")
        return text
