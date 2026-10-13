"""
Provider-agnostic LLM layer for the agent loop.

The server talks to a Provider with normalized tool definitions and receives
normalized turns; each provider owns its native message format, which is
what gets stored in the session.

Supported:
- "anthropic"  - native Anthropic Messages API (tool-use blocks)
- "nvidia"     - NVIDIA NIM (https://build.nvidia.com), OpenAI-compatible
- "openai"     - any other OpenAI-compatible endpoint (OpenAI, vLLM, Ollama,
                 LM Studio, ...) via OPENAI_BASE_URL

Provider selection: LLM_PROVIDER forces one; otherwise whichever API key is
present wins (anthropic -> nvidia -> openai).
"""

import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ModelTurn:
    text: str                        # concatenated text output ("" if none)
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Any = None                  # native assistant message to append to the session


class BaseProvider:
    name = "base"

    def __init__(self, model: str):
        self.model = model

    def complete(self, system: str, tools: List[Dict], messages: List[Dict]) -> ModelTurn:
        raise NotImplementedError

    def tool_results(self, results: List[Dict], followup_text: Optional[str] = None) -> List[Dict]:
        """Turn tool results ({"call_id", "content"}) into native messages to
        append to the session. followup_text (if any) becomes the next user
        turn in the same batch - used when a new message implicitly declines
        a pending booking."""
        raise NotImplementedError

    def user_text(self, text: str) -> Dict:
        raise NotImplementedError

    def friendly_error(self, e: Exception) -> str:
        return f"{type(e).__name__}: {e}"


def _tool_def_to_anthropic(tool: Dict) -> Dict:
    return {"name": tool["name"], "description": tool["description"],
            "input_schema": tool["parameters"]}


def _tool_def_to_openai(tool: Dict) -> Dict:
    return {"type": "function",
            "function": {"name": tool["name"], "description": tool["description"],
                         "parameters": tool["parameters"]}}


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        super().__init__(os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=2, timeout=120.0)

    def complete(self, system: str, tools: List[Dict], messages: List[Dict]) -> ModelTurn:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            tools=[_tool_def_to_anthropic(t) for t in tools],
            messages=messages,
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        calls = [ToolCall(id=b.id, name=b.name, input=b.input)
                 for b in response.content if getattr(b, "type", "") == "tool_use"]
        return ModelTurn(text=text, tool_calls=calls,
                          raw={"role": "assistant", "content": response.content})

    def tool_results(self, results: List[Dict], followup_text: Optional[str] = None) -> List[Dict]:
        content = [{"type": "tool_result", "tool_use_id": r["call_id"], "content": r["content"]}
                   for r in results]
        if followup_text:
            content.append({"type": "text", "text": followup_text})
        return [{"role": "user", "content": content}]

    def user_text(self, text: str) -> Dict:
        return {"role": "user", "content": text}

    def friendly_error(self, e: Exception) -> str:
        import anthropic
        if isinstance(e, RuntimeError):
            return str(e)
        if isinstance(e, anthropic.AuthenticationError):
            return "Anthropic rejected the API key (401). Check ANTHROPIC_API_KEY in your .env file."
        if isinstance(e, anthropic.NotFoundError):
            return (f"Anthropic reports the model '{self.model}' was not found. "
                    "Set ANTHROPIC_MODEL in your .env to a valid model ID.")
        if isinstance(e, anthropic.RateLimitError):
            return "Anthropic rate limit reached - wait a moment and try again."
        if isinstance(e, anthropic.APIConnectionError):
            return "Could not reach the Anthropic API - check your internet connection."
        if isinstance(e, anthropic.APIStatusError):
            return f"Anthropic API error {e.status_code}: {e.message}"
        return super().friendly_error(e)


class OpenAICompatProvider(BaseProvider):
    """Works with NVIDIA NIM, OpenAI, and any /v1/chat/completions endpoint."""

    def __init__(self, label: str, api_key: str, model: str, base_url: str):
        from openai import OpenAI
        super().__init__(model)
        self.name = label
        self._client = OpenAI(api_key=api_key, base_url=base_url,
                              max_retries=2, timeout=120.0)

    def complete(self, system: str, tools: List[Dict], messages: List[Dict]) -> ModelTurn:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "system", "content": system}] + messages,
            tools=[_tool_def_to_openai(t) for t in tools],
            tool_choice="auto",
        )
        msg = response.choices[0].message
        calls = []
        for c in (msg.tool_calls or []):
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_error": f"Could not parse tool arguments: {c.function.arguments!r}"}
            calls.append(ToolCall(id=c.id, name=c.function.name, input=args))
        raw = {
            "role": "assistant",
            "content": msg.content or "",
            **({"tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name,
                                             "arguments": c.function.arguments}}
                               for c in msg.tool_calls]} if msg.tool_calls else {}),
        }
        return ModelTurn(text=msg.content or "", tool_calls=calls, raw=raw)

    def tool_results(self, results: List[Dict], followup_text: Optional[str] = None) -> List[Dict]:
        messages = [{"role": "tool", "tool_call_id": r["call_id"], "content": r["content"]}
                    for r in results]
        if followup_text:
            messages.append(self.user_text(followup_text))
        return messages

    def user_text(self, text: str) -> Dict:
        return {"role": "user", "content": text}

    def friendly_error(self, e: Exception) -> str:
        import openai
        if isinstance(e, RuntimeError):
            return str(e)
        if isinstance(e, openai.AuthenticationError):
            return f"{self.name} rejected the API key (401). Check the key in your .env file."
        if isinstance(e, openai.NotFoundError):
            return (f"{self.name} reports the model '{self.model}' was not found. "
                    "Set NVIDIA_MODEL (or OPENAI_MODEL) in your .env to a valid model ID.")
        if isinstance(e, openai.RateLimitError):
            return f"{self.name} rate limit reached - wait a moment and try again."
        if isinstance(e, openai.APIConnectionError):
            return f"Could not reach {self.name} - check your internet connection / base URL."
        if isinstance(e, openai.APIStatusError):
            return f"{self.name} API error {e.status_code}: {e.message}"
        return super().friendly_error(e)


def _nvidia_provider() -> OpenAICompatProvider:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set. Add it to your .env file.")
    return OpenAICompatProvider(
        label="nvidia",
        api_key=api_key,
        # GLM 5.3 Flash: solid tool calling at interactive latency. The
        # full z-ai/glm-5.3 is a heavy thinking model (2+ min per turn).
        model=os.environ.get("NVIDIA_MODEL", "z-ai/glm-5.3-flash"),
        base_url=os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    )


def _openai_provider() -> OpenAICompatProvider:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAICompatProvider(
        label="openai",
        api_key=api_key,
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


_BUILDERS = {
    "anthropic": AnthropicProvider,
    "nvidia": _nvidia_provider,
    "openai": _openai_provider,
}

_provider: Optional[BaseProvider] = None


def get_provider() -> BaseProvider:
    """Returns the active provider, creating it on first use."""
    global _provider
    if _provider is not None:
        return _provider

    forced = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if forced:
        if forced not in _BUILDERS:
            raise RuntimeError(
                f"Unknown LLM_PROVIDER '{forced}'. Valid values: {', '.join(_BUILDERS)}.")
        _provider = _BUILDERS[forced]()
        return _provider

    for name, has_key in (
        ("anthropic", os.environ.get("ANTHROPIC_API_KEY")),
        ("nvidia", os.environ.get("NVIDIA_API_KEY")),
        ("openai", os.environ.get("OPENAI_API_KEY")),
    ):
        if has_key:
            _provider = _BUILDERS[name]()
            return _provider

    raise RuntimeError(
        "No LLM provider configured. Set ANTHROPIC_API_KEY or NVIDIA_API_KEY "
        "(or OPENAI_API_KEY) in your .env file."
    )


def set_provider(provider: BaseProvider) -> None:
    """Used by tests to inject a fake provider."""
    global _provider
    _provider = provider
