from __future__ import annotations

from researchcrew.llm.base import ResearchLLM


class AnthropicResearchLLM(ResearchLLM):
    """ResearchLLM backed by the Anthropic Python SDK.

    Adaptive thinking is enabled by default for all models that support it
    (opus, sonnet).  Haiku models do not support thinking, so it is
    automatically disabled regardless of the *thinking* argument.
    """

    DEFAULT_MODEL = "claude-opus-4-8"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        thinking: bool = True,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required. Install with: pip install 'researchcrew[llm]'"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        # Haiku does not support thinking; disable automatically.
        self._thinking = thinking and "haiku" not in model.lower()

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        system = "You are a scientific research assistant helping produce high-quality academic work."
        if json_mode:
            system += " Respond with valid JSON only. No markdown fences, no commentary."

        create_kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._thinking:
            create_kwargs["thinking"] = {"type": "adaptive"}

        response = self._client.messages.create(**create_kwargs)
        # Thinking blocks have type "thinking" and contain empty text by default.
        # Return only the text blocks.
        return "".join(
            block.text
            for block in response.content
            if hasattr(block, "text") and block.type == "text"
        )
