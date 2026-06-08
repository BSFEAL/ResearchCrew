from __future__ import annotations

from researchcrew.llm.base import ResearchLLM


class AnthropicResearchLLM(ResearchLLM):
    """ResearchLLM backed by the Anthropic Python SDK."""

    DEFAULT_MODEL = "claude-opus-4-8"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
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

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        system = "You are a scientific research assistant helping produce high-quality academic work."
        if json_mode:
            system += " Respond with valid JSON only. No markdown fences, no commentary."
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
