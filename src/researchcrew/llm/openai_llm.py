from __future__ import annotations

from researchcrew.llm.base import ResearchLLM


class OpenAIResearchLLM(ResearchLLM):
    """ResearchLLM backed by the OpenAI Python SDK."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai package is required. Install with: pip install 'researchcrew[llm]'"
            ) from exc
        self._client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        system = "You are a scientific research assistant helping produce high-quality academic work."
        if json_mode:
            system += " Respond with valid JSON only. No markdown fences, no commentary."
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **kwargs,
        )
        return response.choices[0].message.content or ""
