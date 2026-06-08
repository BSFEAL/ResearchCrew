from __future__ import annotations

from researchcrew.llm.base import ResearchLLM


class GoogleResearchLLM(ResearchLLM):
    """ResearchLLM backed by the Google Gemini SDK (google-genai)."""

    DEFAULT_MODEL = "gemini-2.0-flash"
    _SYSTEM = "You are a scientific research assistant helping produce high-quality academic work."

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
    ) -> None:
        try:
            from google import genai  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "google-genai package is required. Install with: pip install 'researchcrew[llm]'"
            ) from exc
        self._client = genai.Client(api_key=api_key)
        self._genai = genai
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        from google.genai import types  # type: ignore[import]

        system = self._SYSTEM
        if json_mode:
            system += " Respond with valid JSON only. No markdown fences, no commentary."

        config_kwargs: dict = {
            "system_instruction": system,
            "max_output_tokens": self.max_tokens,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text or ""
