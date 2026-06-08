from __future__ import annotations

from typing import Literal

from researchcrew.llm.base import ResearchLLM


LLMProvider = Literal["anthropic", "openai", "google"]


def create_llm(provider: LLMProvider = "anthropic", **kwargs) -> ResearchLLM:  # type: ignore[return]
    """Instantiate a ResearchLLM for the named provider.

    Args:
        provider: One of ``"anthropic"``, ``"openai"``, or ``"google"``.
        **kwargs: Forwarded to the provider constructor (api_key, model,
            max_tokens, and any provider-specific parameters).

    Returns:
        A concrete :class:`ResearchLLM` instance.

    Example::

        llm = create_llm("anthropic", api_key="sk-ant-...")
        llm = create_llm("openai",    api_key="sk-...", model="gpt-4o-mini")
        llm = create_llm("google",    api_key="AIza...", model="gemini-2.0-flash")
    """
    if provider == "anthropic":
        from researchcrew.llm.anthropic_llm import AnthropicResearchLLM

        return AnthropicResearchLLM(**kwargs)
    if provider == "openai":
        from researchcrew.llm.openai_llm import OpenAIResearchLLM

        return OpenAIResearchLLM(**kwargs)
    if provider == "google":
        from researchcrew.llm.google_llm import GoogleResearchLLM

        return GoogleResearchLLM(**kwargs)
    raise ValueError(
        f"Unknown provider {provider!r}. Valid choices: 'anthropic', 'openai', 'google'."
    )
