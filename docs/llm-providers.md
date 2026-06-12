# LLM Providers

ResearchCrew uses a single abstract interface — `ResearchLLM` — so every crew component is provider-agnostic. Switch providers by passing a different object to `create_llm()`.

## Installation

```bash
pip install 'researchcrew[llm]'
```

This installs `anthropic`, `openai`, and `google-genai`.

---

## Factory

```python
from researchcrew.llm import create_llm, LLMProvider

# provider: "anthropic" | "openai" | "google"
llm = create_llm("anthropic")
```

All keyword arguments are forwarded to the provider constructor.

---

## Anthropic Claude

```python
from researchcrew.llm import create_llm

llm = create_llm("anthropic",
    api_key="sk-ant-...",        # or set ANTHROPIC_API_KEY
    model="claude-opus-4-8",     # default
    max_tokens=4096,
    thinking=True,               # adaptive thinking (auto-disabled for haiku)
)
```

**Adaptive thinking** is enabled by default for all non-Haiku models. It lets the model spend variable compute on hard problems before responding. To disable:

```python
llm = create_llm("anthropic", thinking=False)
```

Haiku models do not support thinking — it is automatically disabled regardless of the `thinking` argument.

### Supported Models

| Model | ID |
|---|---|
| Claude Opus 4.8 | `claude-opus-4-8` |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` |
| Claude Haiku 4.5 | `claude-haiku-4-5` |

---

## OpenAI GPT

```python
llm = create_llm("openai",
    api_key="sk-...",            # or set OPENAI_API_KEY
    model="gpt-4o",              # default
    max_tokens=4096,
)
```

JSON mode is enabled automatically when a component calls `complete(..., json_mode=True)`.

---

## Google Gemini

```python
llm = create_llm("google",
    api_key="AIza...",           # or set GOOGLE_API_KEY
    model="gemini-2.0-flash",    # default
    max_tokens=4096,
)
```

Uses the `google-genai` SDK. JSON mode sets `response_mime_type="application/json"`.

---

## Custom Provider

Subclass `ResearchLLM` and implement `complete()`. All factory methods (`section_writer_fn`, `outline_fn`, `review_fn`, etc.) are provided automatically.

```python
from researchcrew.llm.base import ResearchLLM

class OllamaLLM(ResearchLLM):
    def __init__(self, model: str = "llama3"):
        import httpx
        self._client = httpx.Client(base_url="http://localhost:11434")
        self.model = model

    def complete(self, prompt: str, *, json_mode: bool = False) -> str:
        resp = self._client.post("/api/generate", json={
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        })
        return resp.json()["response"]

llm = OllamaLLM()
crew = make_co_scientist_crew(llm=llm)
```

---

## Injection Points

Each crew component accepts a specific callable signature. `ResearchLLM` factory methods return pre-shaped callables:

| Factory Method | Used By | Signature |
|---|---|---|
| `section_writer_fn()` | `SectionWriter` | `(section_name, context) → str` |
| `outline_fn()` | `OutlineBuilder` | `(prompt) → dict` |
| `review_fn()` | `PeerReviewSimulator` | `(section) → dict[str, float]` |
| `rewrite_fn()` | `PeerReviewSimulator` | `(section, score) → str` |
| `screener_fn()` | `TwoPhaseScreener` | `(paper, research_goal) → float` |
| `debate_judge()` | `DebateTool` | Returns `LLMDebateJudge` |

Use them directly:

```python
llm = create_llm("anthropic")

writer_fn   = llm.section_writer_fn()
outline_fn  = llm.outline_fn()
screener_fn = llm.screener_fn()
```
