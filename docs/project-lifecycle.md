# Project Lifecycle

ResearchCrew tracks each research project through a linear state machine. Every state transition can be gated by governance policies.

## Phases

```
IDEATION
   │
   ▼
LITERATURE_REVIEW
   │
   ▼
HYPOTHESIS_GENERATION
   │
   ▼
TOURNAMENT
   │
   ▼
WRITING
   │
   ▼
SUBMITTED
```

## Usage

```python
from researchcrew.project.project import ResearchProject, ProjectStatus
from researchcrew.project.lifecycle import ProjectLifecycle
from researchcrew.project.store import ProjectStore

# Create & persist
store   = ProjectStore()
project = store.create(ResearchProject(
    name="Drug Repurposing Study",
    domain="biomedicine",
    description="Identify approved drugs for AML targeting epigenetic regulators.",
    author="alice@lab.org",
))

print(project.status)          # ProjectStatus.IDEATION
print(project.workspace_dir)   # ~/.researchcrew/workspaces/<id>

# Advance through phases
lc = ProjectLifecycle(project)
lc.advance()   # → LITERATURE_REVIEW
lc.advance()   # → HYPOTHESIS_GENERATION
lc.advance()   # → TOURNAMENT
store.save(project)

# Load later
project = store.load(project.id)
print(project.status)   # ProjectStatus.TOURNAMENT
```

## ResearchProject Fields

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Auto-generated UUID |
| `name` | `str` | Human-readable project name |
| `domain` | `str` | e.g. `"biomedicine"`, `"machine_learning"` |
| `description` | `str` | Research goal / description |
| `author` | `str` | Optional author identifier |
| `status` | `ProjectStatus` | Current lifecycle phase |
| `workspace_dir` | `Path` | Per-project file storage |
| `created_at` | `datetime` | Creation timestamp |
| `updated_at` | `datetime` | Last modified timestamp |

## Governance Gates

Each `advance()` call passes through a `ResearchGatePolicy` that can block transition if quality criteria are not met:

```python
from researchcrew.governance.research_gate import ResearchGatePolicy, GateDecision

class MyPolicy(ResearchGatePolicy):
    def evaluate(self, project, transition) -> GateDecision:
        if transition.target == ProjectStatus.TOURNAMENT:
            n = len(project.hypotheses)
            if n < 5:
                return GateDecision.BLOCK(f"Need ≥5 hypotheses, have {n}.")
        return GateDecision.ALLOW()

lc = ProjectLifecycle(project, gate=MyPolicy())
lc.advance()   # Blocked if < 5 hypotheses
```

## CrewPilot Integration

The `HypothesisVariantBridge` connects each evolved hypothesis to a CrewPilot variant for prompt-level optimisation:

```python
from researchcrew.governance.variant_bridge import HypothesisVariantBridge

bridge = HypothesisVariantBridge()
variant_id = bridge.register(hypothesis)   # creates a CrewPilot variant
bridge.sync_score(hypothesis, elo_delta=+32)   # updates variant score
```

The `MetaReviewOptimizer` uses the meta-review agent's output to trigger a CrewPilot optimisation pass at the end of each tournament:

```python
from researchcrew.governance.meta_optimizer import MetaReviewOptimizer

optimizer = MetaReviewOptimizer()
optimizer.run(project, meta_review_output=crew_result)
```
