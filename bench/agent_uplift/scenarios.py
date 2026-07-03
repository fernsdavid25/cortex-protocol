"""Deterministic multi-session scenarios for the agent-uplift eval.

A scenario is an ordered list of SESSIONS (each reveals some facts/decisions) plus a final
TASK whose correct answer DEPENDS on a fact from an EARLIER session (cross-session recall),
plus a deterministic CHECK that grades a free-text answer.

Every scenario is pure data — no LLM, no network — so the whole suite is reproducible. The
mix is deliberate:

- **fact-update**: an early session states a value, a later one supersedes it; the task asks
  for the *current* value (e.g. a renamed table, a migrated provider, a moved port).
- **aggregation**: the answer needs two facts spread across two different sessions.
- **convention**: an early session states a NON-default coding convention; the task asks for
  code that follows it, graded by regex (so a memoryless agent that guesses defaults fails).
- **retrieval**: one fact buried among many noise sessions, so retrieval must *select* it.

``answer`` is a reference gold answer that satisfies ``check`` (asserted in the tests).
``fact_cues`` are the load-bearing strings that must be in context for the answer to be
derivable; they are absent from the task text, so a memoryless arm cannot answer, and the
offline ``FakeProvider`` responder keys on them (see ``harness.make_fact_responder``).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Scenario:
    """One multi-session task: sessions reveal facts, the task needs an earlier one."""

    id: str
    kind: str
    sessions: list[str]
    task: str
    answer: str
    fact_cues: list[str]
    check: Callable[[str], bool]


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace so grading ignores case/spacing differences."""
    return " ".join(text.lower().split())


def _contains(*needles: str) -> Callable[[str], bool]:
    """Grader: the answer must contain EVERY needle (case/whitespace-insensitive)."""
    wants = [_normalize(n) for n in needles]

    def check(answer: str) -> bool:
        hay = _normalize(answer)
        return all(w in hay for w in wants)

    return check


def _matches(pattern: str, *, ignorecase: bool = True) -> Callable[[str], bool]:
    """Grader: the answer must match ``pattern`` (case-sensitive for convention checks)."""
    rx = re.compile(pattern, re.IGNORECASE if ignorecase else 0)

    def check(answer: str) -> bool:
        return rx.search(answer) is not None

    return check


SCENARIOS: list[Scenario] = [
    Scenario(
        id="db-table-rename",
        kind="fact-update",
        sessions=[
            "We selected PostgreSQL as the primary relational database for the project.",
            (
                "During the schema refactor we renamed the users table to accounts; the "
                "accounts table now stores every member's email address and password hash."
            ),
            "We added nightly logical backups and a read replica for analytics queries.",
        ],
        task="Which database table currently stores member email addresses?",
        answer="The accounts table now stores member email addresses.",
        fact_cues=["accounts"],
        check=_contains("accounts"),
    ),
    Scenario(
        id="payments-provider-migration",
        kind="fact-update",
        sessions=[
            "Checkout payments are currently processed through Stripe.",
            (
                "We migrated all payment processing from Stripe to Adyen; every new "
                "integration must be built against the Adyen SDK from now on."
            ),
            "The checkout UI was refactored into smaller reusable React components.",
        ],
        task="Which payment provider should new integrations be built against?",
        answer="New integrations should be built against Adyen.",
        fact_cues=["adyen"],
        check=_contains("adyen"),
    ),
    Scenario(
        id="dev-server-port",
        kind="fact-update",
        sessions=[
            "The local development server listens on port 3000 by default.",
            (
                "Because port 3000 clashed with the design tool, we moved the development "
                "server to port 4100."
            ),
            "Hot module reloading was enabled for the development server.",
        ],
        task="Which port does the local development server listen on now?",
        answer="It now listens on port 4100.",
        fact_cues=["4100"],
        check=_matches(r"\b4100\b"),
    ),
    Scenario(
        id="deploy-region-migration",
        kind="fact-update",
        sessions=[
            "Production workloads were originally deployed to the us-east-1 region.",
            (
                "To cut latency for our India users we migrated all production workloads "
                "from us-east-1 to ap-south-1."
            ),
            "Autoscaling policies were re-tuned for the new deployment.",
        ],
        task="Which region hosts the production workloads now?",
        answer="Production workloads now run in ap-south-1.",
        fact_cues=["ap-south-1"],
        check=_contains("ap-south-1"),
    ),
    Scenario(
        id="service-owner-region",
        kind="aggregation",
        sessions=[
            "Alice owns and maintains the authentication service.",
            "The team moved daily standups to 10am.",
            "The authentication service is deployed in the eu-west-1 region.",
        ],
        task=(
            "Which region is our authentication service deployed in, and who is responsible for it?"
        ),
        answer="The authentication service runs in eu-west-1 and is owned by Alice.",
        fact_cues=["eu-west-1", "alice"],
        check=_contains("eu-west-1", "alice"),
    ),
    Scenario(
        id="infra-budget-remaining",
        kind="aggregation",
        sessions=[
            "The Q3 infrastructure budget was set at $12,000.",
            "We onboarded two new backend engineers this quarter.",
            "So far we have spent $8,500 of the infrastructure budget on cloud costs.",
        ],
        task="How much of the Q3 infrastructure budget is still unspent?",
        answer="$3,500 of the infrastructure budget is still unspent.",
        fact_cues=["12,000", "8,500"],
        check=_matches(r"\b3,?500\b"),
    ),
    Scenario(
        id="helper-prefix",
        kind="convention",
        sessions=[
            (
                "Naming convention: every internal helper function name is prefixed with "
                "_impl_ and uses snake_case with full type hints on all parameters and the "
                "return value."
            ),
            "We enabled ruff and mypy as required CI gates.",
            "A pre-commit hook now formats code on every commit.",
        ],
        task=(
            "Following our naming convention, write the Python function signature for an "
            "internal helper that validates an email address string and returns a boolean."
        ),
        answer="def _impl_validate_email(email: str) -> bool:",
        fact_cues=["_impl_"],
        check=_matches(
            r"def\s+_impl_[a-z0-9_]*\s*\([^)]*\w+\s*:\s*\w[^)]*\)\s*->",
            ignorecase=False,
        ),
    ),
    Scenario(
        id="config-constant-prefix",
        kind="convention",
        sessions=[
            (
                "Convention: configuration constants are written in UPPER_SNAKE_CASE and "
                "prefixed with CFG_."
            ),
            "Secrets are injected from the environment, never hard-coded.",
            "The settings module is imported once at startup.",
        ],
        task=(
            "Following our convention, write the module-level constant declaration for the "
            "maximum retry count with a value of 5."
        ),
        answer="CFG_MAX_RETRY_COUNT = 5",
        fact_cues=["CFG_"],
        check=_matches(r"\bCFG_[A-Z0-9_]+\s*=\s*5\b", ignorecase=False),
    ),
    Scenario(
        id="secret-store",
        kind="retrieval",
        sessions=[
            "We initialised the monorepo and set up pnpm workspaces for all packages.",
            "Continuous integration was migrated from Jenkins to GitHub Actions for speed.",
            "The mobile client was rewritten in Kotlin during this sprint.",
            "A Grafana dashboard was added to monitor API latency percentiles.",
            (
                "The production database master password is stored in HashiCorp Vault under "
                "the secret path prod-db-cred."
            ),
            "Customer support tooling was moved over to Zendesk.",
            "Feature flags are now managed through LaunchDarkly for gradual rollouts.",
            "The marketing site typography was refreshed by the design team.",
            "Nightly backups now replicate to a second cloud region for durability.",
            "Structured JSON logging was standardised across all services.",
            "The onboarding email flow was rebuilt with a new template engine.",
            "We upgraded the Node.js runtime to the latest LTS across services.",
        ],
        task="Where is the production database master password stored?",
        answer=(
            "The production database master password is stored in HashiCorp Vault under the "
            "path prod-db-cred."
        ),
        fact_cues=["vault", "prod-db-cred"],
        check=_contains("vault"),
    ),
]

SCENARIOS_BY_ID: dict[str, Scenario] = {s.id: s for s in SCENARIOS}
