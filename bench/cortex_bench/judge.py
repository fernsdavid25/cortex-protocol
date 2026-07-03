"""LongMemEval QA judge.

The per-type prompt templates are reproduced VERBATIM from the official LongMemEval
evaluator (`src/evaluation/evaluate_qa.py`, Wu et al., LongMemEval, ICLR 2025) so our
scores stay comparable with published numbers. Backends:

- "offline": deterministic, no-API containment/abstention check — used in tests + cheap
  smoke runs (NOT for headline numbers).
- "gemini":  cheap iteration judge (validate correlation vs gpt-4o before trusting).
- "openai":  the official `gpt-4o-2024-08-06` headline judge (the one sanctioned non-GCP spend).
"""

from __future__ import annotations

from collections.abc import Sequence

from .memory_system import QAInstance

_STANDARD = (
    "I will give you a question, a correct answer, and a response from a model. Please "
    "answer yes if the response contains the correct answer. Otherwise, answer no. If the "
    "response is equivalent to the correct answer or contains all the intermediate steps to "
    "get the correct answer, you should also answer yes. If the response only contains a "
    "subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect "
    "Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
)
_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. Please answer "
    "yes if the response contains the correct answer. Otherwise, answer no. If the response is "
    "equivalent to the correct answer or contains all the intermediate steps to get the correct "
    "answer, you should also answer yes. If the response only contains a subset of the "
    "information required by the answer, answer no. In addition, do not penalize off-by-one "
    "errors for the number of days. If the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is "
    "18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel "
    "Response: {}\n\nIs the model response correct? Answer yes or no only."
)
_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. Please answer "
    "yes if the response contains the correct answer. Otherwise, answer no. If the response "
    "contains some previous information along with an updated answer, the response should be "
    "considered as correct as long as the updated answer is the required answer.\n\nQuestion: "
    "{}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer "
    "yes or no only."
)
_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. Otherwise, "
    "answer no. The model does not need to reflect all the points in the rubric. The response "
    "is correct as long as it recalls and utilizes the user's personal information correctly."
    "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? "
    "Answer yes or no only."
)
_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. The "
    "model could say that the information is incomplete, or some other information is given but "
    "the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer yes or no only."
)

_ABSTENTION_HINTS = (
    "don't know",
    "do not know",
    "cannot",
    "can't",
    "no information",
    "not available",
    "unanswerable",
    "not sure",
    "unable to",
    "no record",
)


def get_anscheck_prompt(
    task: str, question: str, answer: str, response: str, abstention: bool = False
) -> str:
    """Build the judge prompt with the correct per-type template (verbatim upstream)."""
    if abstention:
        return _ABSTENTION.format(question, answer, response)
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return _STANDARD.format(question, answer, response)
    if task == "temporal-reasoning":
        return _TEMPORAL.format(question, answer, response)
    if task == "knowledge-update":
        return _KNOWLEDGE_UPDATE.format(question, answer, response)
    if task == "single-session-preference":
        return _PREFERENCE.format(question, answer, response)
    # LoCoMo categories (adversarial cat-5 is graded via the abstention branch above):
    if task == "locomo-temporal":
        return _TEMPORAL.format(question, answer, response)
    if task.startswith("locomo-"):  # multihop / singlehop / opendomain / adversarial
        return _STANDARD.format(question, answer, response)
    raise ValueError(f"unknown question_type: {task!r}")


def offline_label(instance: QAInstance, hypothesis: str) -> bool:
    """Deterministic no-API judge for tests/smoke runs. Not for headline numbers."""
    h = hypothesis.lower()
    if instance.is_abstention:
        return any(hint in h for hint in _ABSTENTION_HINTS)
    return instance.answer.strip().lower() in h


def majority_vote(labels: Sequence[bool]) -> bool:
    """Majority label over a non-empty list of yes/no votes (ties -> True).

    Used for Gemini self-consistency: the judge is non-deterministic even at temperature 0, so
    calling it an odd number of times and taking the majority dampens run-to-run flip-flop.
    """
    if not labels:
        raise ValueError("majority_vote needs at least one label")
    yes = sum(1 for x in labels if x)
    return yes * 2 >= len(labels)


class Judge:
    """Grades a hypothesis against an instance, returning True if correct.

    ``votes`` only affects the non-deterministic ``gemini`` backend: when ``votes > 1`` the judge
    is queried ``votes`` times and the MAJORITY label is returned (self-consistency). ``offline``
    is deterministic and ``openai`` is the sanctioned headline judge, so both ignore ``votes``.
    """

    def __init__(self, backend: str = "offline", model: str | None = None, votes: int = 1) -> None:
        if votes < 1:
            raise ValueError(f"votes must be >= 1, got {votes}")
        self.backend = backend
        self.votes = votes
        self.model = model or {
            "openai": "gpt-4o-2024-08-06",
            "gemini": "gemini-3.5-flash",  # GCP-native judge (per user; stronger than gpt-4o)
        }.get(backend)

    def grade(self, instance: QAInstance, hypothesis: str) -> bool:
        if self.backend == "offline":
            return offline_label(instance, hypothesis)
        prompt = get_anscheck_prompt(
            instance.question_type,
            instance.question,
            instance.answer,
            hypothesis,
            abstention=instance.is_abstention,
        )
        model = self.model
        if model is None:
            raise ValueError(f"no model configured for backend {self.backend!r}")
        if self.backend == "gemini":
            if self.votes > 1:
                return majority_vote([self._gemini_grade(prompt, model) for _ in range(self.votes)])
            return self._gemini_grade(prompt, model)
        if self.backend == "openai":
            return _openai_yes(prompt, model)
        raise ValueError(f"unknown judge backend: {self.backend!r}")

    def _gemini_grade(self, prompt: str, model: str) -> bool:
        """One Gemini judge call. Overridable in tests to avoid live API calls."""
        return _gemini_yes(prompt, model)


def _gemini_yes(prompt: str, model: str) -> bool:
    import os

    from google import genai  # lazy: tests/offline runs don't need the SDK
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=10,
            # Gemini 3.x "thinking" consumes the tiny 10-token budget and emits empty text,
            # making every grade read as "no". Disable it so the judge returns a bare yes/no.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return "yes" in (resp.text or "").lower()


def _openai_yes(prompt: str, model: str) -> bool:
    import os

    from openai import OpenAI  # lazy

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        n=1,
        temperature=0,
        max_tokens=10,
    )
    return "yes" in (resp.choices[0].message.content or "").lower()
