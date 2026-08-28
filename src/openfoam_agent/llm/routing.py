from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .protocol import StructuredLLM


@dataclass(frozen=True)
class WorkflowLLMs:
    """Role-specific structured LLMs used by one CFD workflow.

    A single StructuredLLM can still be supplied to the public workflow API;
    :meth:`coerce` expands it to all roles for backward compatibility.
    """

    intake: StructuredLLM
    engineering: StructuredLLM
    postprocessing: StructuredLLM
    review: StructuredLLM

    @classmethod
    def uniform(cls, llm: StructuredLLM) -> "WorkflowLLMs":
        return cls(
            intake=llm,
            engineering=llm,
            postprocessing=llm,
            review=llm,
        )

    @classmethod
    def coerce(cls, value: StructuredLLM | "WorkflowLLMs") -> "WorkflowLLMs":
        return value if isinstance(value, cls) else cls.uniform(value)

    def model_names(self) -> dict[str, str | None]:
        def model_name(llm: Any) -> str | None:
            value = getattr(llm, "model", None)
            return value if isinstance(value, str) and value.strip() else None

        return {
            "intake": model_name(self.intake),
            "engineering": model_name(self.engineering),
            "postprocessing": model_name(self.postprocessing),
            "review": model_name(self.review),
        }
