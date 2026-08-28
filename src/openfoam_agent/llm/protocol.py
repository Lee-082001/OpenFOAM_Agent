from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    def generate(
        self,
        schema: type[T],
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> T:
        ...
