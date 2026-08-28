from __future__ import annotations

import hashlib
import json
from pathlib import Path

from openfoam_agent.schemas.capability import CapabilityGraphSpec, CapabilityProvider


class CapabilityGraph:
    """Read-only capability evidence graph used as an agent tool."""

    def __init__(self, spec: CapabilityGraphSpec):
        self.spec = spec

    @classmethod
    def from_json(cls, path: str | Path) -> "CapabilityGraph":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(CapabilityGraphSpec.model_validate(payload))

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.spec.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def provider(self, provider_id: str) -> CapabilityProvider:
        for provider in self.spec.providers:
            if provider.id == provider_id:
                return provider
        raise KeyError(f"Unknown capability provider: {provider_id}")

    def providers_for(
        self,
        capability_id: str,
        *,
        include_unverified: bool = False,
    ) -> list[CapabilityProvider]:
        return sorted(
            (
                provider
                for provider in self.spec.providers
                if capability_id in provider.capabilities
                and (include_unverified or provider.verified)
            ),
            key=lambda provider: provider.id,
        )
