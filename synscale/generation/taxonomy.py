"""Topic taxonomy: domains → subdomains → leaf topics (spec §1.4).

YAML layout::

    version: taxonomy-example-v0
    science_domains: [physical_sciences, life_sciences]
    domains:
      physical_sciences:
        physics: ["Newton's laws of motion", "thermodynamics", ...]
        chemistry: [...]

Every templated / seeded / writer prompt records ``(domain, subdomain, leaf)`` and per-leaf caps
(≤ 0.2 % of the pool by default) prevent topic collapse.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import yaml


@dataclass(frozen=True)
class Leaf:
    domain: str
    subdomain: str
    name: str

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.subdomain}/{self.name}"

    @property
    def subcategory(self) -> str:
        """``domain.subdomain`` as stored in the pool record."""
        return f"{self.domain}.{self.subdomain}"

    @property
    def domain_h(self) -> str:
        return self.domain.replace("_", " ")

    @property
    def subdomain_h(self) -> str:
        return self.subdomain.replace("_", " ")


class Taxonomy:
    """Immutable list of leaves with lookup helpers and a content hash."""

    def __init__(self, leaves: Iterable[Leaf], version: str = "taxonomy-v0",
                 science_domains: Optional[Iterable[str]] = None):
        self.leaves: tuple[Leaf, ...] = tuple(leaves)
        if not self.leaves:
            raise ValueError("taxonomy has no leaves")
        self.version = version
        self._by_domain: dict[str, list[Leaf]] = {}
        self._by_subdomain: dict[tuple[str, str], list[Leaf]] = {}
        for lf in self.leaves:
            self._by_domain.setdefault(lf.domain, []).append(lf)
            self._by_subdomain.setdefault((lf.domain, lf.subdomain), []).append(lf)
        sd = list(science_domains) if science_domains else list(self._by_domain)
        self.science_domains: tuple[str, ...] = tuple(d for d in sd if d in self._by_domain)
        payload = "\n".join(lf.key for lf in self.leaves).encode("utf-8")
        self.sha256 = hashlib.sha256(payload).hexdigest()

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping) -> "Taxonomy":
        leaves: list[Leaf] = []
        for domain, subs in (data.get("domains") or {}).items():
            for sub, names in (subs or {}).items():
                for name in names or []:
                    leaves.append(Leaf(str(domain), str(sub), str(name)))
        return cls(leaves, version=str(data.get("version", "taxonomy-v0")),
                   science_domains=data.get("science_domains"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Taxonomy":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    # ---- lookups -----------------------------------------------------------------------
    @property
    def domains(self) -> list[str]:
        return list(self._by_domain)

    def leaves_in_domain(self, domain: str) -> list[Leaf]:
        return list(self._by_domain.get(domain, []))

    def leaves_in_domains(self, domains: Iterable[str]) -> list[Leaf]:
        out: list[Leaf] = []
        for d in domains:
            out.extend(self._by_domain.get(d, []))
        return out

    def siblings(self, leaf: Leaf) -> list[Leaf]:
        return [lf for lf in self._by_subdomain[(leaf.domain, leaf.subdomain)] if lf != leaf]

    def science_leaves(self) -> list[Leaf]:
        return self.leaves_in_domains(self.science_domains)

    def __len__(self) -> int:
        return len(self.leaves)


def leaf_entropy(counts: Mapping[str, int]) -> dict[str, float]:
    """Shannon entropy (bits) of a leaf-count distribution and its normalised form."""
    total = sum(counts.values())
    if total == 0:
        return {"bits": 0.0, "normalised": 0.0, "n_leaves_used": 0}
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    n = sum(1 for c in counts.values() if c > 0)
    return {"bits": h, "normalised": (h / math.log2(n)) if n > 1 else 0.0, "n_leaves_used": n}


def leaf_counts(leaf_keys: Iterable[Optional[str]]) -> Counter[str]:
    return Counter(k for k in leaf_keys if k)
