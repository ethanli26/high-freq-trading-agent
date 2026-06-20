"""Factor interface + registry: vet a signal's predictive power before building it.

A Factor turns aligned price/volume panels into a date x symbol panel of factor
values. The whole point of this layer is to measure WHETHER a factor predicts returns
(IC, quantile spread) cheaply, BEFORE any strategy is built on it.

POINT-IN-TIME CONTRACT (every factor must honor):
  The factor value on date ``t`` uses ONLY data up to and including ``t`` — no future
  bar, no future revision. Price/volume factors satisfy this with backward-looking
  rolling/shift operations. FUNDAMENTAL factors must set ``point_in_time_provider =
  True`` and be fed as-reported values dated to their FILING date; the current free
  data source cannot honor that, so fundamental factors are declared but not run.

Read-only research: no IBKR, no orders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class FactorData:
    """Aligned ``date x symbol`` panels of price/volume data, plus optional fundamentals.

    ``market`` is the benchmark (SPY) close as a date-indexed Series aligned to the
    panels — supplied for factors that need a market reference (beta, idio vol).

    ``fundamentals`` is an optional ``{field: date x symbol DataFrame}`` of POINT-IN-TIME
    fundamentals. Each panel is built by placing a filing's value on its FILING date
    (datekey) and forward-filling, so ``fundamentals[field].loc[t]`` reflects only
    statements filed on or before ``t`` (no look-ahead). It is ``None`` on the free data
    path, where fundamental factors are deferred (see ``point_in_time_provider``).
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    market: pd.Series | None = None
    fundamentals: dict[str, pd.DataFrame] | None = None


class Factor(ABC):
    """A cross-sectional factor: ``compute`` returns date x symbol factor values."""

    name: str
    category: str  # "price" | "volume" | "fundamental"
    requires: tuple[str, ...]
    point_in_time_provider: bool = False  # True if it needs PIT fundamentals (paid)

    @abstractmethod
    def compute(self, data: FactorData) -> pd.DataFrame:
        """Return date x symbol factor values; value[t] uses only data <= t (no look-ahead)."""


_REGISTRY: dict[str, type] = {}


def register(factor_cls: type) -> type:
    """Class decorator: register a Factor subclass under its ``name``."""
    name = getattr(factor_cls, "name", None)
    if not name:
        raise ValueError(f"{factor_cls.__name__} must define a non-empty 'name'.")
    _REGISTRY[name] = factor_cls
    return factor_cls


def get(name: str) -> type:
    """Return the registered factor class for ``name``."""
    return _REGISTRY[name]


def all_factors() -> dict[str, type]:
    """Return a copy of the registry mapping name -> class."""
    return dict(_REGISTRY)


def names() -> list[str]:
    """List registered factor names (registration order)."""
    return list(_REGISTRY)
