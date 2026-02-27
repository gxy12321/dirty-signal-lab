from __future__ import annotations

import pandas as pd
from pathlib import Path


class KdbMock:
    """A tiny kdb+ style mock to demonstrate q-like querying from Python."""

    def __init__(self, tables: dict[str, pd.DataFrame]):
        self.tables = tables

    @classmethod
    def from_csv(cls, path: str | Path, table: str = "ticks") -> "KdbMock":
        df = pd.read_csv(path, parse_dates=["ts"])
        return cls({table: df})

    def select(
        self,
        table: str,
        sym: str | None = None,
        start: str | None = None,
        end: str | None = None,
        cols: list[str] | None = None,
    ) -> pd.DataFrame:
        df = self.tables[table]
        if sym is not None:
            df = df[df["sym"] == sym]
        if start is not None:
            df = df[df["ts"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["ts"] <= pd.to_datetime(end)]
        if cols is not None:
            df = df[cols]
        return df.copy()

    def schema(self) -> dict:
        return {name: df.dtypes.astype(str).to_dict() for name, df in self.tables.items()}
