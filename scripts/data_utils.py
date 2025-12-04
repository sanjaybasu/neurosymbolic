"""Utility helpers for loading CSVs with unescaped newlines/quotes."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def parse_multiline_csv(path: Path, record_prefixes: Iterable[str]) -> pd.DataFrame:
    """
    Load CSV files where the `context_text` column contains unescaped newlines/quotes.

    The files in `prospective_eval/` mark each row with a prefix (e.g., "harm_candidate").
    We treat any line beginning with one of these prefixes as the start of a new record and
    join all subsequent lines until the next prefix.
    """
    prefixes = tuple(record_prefixes)
    lines: List[str] = Path(path).read_text().splitlines()
    if not lines:
        raise ValueError(f"{path} is empty.")

    header = lines[0]
    records: List[str] = []
    current: List[str] = []

    for line in lines[1:]:
        if line.startswith(prefixes):
            if current:
                records.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        records.append("\n".join(current))

    rows = [
        next(csv.DictReader(io.StringIO(f"{header}\n{record}")))
        for record in records
        if record.strip()
    ]
    return pd.DataFrame(rows)
