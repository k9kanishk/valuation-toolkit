from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def slugify(value: str) -> str:
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in value).strip('_')


def cache_key(*parts: Any) -> str:
    payload = '||'.join(str(p) for p in parts)
    return hashlib.md5(payload.encode('utf-8')).hexdigest()


def read_json_cache(path: Path, ttl_hours: int) -> Any | None:
    if not path.exists():
        return None
    max_age = ttl_hours * 3600
    if time.time() - path.stat().st_mtime > max_age:
        return None
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)



def write_json_cache(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)



def pick(record: dict[str, Any], keys: Iterable[str], default: float | str | None = np.nan) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, '', 'None'):
            return record[key]
    return default



def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value in (None, '', 'None'):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default



def safe_div(numerator: Any, denominator: Any, default: float = np.nan) -> float:
    num = safe_float(numerator, default=np.nan)
    den = safe_float(denominator, default=np.nan)
    if np.isnan(num) or np.isnan(den) or abs(den) < 1e-12:
        return default
    return num / den



def winsorize_series(series: pd.Series, lower: float = 0.05, upper: float = 0.95) -> pd.Series:
    clean = series.astype(float).copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    if clean.dropna().empty:
        return clean
    lo = clean.quantile(lower)
    hi = clean.quantile(upper)
    return clean.clip(lower=lo, upper=hi)



def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    aligned = pd.concat([values, weights], axis=1).dropna()
    if aligned.empty:
        return np.nan
    total_weight = aligned.iloc[:, 1].sum()
    if total_weight <= 0:
        return np.nan
    return float((aligned.iloc[:, 0] * aligned.iloc[:, 1]).sum() / total_weight)



def first_valid(*values: Any, default: float | str | None = np.nan) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if value == '':
            continue
        return value
    return default



def as_percent(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return 'n.a.'
    return f'{value * 100:.{decimals}f}%'



def as_multiple(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return 'n.a.'
    return f'{value:.{decimals}f}x'



def as_currency(value: float, decimals: int = 0, symbol: str = '$') -> str:
    if pd.isna(value):
        return 'n.a.'
    return f'{symbol}{value:,.{decimals}f}'
