from __future__ import annotations

import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf
from requests import HTTPError

from .config import CACHE_DIR, settings

logger = logging.getLogger(__name__)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def _cache_key(prefix: str, url: str, params: dict[str, Any] | None = None) -> str:
    payload = json.dumps(
        {"prefix": prefix, "url": url, "params": params or {}},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _read_json_cache(path: Path, ttl_hours: int) -> Any | None:
    if not path.exists():
        return None
    age_seconds = time.time() - path.stat().st_mtime
    if age_seconds > ttl_hours * 3600:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json_cache(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")


def _read_text_cache(path: Path, ttl_hours: int) -> str | None:
    if not path.exists():
        return None
    age_seconds = time.time() - path.stat().st_mtime
    if age_seconds > ttl_hours * 3600:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _write_text_cache(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


class HTTPClient:
    def __init__(self, cache_subdir: str, ttl_hours: int | None = None):
        self.cache_dir = CACHE_DIR / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours or settings.cache_ttl_hours

    def _cache_path(self, prefix: str, url: str, params: dict[str, Any] | None, suffix: str) -> Path:
        return self.cache_dir / f"{_cache_key(prefix, url, params)}.{suffix}"

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        prefix: str = "request",
        timeout: int = 30,
        force_refresh: bool = False,
    ) -> Any:
        cache_path = self._cache_path(prefix, url, params, "json")

        if not force_refresh:
            cached = _read_json_cache(cache_path, self.ttl_hours)
            if cached is not None:
                return cached

        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            _write_json_cache(cache_path, payload)
            time.sleep(0.10)
            return payload
        except Exception:
            cached = _read_json_cache(cache_path, ttl_hours=10_000)
            if cached is not None:
                return cached
            raise

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        prefix: str = "request",
        timeout: int = 30,
        force_refresh: bool = False,
    ) -> str:
        cache_path = self._cache_path(prefix, url, params, "txt")

        if not force_refresh:
            cached = _read_text_cache(cache_path, self.ttl_hours)
            if cached is not None:
                return cached

        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            payload = response.text
            _write_text_cache(cache_path, payload)
            time.sleep(0.10)
            return payload
        except Exception:
            cached = _read_text_cache(cache_path, ttl_hours=10_000)
            if cached is not None:
                return cached
            raise


class SECClient(HTTPClient):
    def __init__(self):
        super().__init__(cache_subdir="sec", ttl_hours=24)
        self.headers = {
            "User-Agent": settings.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def ticker_map(self) -> pd.DataFrame:
        payload = self.get_json(
            settings.sec_ticker_url,
            headers=self.headers,
            prefix="sec_ticker_map",
        )

        if isinstance(payload, dict):
            rows = list(payload.values())
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError("Unexpected SEC ticker map format.")

        df = pd.DataFrame(rows)
        if "title" in df.columns and "name" not in df.columns:
            df["name"] = df["title"]
        if "cik_str" in df.columns:
            df["cik_str"] = df["cik_str"].astype(str).str.zfill(10)
        elif "cik" in df.columns:
            df["cik_str"] = df["cik"].astype(str).str.zfill(10)

        if "ticker" not in df.columns:
            raise ValueError("SEC ticker map missing ticker column.")

        df["ticker"] = df["ticker"].astype(str).str.upper()
        return df

    def lookup_ticker(self, symbol: str) -> dict[str, Any] | None:
        df = self.ticker_map()
        matched = df[df["ticker"] == symbol.upper()]
        if matched.empty:
            return None
        return matched.iloc[0].to_dict()

    def company_facts(self, cik: str) -> dict[str, Any]:
        cik_padded = str(cik).zfill(10)
        url = f"{settings.sec_companyfacts_base}/CIK{cik_padded}.json"
        return self.get_json(url, headers=self.headers, prefix=f"sec_companyfacts_{cik_padded}")


class TreasuryClient(HTTPClient):
    TENOR_MAP = {
        "1 mo": {"bc1month", "bc_1month", "d_bc_1month"},
        "2 mo": {"bc2month", "bc_2month", "d_bc_2month"},
        "3 mo": {"bc3month", "bc_3month", "d_bc_3month"},
        "4 mo": {"bc4month", "bc_4month", "d_bc_4month"},
        "6 mo": {"bc6month", "bc_6month", "d_bc_6month"},
        "1 yr": {"bc1year", "bc_1year", "d_bc_1year"},
        "2 yr": {"bc2year", "bc_2year", "d_bc_2year"},
        "3 yr": {"bc3year", "bc_3year", "d_bc_3year"},
        "5 yr": {"bc5year", "bc_5year", "d_bc_5year"},
        "7 yr": {"bc7year", "bc_7year", "d_bc_7year"},
        "10 yr": {"bc10year", "bc_10year", "d_bc_10year"},
        "20 yr": {"bc20year", "bc_20year", "d_bc_20year"},
        "30 yr": {"bc30year", "bc_30year", "d_bc_30year"},
    }

    def __init__(self):
        super().__init__(cache_subdir="treasury", ttl_hours=12)

    @staticmethod
    def _norm(tag: str) -> str:
        tag = tag.split("}")[-1]
        return "".join(ch for ch in tag.lower() if ch.isalnum() or ch == "_")

    def current_risk_free_rate(self, tenor: str = "10 yr") -> float:
        xml_text = self.get_text(
            settings.treasury_yield_xml_url,
            prefix="treasury_yield_xml",
        )

        root = ET.fromstring(xml_text)
        rows: list[dict[str, Any]] = []

        for node in root.iter():
            if self._norm(node.tag).endswith("properties"):
                row: dict[str, Any] = {}
                for child in list(node):
                    row[self._norm(child.tag)] = child.text
                if row:
                    rows.append(row)

        if not rows:
            raise ValueError("Treasury XML feed returned no rows.")

        date_candidates = {"new_date", "newdate", "date"}
        value_candidates = self.TENOR_MAP.get(tenor.lower(), self.TENOR_MAP["10 yr"])

        parsed: list[tuple[pd.Timestamp, float]] = []
        for row in rows:
            date_val = None
            for k in date_candidates:
                if k in row and row[k]:
                    date_val = row[k]
                    break

            rate_val = None
            for k in value_candidates:
                if k in row and row[k] not in (None, ""):
                    rate_val = row[k]
                    break

            if date_val is None or rate_val is None:
                continue

            dt = pd.to_datetime(date_val, errors="coerce")
            rv = pd.to_numeric(rate_val, errors="coerce")

            if pd.notna(dt) and pd.notna(rv):
                parsed.append((dt, float(rv) / 100.0))

        if not parsed:
            raise ValueError(f"No Treasury values found for tenor={tenor}")

        parsed.sort(key=lambda x: x[0])
        return parsed[-1][1]


class YahooClient:
    def __init__(self):
        self._cache: dict[str, Any] = {}

    def _ticker(self, symbol: str):
        symbol = symbol.upper().strip()
        if symbol not in self._cache:
            self._cache[symbol] = yf.Ticker(symbol)
        return self._cache[symbol]

    def quote_summary(self, symbol: str) -> dict[str, Any]:
        t = self._ticker(symbol)
        info = t.info or {}
        fast = getattr(t, "fast_info", {}) or {}

        def fast_get(name: str, default: Any = None) -> Any:
            try:
                return fast.get(name, default)
            except Exception:
                return default

        return {
            "symbol": symbol.upper(),
            "name": info.get("shortName") or info.get("longName") or symbol.upper(),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "currency": info.get("financialCurrency") or info.get("currency") or "USD",
            "price": info.get("currentPrice") or fast_get("lastPrice"),
            "marketCap": info.get("marketCap") or fast_get("marketCap"),
            "sharesOutstanding": info.get("sharesOutstanding") or fast_get("shares"),
            "enterpriseValue": info.get("enterpriseValue"),
            "beta": info.get("beta"),
            "totalRevenue": info.get("totalRevenue"),
            "ebitda": info.get("ebitda"),
            "netIncome": info.get("netIncomeToCommon"),
            "totalCash": info.get("totalCash"),
            "totalDebt": info.get("totalDebt"),
        }

    def quarterly_income_stmt(self, symbol: str) -> pd.DataFrame | None:
        try:
            return self._ticker(symbol).quarterly_income_stmt
        except Exception:
            return None

    def annual_income_stmt(self, symbol: str) -> pd.DataFrame | None:
        try:
            return self._ticker(symbol).income_stmt
        except Exception:
            return None

    def quarterly_balance_sheet(self, symbol: str) -> pd.DataFrame | None:
        try:
            return self._ticker(symbol).quarterly_balance_sheet
        except Exception:
            return None

    def annual_balance_sheet(self, symbol: str) -> pd.DataFrame | None:
        try:
            return self._ticker(symbol).balance_sheet
        except Exception:
            return None

    def quarterly_cashflow(self, symbol: str) -> pd.DataFrame | None:
        try:
            return self._ticker(symbol).quarterly_cashflow
        except Exception:
            return None

    def annual_cashflow(self, symbol: str) -> pd.DataFrame | None:
        try:
            return self._ticker(symbol).cashflow
        except Exception:
            return None


class NullOptionalClient:
    disabled = True

    def profile(self, symbol: str) -> dict[str, Any]:
        return {}

    def quote(self, symbol: str) -> dict[str, Any]:
        return {}

    def enterprise_values(self, symbol: str, limit: int = 1) -> list[dict[str, Any]]:
        return []

    def ratios_ttm(self, symbol: str) -> dict[str, Any]:
        return {}

    def key_metrics_ttm(self, symbol: str) -> dict[str, Any]:
        return {}

    def stock_peers(self, symbol: str) -> list[str]:
        return []

    def screener(
        self,
        sector: str | None = None,
        market_cap_min: float | None = None,
        market_cap_max: float | None = None,
        country: str = "US",
        exchange: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return []


class FMPOptionalClient(HTTPClient):
    def __init__(self):
        super().__init__(cache_subdir="fmp_optional", ttl_hours=12)
        self.base_url = settings.fmp_base_url
        self.api_key = settings.fmp_api_key
        self.disabled = False

    def _get_optional(self, endpoint: str, default: Any, **params: Any) -> Any:
        if self.disabled or not self.api_key:
            return default

        query = {k: v for k, v in params.items() if v is not None}
        query["apikey"] = self.api_key

        try:
            return self.get_json(
                f"{self.base_url}/{endpoint}",
                params=query,
                prefix=f"fmp_{endpoint}",
            )
        except HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in (402, 429):
                logger.warning("Disabling optional FMP provider due to status=%s", status)
                self.disabled = True
                return default
            return default
        except Exception:
            return default

    def profile(self, symbol: str) -> dict[str, Any]:
        payload = self._get_optional("profile", [], symbol=symbol.upper())
        return payload[0] if payload else {}

    def quote(self, symbol: str) -> dict[str, Any]:
        payload = self._get_optional("quote", [], symbol=symbol.upper())
        return payload[0] if payload else {}

    def enterprise_values(self, symbol: str, limit: int = 1) -> list[dict[str, Any]]:
        return self._get_optional("enterprise-values", [], symbol=symbol.upper(), limit=limit)

    def ratios_ttm(self, symbol: str) -> dict[str, Any]:
        payload = self._get_optional("ratios-ttm", [], symbol=symbol.upper())
        return payload[0] if payload else {}

    def key_metrics_ttm(self, symbol: str) -> dict[str, Any]:
        payload = self._get_optional("key-metrics-ttm", [], symbol=symbol.upper())
        return payload[0] if payload else {}

    def stock_peers(self, symbol: str) -> list[str]:
        payload = self._get_optional("stock-peers", [], symbol=symbol.upper())
        if not payload:
            return []
        row = payload[0]
        peers = row.get("peersList") or row.get("peers") or []
        return [p for p in peers if isinstance(p, str)]

    def screener(
        self,
        sector: str | None = None,
        market_cap_min: float | None = None,
        market_cap_max: float | None = None,
        country: str = "US",
        exchange: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return self._get_optional(
            "company-screener",
            [],
            sector=sector,
            marketCapMoreThan=int(market_cap_min) if market_cap_min else None,
            marketCapLowerThan=int(market_cap_max) if market_cap_max else None,
            country=country,
            exchange=exchange,
            isEtf=False,
            isFund=False,
            limit=limit,
        )


def build_optional_client(enabled: bool | None = None):
    use_it = settings.use_optional_provider if enabled is None else enabled
    if not use_it:
        return NullOptionalClient()

    if settings.optional_provider == "fmp" and settings.fmp_api_key:
        return FMPOptionalClient()

    return NullOptionalClient()
