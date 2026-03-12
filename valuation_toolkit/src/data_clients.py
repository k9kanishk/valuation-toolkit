from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests import HTTPError

from .config import CACHE_DIR, settings
from .utils import cache_key, read_json_cache, safe_float, write_json_cache

logger = logging.getLogger(__name__)


class PaidPlanRequiredError(RuntimeError):
    pass


class RateLimitExceededError(RuntimeError):
    pass


class HTTPClient:
    def __init__(self, cache_subdir: str, ttl_hours: int | None = None):
        self.cache_dir = CACHE_DIR / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours or settings.cache_ttl_hours

    def _cache_path(self, prefix: str, url: str, params: dict[str, Any] | None) -> Path:
        key = cache_key(prefix, url, sorted((params or {}).items()))
        return self.cache_dir / f"{key}.json"

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
        prefix: str = "request",
        force_refresh: bool = False,
    ) -> Any:
        cache_path = self._cache_path(prefix, url, params)
        if not force_refresh:
            cached = read_json_cache(cache_path, ttl_hours=self.ttl_hours)
            if cached is not None:
                return cached

        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
        except HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 402:
                raise PaidPlanRequiredError(
                    f"402 Payment Required for {url}. Endpoint not available on current plan."
                ) from exc
            if status == 429:
                raise RateLimitExceededError(
                    f"429 Too Many Requests for {url}. Daily quota/rate limit exceeded."
                ) from exc
            raise

        payload = response.json()
        write_json_cache(cache_path, payload)
        time.sleep(0.15)
        return payload


class FMPClient(HTTPClient):
    def __init__(self):
        super().__init__(cache_subdir="fmp")
        if not settings.fmp_api_key:
            raise ValueError("Missing FMP_API_KEY in environment.")
        self.base_url = settings.fmp_base_url
        self.disabled = False

    def _get(self, endpoint: str, **params: Any) -> Any:
        merged = {k: v for k, v in params.items() if v is not None}
        merged["apikey"] = settings.fmp_api_key
        return self.get_json(
            url=f"{self.base_url}/{endpoint}",
            params=merged,
            prefix=f"fmp_{endpoint}",
        )

    def _get_optional(self, endpoint: str, default: Any, **params: Any) -> Any:
        if self.disabled:
            return default

        try:
            return self._get(endpoint, **params)
        except PaidPlanRequiredError as exc:
            logger.warning("Skipping paid/gated FMP endpoint %s: %s", endpoint, exc)
            return default
        except RateLimitExceededError as exc:
            logger.warning("FMP rate limit hit. Disabling FMP for this run: %s", exc)
            self.disabled = True
            return default
        except Exception as exc:
            logger.warning("FMP endpoint failed %s: %s", endpoint, exc)
            return default

    def profile(self, symbol: str) -> dict[str, Any]:
        payload = self._get_optional("profile", [], symbol=symbol.upper())
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
        limit: int = 200,
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

    def quote(self, symbol: str) -> dict[str, Any]:
        payload = self._get_optional("quote", [], symbol=symbol.upper())
        return payload[0] if payload else {}

    def enterprise_values(self, symbol: str, limit: int = 8) -> list[dict[str, Any]]:
        return self._get_optional("enterprise-values", [], symbol=symbol.upper(), limit=limit)

    def income_statement(self, symbol: str, period: str = "annual", limit: int = 8) -> list[dict[str, Any]]:
        return self._get_optional("income-statement", [], symbol=symbol.upper(), period=period, limit=limit)

    def balance_sheet(self, symbol: str, period: str = "annual", limit: int = 8) -> list[dict[str, Any]]:
        return self._get_optional("balance-sheet-statement", [], symbol=symbol.upper(), period=period, limit=limit)

    def cash_flow(self, symbol: str, period: str = "annual", limit: int = 8) -> list[dict[str, Any]]:
        return self._get_optional("cash-flow-statement", [], symbol=symbol.upper(), period=period, limit=limit)

    def ratios_ttm(self, symbol: str) -> dict[str, Any]:
        payload = self._get_optional("ratios-ttm", [], symbol=symbol.upper())
        return payload[0] if payload else {}

    def key_metrics_ttm(self, symbol: str) -> dict[str, Any]:
        payload = self._get_optional("key-metrics-ttm", [], symbol=symbol.upper())
        return payload[0] if payload else {}


class SECClient(HTTPClient):
    def __init__(self):
        super().__init__(cache_subdir="sec", ttl_hours=24)

        # Do NOT hardcode Host. Let requests set it from the URL.
        self.headers = {
            "User-Agent": settings.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def ticker_map(self) -> pd.DataFrame:
        # Official current path per SEC docs
        primary_url = "https://www.sec.gov/files/company_tickers_exchange.json"
        fallback_url = "https://www.sec.gov/files/company_tickers.json"

        try:
            payload = self.get_json(primary_url, headers=self.headers, prefix="sec_ticker_map_exchange")
            columns = payload.get("fields", [])
            rows = payload.get("data", [])
            df = pd.DataFrame(rows, columns=columns)

            # normalize expected columns
            rename_map = {}
            if "cik" in df.columns and "cik_str" not in df.columns:
                rename_map["cik"] = "cik"
            if rename_map:
                df = df.rename(columns=rename_map)

            if "ticker" not in df.columns:
                raise ValueError("SEC exchange ticker file did not contain expected 'ticker' column.")
            return df

        except Exception:
            # Fallback to older/simple ticker mapping structure
            payload = self.get_json(fallback_url, headers=self.headers, prefix="sec_ticker_map_basic")

            # company_tickers.json is usually keyed dict: {"0": {...}, "1": {...}}
            if isinstance(payload, dict):
                rows = list(payload.values())
            elif isinstance(payload, list):
                rows = payload
            else:
                raise ValueError("Unexpected SEC ticker file format.")

            df = pd.DataFrame(rows)

            # normalize columns to what lookup_ticker expects
            if "cik_str" not in df.columns and "cik" in df.columns:
                df["cik_str"] = df["cik"].astype(str).str.zfill(10)
            elif "cik_str" in df.columns:
                df["cik_str"] = df["cik_str"].astype(str).str.zfill(10)

            if "ticker" in df.columns:
                df["ticker"] = df["ticker"].astype(str)
            else:
                raise ValueError("SEC basic ticker file missing 'ticker' column.")

            if "title" in df.columns and "name" not in df.columns:
                df["name"] = df["title"]

            if "exchange" not in df.columns:
                df["exchange"] = None

            return df

    def lookup_ticker(self, symbol: str) -> dict[str, Any] | None:
        df = self.ticker_map()
        matched = df[df["ticker"].astype(str).str.upper() == symbol.upper()]
        if matched.empty:
            return None

        row = matched.iloc[0].to_dict()

        # normalize cik field
        cik_val = row.get("cik_str") or row.get("cik")
        row["cik_str"] = str(cik_val).zfill(10) if cik_val is not None else None
        return row

    def company_facts(self, cik: str) -> dict[str, Any]:
        cik_padded = str(cik).zfill(10)
        url = f"{settings.sec_base_url}/api/xbrl/companyfacts/CIK{cik_padded}.json"
        return self.get_json(url, headers=self.headers, prefix=f"sec_companyfacts_{cik_padded}")


class TreasuryClient(HTTPClient):
    def __init__(self):
        super().__init__(cache_subdir="treasury", ttl_hours=24)

    def current_risk_free_rate(self, tenor: str = "10 yr") -> float:
        archive_urls = [
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives/par-yield-curve-rates-2020-2023.csv",
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rate-archives/par-yield-curve-rates-1990-2023.csv",
        ]

        tenor_map = {
            "1 mo": "1 Mo",
            "2 mo": "2 Mo",
            "3 mo": "3 Mo",
            "4 mo": "4 Mo",
            "6 mo": "6 Mo",
            "1 yr": "1 Yr",
            "2 yr": "2 Yr",
            "3 yr": "3 Yr",
            "5 yr": "5 Yr",
            "7 yr": "7 Yr",
            "10 yr": "10 Yr",
            "20 yr": "20 Yr",
            "30 yr": "30 Yr",
        }

        col = tenor_map.get(tenor, "10 Yr")

        for url in archive_urls:
            try:
                response = requests.get(url, timeout=20)
                response.raise_for_status()

                from io import StringIO

                df = pd.read_csv(StringIO(response.text))

                if "Date" not in df.columns or col not in df.columns:
                    continue

                df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=[col])

                if not df.empty:
                    latest = df.iloc[-1]
                    return float(latest[col]) / 100.0
            except Exception:
                continue

        raise ValueError("Unable to retrieve Treasury yield curve data.")


class YahooClient:
    def __init__(self):
        import yfinance as yf

        self.yf = yf

    def _ticker(self, symbol: str):
        return self.yf.Ticker(symbol.upper())

    def quote_summary(self, symbol: str) -> dict[str, Any]:
        ticker = self._ticker(symbol)
        info = ticker.info or {}
        fast = getattr(ticker, "fast_info", {}) or {}

        def fast_get(name: str, default: Any = None):
            try:
                return fast.get(name, default)
            except Exception:
                return default

        return {
            "symbol": symbol.upper(),
            "companyName": info.get("shortName") or info.get("longName") or symbol.upper(),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
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

    def quarterly_income_stmt(self, symbol: str):
        ticker = self._ticker(symbol)
        try:
            return ticker.quarterly_income_stmt
        except Exception:
            return None

    def annual_income_stmt(self, symbol: str):
        ticker = self._ticker(symbol)
        try:
            return ticker.income_stmt
        except Exception:
            return None

    def quarterly_balance_sheet(self, symbol: str):
        ticker = self._ticker(symbol)
        try:
            return ticker.quarterly_balance_sheet
        except Exception:
            return None

    def annual_balance_sheet(self, symbol: str):
        ticker = self._ticker(symbol)
        try:
            return ticker.balance_sheet
        except Exception:
            return None

    def quarterly_cashflow(self, symbol: str):
        ticker = self._ticker(symbol)
        try:
            return ticker.quarterly_cashflow
        except Exception:
            return None

    def annual_cashflow(self, symbol: str):
        ticker = self._ticker(symbol)
        try:
            return ticker.cashflow
        except Exception:
            return None
