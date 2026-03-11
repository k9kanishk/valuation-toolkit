from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import CACHE_DIR, settings
from .utils import cache_key, read_json_cache, safe_float, write_json_cache


class HTTPClient:
    def __init__(self, cache_subdir: str, ttl_hours: int | None = None):
        self.cache_dir = CACHE_DIR / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_hours = ttl_hours or settings.cache_ttl_hours

    def _cache_path(self, prefix: str, url: str, params: dict[str, Any] | None) -> Path:
        key = cache_key(prefix, url, sorted((params or {}).items()))
        return self.cache_dir / f'{key}.json'

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: int = 30,
        prefix: str = 'request',
        force_refresh: bool = False,
    ) -> Any:
        cache_path = self._cache_path(prefix, url, params)
        if not force_refresh:
            cached = read_json_cache(cache_path, ttl_hours=self.ttl_hours)
            if cached is not None:
                return cached

        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        write_json_cache(cache_path, payload)
        time.sleep(0.15)
        return payload


class FMPClient(HTTPClient):
    def __init__(self):
        super().__init__(cache_subdir='fmp')
        if not settings.fmp_api_key:
            raise ValueError('Missing FMP_API_KEY in environment.')
        self.base_url = settings.fmp_base_url

    def _get(self, endpoint: str, **params: Any) -> Any:
        merged = {k: v for k, v in params.items() if v is not None}
        merged['apikey'] = settings.fmp_api_key
        return self.get_json(
            url=f'{self.base_url}/{endpoint}',
            params=merged,
            prefix=f'fmp_{endpoint}',
        )

    def profile(self, symbol: str) -> dict[str, Any]:
        payload = self._get('profile', symbol=symbol.upper())
        return payload[0] if payload else {}

    def stock_peers(self, symbol: str) -> list[str]:
        payload = self._get('stock-peers', symbol=symbol.upper())
        if not payload:
            return []
        row = payload[0]
        peers = row.get('peersList') or row.get('peers') or []
        return [p for p in peers if isinstance(p, str)]

    def screener(
        self,
        sector: str | None = None,
        market_cap_min: float | None = None,
        market_cap_max: float | None = None,
        country: str = 'US',
        exchange: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        return self._get(
            'company-screener',
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
        payload = self._get('quote', symbol=symbol.upper())
        return payload[0] if payload else {}

    def enterprise_values(self, symbol: str, limit: int = 8) -> list[dict[str, Any]]:
        return self._get('enterprise-values', symbol=symbol.upper(), limit=limit)

    def income_statement(self, symbol: str, period: str = 'annual', limit: int = 8) -> list[dict[str, Any]]:
        return self._get('income-statement', symbol=symbol.upper(), period=period, limit=limit)

    def balance_sheet(self, symbol: str, period: str = 'annual', limit: int = 8) -> list[dict[str, Any]]:
        return self._get('balance-sheet-statement', symbol=symbol.upper(), period=period, limit=limit)

    def cash_flow(self, symbol: str, period: str = 'annual', limit: int = 8) -> list[dict[str, Any]]:
        return self._get('cash-flow-statement', symbol=symbol.upper(), period=period, limit=limit)

    def ratios_ttm(self, symbol: str) -> dict[str, Any]:
        payload = self._get('ratios-ttm', symbol=symbol.upper())
        return payload[0] if payload else {}

    def key_metrics_ttm(self, symbol: str) -> dict[str, Any]:
        payload = self._get('key-metrics-ttm', symbol=symbol.upper())
        return payload[0] if payload else {}


class SECClient(HTTPClient):
    def __init__(self):
        super().__init__(cache_subdir='sec', ttl_hours=24)
        self.headers = {
            'User-Agent': settings.sec_user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov',
        }

    def ticker_map(self) -> pd.DataFrame:
        url = 'https://www.sec.gov/files/company_tickers_exchange.json'
        payload = self.get_json(url, headers=self.headers, prefix='sec_ticker_map')
        columns = payload.get('fields', [])
        rows = payload.get('data', [])
        return pd.DataFrame(rows, columns=columns)

    def lookup_ticker(self, symbol: str) -> dict[str, Any] | None:
        df = self.ticker_map()
        matched = df[df['ticker'].str.upper() == symbol.upper()]
        if matched.empty:
            return None
        row = matched.iloc[0].to_dict()
        row['cik_str'] = str(row.get('cik', '')).zfill(10)
        return row

    def company_facts(self, cik: str) -> dict[str, Any]:
        cik_padded = str(cik).zfill(10)
        url = f'{settings.sec_base_url}/api/xbrl/companyfacts/CIK{cik_padded}.json'
        return self.get_json(url, headers=self.headers, prefix=f'sec_companyfacts_{cik_padded}')


class TreasuryClient(HTTPClient):
    def __init__(self):
        super().__init__(cache_subdir='treasury', ttl_hours=24)

    def current_risk_free_rate(self, tenor: str = '10 yr') -> float:
        year = datetime.utcnow().year
        frames: list[pd.DataFrame] = []
        for candidate_year in (year, year - 1):
            tables = pd.read_html(settings.treasury_url.format(year=candidate_year))
            if tables:
                for table in tables:
                    if 'Date' in table.columns:
                        frames.append(table)
                        break
            if frames:
                break

        if not frames:
            raise ValueError('Could not retrieve Treasury yield table.')

        rates = frames[0].copy()
        rates.columns = [str(col).strip().lower() for col in rates.columns]
        tenor_col = tenor.strip().lower()
        if tenor_col not in rates.columns:
            tenor_map = {
                '10 yr': '10 yr',
                '20 yr': '20 yr',
                '30 yr': '30 yr',
                '5 yr': '5 yr',
            }
            tenor_col = tenor_map.get(tenor.lower(), '10 yr')

        rates[tenor_col] = pd.to_numeric(rates[tenor_col], errors='coerce')
        latest = rates.dropna(subset=[tenor_col]).iloc[-1]
        return safe_float(latest[tenor_col]) / 100.0
