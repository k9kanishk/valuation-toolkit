from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
CACHE_DIR = DATA_DIR / 'cache'
OUTPUT_DIR = BASE_DIR / 'outputs'

for directory in (DATA_DIR, CACHE_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    fmp_api_key: str = os.getenv('FMP_API_KEY', '').strip()
    sec_user_agent: str = os.getenv('SEC_USER_AGENT', 'ValuationToolkit contact@example.com').strip()
    cache_ttl_hours: int = int(os.getenv('CACHE_TTL_HOURS', '12'))
    default_currency: str = os.getenv('DEFAULT_CURRENCY', 'USD').strip().upper()
    use_fmp: bool = os.getenv('USE_FMP', 'true').lower() == 'true'
    fmp_base_url: str = 'https://financialmodelingprep.com/stable'
    sec_base_url: str = 'https://data.sec.gov'
    treasury_url: str = (
        'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/'
        'TextView?type=daily_treasury_yield_curve&field_tdr_date_value={year}'
    )


settings = Settings()
