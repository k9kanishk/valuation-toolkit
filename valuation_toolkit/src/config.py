from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    cache_ttl_hours: int = int(os.getenv("CACHE_TTL_HOURS", "24"))

    sec_user_agent: str = os.getenv(
        "SEC_USER_AGENT",
        "Your Name your_email@example.com",
    )
    sec_ticker_url: str = "https://www.sec.gov/files/company_tickers.json"
    sec_companyfacts_base: str = "https://data.sec.gov/api/xbrl/companyfacts"

    treasury_yield_xml_url: str = (
        "https://home.treasury.gov/sites/default/files/interest-rates/yield.xml"
    )

    use_optional_provider: bool = os.getenv(
        "USE_OPTIONAL_PROVIDER", "false"
    ).lower() == "true"
    optional_provider: str = os.getenv("OPTIONAL_PROVIDER", "none").lower()

    fmp_api_key: str = os.getenv("FMP_API_KEY", "").strip()
    fmp_base_url: str = os.getenv(
        "FMP_BASE_URL",
        "https://financialmodelingprep.com/stable",
    ).rstrip("/")


settings = Settings()
