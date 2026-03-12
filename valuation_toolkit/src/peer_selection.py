from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .fundamentals import CompanySnapshot, FundamentalsBuilder
from .utils import safe_float

logger = logging.getLogger(__name__)

# Last-resort liquid US non-financial universe
LIQUID_US_FALLBACK = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL", "CRM", "ADBE",
    "INTU", "CSCO", "AMD", "QCOM", "TXN", "IBM", "NOW", "NFLX", "UBER", "PYPL",
    "WMT", "COST", "HD", "LOW", "TGT", "NKE", "SBUX", "MCD", "DIS", "CMCSA",
    "TMUS", "VZ", "T", "KO", "PEP", "PG", "CL", "KMB", "GIS", "MDLZ",
    "JNJ", "MRK", "PFE", "ABBV", "LLY", "ABT", "TMO", "DHR", "AMGN", "GILD",
    "CAT", "DE", "HON", "GE", "RTX", "LMT", "UPS", "FDX", "UNP", "ETN",
    "PH", "EMR", "ITW", "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "LIN",
    "APD", "FCX", "NEM", "DD", "NEE", "DUK", "SO", "AEP", "XEL"
]


class PeerSelector:
    def __init__(self, builder: FundamentalsBuilder):
        self.builder = builder

    def build_peer_set(self, target: CompanySnapshot, max_peers: int = 8) -> pd.DataFrame:
        candidates = self._get_candidates(target)

        if not candidates:
            raise ValueError(
                "Could not assemble peer candidates even after fallbacks. "
                "Check FMP key/limits and try clearing cache."
            )

        candidate_df = pd.DataFrame(candidates).drop_duplicates(subset=["symbol"]).copy()
        candidate_df = candidate_df[candidate_df["symbol"] != target.symbol].copy()

        if candidate_df.empty:
            raise ValueError("Peer candidate list ended up empty after filtering.")

        candidate_df["industry_match"] = np.where(candidate_df["industry"] == target.industry, 1.0, 0.4)
        candidate_df["sector_match"] = np.where(candidate_df["sector"] == target.sector, 1.0, 0.5)

        target_market_cap = max(safe_float(target.market_cap), 1.0)
        candidate_df["market_cap"] = candidate_df["market_cap"].fillna(target_market_cap)

        candidate_df["size_score"] = 1 / (
            1 + np.abs(np.log((candidate_df["market_cap"] + 1.0) / (target_market_cap + 1.0)))
        )

        candidate_df["rough_score"] = (
            0.40 * candidate_df["industry_match"]
            + 0.20 * candidate_df["sector_match"]
            + 0.40 * candidate_df["size_score"]
        )

        shortlist_n = min(max(8, max_peers * 2), 10)
        shortlist_symbols = (
            candidate_df.sort_values(["rough_score", "market_cap"], ascending=[False, False])
            .head(shortlist_n)["symbol"]
            .tolist()
        )

        peer_snapshots: list[CompanySnapshot] = []
        for symbol in shortlist_symbols:
            try:
                peer = self.builder.build_snapshot(symbol)
                peer_snapshots.append(peer)
            except Exception as exc:
                logger.warning("Failed to build snapshot for %s: %s", symbol, exc)
                continue

        # One more last resort if hydration fails badly
        if not peer_snapshots:
            logger.warning("Shortlist hydration failed. Trying fallback universe directly.")
            for symbol in LIQUID_US_FALLBACK[:10]:
                if symbol == target.symbol:
                    continue
                try:
                    peer = self.builder.build_snapshot(symbol)
                    peer_snapshots.append(peer)
                except Exception as exc:
                    logger.warning("Fallback snapshot failed for %s: %s", symbol, exc)
                    continue

        if not peer_snapshots:
            raise ValueError(
                "Could not assemble peer set after shortlist and fallback hydration. "
                "Your upstream data pulls are likely failing."
            )

        peers = pd.DataFrame([p.to_dict() for p in peer_snapshots])
        peers = peers.drop_duplicates(subset=["symbol"])
        peers = peers[peers["symbol"] != target.symbol].copy()
        peers = peers[peers["market_cap"] > 0].copy()

        if peers.empty:
            raise ValueError("Hydrated peers were empty or invalid.")

        peers["industry_match"] = np.where(peers["industry"] == target.industry, 1.0, 0.4)
        peers["sector_match"] = np.where(peers["sector"] == target.sector, 1.0, 0.5)
        peers["size_score"] = 1 / (
            1 + np.abs(np.log((peers["market_cap"] + 1.0) / (target_market_cap + 1.0)))
        )

        peers["growth_score"] = 1 / (
            1 + np.abs(peers["revenue_growth"] - target.revenue_growth).fillna(1.0) * 8
        )
        peers["margin_score"] = 1 / (
            1 + np.abs(peers["ebitda_margin"] - target.ebitda_margin).fillna(1.0) * 10
        )

        target_leverage = (
            target.net_debt / target.ebitda_ltm
            if target.ebitda_ltm and target.ebitda_ltm > 0
            else np.nan
        )

        peers["leverage"] = np.where(
            peers["ebitda_ltm"] > 0,
            peers["net_debt"] / peers["ebitda_ltm"],
            np.nan,
        )
        peers["leverage_score"] = 1 / (
            1 + np.abs(peers["leverage"] - target_leverage).fillna(2.0) * 2
        )

        peers["similarity_score"] = (
            0.20 * peers["industry_match"]
            + 0.15 * peers["sector_match"]
            + 0.25 * peers["size_score"]
            + 0.15 * peers["growth_score"]
            + 0.15 * peers["margin_score"]
            + 0.10 * peers["leverage_score"]
        )

        peers = peers.sort_values(["similarity_score", "market_cap"], ascending=[False, False]).head(max_peers)
        peers["selection_rationale"] = peers.apply(self._rationale, axis=1, args=(target,))
        return peers.reset_index(drop=True)

    def _get_candidates(self, target: CompanySnapshot) -> list[dict[str, Any]]:
        stock_peer_symbols = set()
        if self.builder.fmp:
            try:
                stock_peer_symbols = set(self.builder.fmp.stock_peers(target.symbol))
            except Exception as exc:
                logger.warning("stock_peers failed for %s: %s", target.symbol, exc)

        target_market_cap = safe_float(target.market_cap)
        sector = target.sector if target.sector and target.sector != "Unknown" else None

        attempts: list[list[dict[str, Any]]] = []

        # Attempt 1: sector + wider market cap band
        if sector and target_market_cap > 0:
            attempts.append(
                self._safe_screen(
                    sector=sector,
                    market_cap_min=max(target_market_cap * 0.10, 50_000_000),
                    market_cap_max=max(target_market_cap * 10.0, 500_000_000),
                    limit=80,
                )
            )

        # Attempt 2: sector only
        if sector:
            attempts.append(
                self._safe_screen(
                    sector=sector,
                    market_cap_min=None,
                    market_cap_max=None,
                    limit=100,
                )
            )

        # Attempt 3: size only
        if target_market_cap > 0:
            attempts.append(
                self._safe_screen(
                    sector=None,
                    market_cap_min=max(target_market_cap * 0.10, 50_000_000),
                    market_cap_max=max(target_market_cap * 10.0, 500_000_000),
                    limit=100,
                )
            )

        # Attempt 4: broad US universe
        attempts.append(
            self._safe_screen(
                sector=None,
                market_cap_min=100_000_000,
                market_cap_max=None,
                limit=120,
            )
        )

        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        for result in attempts:
            for row in result:
                symbol = str(row.get("symbol", "")).upper().strip()
                if not symbol or symbol == target.symbol or symbol in seen:
                    continue

                seen.add(symbol)
                candidates.append(
                    {
                        "symbol": symbol,
                        "name": row.get("companyName") or row.get("name") or symbol,
                        "sector": row.get("sector") or "Unknown",
                        "industry": row.get("industry") or "Unknown",
                        "market_cap": safe_float(row.get("marketCap")),
                    }
                )

        # Add API peer suggestions if available
        for symbol in stock_peer_symbols:
            symbol = str(symbol).upper().strip()
            if not symbol or symbol == target.symbol or symbol in seen:
                continue
            seen.add(symbol)
            candidates.append(
                {
                    "symbol": symbol,
                    "name": symbol,
                    "sector": "Unknown",
                    "industry": "Unknown",
                    "market_cap": np.nan,
                }
            )

        # Final fallback: liquid universe
        if not candidates:
            logger.warning("FMP screener returned no candidates. Falling back to static liquid US universe.")
            for symbol in LIQUID_US_FALLBACK:
                if symbol == target.symbol or symbol in seen:
                    continue
                seen.add(symbol)
                candidates.append(
                    {
                        "symbol": symbol,
                        "name": symbol,
                        "sector": "Unknown",
                        "industry": "Unknown",
                        "market_cap": np.nan,
                    }
                )

        return candidates

    def _safe_screen(
        self,
        sector: str | None,
        market_cap_min: float | None,
        market_cap_max: float | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not self.builder.fmp:
            return []

        try:
            rows = self.builder.fmp.screener(
                sector=sector,
                market_cap_min=market_cap_min,
                market_cap_max=market_cap_max,
                country="US",
                limit=limit,
            )
            logger.info(
                "Screener returned %s rows | sector=%s market_cap_min=%s market_cap_max=%s limit=%s",
                len(rows) if rows else 0,
                sector,
                market_cap_min,
                market_cap_max,
                limit,
            )
            return rows or []
        except Exception as exc:
            logger.warning(
                "Screener failed | sector=%s market_cap_min=%s market_cap_max=%s limit=%s | %s",
                sector,
                market_cap_min,
                market_cap_max,
                limit,
                exc,
            )
            return []

    @staticmethod
    def _rationale(row: pd.Series, target: CompanySnapshot) -> str:
        notes = []

        if row["industry"] == target.industry:
            notes.append("same industry")
        elif row["sector"] == target.sector:
            notes.append("same sector")

        if 0.5 <= row["market_cap"] / max(target.market_cap, 1) <= 2.0:
            notes.append("similar size")

        if pd.notna(row["revenue_growth"]) and pd.notna(target.revenue_growth):
            if abs(row["revenue_growth"] - target.revenue_growth) < 0.05:
                notes.append("similar growth")

        if pd.notna(row["ebitda_margin"]) and pd.notna(target.ebitda_margin):
            if abs(row["ebitda_margin"] - target.ebitda_margin) < 0.05:
                notes.append("similar margin")

        return ", ".join(notes[:3]) if notes else "closest available public comp"
