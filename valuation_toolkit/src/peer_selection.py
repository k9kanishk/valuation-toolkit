from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from .fundamentals import CompanySnapshot, FundamentalsBuilder

logger = logging.getLogger(__name__)

CURATED_PEERS = {
    "AAPL": ["MSFT", "GOOGL", "META", "AMZN", "ORCL", "ADBE", "CRM", "INTU", "QCOM", "CSCO", "NVDA", "AVGO"],
    "MSFT": ["GOOGL", "AAPL", "ORCL", "ADBE", "CRM", "NOW", "INTU", "IBM", "CSCO", "AMZN"],
    "GOOGL": ["META", "MSFT", "AMZN", "AAPL", "NFLX", "CRM", "ORCL", "ADBE"],
    "AMZN": ["WMT", "COST", "TGT", "EBAY", "SHOP", "GOOGL", "MSFT", "AAPL"],
    "META": ["GOOGL", "NFLX", "SNAP", "PINS", "AAPL", "MSFT", "AMZN"],
}

SECTOR_FALLBACK = {
    "Technology": ["MSFT", "GOOGL", "META", "ORCL", "ADBE", "CRM", "INTU", "CSCO", "QCOM", "IBM", "NOW", "TXN"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ"],
    "Consumer Cyclical": ["AMZN", "HD", "LOW", "NKE", "SBUX", "MCD", "TGT"],
    "Consumer Defensive": ["WMT", "COST", "KO", "PEP", "PG", "MDLZ", "GIS"],
    "Healthcare": ["JNJ", "MRK", "ABBV", "PFE", "LLY", "ABT", "TMO", "DHR"],
    "Industrials": ["CAT", "DE", "HON", "GE", "RTX", "LMT", "UPS", "FDX"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "OXY"],
    "Utilities": ["NEE", "DUK", "SO", "AEP", "XEL"],
}

GENERAL_FALLBACK = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "ORCL", "ADBE", "CRM", "INTU", "CSCO",
    "NVDA", "AVGO", "QCOM", "IBM", "NOW", "NFLX", "WMT", "COST", "HD", "JNJ",
    "MRK", "ABBV", "LLY", "XOM", "CVX", "CAT", "HON", "GE", "LIN", "NKE",
]


class PeerSelector:
    def __init__(self, builder: FundamentalsBuilder):
        self.builder = builder

    def build_peer_set(self, target: CompanySnapshot, max_peers: int = 6) -> pd.DataFrame:
        candidate_symbols = self._candidate_symbols(target)

        peer_snapshots: list[CompanySnapshot] = []
        seen: set[str] = set()

        # Hydrate a limited number of candidates, but stop once we have enough usable peers
        hydrate_limit = max(10, max_peers * 3)

        for symbol in candidate_symbols[:hydrate_limit]:
            symbol = str(symbol).upper().strip()
            if not symbol or symbol == target.symbol or symbol in seen:
                continue
            seen.add(symbol)

            snapshot = self._safe_build_snapshot(symbol)
            if snapshot is None:
                continue
            if not self._is_usable_snapshot(snapshot):
                continue

            peer_snapshots.append(snapshot)

            if len(peer_snapshots) >= max(max_peers, 4):
                # enough to proceed
                pass

        if not peer_snapshots:
            raise ValueError(
                "Could not hydrate any usable peers. "
                "Try clearing cache, or use manual peer overrides for now."
            )

        peers = pd.DataFrame([p.to_dict() for p in peer_snapshots]).drop_duplicates(subset=["symbol"]).copy()

        # Numeric cleanup
        numeric_cols = [
            "price", "shares_outstanding", "market_cap", "enterprise_value", "total_debt", "cash", "net_debt",
            "revenue_ltm", "ebitda_ltm", "net_income_ltm", "revenue_growth", "ebitda_margin", "beta"
        ]
        for col in numeric_cols:
            if col in peers.columns:
                peers[col] = pd.to_numeric(peers[col], errors="coerce")

        peers = peers[peers["symbol"] != target.symbol].copy()

        # Fallback market cap derivation
        peers["market_cap"] = peers["market_cap"].where(peers["market_cap"] > 0, np.nan)

        if "enterprise_value" in peers.columns and "net_debt" in peers.columns:
            alt_market_cap = peers["enterprise_value"] - peers["net_debt"]
            peers["market_cap"] = peers["market_cap"].fillna(alt_market_cap)

        if "price" in peers.columns and "shares_outstanding" in peers.columns:
            px_shares_mc = peers["price"] * peers["shares_outstanding"]
            peers["market_cap"] = peers["market_cap"].fillna(px_shares_mc)

        # Keep peers if they have at least one valuation base
        peers = peers[
            (
                peers["market_cap"].fillna(0) > 0
            ) | (
                peers["enterprise_value"].fillna(0) > 0
            )
        ].copy()

        # And at least one operating base
        peers = peers[
            (
                peers["revenue_ltm"].fillna(0) > 0
            ) | (
                peers["ebitda_ltm"].fillna(0) > 0
            ) | (
                peers["net_income_ltm"].fillna(0) > 0
            )
        ].copy()

        if peers.empty:
            raise ValueError(
                "Hydrated peers were empty or invalid after valuation-base checks. "
                "Your snapshot fallbacks are still too weak."
            )

        target_market_cap = self._target_market_cap(target)

        peers["sector"] = peers["sector"].fillna("Unknown")
        peers["industry"] = peers["industry"].fillna("Unknown")
        peers["revenue_growth"] = peers["revenue_growth"].fillna(target.revenue_growth if pd.notna(target.revenue_growth) else 0.05)
        peers["ebitda_margin"] = peers["ebitda_margin"].fillna(target.ebitda_margin if pd.notna(target.ebitda_margin) else 0.15)
        peers["market_cap"] = peers["market_cap"].fillna(target_market_cap)
        peers["beta"] = peers["beta"].fillna(1.0)

        peers["industry_match"] = np.where(peers["industry"] == target.industry, 1.0, 0.4)
        peers["sector_match"] = np.where(peers["sector"] == target.sector, 1.0, 0.5)

        peers["size_score"] = 1 / (
            1 + np.abs(np.log((peers["market_cap"] + 1.0) / (target_market_cap + 1.0)))
        )

        target_growth = target.revenue_growth if pd.notna(target.revenue_growth) else 0.05
        target_margin = target.ebitda_margin if pd.notna(target.ebitda_margin) else 0.15

        peers["growth_score"] = 1 / (
            1 + np.abs(peers["revenue_growth"] - target_growth) * 8
        )
        peers["margin_score"] = 1 / (
            1 + np.abs(peers["ebitda_margin"] - target_margin) * 10
        )

        target_leverage = np.nan
        if getattr(target, "ebitda_ltm", 0) and target.ebitda_ltm > 0:
            target_leverage = target.net_debt / target.ebitda_ltm

        peers["leverage"] = np.where(
            peers["ebitda_ltm"].fillna(0) > 0,
            peers["net_debt"].fillna(0) / peers["ebitda_ltm"],
            np.nan,
        )

        if pd.isna(target_leverage):
            peers["leverage_score"] = 0.7
        else:
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

        peers = peers.sort_values(["similarity_score", "market_cap"], ascending=[False, False]).head(max_peers).copy()
        peers["selection_rationale"] = peers.apply(self._rationale, axis=1, args=(target,))
        return peers.reset_index(drop=True)

    def _candidate_symbols(self, target: CompanySnapshot) -> list[str]:
        symbols: list[str] = []

        # 1) Curated target-specific peers
        symbols.extend(CURATED_PEERS.get(target.symbol, []))

        # 2) Curated sector fallbacks
        if target.sector in SECTOR_FALLBACK:
            symbols.extend(SECTOR_FALLBACK[target.sector])

        # 3) FMP suggestions if available
        if getattr(self.builder, "fmp", None) is not None:
            try:
                symbols.extend(self.builder.fmp.stock_peers(target.symbol) or [])
            except Exception:
                pass

            try:
                screen = self.builder.fmp.screener(
                    sector=target.sector if target.sector != "Unknown" else None,
                    market_cap_min=max(self._target_market_cap(target) * 0.10, 50_000_000),
                    market_cap_max=max(self._target_market_cap(target) * 10.0, 500_000_000),
                    country="US",
                    limit=50,
                )
                for row in screen or []:
                    symbol = str(row.get("symbol", "")).upper().strip()
                    if symbol:
                        symbols.append(symbol)
            except Exception:
                pass

        # 4) General fallback
        symbols.extend(GENERAL_FALLBACK)

        # de-dup but preserve order
        deduped: list[str] = []
        seen: set[str] = set()
        for s in symbols:
            s = str(s).upper().strip()
            if not s or s in seen or s == target.symbol:
                continue
            seen.add(s)
            deduped.append(s)
        return deduped

    def _safe_build_snapshot(self, symbol: str) -> CompanySnapshot | None:
        try:
            return self.builder.build_snapshot(symbol)
        except Exception as exc:
            logger.warning("Failed to build snapshot for %s: %s", symbol, exc)
            return None

    @staticmethod
    def _is_usable_snapshot(snapshot: CompanySnapshot) -> bool:
        valuation_base = any([
            getattr(snapshot, "market_cap", 0) and snapshot.market_cap > 0,
            getattr(snapshot, "enterprise_value", 0) and snapshot.enterprise_value > 0,
            (getattr(snapshot, "price", 0) and snapshot.price > 0 and getattr(snapshot, "shares_outstanding", 0) and snapshot.shares_outstanding > 0),
        ])
        operating_base = any([
            getattr(snapshot, "revenue_ltm", 0) and snapshot.revenue_ltm > 0,
            getattr(snapshot, "ebitda_ltm", 0) and snapshot.ebitda_ltm > 0,
            getattr(snapshot, "net_income_ltm", 0) and snapshot.net_income_ltm > 0,
        ])
        return valuation_base and operating_base

    @staticmethod
    def _target_market_cap(target: CompanySnapshot) -> float:
        if getattr(target, "market_cap", 0) and target.market_cap > 0:
            return float(target.market_cap)
        if getattr(target, "enterprise_value", 0) and target.enterprise_value > 0:
            mc = target.enterprise_value - getattr(target, "net_debt", 0)
            if mc > 0:
                return float(mc)
        if getattr(target, "price", 0) and target.price > 0 and getattr(target, "shares_outstanding", 0) and target.shares_outstanding > 0:
            return float(target.price * target.shares_outstanding)
        return 1.0

    @staticmethod
    def _rationale(row: pd.Series, target: CompanySnapshot) -> str:
        notes: list[str] = []

        if row.get("industry") == target.industry:
            notes.append("same industry")
        elif row.get("sector") == target.sector:
            notes.append("same sector")

        row_mc = row.get("market_cap", np.nan)
        target_mc = target.market_cap if getattr(target, "market_cap", 0) and target.market_cap > 0 else np.nan
        if pd.notna(row_mc) and pd.notna(target_mc) and target_mc > 0:
            ratio = row_mc / target_mc
            if 0.4 <= ratio <= 2.5:
                notes.append("similar size")

        if pd.notna(row.get("revenue_growth")) and pd.notna(target.revenue_growth):
            if abs(row["revenue_growth"] - target.revenue_growth) < 0.05:
                notes.append("similar growth")

        if pd.notna(row.get("ebitda_margin")) and pd.notna(target.ebitda_margin):
            if abs(row["ebitda_margin"] - target.ebitda_margin) < 0.05:
                notes.append("similar margin")

        return ", ".join(notes[:3]) if notes else "closest available public comp"
