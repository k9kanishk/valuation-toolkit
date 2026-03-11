from __future__ import annotations

import numpy as np
import pandas as pd

from .fundamentals import CompanySnapshot, FundamentalsBuilder


class PeerSelector:
    def __init__(self, builder: FundamentalsBuilder):
        self.builder = builder

    def build_peer_set(self, target: CompanySnapshot, max_peers: int = 8) -> pd.DataFrame:
        peer_symbols = set(self.builder.fmp.stock_peers(target.symbol))

        screen = self.builder.fmp.screener(
            sector=target.sector if target.sector != 'Unknown' else None,
            market_cap_min=max(target.market_cap * 0.25, 100_000_000),
            market_cap_max=max(target.market_cap * 4.0, 500_000_000),
            country='US',
            limit=100,
        )
        for row in screen:
            symbol = row.get('symbol')
            if symbol and symbol != target.symbol:
                peer_symbols.add(symbol)

        peer_snapshots: list[CompanySnapshot] = []
        for symbol in sorted(peer_symbols):
            try:
                peer = self.builder.build_snapshot(symbol)
                peer_snapshots.append(peer)
            except Exception:
                continue

        if not peer_snapshots:
            raise ValueError('Could not assemble peer set. Try a larger, liquid US ticker first.')

        peers = pd.DataFrame([p.to_dict() for p in peer_snapshots])
        peers = peers.drop_duplicates(subset=['symbol'])
        peers = peers[peers['symbol'] != target.symbol].copy()
        peers = peers[peers['market_cap'] > 0].copy()

        peers['industry_match'] = np.where(peers['industry'] == target.industry, 1.0, 0.4)
        peers['sector_match'] = np.where(peers['sector'] == target.sector, 1.0, 0.5)
        peers['size_score'] = 1 / (1 + np.abs(np.log((peers['market_cap'] + 1) / (target.market_cap + 1))))
        peers['growth_score'] = 1 / (1 + np.abs(peers['revenue_growth'] - target.revenue_growth).fillna(1.0) * 8)
        peers['margin_score'] = 1 / (1 + np.abs(peers['ebitda_margin'] - target.ebitda_margin).fillna(1.0) * 10)

        target_leverage = target.net_debt / target.ebitda_ltm if target.ebitda_ltm and target.ebitda_ltm > 0 else np.nan
        peers['leverage'] = np.where(peers['ebitda_ltm'] > 0, peers['net_debt'] / peers['ebitda_ltm'], np.nan)
        peers['leverage_score'] = 1 / (1 + np.abs(peers['leverage'] - target_leverage).fillna(2.0) * 2)

        peers['similarity_score'] = (
            0.20 * peers['industry_match']
            + 0.15 * peers['sector_match']
            + 0.25 * peers['size_score']
            + 0.15 * peers['growth_score']
            + 0.15 * peers['margin_score']
            + 0.10 * peers['leverage_score']
        )

        peers = peers.sort_values(['similarity_score', 'market_cap'], ascending=[False, False]).head(max_peers)
        peers['selection_rationale'] = peers.apply(self._rationale, axis=1, args=(target,))
        return peers.reset_index(drop=True)

    @staticmethod
    def _rationale(row: pd.Series, target: CompanySnapshot) -> str:
        notes = []
        if row['industry'] == target.industry:
            notes.append('same industry')
        elif row['sector'] == target.sector:
            notes.append('same sector')

        if 0.5 <= row['market_cap'] / max(target.market_cap, 1) <= 2.0:
            notes.append('similar size')
        if pd.notna(row['revenue_growth']) and pd.notna(target.revenue_growth):
            if abs(row['revenue_growth'] - target.revenue_growth) < 0.05:
                notes.append('similar growth')
        if pd.notna(row['ebitda_margin']) and pd.notna(target.ebitda_margin):
            if abs(row['ebitda_margin'] - target.ebitda_margin) < 0.05:
                notes.append('similar margin')

        return ', '.join(notes[:3]) if notes else 'closest available public comp'
