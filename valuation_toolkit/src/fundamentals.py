from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_clients import FMPClient, SECClient, TreasuryClient, YahooClient

from .config import settings
from .utils import safe_div, safe_float


@dataclass
class CompanySnapshot:
    symbol: str
    name: str
    sector: str
    industry: str
    currency: str
    price: float
    shares_out: float
    market_cap: float
    total_debt: float
    cash: float
    net_debt: float
    enterprise_value: float
    revenue_ltm: float
    ebitda_ltm: float
    net_income_ltm: float
    revenue_growth: float
    ebitda_margin: float
    tax_rate: float
    da_pct_sales: float
    capex_pct_sales: float
    nwc_pct_sales: float
    beta: float
    interest_expense: float
    cik: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class FundamentalsBuilder:
    def __init__(self, use_fmp: bool = True):
        self.sec = SECClient()
        self.treasury = TreasuryClient()
        self.yahoo = YahooClient()

        self.fmp = None
        if use_fmp and settings.use_fmp:
            try:
                self.fmp = FMPClient()
            except Exception:
                self.fmp = None

        self._snapshot_cache: dict[str, CompanySnapshot] = {}

    @staticmethod
    def _df_row_sum(df: pd.DataFrame | None, candidates: list[str]) -> float:
        if df is None or df.empty:
            return np.nan
        for name in candidates:
            if name in df.index:
                try:
                    vals = pd.to_numeric(df.loc[name], errors='coerce').dropna()
                    if not vals.empty:
                        return float(vals.iloc[:4].sum()) if len(vals) >= 4 else float(vals.iloc[0])
                except Exception:
                    continue
        return np.nan

    @staticmethod
    def _df_latest_value(df: pd.DataFrame | None, candidates: list[str]) -> float:
        if df is None or df.empty:
            return np.nan
        for name in candidates:
            if name in df.index:
                try:
                    vals = pd.to_numeric(df.loc[name], errors='coerce').dropna()
                    if not vals.empty:
                        return float(vals.iloc[0])
                except Exception:
                    continue
        return np.nan

    def build_snapshot(self, symbol: str) -> CompanySnapshot:
        symbol = symbol.upper().strip()
        if symbol in self._snapshot_cache:
            return self._snapshot_cache[symbol]

        profile = self.fmp.profile(symbol) if self.fmp else {}
        quote = self.fmp.quote(symbol) if self.fmp else {}
        ratios = self.fmp.ratios_ttm(symbol) if self.fmp else {}
        metrics = self.fmp.key_metrics_ttm(symbol) if self.fmp else {}
        ev_rows = self.fmp.enterprise_values(symbol, limit=1) if self.fmp else []
        annual_cf = self.fmp.cash_flow(symbol, period='annual', limit=5) if self.fmp else []

        yq = self.yahoo.quote_summary(symbol)
        q_is = self.yahoo.quarterly_income_stmt(symbol)
        a_is = self.yahoo.annual_income_stmt(symbol)
        q_bs = self.yahoo.quarterly_balance_sheet(symbol)
        a_bs = self.yahoo.annual_balance_sheet(symbol)
        q_cf = self.yahoo.quarterly_cashflow(symbol)
        a_cf = self.yahoo.annual_cashflow(symbol)

        sec_row = self.sec.lookup_ticker(symbol)
        cik = sec_row.get('cik_str') if sec_row else None

        name = profile.get('companyName') or profile.get('name') or yq.get('companyName') or symbol
        sector = profile.get('sector') or yq.get('sector') or 'Unknown'
        industry = profile.get('industry') or yq.get('industry') or 'Unknown'
        currency = profile.get('currency') or 'USD'

        shares_outstanding = (
            safe_float(metrics.get('sharesOutstanding'))
            or safe_float(quote.get('sharesOutstanding'))
            or safe_float(yq.get('sharesOutstanding'))
        )

        price = safe_float(quote.get('price')) or safe_float(yq.get('price'))

        market_cap = (
            safe_float(quote.get('marketCap'))
            or safe_float(yq.get('marketCap'))
        )

        total_debt = (
            safe_float(yq.get('totalDebt'))
            or self._df_latest_value(q_bs, ['Total Debt', 'Current Debt And Capital Lease Obligation'])
            or self._df_latest_value(a_bs, ['Total Debt', 'Current Debt And Capital Lease Obligation'])
            or 0.0
        )
        cash = (
            safe_float(yq.get('totalCash'))
            or self._df_latest_value(
                q_bs,
                ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments'],
            )
            or self._df_latest_value(
                a_bs,
                ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments'],
            )
            or 0.0
        )
        net_debt = max(total_debt - cash, 0.0)

        revenue_ltm = (
            safe_float(ratios.get('revenuePerShareTTM')) * shares_outstanding
            if safe_float(ratios.get('revenuePerShareTTM')) and shares_outstanding
            else np.nan
        )
        if pd.isna(revenue_ltm):
            revenue_ltm = (
                safe_float(yq.get('totalRevenue'))
                or self._df_row_sum(q_is, ['Total Revenue', 'Revenue'])
                or self._df_latest_value(a_is, ['Total Revenue', 'Revenue'])
                or 0.0
            )

        ebitda_ltm = (
            safe_float(yq.get('ebitda'))
            or self._df_row_sum(q_is, ['EBITDA', 'Normalized EBITDA'])
            or self._df_latest_value(a_is, ['EBITDA', 'Normalized EBITDA'])
            or 0.0
        )

        net_income_ltm = (
            safe_float(yq.get('netIncome'))
            or self._df_row_sum(q_is, ['Net Income', 'Net Income Common Stockholders'])
            or self._df_latest_value(a_is, ['Net Income', 'Net Income Common Stockholders'])
            or 0.0
        )

        enterprise_value = safe_float(ev_rows[0].get('enterpriseValue')) if ev_rows else np.nan
        if pd.isna(enterprise_value):
            enterprise_value = safe_float(yq.get('enterpriseValue')) or (market_cap + total_debt - cash if market_cap else np.nan)

        price = float(price or 0.0)
        market_cap = float(market_cap or 0.0)
        shares_outstanding = float(shares_outstanding or 0.0)
        enterprise_value = float(enterprise_value or 0.0)
        total_debt = float(total_debt or 0.0)
        cash = float(cash or 0.0)
        net_debt = float(net_debt or 0.0)

        # Derive missing shares outstanding from market cap / price
        if shares_outstanding <= 0 and market_cap > 0 and price > 0:
            shares_outstanding = market_cap / price

        # Derive missing market cap from price * shares
        if market_cap <= 0 and price > 0 and shares_outstanding > 0:
            market_cap = price * shares_outstanding

        # Derive missing market cap from EV - net debt
        if market_cap <= 0 and enterprise_value > 0:
            derived_mc = enterprise_value - net_debt
            if derived_mc > 0:
                market_cap = derived_mc

        # Derive missing EV from market cap + net debt
        if enterprise_value <= 0 and market_cap > 0:
            enterprise_value = market_cap + net_debt

        revenue_growth = np.nan
        if q_is is not None and not q_is.empty:
            for revenue_label in ['Total Revenue', 'Revenue']:
                if revenue_label in q_is.index:
                    try:
                        vals = pd.to_numeric(q_is.loc[revenue_label], errors='coerce').dropna()
                        if len(vals) >= 8:
                            recent4 = vals.iloc[:4].sum()
                            prior4 = vals.iloc[4:8].sum()
                            if prior4 and prior4 != 0:
                                revenue_growth = (recent4 / prior4) - 1
                                break
                    except Exception:
                        pass
        if pd.isna(revenue_growth):
            revenue_growth = 0.05

        ebitda_margin = (ebitda_ltm / revenue_ltm) if revenue_ltm and revenue_ltm > 0 else 0.0
        beta = safe_float(yq.get('beta')) or safe_float(profile.get('beta')) or safe_float(metrics.get('beta')) or 1.0

        effective_tax = safe_float(ratios.get('effectiveTaxRateTTM')) or safe_float(ratios.get('effectiveTaxRate')) or 0.25
        tax_rate = float(np.clip(effective_tax, 0.0, 0.40))

        fmp_da = safe_float(annual_cf[0].get('depreciationAndAmortization')) if annual_cf else np.nan
        fmp_capex = safe_float(annual_cf[0].get('capitalExpenditure')) if annual_cf else np.nan

        da_pct_sales = safe_div(
            self._df_latest_value(q_cf, ['Depreciation Amortization Depletion', 'Depreciation And Amortization'])
            or self._df_latest_value(a_cf, ['Depreciation Amortization Depletion', 'Depreciation And Amortization'])
            or fmp_da,
            revenue_ltm,
            default=0.03,
        )
        capex_pct_sales = abs(
            safe_div(
                self._df_latest_value(q_cf, ['Capital Expenditure'])
                or self._df_latest_value(a_cf, ['Capital Expenditure'])
                or fmp_capex,
                revenue_ltm,
                default=0.04,
            )
        )

        nwc = self._df_latest_value(q_bs, ['Working Capital']) or self._df_latest_value(a_bs, ['Working Capital'])
        nwc_pct_sales = safe_div(nwc, revenue_ltm, default=0.12)

        interest_expense = abs(
            self._df_latest_value(q_is, ['Interest Expense', 'Interest Expense Non Operating'])
            or self._df_latest_value(a_is, ['Interest Expense', 'Interest Expense Non Operating'])
            or 0.0
        )

        snapshot = CompanySnapshot(
            symbol=symbol,
            name=str(name),
            sector=str(sector),
            industry=str(industry),
            currency=str(currency),
            price=float(price or 0.0),
            shares_out=float(shares_outstanding or 0.0),
            market_cap=float(market_cap or 0.0),
            total_debt=float(total_debt or 0.0),
            cash=float(cash or 0.0),
            net_debt=float(net_debt or 0.0),
            enterprise_value=float(enterprise_value or 0.0),
            revenue_ltm=float(revenue_ltm or 0.0),
            ebitda_ltm=float(ebitda_ltm or 0.0),
            net_income_ltm=float(net_income_ltm or 0.0),
            revenue_growth=float(revenue_growth or 0.0),
            ebitda_margin=float(ebitda_margin or 0.0),
            tax_rate=float(tax_rate or 0.25),
            da_pct_sales=float(da_pct_sales or 0.03),
            capex_pct_sales=float(capex_pct_sales or 0.04),
            nwc_pct_sales=float(nwc_pct_sales or 0.12),
            beta=float(beta or 1.0),
            interest_expense=float(interest_expense or 0.0),
            cik=cik,
        )
        self._snapshot_cache[symbol] = snapshot
        return snapshot
