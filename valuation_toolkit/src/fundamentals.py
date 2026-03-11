from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_clients import FMPClient, SECClient, TreasuryClient
from .utils import first_valid, pick, safe_div, safe_float


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
    def __init__(self):
        self.fmp = FMPClient()
        self.sec = SECClient()
        self.treasury = TreasuryClient()

    def build_snapshot(self, symbol: str) -> CompanySnapshot:
        symbol = symbol.upper().strip()
        profile = self.fmp.profile(symbol)
        if not profile:
            raise ValueError(f'No profile returned for {symbol}.')

        quote = self.fmp.quote(symbol)
        ratios_ttm = self.fmp.ratios_ttm(symbol)
        key_metrics_ttm = self.fmp.key_metrics_ttm(symbol)
        annual_is = self.fmp.income_statement(symbol, period='annual', limit=5)
        quarterly_is = self.fmp.income_statement(symbol, period='quarter', limit=8)
        annual_bs = self.fmp.balance_sheet(symbol, period='annual', limit=5)
        quarterly_bs = self.fmp.balance_sheet(symbol, period='quarter', limit=4)
        annual_cf = self.fmp.cash_flow(symbol, period='annual', limit=5)
        quarterly_cf = self.fmp.cash_flow(symbol, period='quarter', limit=4)
        ev_history = self.fmp.enterprise_values(symbol, limit=4)

        sec_row = self.sec.lookup_ticker(symbol)
        cik = sec_row.get('cik_str') if sec_row else None

        revenue_ltm = self._ltm_sum(quarterly_is, ['revenue'])
        ebitda_ltm = self._ltm_sum(quarterly_is, ['ebitda', 'EBITDA'])
        net_income_ltm = self._ltm_sum(quarterly_is, ['netIncome'])

        annual_latest = annual_is[0] if annual_is else {}
        annual_prev = annual_is[1] if len(annual_is) > 1 else {}
        latest_bs = quarterly_bs[0] if quarterly_bs else (annual_bs[0] if annual_bs else {})
        latest_cf = quarterly_cf[0] if quarterly_cf else (annual_cf[0] if annual_cf else {})

        revenue_ltm = first_valid(revenue_ltm, safe_float(annual_latest.get('revenue')), default=np.nan)
        ebitda_ltm = first_valid(ebitda_ltm, safe_float(annual_latest.get('ebitda')), default=np.nan)
        net_income_ltm = first_valid(net_income_ltm, safe_float(annual_latest.get('netIncome')), default=np.nan)

        revenue_prev = safe_float(annual_prev.get('revenue'))
        revenue_growth = safe_div(revenue_ltm - revenue_prev, revenue_prev, default=np.nan)

        total_debt = safe_float(
            first_valid(
                latest_bs.get('totalDebt'),
                latest_bs.get('shortTermDebt', 0) + latest_bs.get('longTermDebt', 0),
                profile.get('debtToEquity'),
                default=np.nan,
            )
        )
        cash = safe_float(
            first_valid(
                latest_bs.get('cashAndCashEquivalents'),
                latest_bs.get('cashAndShortTermInvestments'),
                latest_bs.get('cashAndShortTermInvestmentsUSD'),
                default=np.nan,
            )
        )
        if np.isnan(total_debt):
            total_debt = 0.0
        if np.isnan(cash):
            cash = 0.0
        net_debt = max(total_debt - cash, 0.0)

        price = safe_float(first_valid(quote.get('price'), profile.get('price')))
        market_cap = safe_float(first_valid(quote.get('marketCap'), profile.get('mktCap'), profile.get('marketCap')))
        shares_out = safe_float(
            first_valid(
                quote.get('sharesOutstanding'),
                profile.get('sharesOutstanding'),
                safe_div(market_cap, price),
                default=np.nan,
            )
        )

        ev_from_history = safe_float(ev_history[0].get('enterpriseValue')) if ev_history else np.nan
        enterprise_value = first_valid(ev_from_history, market_cap + total_debt - cash, default=np.nan)
        enterprise_value = safe_float(enterprise_value)

        tax_rate = safe_float(
            first_valid(
                ratios_ttm.get('effectiveTaxRateTTM'),
                ratios_ttm.get('effectiveTaxRate'),
                safe_div(annual_latest.get('incomeTaxExpense'), annual_latest.get('incomeBeforeTax')),
                default=0.25,
            )
        )
        if np.isnan(tax_rate):
            tax_rate = 0.25
        tax_rate = float(np.clip(tax_rate, 0.0, 0.40))

        da_pct_sales = safe_div(
            first_valid(
                annual_cf[0].get('depreciationAndAmortization') if annual_cf else np.nan,
                annual_cf[0].get('depreciationAndAmortizationExpense') if annual_cf else np.nan,
                annual_latest.get('depreciationAndAmortization') if annual_latest else np.nan,
                default=np.nan,
            ),
            revenue_ltm,
            default=0.03,
        )
        capex_pct_sales = abs(
            safe_div(
                first_valid(
                    latest_cf.get('capitalExpenditure'),
                    annual_cf[0].get('capitalExpenditure') if annual_cf else np.nan,
                    default=np.nan,
                ),
                revenue_ltm,
                default=0.04,
            )
        )
        nwc = safe_float(
            first_valid(
                latest_bs.get('netWorkingCapital'),
                latest_bs.get('totalCurrentAssets', 0) - latest_bs.get('totalCurrentLiabilities', 0),
                default=np.nan,
            )
        )
        nwc_pct_sales = safe_div(nwc, revenue_ltm, default=0.12)

        beta = safe_float(first_valid(profile.get('beta'), key_metrics_ttm.get('beta'), default=1.0))
        if np.isnan(beta):
            beta = 1.0

        interest_expense = abs(
            safe_float(
                first_valid(
                    annual_latest.get('interestExpense'),
                    annual_cf[0].get('interestExpense') if annual_cf else np.nan,
                    default=np.nan,
                )
            )
        )

        ebitda_margin = safe_div(ebitda_ltm, revenue_ltm, default=np.nan)
        if np.isnan(ebitda_margin):
            ebitda_margin = safe_float(first_valid(ratios_ttm.get('ebitdaMarginTTM'), ratios_ttm.get('ebitdaMargin')))

        return CompanySnapshot(
            symbol=symbol,
            name=str(first_valid(profile.get('companyName'), profile.get('name'), symbol)),
            sector=str(first_valid(profile.get('sector'), 'Unknown')),
            industry=str(first_valid(profile.get('industry'), 'Unknown')),
            currency=str(first_valid(profile.get('currency'), 'USD')),
            price=price,
            shares_out=shares_out,
            market_cap=market_cap,
            total_debt=total_debt,
            cash=cash,
            net_debt=net_debt,
            enterprise_value=enterprise_value,
            revenue_ltm=revenue_ltm,
            ebitda_ltm=ebitda_ltm,
            net_income_ltm=net_income_ltm,
            revenue_growth=revenue_growth,
            ebitda_margin=ebitda_margin,
            tax_rate=tax_rate,
            da_pct_sales=da_pct_sales,
            capex_pct_sales=capex_pct_sales,
            nwc_pct_sales=nwc_pct_sales,
            beta=beta,
            interest_expense=interest_expense,
            cik=cik,
        )

    @staticmethod
    def _ltm_sum(records: list[dict[str, Any]], keys: list[str]) -> float:
        if not records:
            return np.nan
        values: list[float] = []
        for record in records[:4]:
            values.append(safe_float(pick(record, keys, default=np.nan)))
        if len(values) < 4 or any(np.isnan(values)):
            return np.nan
        return float(np.sum(values))
