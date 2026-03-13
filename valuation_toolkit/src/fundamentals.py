from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_clients import (
    SECClient,
    TreasuryClient,
    YahooClient,
    build_optional_client,
    safe_float,
)


ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
QUARTERLY_FORMS = {"10-Q", "10-Q/A"}


@dataclass
class CompanySnapshot:
    symbol: str
    name: str
    sector: str
    industry: str
    currency: str
    price: float
    shares_outstanding: float
    market_cap: float
    enterprise_value: float
    total_debt: float
    cash: float
    net_debt: float
    revenue_ltm: float
    ebitda_ltm: float
    net_income_ltm: float
    revenue_growth: float
    ebitda_margin: float
    beta: float

    @property
    def company_name(self) -> str:
        return self.name

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["company_name"] = self.name
        return payload


class FundamentalsBuilder:
    def __init__(self, use_optional_provider: bool = False):
        self.sec = SECClient()
        self.treasury = TreasuryClient()
        self.yahoo = YahooClient()

        self.optional_client = build_optional_client(enabled=use_optional_provider)
        self.fmp = self.optional_client  # backward-compatible alias for older code

        self._snapshot_cache: dict[str, CompanySnapshot] = {}

    @staticmethod
    def _stmt_ltm(df: pd.DataFrame | None, labels: list[str]) -> float | None:
        if df is None or df.empty:
            return None
        for label in labels:
            if label in df.index:
                vals = pd.to_numeric(df.loc[label], errors="coerce").dropna()
                if len(vals) >= 4:
                    return float(vals.iloc[:4].sum())
                if len(vals) >= 1:
                    return float(vals.iloc[0])
        return None

    @staticmethod
    def _stmt_latest(df: pd.DataFrame | None, labels: list[str]) -> float | None:
        if df is None or df.empty:
            return None
        for label in labels:
            if label in df.index:
                vals = pd.to_numeric(df.loc[label], errors="coerce").dropna()
                if len(vals) >= 1:
                    return float(vals.iloc[0])
        return None

    @staticmethod
    def _stmt_growth_from_quarters(df: pd.DataFrame | None, labels: list[str]) -> float | None:
        if df is None or df.empty:
            return None
        for label in labels:
            if label in df.index:
                vals = pd.to_numeric(df.loc[label], errors="coerce").dropna()
                if len(vals) >= 8:
                    recent = float(vals.iloc[:4].sum())
                    prior = float(vals.iloc[4:8].sum())
                    if prior != 0:
                        return recent / prior - 1.0
        return None

    @staticmethod
    def _stmt_growth_from_annual(df: pd.DataFrame | None, labels: list[str]) -> float | None:
        if df is None or df.empty:
            return None
        for label in labels:
            if label in df.index:
                vals = pd.to_numeric(df.loc[label], errors="coerce").dropna()
                if len(vals) >= 2 and vals.iloc[1] != 0:
                    return float(vals.iloc[0] / vals.iloc[1] - 1.0)
        return None

    @staticmethod
    def _company_facts_values(
        company_facts: dict[str, Any] | None,
        *,
        taxonomy: str,
        concepts: list[str],
        unit: str,
        forms: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not company_facts:
            return []

        facts = company_facts.get("facts", {}).get(taxonomy, {})
        rows: list[dict[str, Any]] = []

        for concept in concepts:
            concept_block = facts.get(concept, {})
            units_block = concept_block.get("units", {})
            items = units_block.get(unit, [])
            for item in items:
                val = safe_float(item.get("val"))
                if val is None:
                    continue
                form = str(item.get("form", "")).upper()
                if forms and form not in forms:
                    continue
                end_dt = pd.to_datetime(item.get("end"), errors="coerce")
                filed_dt = pd.to_datetime(item.get("filed"), errors="coerce")
                rows.append(
                    {
                        "concept": concept,
                        "val": float(val),
                        "form": form,
                        "end": end_dt,
                        "filed": filed_dt,
                        "fy": item.get("fy"),
                        "fp": item.get("fp"),
                    }
                )

        rows.sort(
            key=lambda x: (
                pd.Timestamp.min if pd.isna(x["end"]) else x["end"],
                pd.Timestamp.min if pd.isna(x["filed"]) else x["filed"],
            ),
            reverse=True,
        )
        return rows

    def _sec_latest_value(
        self,
        company_facts: dict[str, Any] | None,
        *,
        taxonomy: str,
        concepts: list[str],
        unit: str,
        forms: set[str] | None = None,
    ) -> float | None:
        rows = self._company_facts_values(
            company_facts,
            taxonomy=taxonomy,
            concepts=concepts,
            unit=unit,
            forms=forms,
        )
        return rows[0]["val"] if rows else None

    def _sec_annual_growth(
        self,
        company_facts: dict[str, Any] | None,
        *,
        taxonomy: str,
        concepts: list[str],
        unit: str,
    ) -> float | None:
        rows = self._company_facts_values(
            company_facts,
            taxonomy=taxonomy,
            concepts=concepts,
            unit=unit,
            forms=ANNUAL_FORMS,
        )
        if len(rows) >= 2 and rows[1]["val"] != 0:
            return rows[0]["val"] / rows[1]["val"] - 1.0
        return None

    def build_snapshot(self, symbol: str) -> CompanySnapshot:
        symbol = symbol.upper().strip()
        if symbol in self._snapshot_cache:
            return self._snapshot_cache[symbol]

        optional_profile = self.optional_client.profile(symbol)
        optional_quote = self.optional_client.quote(symbol)
        optional_ratios = self.optional_client.ratios_ttm(symbol)
        optional_metrics = self.optional_client.key_metrics_ttm(symbol)
        optional_ev_rows = self.optional_client.enterprise_values(symbol, limit=1)

        yahoo = self.yahoo.quote_summary(symbol)
        q_is = self.yahoo.quarterly_income_stmt(symbol)
        a_is = self.yahoo.annual_income_stmt(symbol)
        q_bs = self.yahoo.quarterly_balance_sheet(symbol)
        a_bs = self.yahoo.annual_balance_sheet(symbol)

        sec_row = self.sec.lookup_ticker(symbol)
        company_facts = None
        if sec_row and sec_row.get("cik_str"):
            try:
                company_facts = self.sec.company_facts(sec_row["cik_str"])
            except Exception:
                company_facts = None

        sec_name = sec_row.get("name") if sec_row else None
        sec_shares = self._sec_latest_value(
            company_facts,
            taxonomy="dei",
            concepts=["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
            unit="shares",
            forms=None,
        )

        sec_revenue = self._sec_latest_value(
            company_facts,
            taxonomy="us-gaap",
            concepts=[
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet",
                "Revenues",
            ],
            unit="USD",
            forms=ANNUAL_FORMS,
        )

        sec_net_income = self._sec_latest_value(
            company_facts,
            taxonomy="us-gaap",
            concepts=["NetIncomeLoss", "ProfitLoss"],
            unit="USD",
            forms=ANNUAL_FORMS,
        )

        sec_cash = self._sec_latest_value(
            company_facts,
            taxonomy="us-gaap",
            concepts=[
                "CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                "CashCashEquivalentsAndShortTermInvestments",
            ],
            unit="USD",
            forms=None,
        )

        sec_total_debt = self._sec_latest_value(
            company_facts,
            taxonomy="us-gaap",
            concepts=[
                "DebtAndFinanceLeaseObligations",
                "LongTermDebtAndCapitalLeaseObligation",
                "LongTermDebtNoncurrent",
                "LongTermDebt",
            ],
            unit="USD",
            forms=None,
        )

        name = (
            optional_profile.get("companyName")
            or yahoo.get("name")
            or sec_name
            or symbol
        )
        sector = (
            optional_profile.get("sector")
            or yahoo.get("sector")
            or "Unknown"
        )
        industry = (
            optional_profile.get("industry")
            or yahoo.get("industry")
            or "Unknown"
        )
        currency = yahoo.get("currency") or "USD"

        price = safe_float(optional_quote.get("price")) or safe_float(yahoo.get("price")) or 0.0

        shares_outstanding = (
            safe_float(optional_metrics.get("sharesOutstanding"))
            or safe_float(optional_quote.get("sharesOutstanding"))
            or safe_float(yahoo.get("sharesOutstanding"))
            or sec_shares
            or 0.0
        )

        market_cap = (
            safe_float(optional_quote.get("marketCap"))
            or safe_float(yahoo.get("marketCap"))
            or 0.0
        )

        if market_cap <= 0 and price > 0 and shares_outstanding > 0:
            market_cap = price * shares_outstanding

        if shares_outstanding <= 0 and market_cap > 0 and price > 0:
            shares_outstanding = market_cap / price

        total_debt = (
            safe_float(yahoo.get("totalDebt"))
            or self._stmt_latest(a_bs, ["Total Debt", "Current Debt And Capital Lease Obligation"])
            or sec_total_debt
            or 0.0
        )

        cash = (
            safe_float(yahoo.get("totalCash"))
            or self._stmt_latest(
                a_bs,
                ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
            )
            or sec_cash
            or 0.0
        )

        net_debt = max(float(total_debt) - float(cash), 0.0)

        revenue_ltm = (
            safe_float(yahoo.get("totalRevenue"))
            or self._stmt_ltm(q_is, ["Total Revenue", "Revenue"])
            or self._stmt_latest(a_is, ["Total Revenue", "Revenue"])
            or sec_revenue
            or 0.0
        )

        ebitda_ltm = (
            safe_float(yahoo.get("ebitda"))
            or self._stmt_ltm(q_is, ["EBITDA", "Normalized EBITDA"])
            or self._stmt_latest(a_is, ["EBITDA", "Normalized EBITDA"])
            or 0.0
        )

        net_income_ltm = (
            safe_float(yahoo.get("netIncome"))
            or self._stmt_ltm(q_is, ["Net Income", "Net Income Common Stockholders"])
            or self._stmt_latest(a_is, ["Net Income", "Net Income Common Stockholders"])
            or sec_net_income
            or 0.0
        )

        enterprise_value = (
            safe_float(optional_ev_rows[0].get("enterpriseValue")) if optional_ev_rows else None
        )
        enterprise_value = enterprise_value or safe_float(yahoo.get("enterpriseValue")) or 0.0

        if enterprise_value <= 0 and market_cap > 0:
            enterprise_value = market_cap + net_debt

        revenue_growth = (
            self._stmt_growth_from_quarters(q_is, ["Total Revenue", "Revenue"])
            or self._stmt_growth_from_annual(a_is, ["Total Revenue", "Revenue"])
            or self._sec_annual_growth(
                company_facts,
                taxonomy="us-gaap",
                concepts=[
                    "RevenueFromContractWithCustomerExcludingAssessedTax",
                    "SalesRevenueNet",
                    "Revenues",
                ],
                unit="USD",
            )
            or 0.05
        )

        ebitda_margin = (ebitda_ltm / revenue_ltm) if revenue_ltm and revenue_ltm > 0 else 0.15
        beta = (
            safe_float(yahoo.get("beta"))
            or safe_float(optional_profile.get("beta"))
            or 1.0
        )

        snapshot = CompanySnapshot(
            symbol=symbol,
            name=name,
            sector=sector,
            industry=industry,
            currency=currency,
            price=float(price or 0.0),
            shares_outstanding=float(shares_outstanding or 0.0),
            market_cap=float(market_cap or 0.0),
            enterprise_value=float(enterprise_value or 0.0),
            total_debt=float(total_debt or 0.0),
            cash=float(cash or 0.0),
            net_debt=float(net_debt or 0.0),
            revenue_ltm=float(revenue_ltm or 0.0),
            ebitda_ltm=float(ebitda_ltm or 0.0),
            net_income_ltm=float(net_income_ltm or 0.0),
            revenue_growth=float(revenue_growth or 0.0),
            ebitda_margin=float(ebitda_margin or 0.0),
            beta=float(beta or 1.0),
        )

        self._snapshot_cache[symbol] = snapshot
        return snapshot
