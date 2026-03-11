from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .fundamentals import CompanySnapshot
from .utils import safe_div, weighted_average, winsorize_series


@dataclass
class ValuationOutput:
    target: CompanySnapshot
    peers: pd.DataFrame
    comps_table: tuple[pd.DataFrame, pd.DataFrame]
    implied_values: pd.DataFrame
    dcf_summary: pd.DataFrame
    forecast: pd.DataFrame
    wacc_summary: pd.DataFrame
    sensitivity: pd.DataFrame
    commentary: list[str]


class ValuationEngine:
    def __init__(self, risk_free_rate: float, equity_risk_premium: float = 0.045, terminal_growth: float = 0.025):
        self.risk_free_rate = risk_free_rate
        self.equity_risk_premium = equity_risk_premium
        self.terminal_growth = terminal_growth

    def run(self, target: CompanySnapshot, peers: pd.DataFrame) -> ValuationOutput:
        comps_table = self._build_comps_table(target, peers)
        implied_values = self._build_implied_valuation(target, comps_table)
        dcf_summary, forecast, wacc_summary, sensitivity = self._run_dcf(target, peers)
        commentary = self._commentary(target, comps_table, implied_values, dcf_summary)
        return ValuationOutput(
            target=target,
            peers=peers,
            comps_table=comps_table,
            implied_values=implied_values,
            dcf_summary=dcf_summary,
            forecast=forecast,
            wacc_summary=wacc_summary,
            sensitivity=sensitivity,
            commentary=commentary,
        )

    def _build_comps_table(self, target: CompanySnapshot, peers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        comps = peers.copy()
        comps['ev_revenue'] = safe_div_series(comps['enterprise_value'], comps['revenue_ltm'])
        comps['ev_ebitda'] = safe_div_series(comps['enterprise_value'], comps['ebitda_ltm'])
        comps['pe'] = safe_div_series(comps['market_cap'], comps['net_income_ltm'])

        for col in ('ev_revenue', 'ev_ebitda', 'pe'):
            comps.loc[~np.isfinite(comps[col]), col] = np.nan
            comps[col] = winsorize_series(comps[col])

        summary_rows = []
        metrics = {
            'EV / Revenue': 'ev_revenue',
            'EV / EBITDA': 'ev_ebitda',
            'P / E': 'pe',
        }
        weights = comps['similarity_score']

        for label, col in metrics.items():
            series = comps[col].replace([np.inf, -np.inf], np.nan).dropna()
            row = {
                'metric': label,
                'min': series.min() if not series.empty else np.nan,
                'p25': series.quantile(0.25) if not series.empty else np.nan,
                'median': series.median() if not series.empty else np.nan,
                'p75': series.quantile(0.75) if not series.empty else np.nan,
                'max': series.max() if not series.empty else np.nan,
                'weighted_mean': weighted_average(comps[col], weights),
            }
            summary_rows.append(row)

        summary = pd.DataFrame(summary_rows)
        return comps, summary

    def _build_implied_valuation(self, target: CompanySnapshot, comps_pack: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
        _, summary = comps_pack
        rows: list[dict[str, Any]] = []

        metric_to_fundamental = {
            'EV / Revenue': ('revenue_ltm', 'enterprise'),
            'EV / EBITDA': ('ebitda_ltm', 'enterprise'),
            'P / E': ('net_income_ltm', 'equity'),
        }

        for _, row in summary.iterrows():
            metric = row['metric']
            base_metric_name, value_type = metric_to_fundamental[metric]
            base_value = getattr(target, base_metric_name)
            if pd.isna(base_value) or base_value <= 0:
                continue

            for stat in ('p25', 'median', 'p75', 'weighted_mean'):
                multiple = row[stat]
                if pd.isna(multiple):
                    continue
                if value_type == 'enterprise':
                    implied_ev = multiple * base_value
                    implied_equity = implied_ev - target.net_debt
                else:
                    implied_equity = multiple * base_value
                    implied_ev = implied_equity + target.net_debt
                implied_price = safe_div(implied_equity, target.shares_out)
                rows.append(
                    {
                        'method': metric,
                        'statistic': stat,
                        'multiple': multiple,
                        'implied_enterprise_value': implied_ev,
                        'implied_equity_value': implied_equity,
                        'implied_price_per_share': implied_price,
                    }
                )

        return pd.DataFrame(rows)

    def _run_dcf(
        self,
        target: CompanySnapshot,
        peers: pd.DataFrame,
        years: int = 5,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        peer_margin = peers['ebitda_margin'].median(skipna=True)
        target_growth = target.revenue_growth if pd.notna(target.revenue_growth) else 0.06
        base_growth = float(np.clip(target_growth, -0.05, 0.15))
        end_growth = self.terminal_growth + 0.01

        base_margin = target.ebitda_margin if pd.notna(target.ebitda_margin) else 0.20
        terminal_margin = float(np.nanmedian([base_margin, peer_margin if pd.notna(peer_margin) else base_margin]))

        cost_of_equity = self.risk_free_rate + target.beta * self.equity_risk_premium
        debt_weight = safe_div(target.total_debt, target.total_debt + target.market_cap, default=0.15)
        equity_weight = 1 - debt_weight
        pre_tax_cost_of_debt = safe_div(target.interest_expense, max(target.total_debt, 1), default=self.risk_free_rate + 0.015)
        if pd.isna(pre_tax_cost_of_debt) or pre_tax_cost_of_debt <= 0:
            pre_tax_cost_of_debt = self.risk_free_rate + 0.015
        after_tax_cost_of_debt = pre_tax_cost_of_debt * (1 - target.tax_rate)
        wacc = equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt
        wacc = max(wacc, self.terminal_growth + 0.01)

        forecast_rows: list[dict[str, Any]] = []
        revenue_prev = target.revenue_ltm
        for year in range(1, years + 1):
            fade = year / years
            growth = base_growth + fade * (end_growth - base_growth)
            margin = base_margin + fade * (terminal_margin - base_margin)
            revenue = revenue_prev * (1 + growth)
            ebitda = revenue * margin
            da = revenue * abs(target.da_pct_sales)
            ebit = ebitda - da
            tax = max(ebit, 0) * target.tax_rate
            capex = revenue * abs(target.capex_pct_sales)
            delta_nwc = max(revenue - revenue_prev, 0) * abs(target.nwc_pct_sales)
            fcff = ebit * (1 - target.tax_rate) + da - capex - delta_nwc
            discount_factor = 1 / ((1 + wacc) ** year)
            pv_fcff = fcff * discount_factor
            forecast_rows.append(
                {
                    'year': year,
                    'revenue': revenue,
                    'growth': growth,
                    'ebitda_margin': margin,
                    'ebitda': ebitda,
                    'da': da,
                    'ebit': ebit,
                    'tax': tax,
                    'capex': capex,
                    'delta_nwc': delta_nwc,
                    'fcff': fcff,
                    'discount_factor': discount_factor,
                    'pv_fcff': pv_fcff,
                }
            )
            revenue_prev = revenue

        forecast = pd.DataFrame(forecast_rows)
        terminal_fcf = forecast.iloc[-1]['fcff'] * (1 + self.terminal_growth)
        terminal_value = terminal_fcf / (wacc - self.terminal_growth)
        terminal_pv = terminal_value / ((1 + wacc) ** years)
        enterprise_value = forecast['pv_fcff'].sum() + terminal_pv
        equity_value = enterprise_value - target.net_debt
        price_per_share = safe_div(equity_value, target.shares_out)

        dcf_summary = pd.DataFrame(
            [
                {'item': 'WACC', 'value': wacc},
                {'item': 'Terminal growth', 'value': self.terminal_growth},
                {'item': 'PV of explicit FCFF', 'value': forecast['pv_fcff'].sum()},
                {'item': 'PV of terminal value', 'value': terminal_pv},
                {'item': 'Enterprise value', 'value': enterprise_value},
                {'item': 'Net debt', 'value': target.net_debt},
                {'item': 'Equity value', 'value': equity_value},
                {'item': 'Value per share', 'value': price_per_share},
            ]
        )

        wacc_summary = pd.DataFrame(
            [
                {'component': 'Risk-free rate', 'value': self.risk_free_rate},
                {'component': 'Beta', 'value': target.beta},
                {'component': 'Equity risk premium', 'value': self.equity_risk_premium},
                {'component': 'Cost of equity', 'value': cost_of_equity},
                {'component': 'Pre-tax cost of debt', 'value': pre_tax_cost_of_debt},
                {'component': 'After-tax cost of debt', 'value': after_tax_cost_of_debt},
                {'component': 'Equity weight', 'value': equity_weight},
                {'component': 'Debt weight', 'value': debt_weight},
                {'component': 'WACC', 'value': wacc},
            ]
        )

        sensitivity = self._build_sensitivity(target, forecast, wacc)
        return dcf_summary, forecast, wacc_summary, sensitivity

    def _build_sensitivity(self, target: CompanySnapshot, forecast: pd.DataFrame, base_wacc: float) -> pd.DataFrame:
        wacc_range = [base_wacc - 0.01, base_wacc - 0.005, base_wacc, base_wacc + 0.005, base_wacc + 0.01]
        tg_range = [self.terminal_growth - 0.01, self.terminal_growth - 0.005, self.terminal_growth, self.terminal_growth + 0.005, self.terminal_growth + 0.01]

        rows = []
        last_fcff = forecast.iloc[-1]['fcff']
        explicit_pv = forecast['pv_fcff'].sum()
        last_year = int(forecast.iloc[-1]['year'])

        for tg in tg_range:
            row = {'terminal_growth': tg}
            for wacc in wacc_range:
                if wacc <= tg:
                    row[f'{wacc:.3%}'] = np.nan
                    continue
                tv = last_fcff * (1 + tg) / (wacc - tg)
                pv_tv = tv / ((1 + wacc) ** last_year)
                ev = explicit_pv + pv_tv
                eq = ev - target.net_debt
                row[f'{wacc:.3%}'] = safe_div(eq, target.shares_out)
            rows.append(row)
        return pd.DataFrame(rows)

    def _commentary(
        self,
        target: CompanySnapshot,
        comps_pack: tuple[pd.DataFrame, pd.DataFrame],
        implied_values: pd.DataFrame,
        dcf_summary: pd.DataFrame,
    ) -> list[str]:
        comps, summary = comps_pack
        median_ebitda = summary.loc[summary['metric'] == 'EV / EBITDA', 'median']
        median_ebitda_multiple = median_ebitda.iloc[0] if not median_ebitda.empty else np.nan
        target_multiple = safe_div(target.enterprise_value, target.ebitda_ltm)
        dcf_price = dcf_summary.loc[dcf_summary['item'] == 'Value per share', 'value'].iloc[0]
        comps_median_price = implied_values.loc[
            (implied_values['method'] == 'EV / EBITDA') & (implied_values['statistic'] == 'median'),
            'implied_price_per_share',
        ]
        comps_median_price_value = comps_median_price.iloc[0] if not comps_median_price.empty else np.nan

        commentary = []
        if pd.notna(target_multiple) and pd.notna(median_ebitda_multiple):
            if target_multiple > median_ebitda_multiple:
                commentary.append(
                    f'{target.symbol} screens at a premium to the peer median on EV/EBITDA '
                    f'({target_multiple:.1f}x vs {median_ebitda_multiple:.1f}x).'
                )
            else:
                commentary.append(
                    f'{target.symbol} screens at a discount to the peer median on EV/EBITDA '
                    f'({target_multiple:.1f}x vs {median_ebitda_multiple:.1f}x).'
                )

        if pd.notna(dcf_price) and pd.notna(comps_median_price_value) and comps_median_price_value != 0:
            diff = safe_div(dcf_price - comps_median_price_value, comps_median_price_value)
            direction = 'above' if diff >= 0 else 'below'
            commentary.append(
                f'The DCF value is {abs(diff) * 100:.1f}% {direction} the median EV/EBITDA-derived value, '
                'mainly reflecting margin and terminal assumptions.'
            )

        if not comps.empty:
            top_peer = comps.sort_values('similarity_score', ascending=False).iloc[0]
            commentary.append(
                f"The highest-ranked comp is {top_peer['symbol']} because it is {top_peer['selection_rationale']}."
            )
        commentary.append(
            'Use the weighted multiple outputs as the primary market-based indication when peer dispersion is wide.'
        )
        return commentary


def safe_div_series(num: pd.Series, den: pd.Series) -> pd.Series:
    result = num.astype(float) / den.astype(float)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result
