from __future__ import annotations

from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from src.fundamentals import FundamentalsBuilder
from src.peer_selection import PeerSelector
from src.reporting import ReportBuilder
from src.valuation import ValuationEngine

DEFAULT_TICKER = "AAPL"

st.set_page_config(page_title="Comparable Company + DCF Valuation Toolkit", layout="wide")
st.title("Comparable Company + DCF Valuation Toolkit")
st.caption(
    "SEC-backed U.S. public comps + DCF toolkit. "
    "Treasury provides the risk-free rate, yfinance provides market convenience fields, "
    "and an optional provider hook can be enabled later."
)

with st.sidebar:
    st.header("Inputs")
    ticker_input = st.text_input("Target ticker", value=DEFAULT_TICKER).upper().strip()
    max_peers = st.slider("Max peers", min_value=4, max_value=8, value=5)
    manual_peers = st.text_input("Manual peer override (comma-separated)", value="")
    equity_risk_premium = st.number_input("Equity risk premium", value=0.0450, step=0.0050, format="%.4f")
    terminal_growth = st.number_input("Terminal growth", value=0.0250, step=0.0025, format="%.4f")
    use_optional_provider = st.checkbox("Use optional provider", value=False)
    run_clicked = st.button("Run valuation", type="primary")


if "run_inputs" not in st.session_state:
    st.session_state.run_inputs = None

if run_clicked:
    st.session_state.run_inputs = {
        "ticker": ticker_input,
        "max_peers": max_peers,
        "manual_peers": manual_peers,
        "equity_risk_premium": equity_risk_premium,
        "terminal_growth": terminal_growth,
        "use_optional_provider": use_optional_provider,
    }

if st.session_state.run_inputs is None:
    st.info("Set inputs and click Run valuation.")
    st.stop()


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def run_model_cached(ticker: str, max_peers: int, use_optional_provider: bool):
    builder = FundamentalsBuilder(use_optional_provider=use_optional_provider)
    target = builder.build_snapshot(ticker)
    peers = PeerSelector(builder).build_peer_set(target, max_peers=max_peers)
    return builder, target, peers


def fmt_bn(x: float) -> str:
    if pd.isna(x):
        return '-'
    return f'${x / 1e9:,.1f}bn'


def fmt_mm(x: float) -> str:
    if pd.isna(x):
        return '-'
    return f'${x / 1e6:,.1f}m'


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return '-'
    return f'{x:.1%}'


def fmt_px(x: float) -> str:
    if pd.isna(x):
        return '-'
    return f'${x:,.2f}'


inputs = st.session_state.run_inputs

try:
    with st.spinner('Pulling live data and building valuation...'):
        builder, target, peers = run_model_cached(
            ticker=inputs["ticker"],
            max_peers=inputs["max_peers"],
            use_optional_provider=inputs["use_optional_provider"],
        )

        if inputs["manual_peers"].strip():
            manual_symbols = [x.strip().upper() for x in inputs["manual_peers"].split(",") if x.strip()]
            peer_snapshots = []
            for symbol in manual_symbols:
                try:
                    peer_snapshots.append(builder.build_snapshot(symbol))
                except Exception:
                    pass
            if peer_snapshots:
                peers = pd.DataFrame([peer.to_dict() for peer in peer_snapshots])
        try:
            risk_free_rate = builder.treasury.current_risk_free_rate('10 yr')
            st.caption(f'Using live 10Y Treasury: {risk_free_rate:.2%}')
        except Exception as exc:
            st.warning(
                f'Could not fetch live Treasury rate ({exc}). Falling back to manual risk-free rate.'
            )
            risk_free_rate = (
                st.sidebar.number_input(
                    'Risk-free rate (%)',
                    min_value=0.0,
                    max_value=15.0,
                    value=4.25,
                    step=0.05,
                )
                / 100.0
            )
        output = ValuationEngine(
            risk_free_rate=risk_free_rate,
            equity_risk_premium=inputs['equity_risk_premium'],
            terminal_growth=inputs['terminal_growth'],
        ).run(target, peers)
        report_path = ReportBuilder().build_excel(output)

    comps_detail, comps_summary = output.comps_table

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Share price', fmt_px(target.price))
    col2.metric('Market cap', fmt_bn(target.market_cap))
    col3.metric('Enterprise value', fmt_bn(target.enterprise_value))
    col4.metric('10Y risk-free', f'{risk_free_rate:.2%}')

    st.subheader('Target snapshot')
    target_df = pd.DataFrame([target.to_dict()])
    st.dataframe(
        target_df.style.format(
            {
                'price': fmt_px,
                'shares_outstanding': '{:,.0f}',
                'market_cap': fmt_bn,
                'enterprise_value': fmt_bn,
                'total_debt': fmt_bn,
                'cash': fmt_bn,
                'net_debt': fmt_bn,
                'revenue_ltm': fmt_bn,
                'ebitda_ltm': fmt_bn,
                'net_income_ltm': fmt_bn,
                'revenue_growth': fmt_pct,
                'ebitda_margin': fmt_pct,
            }
        ),
        use_container_width=True,
    )

    st.subheader('Selected peers')
    peer_view = comps_detail[
        [
            'symbol', 'name', 'sector', 'industry', 'market_cap', 'revenue_growth',
            'ebitda_margin', 'similarity_score', 'selection_rationale', 'ev_revenue', 'ev_ebitda', 'pe'
        ]
    ]
    st.dataframe(
        peer_view.style.format(
            {
                'market_cap': fmt_bn,
                'revenue_growth': fmt_pct,
                'ebitda_margin': fmt_pct,
                'similarity_score': '{:.2f}',
                'ev_revenue': '{:.2f}x',
                'ev_ebitda': '{:.2f}x',
                'pe': '{:.2f}x',
            }
        ),
        use_container_width=True,
    )

    st.subheader('Multiples summary')
    st.dataframe(comps_summary.style.format({'min': '{:.2f}x', 'p25': '{:.2f}x', 'median': '{:.2f}x', 'p75': '{:.2f}x', 'max': '{:.2f}x', 'weighted_mean': '{:.2f}x'}), use_container_width=True)

    chart_df = comps_detail[['symbol', 'ev_revenue', 'ev_ebitda', 'pe']].melt(id_vars='symbol', var_name='metric', value_name='multiple')
    fig = px.bar(chart_df, x='symbol', y='multiple', color='metric', barmode='group', title='Peer trading multiples')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('Implied valuation range')
    st.dataframe(
        output.implied_values.style.format(
            {
                'multiple': '{:.2f}x',
                'implied_enterprise_value': fmt_bn,
                'implied_equity_value': fmt_bn,
                'implied_price_per_share': fmt_px,
            }
        ),
        use_container_width=True,
    )
    implied_fig = px.bar(
        output.implied_values,
        x='statistic',
        y='implied_price_per_share',
        color='method',
        barmode='group',
        title='Implied price per share by method',
    )
    st.plotly_chart(implied_fig, use_container_width=True)

    st.subheader('DCF summary')
    left, right = st.columns([1, 2])
    dcf_display = output.dcf_summary.copy()
    percent_items = {'WACC', 'Terminal growth'}
    per_share_items = {'Value per share'}
    dcf_display['value'] = dcf_display.apply(
        lambda r: fmt_pct(r['value']) if r['item'] in percent_items else (fmt_px(r['value']) if r['item'] in per_share_items else fmt_bn(r['value'])),
        axis=1,
    )
    left.dataframe(dcf_display, use_container_width=True)
    right.plotly_chart(
        px.line(output.forecast, x='year', y=['revenue', 'ebitda', 'fcff'], markers=True, title='DCF forecast'),
        use_container_width=True,
    )

    st.subheader('WACC build')
    st.dataframe(output.wacc_summary.style.format({'value': fmt_pct}), use_container_width=True)

    st.subheader('DCF sensitivity')
    st.dataframe(output.sensitivity.style.format({col: fmt_px for col in output.sensitivity.columns if col != 'terminal_growth'} | {'terminal_growth': fmt_pct}), use_container_width=True)

    st.subheader('Summary commentary')
    for bullet in output.commentary:
        st.write(f'- {bullet}')

    with open(report_path, 'rb') as f:
        st.download_button(
            label='Download Excel valuation pack',
            data=BytesIO(f.read()).getvalue(),
            file_name=report_path.name,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )

except Exception as exc:
    st.error(str(exc))
    st.exception(exc)
