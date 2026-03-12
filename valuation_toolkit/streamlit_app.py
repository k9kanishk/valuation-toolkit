from __future__ import annotations

from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from src.fundamentals import FundamentalsBuilder
from src.peer_selection import PeerSelector
from src.reporting import ReportBuilder
from src.valuation import ValuationEngine

st.set_page_config(page_title='Comparable Company + DCF Toolkit', layout='wide')
st.title('Comparable Company + DCF Valuation Toolkit')
st.caption(
    'Real-data U.S. public comps, automated peer selection, trading multiples, and DCF output pack. '
    'Free FMP keys may fall back to annual-only data where quarterly/peer endpoints are unavailable.'
)

with st.sidebar:
    st.header('Inputs')
    ticker = st.text_input('Target ticker', value='AAPL').upper().strip()
    max_peers = st.slider('Max peers', min_value=4, max_value=10, value=6)
    erp = st.number_input('Equity risk premium', min_value=0.02, max_value=0.08, value=0.045, step=0.0025, format='%.4f')
    terminal_growth = st.number_input('Terminal growth', min_value=0.01, max_value=0.04, value=0.025, step=0.0025, format='%.4f')
    run = st.button('Run valuation', type='primary')


def fmt_currency(x: float) -> str:
    if pd.isna(x):
        return 'n.a.'
    return f'${x:,.1f}'


if run:
    try:
        with st.spinner('Pulling live data and building valuation...'):
            builder = FundamentalsBuilder()
            target = builder.build_snapshot(ticker)
            peers = PeerSelector(builder).build_peer_set(target, max_peers=max_peers)
            try:
                risk_free_rate = builder.treasury.current_risk_free_rate("10 yr")
                st.caption(f"Using live 10Y Treasury: {risk_free_rate:.2%}")
            except Exception as exc:
                st.warning(
                    f"Could not fetch live Treasury rate ({exc}). Falling back to manual risk-free rate."
                )
                risk_free_rate = (
                    st.sidebar.number_input(
                        "Risk-free rate (%)",
                        min_value=0.0,
                        max_value=15.0,
                        value=4.25,
                        step=0.05,
                    )
                    / 100.0
                )
            output = ValuationEngine(
                risk_free_rate=risk_free_rate,
                equity_risk_premium=erp,
                terminal_growth=terminal_growth,
            ).run(target, peers)
            report_path = ReportBuilder().build_excel(output)

        comps_detail, comps_summary = output.comps_table

        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Share price', fmt_currency(target.price))
        col2.metric('Market cap', fmt_currency(target.market_cap / 1e9) + 'bn')
        col3.metric('Enterprise value', fmt_currency(target.enterprise_value / 1e9) + 'bn')
        col4.metric('10Y risk-free', f'{risk_free_rate:.2%}')

        st.subheader('Target snapshot')
        st.dataframe(pd.DataFrame([target.to_dict()]), use_container_width=True)

        st.subheader('Selected peers')
        st.dataframe(
            comps_detail[
                [
                    'symbol', 'name', 'sector', 'industry', 'market_cap', 'revenue_growth',
                    'ebitda_margin', 'similarity_score', 'selection_rationale', 'ev_revenue', 'ev_ebitda', 'pe'
                ]
            ],
            use_container_width=True,
        )

        st.subheader('Multiples summary')
        st.dataframe(comps_summary, use_container_width=True)

        chart_df = comps_detail[['symbol', 'ev_revenue', 'ev_ebitda', 'pe']].melt(id_vars='symbol', var_name='metric', value_name='multiple')
        fig = px.bar(chart_df, x='symbol', y='multiple', color='metric', barmode='group', title='Peer trading multiples')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Implied valuation range')
        st.dataframe(output.implied_values, use_container_width=True)
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
        left.dataframe(output.dcf_summary, use_container_width=True)
        right.plotly_chart(
            px.line(output.forecast, x='year', y=['revenue', 'ebitda', 'fcff'], markers=True, title='DCF forecast'),
            use_container_width=True,
        )

        st.subheader('WACC build')
        st.dataframe(output.wacc_summary, use_container_width=True)

        st.subheader('DCF sensitivity')
        st.dataframe(output.sensitivity, use_container_width=True)

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
else:
    st.info('Enter a U.S. ticker, keep the default assumptions for the first run, and click **Run valuation**.')
