from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import BASE_DIR
from .utils import slugify
from .valuation import ValuationOutput


class ReportBuilder:
    def build_excel(self, output: ValuationOutput, output_dir: Path | None = None) -> Path:
        output_dir = output_dir or (BASE_DIR / "outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        file_path = output_dir / f"{slugify(output.target.symbol)}_valuation_pack_{stamp}.xlsx"

        peers, summary = output.comps_table
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            money_fmt = workbook.add_format({'num_format': '$#,##0'})
            money_1_fmt = workbook.add_format({'num_format': '$#,##0.0'})
            pct_fmt = workbook.add_format({'num_format': '0.0%'})
            x_fmt = workbook.add_format({'num_format': '0.0x'})
            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D9E2F3'})

            snapshot = pd.DataFrame([output.target.to_dict()])
            snapshot.to_excel(writer, sheet_name='Summary', index=False, startrow=0)
            implied_start = len(snapshot) + 3
            output.implied_values.to_excel(writer, sheet_name='Summary', index=False, startrow=implied_start)
            commentary_start = implied_start + len(output.implied_values) + 3
            pd.DataFrame({'commentary': output.commentary}).to_excel(
                writer,
                sheet_name='Summary',
                index=False,
                startrow=commentary_start,
            )

            peers.to_excel(writer, sheet_name='Peers', index=False)
            summary.to_excel(writer, sheet_name='Comps Summary', index=False)
            output.dcf_summary.to_excel(writer, sheet_name='DCF', index=False, startrow=0)
            output.wacc_summary.to_excel(writer, sheet_name='DCF', index=False, startrow=len(output.dcf_summary) + 3)
            output.forecast.to_excel(writer, sheet_name='Forecast', index=False)
            output.sensitivity.to_excel(writer, sheet_name='Sensitivity', index=False)

            for sheet_name in ['Summary', 'Peers', 'Comps Summary', 'DCF', 'Forecast', 'Sensitivity']:
                sheet = writer.sheets[sheet_name]
                sheet.set_row(0, None, header_fmt)
                sheet.freeze_panes(1, 0)
                sheet.set_column(0, 0, 18)
                sheet.set_column(1, 12, 16)

            summary_sheet = writer.sheets['Summary']
            peer_sheet = writer.sheets['Peers']
            comps_summary_sheet = writer.sheets['Comps Summary']

            peer_chart = workbook.add_chart({'type': 'column'})
            peer_chart.add_series(
                {
                    'name': 'EV / EBITDA',
                    'categories': ['Peers', 1, 0, len(peers), 0],
                    'values': ['Peers', 1, peers.columns.get_loc('ev_ebitda'), len(peers), peers.columns.get_loc('ev_ebitda')],
                }
            )
            peer_chart.set_title({'name': 'Peer EV / EBITDA'})
            peer_chart.set_y_axis({'num_format': '0.0x'})
            peer_sheet.insert_chart('N2', peer_chart)

            implied_chart = workbook.add_chart({'type': 'column'})
            implied_chart.add_series(
                {
                    'name': 'Implied price per share',
                    'categories': ['Summary', implied_start + 1, 0, implied_start + len(output.implied_values), 1],
                    'values': ['Summary', implied_start + 1, 5, implied_start + len(output.implied_values), 5],
                }
            )
            implied_chart.set_title({'name': 'Implied Price Range'})
            implied_chart.set_y_axis({'num_format': '$#,##0.0'})
            summary_sheet.insert_chart('J2', implied_chart)

            sensitivity_sheet = writer.sheets['Sensitivity']
            sensitivity_sheet.conditional_format(
                1,
                1,
                len(output.sensitivity),
                len(output.sensitivity.columns) - 1,
                {'type': '3_color_scale'},
            )

            # light formatting
            for ws in [summary_sheet, peer_sheet, comps_summary_sheet, writer.sheets['DCF'], writer.sheets['Forecast']]:
                ws.autofilter(0, 0, 200, 20)

        return file_path
