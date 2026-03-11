# Comparable Company + DCF Valuation Toolkit

This project gives you a working end-to-end valuation workflow for U.S.-listed companies:

- live company/fundamental data pull,
- automated peer candidate generation,
- peer similarity scoring,
- trading multiples valuation,
- DCF valuation,
- downloadable Excel output pack,
- Streamlit front end for demo/interview use.

## What this version does

### Data sources
- **FMP** for company profiles, peer seeds, statements, ratios, quotes, and enterprise values.
- **U.S. Treasury** for the live risk-free rate.
- **SEC** for ticker-to-CIK mapping and optional company-facts extension.

### Valuation logic
- Peer selection starts from FMP peer suggestions and a same-sector/size screen.
- Peers are scored on:
  - industry / sector match,
  - size,
  - revenue growth,
  - EBITDA margin,
  - leverage.
- Trading multiples:
  - EV / Revenue,
  - EV / EBITDA,
  - P / E.
- DCF includes:
  - 5-year revenue forecast,
  - margin fade,
  - FCFF build,
  - WACC,
  - Gordon-growth terminal value,
  - sensitivity table.

## Exactly what you need to do

### 1) Install Python
Use **Python 3.11 or 3.12**.

### 2) Create an FMP API key
Create an account with Financial Modeling Prep and get an API key.

### 3) Clone or unzip the project
Put the project in a folder, then open a terminal inside that folder.

### 4) Create a virtual environment
On macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 5) Install packages
```bash
pip install -r requirements.txt
```

### 6) Create your environment file
Copy `.env.example` to `.env`.

macOS / Linux:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
copy .env.example .env
```

Then edit `.env` and set:

```env
FMP_API_KEY=your_real_key_here
SEC_USER_AGENT=Your Name your_email@example.com
```

### 7) Run the app
```bash
streamlit run streamlit_app.py
```

### 8) First test tickers
Use these first because they are liquid and have clean public data:
- AAPL
- MSFT
- ADBE
- CRM
- INTU
- AMZN

Avoid banks, insurers, REITs, and weird microcaps for your first demo.

## Repo structure

```text
valuation_toolkit/
  data/
    cache/
  outputs/
  src/
    config.py
    data_clients.py
    fundamentals.py
    peer_selection.py
    reporting.py
    utils.py
    valuation.py
  .env.example
  requirements.txt
  README.md
  streamlit_app.py
```

## What to improve next

### High priority
1. Add manual peer override in the UI.
2. Add country/sector exclusions for financials and REITs.
3. Add football-field chart in Excel and Streamlit.
4. Add exit-multiple terminal value in addition to Gordon growth.
5. Add SEC company-facts fallback when FMP fields are missing.

### Good interview upgrades
1. Save peer selection audit trail.
2. Add precedent transactions module.
3. Add scenario cases: bear / base / bull.
4. Add PDF export layer.
5. Add unit tests around DCF math and comp filters.

## Notes
- This is a **strong v1**, not a perfect institutional platform.
- The project is intentionally scoped to **U.S. public companies first**.
- Some API fields vary by company, so the code uses fallback logic across similar field names.
