# Rice Investment Fund — Project State
**Last updated:** July 1, 2026 (session 3)  
**Repo:** github.com/chego-1/riceinvestmentproject  
**Local path:** ~/Desktop/Coding/ai-fund  
**Python:** 3.11 (venv at ~/Desktop/Coding/ai-fund/venv)   

---

## Session Update Instructions
> At the end of every Claude Code session, update this file with:
> 1. What was changed or built
> 2. Any new blockers
> 3. Next steps
> Leave committing to the user — do not run `git commit` automatically.

---

## Project Description
A systematic, ML-driven long-short equity quant fund built in Python. Uses cross-sectional factor signals from OpenAssetPricing to predict stock returns via LightGBM, with NLP sentiment integration via FinBERT. Multi-agent architecture (researcher, portfolio manager, trader, auditor).

---

## What's Been Built

### Infrastructure
- Python 3.11 venv at `~/Desktop/Coding/ai-fund/venv`
- `.env` config with Finnhub + Anthropic API keys
- GitHub repo: `chego-1/riceinvestmentproject`
- `.gitignore` excludes `/data/` and large files (resolved 100MB GitHub limit)
- VS Code Python interpreter pointed at venv

### Activate venv (run every new terminal):
```bash
source ~/Desktop/Coding/ai-fund/venv/bin/activate
```

---

### Components

#### 1. News Agent — `agents/news_agent/news_agent.py`
- Pulls Finnhub headlines + summaries for a watchlist of tickers
- Uses Claude Sonnet (via Anthropic API) to summarize and generate structured HTML report
- **Updated (session 3):** AI summary now also incorporates watchlist-specific headlines (tagged by ticker), not just general market news; prompt explicitly asks Claude to call out watchlist news or note when there's none
- Run: `python agents/news_agent/news_agent.py`

#### 2. Researcher Agent — `agents/researcher/researcher_agent.py`
- Reads academic PDFs dropped into `papers/` folder
- Uses pdfplumber + Claude to extract and summarize content
- Run: `python agents/researcher/researcher_agent.py`

#### 3. LightGBM Model — `models/train_model.py`
- Trained on 30 signals from OpenAssetPricing (1986–1999 training period)
- **Updated (session 3):** added 6 factor-momentum features (1-month and 12-month trailing Mkt_RF, SMB, HML from Fama-French data) per the "Factor Momentum Everywhere" paper takeaway — 36 features total now
- **RMSE: 0.4007** (down from 0.4159 baseline). The 6 new factor-momentum features ranked #4-#9 out of 36 in feature importance, ahead of most original characteristics — a real improvement, not noise
- Saved model: `models/lgbm_model.pkl`
- Run: `python models/train_model.py`
- **Caveat:** `backtest.py` and `alpha_test.py` don't consume this model's predictions at all — they equal-weight the 30 raw OpenAssetPricing signal portfolios directly. This RMSE gain won't show up in the 231.9%/0.88 Sharpe/5.82% alpha numbers until a portfolio-construction step is built that actually ranks stocks by the model's output

#### 4. Backtest — `models/backtest.py`
- Uses pre-computed OpenAssetPricing portfolio returns
- Out-of-sample period: 2000–2020
- **Results: 231.9% total return, 0.88 Sharpe ratio**
- Run: `python models/backtest.py`

#### 5. CAPM Alpha Test — `models/alpha_test.py`
- Uses Ken French data library (risk-free rates + market returns)
- CAPM regression on long-short strategy with Newey-West (HAC, 6 lags) standard errors
- **Results: 5.82% annualized alpha, t-stat 4.70, p-value ~0.000 (significant at 1% level)**
- Beta: -0.22 (slightly negatively correlated with market — defensive)
- Run: `python models/alpha_test.py`

#### 6. FinBERT Sentiment Agent — `agents/researcher/sentiment_agent.py`
- Processed FNSPID financial news dataset
- Generated monthly sentiment scores for 100 tickers (2009–2020)
- Output: `data/processed/sentiment_scores.csv`
- Run: `python agents/researcher/sentiment_agent.py`

#### 7. StockTwits Sentiment Agent — `agents/researcher/stocktwits_agent.py` (NEW, session 3)
- Pulls from the public `stocktwits-nyu` S3 bucket (Li, Al Ansari & Kaufman 2025 dataset — [repo](https://github.com/Jaxingjili/StockTwits-from-2008-to-2022)), no AWS credentials needed
- Samples 150k rows/shard (full dataset is ~550M messages — sampling for prototyping, same pattern as `sentiment_agent.py`'s FNSPID subsample), filters to the same ticker universe + 2009–2020 window as FinBERT
- Aggregates monthly `bullish_rate` / `avg_sentiment` / `n_messages` per ticker (user-tagged bullish/bearish, not model-inferred)
- Output: `data/processed/stocktwits_scores.csv`
- Run: `python agents/researcher/stocktwits_agent.py`
- **Test run 1 (150k rows/shard):** 1,365 ticker-months, 91 tickers. Correlation with FinBERT sentiment on 923 overlapping ticker-months was ~0.02–0.12 — too noisy to read (median 3 messages/ticker-month).
- **Test run 2 (1M rows/shard):** 3,585 ticker-months, 92 tickers, 2,467 overlapping with FinBERT. Correlation stabilized at ~0.09–0.15 (peaks at n_messages≥10 cutoff, corr=0.151) — small but consistently positive across message-count cutoffs, no sign flips. Reading: StockTwits retail sentiment and FinBERT news-based sentiment are weakly related but largely distinct signals, not redundant — supports including both as separate model features rather than picking one.
- Like FinBERT sentiment, this is **not yet merged into the model** — same permno/ticker blocker applies. It's a second candidate sentiment feature waiting at the same merge step.

---

## Current Blockers

### 🔴 WRDS / CRSP Linking Table (MAIN BLOCKER)
- **Problem:** FinBERT sentiment scores use ticker symbols; OpenAssetPricing uses CRSP permno IDs — can't merge without a linking table
- **Why it matters:** Merging sentiment into the LightGBM model as an additional feature requires this mapping
- **Status:** UT Austin denied WRDS access (undergrad license restriction). Request sent to course supervisor — access agreed, pending receipt
- **Workaround considered:** yfinance / Simfin — rejected because they lack the depth of fundamental signals in OpenAssetPricing

### 🔴 Model-driven backtest also needs WRDS (found session 3)
- **Problem:** To actually rank individual stocks by the LightGBM model's predicted score and backtest that portfolio (rather than the naive equal-weight of the 30 raw signal portfolios `backtest.py` currently uses), we need real permno-level monthly returns
- **Why it matters:** Checked the installed `openassetpricing` package source (`openap_download.py`) — the only place it fetches raw per-stock returns (`ret` from `crsp.msf`) is `_dl_signal_crsp3()`, which opens a live `wrds.Connection()`. `dl_port('op', ...)` (what `backtest.py`/`alpha_test.py` use) only returns OpenAssetPricing's own pre-computed portfolio-level returns per signal, not raw stock-level returns re-rankable by the model's blended score
- **Status:** Same WRDS blocker as above — waiting on supervisor access, no separate workaround in progress

---

## Next Steps (Priority Order)
1. **Wait for supervisor reply** on WRDS/CRSP access — unblocks both the sentiment merge AND the model-driven backtest (real permno-level returns)
2. **Once linking table received:** merge sentiment scores (FinBERT + StockTwits) into signal data → retrain LightGBM with sentiment as features → re-run backtest and CAPM test → ablate to see if either/both sentiment sources actually improve Sharpe/alpha
3. **Once WRDS access received:** build the model-driven backtest — rank stocks monthly by LightGBM predicted score, form long-short portfolio from real permno-level returns, compare vs. the naive equal-weight `backtest.py` baseline
4. **Portfolio manager agent** — risk model, portfolio construction layer
5. **Trader agent** — execution logic
6. **Auditor agent** — performance monitoring and reporting
7. **Full multi-agent orchestration** — connect all agents into unified pipeline

---

## Key Architecture Decisions
- **LightGBM over neural nets** — compute efficiency, advisor recommendation
- **OpenAssetPricing** — 200+ pre-computed research-grade factors (1986–present), best free source
- **FinBERT over generic sentiment** — trained specifically on financial text
- **FNSPID dataset** — large financial news corpus used for sentiment processing
- **Claude Sonnet** — used for news summarization and report generation

---

## Environment & Dependencies
```
lightgbm
pandas
numpy
scikit-learn
pdfplumber
anthropic
python-dotenv
requests
finnhub-python
transformers  # for FinBERT
torch
boto3  # for StockTwits agent (public S3 access)
s3fs   # for StockTwits agent (pandas read_csv from S3)
```

---

## Known Debugging History
- Python 3.14 caused segfaults with key packages → downgraded to 3.11
- Broken venv paths after moving folder → rebuilt venv in place
- SSL certificate errors → resolved via pip cert config
- GitHub 100MB limit on data files → added `/data/` to `.gitignore`, rewrote history with `filter-branch`
- Remote URL was wrong after VS Code created new repo → fixed with `git remote set-url`

---