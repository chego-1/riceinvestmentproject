# Rice Investment Fund — Project State
**Last updated:** June 24, 2026  
**Repo:** github.com/chego-1/riceinvestmentproject  
**Local path:** ~/Desktop/Coding/ai-fund  
**Python:** 3.11 (venv at ~/Desktop/Coding/ai-fund/venv)  
**Supervisor:** Professor Back, Rice University  

---

## Session Update Instructions
> At the end of every Claude Code session, update this file with:
> 1. What was changed or built
> 2. Any new blockers
> 3. Next steps
> Then commit with message: `update CLAUDE.md — [date]`

---

## Project Description
A systematic, ML-driven long-short equity quant fund built in Python. Uses cross-sectional factor signals from OpenAssetPricing to predict stock returns via LightGBM, with NLP sentiment integration via FinBERT. Multi-agent architecture (researcher, portfolio manager, trader, auditor). Assigned by Professor Back at Rice University.

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
- Run: `python agents/news_agent/news_agent.py`

#### 2. Researcher Agent — `agents/researcher/researcher_agent.py`
- Reads academic PDFs dropped into `papers/` folder
- Uses pdfplumber + Claude to extract and summarize content
- Run: `python agents/researcher/researcher_agent.py`

#### 3. LightGBM Model — `models/train_model.py`
- Trained on 30 signals from OpenAssetPricing (1986–1999 training period)
- RMSE: 0.4159
- Saved model: `models/lgbm_model.pkl`
- Run: `python models/train_model.py`

#### 4. Backtest — `models/backtest.py`
- Uses pre-computed OpenAssetPricing portfolio returns
- Out-of-sample period: 2000–2020
- **Results: 156% total return, 0.52 Sharpe ratio**
- Run: `python models/backtest.py`

#### 5. CAPM Alpha Test — `models/alpha_test.py`
- Uses Ken French data library (risk-free rates + market returns)
- CAPM regression on long-short strategy
- **Results: 5.83% annualized alpha, t-stat 3.04, p-value 0.003 (significant at 1% level)**
- Beta: -0.29 (slightly negatively correlated with market — defensive)
- Run: `python models/alpha_test.py`

#### 6. FinBERT Sentiment Agent — `agents/researcher/sentiment_agent.py`
- Processed FNSPID financial news dataset
- Generated monthly sentiment scores for 100 tickers (2009–2020)
- Output: `data/processed/sentiment_scores.csv`
- Run: `python agents/researcher/sentiment_agent.py`

---

## Current Blockers

### 🔴 WRDS / CRSP Linking Table (MAIN BLOCKER)
- **Problem:** FinBERT sentiment scores use ticker symbols; OpenAssetPricing uses CRSP permno IDs — can't merge without a linking table
- **Why it matters:** Merging sentiment into the LightGBM model as an additional feature requires this mapping
- **Status:** UT Austin denied WRDS access (undergrad license restriction). Email sent to Professor Back requesting either Rice WRDS access or the linking table directly
- **Workaround considered:** yfinance / Simfin — rejected because they lack the depth of fundamental signals in OpenAssetPricing

---

## Next Steps (Priority Order)
1. **Wait for Professor Back's reply** on WRDS/CRSP access
2. **Once linking table received:** merge sentiment scores into signal data → retrain LightGBM with sentiment as feature → re-run backtest and CAPM test
3. **Portfolio manager agent** — risk model, portfolio construction layer
4. **Trader agent** — execution logic
5. **Auditor agent** — performance monitoring and reporting
6. **Full multi-agent orchestration** — connect all agents into unified pipeline

---

## Key Architecture Decisions
- **LightGBM over neural nets** — compute efficiency, professor's recommendation
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
```

---

## Known Debugging History
- Python 3.14 caused segfaults with key packages → downgraded to 3.11
- Broken venv paths after moving folder → rebuilt venv in place
- SSL certificate errors → resolved via pip cert config
- GitHub 100MB limit on data files → added `/data/` to `.gitignore`, rewrote history with `filter-branch`
- Remote URL was wrong after VS Code created new repo → fixed with `git remote set-url`

---

## Contact
- **Professor Back** — Rice University (supervising)
- **Emil Banchs** — UT Austin, freshman, Economics + Stats & Data Science minor
- **GitHub:** chego-1
