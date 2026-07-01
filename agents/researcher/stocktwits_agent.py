import pandas as pd
import numpy as np
import ast
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Public S3 bucket for Li, Al Ansari & Kaufman (2025) StockTwits dataset.
# https://github.com/Jaxingjili/StockTwits-from-2008-to-2022 — no AWS credentials needed.
BUCKET = "stocktwits-nyu"
PREFIX = "dataset/v1/data/csv/symbol_sentiments/"

# Sample cap per shard. The dataset is ~550M messages across many shards;
# pulling everything isn't practical for a first pass. Taking the first N rows
# of every shard gives a time-spread sample since shards are chronological,
# same "subsample for prototyping" approach used in sentiment_agent.py.
ROWS_PER_SHARD = 150_000

# Match the FNSPID/FinBERT window so the two sentiment sources are directly comparable.
DATE_START = "2009-01-01"
DATE_END = "2020-12-31"

print("Loading existing FinBERT ticker universe (for apples-to-apples comparison)...")
finbert = pd.read_csv("data/processed/sentiment_scores.csv")
target_tickers = set(finbert["ticker"].unique())
print(f"Target tickers: {len(target_tickers)}")

print("Discovering available shards on S3 (public bucket, no credentials needed)...")
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
paginator = s3.get_paginator("list_objects_v2")
shard_keys = []
for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
    for obj in page.get("Contents", []):
        if obj["Key"].endswith(".csv"):
            shard_keys.append(obj["Key"])
print(f"Found {len(shard_keys)} shard files")

print(f"Pulling first {ROWS_PER_SHARD:,} rows from each shard...")
frames = []
for i, key in enumerate(shard_keys):
    url = f"s3://{BUCKET}/{key}"
    print(f"  [{i + 1}/{len(shard_keys)}] {key.split('/')[-1]}...")
    try:
        df = pd.read_csv(
            url,
            storage_options={"anon": True},
            dtype={"message_id": "object", "sentiment": "object"},
            usecols=["created_at", "sentiment", "symbol_list"],
            nrows=ROWS_PER_SHARD,
        )
        frames.append(df)
    except Exception as e:
        print(f"    skipped ({e})")

if not frames:
    raise RuntimeError("No shards could be read — check S3 access / bucket path.")

raw = pd.concat(frames, ignore_index=True)
print(f"Raw rows pulled: {len(raw)}")

print("Cleaning and filtering to date window + target tickers...")
raw["created_at"] = pd.to_datetime(raw["created_at"], utc=True, errors="coerce")
raw["sentiment"] = pd.to_numeric(raw["sentiment"], errors="coerce")
raw = raw.dropna(subset=["created_at", "sentiment", "symbol_list"])
raw = raw[(raw["created_at"] >= DATE_START) & (raw["created_at"] <= DATE_END)]

# symbol_list arrives as a stringified Python list, e.g. "['AAPL', 'GS']"
raw["symbol_list"] = raw["symbol_list"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else []
)
raw = raw.explode("symbol_list").rename(columns={"symbol_list": "ticker"})
raw = raw[raw["ticker"].isin(target_tickers)]
print(f"Rows after ticker filter: {len(raw)}")

raw["yyyymm"] = raw["created_at"].dt.strftime("%Y%m").astype(int)

print("Aggregating monthly bullish rate + volume per ticker...")
grouped = (
    raw.groupby(["ticker", "yyyymm"])
    .agg(
        bullish_rate=("sentiment", lambda s: (s > 0).mean()),
        avg_sentiment=("sentiment", "mean"),
        n_messages=("sentiment", "count"),
    )
    .reset_index()
)

os.makedirs("data/processed", exist_ok=True)
grouped.to_csv("data/processed/stocktwits_scores.csv", index=False)
print(f"\nDone! Saved {len(grouped)} ticker-month StockTwits sentiment rows")
print(grouped.head(10))
