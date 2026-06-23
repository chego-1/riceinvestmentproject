import pandas as pd
import numpy as np
from transformers import pipeline
import openassetpricing as oap
import os
import json
from datetime import datetime

print("Loading sentiment model...")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    truncation=True,
    max_length=512
)

print("Loading FNSPID news data...")
news = pd.read_csv('data/raw/fnspid_news.csv', 
                   usecols=['Date', 'Article_title', 'Stock_symbol', 'Lsa_summary'],
                   nrows=500000)

news['Date'] = pd.to_datetime(news['Date'], utc=True)
news['yyyymm'] = news['Date'].dt.strftime('%Y%m').astype(int)
news = news[(news['yyyymm'] >= 200001) & (news['yyyymm'] <= 202012)]
news = news.dropna(subset=['Article_title'])

print(f"News rows after filter: {len(news)}")

print("Getting overlapping tickers with OpenAssetPricing...")
openap = oap.OpenAP()
signal_data = openap.dl_signal('pandas', ['BM', 'Mom12m'])
valid_permnos = signal_data['permno'].unique()

news_tickers = news['Stock_symbol'].unique()
print(f"News tickers: {len(news_tickers)}")

overlapping = [t for t in news_tickers if t in news['Stock_symbol'].values]
print(f"Processing {len(overlapping)} tickers")

news['text'] = news['Article_title'].fillna('') + '. ' + news['Lsa_summary'].fillna('')
news['text'] = news['text'].str[:512]

print("Scoring sentiment...")
results = []
tickers_to_process = news['Stock_symbol'].unique()[:100]

for i, ticker in enumerate(tickers_to_process):
    ticker_news = news[news['Stock_symbol'] == ticker]
    
    monthly = ticker_news.groupby('yyyymm')
    
    for month, group in monthly:
        texts = group['text'].tolist()[:10]
        
        try:
            scores = sentiment_model(texts)
            
            sentiment_scores = []
            for s in scores:
                if s['label'] == 'positive':
                    sentiment_scores.append(s['score'])
                elif s['label'] == 'negative':
                    sentiment_scores.append(-s['score'])
                else:
                    sentiment_scores.append(0)
            
            avg_sentiment = np.mean(sentiment_scores)
            
            results.append({
                'ticker': ticker,
                'yyyymm': month,
                'sentiment': avg_sentiment,
                'n_articles': len(texts)
            })
        except Exception as e:
            continue
    
    if i % 10 == 0:
        print(f"  Processed {i}/{len(tickers_to_process)} tickers...")

sentiment_df = pd.DataFrame(results)
sentiment_df.to_csv('data/processed/sentiment_scores.csv', index=False)
print(f"\nDone! Saved {len(sentiment_df)} sentiment scores")
print(sentiment_df.head(10))