import finnhub
import os
import anthropic
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

WATCHLIST = ["AAPL", "NVDA", "MSFT", "TSLA", "TSMC", "ASML"]

def get_general_news():
    news = finnhub_client.general_news('general', min_id=0)
    return news[:10]

def get_stock_news(ticker):
    today = datetime.now().strftime("%Y-%m-%d")
    news = finnhub_client.company_news(ticker, _from=today, to=today)
    return news[:3]

def get_ai_summary(headlines):
    headline_text = "\n".join([h['headline'] for h in headlines])
    
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"Based on these market headlines, write a 2-3 sentence summary of what's happening in markets today. Be concise and factual.\n\n{headline_text}"
        }]
    )
    return message.content[0].text

def build_html(general_news, stock_news, ai_summary):
    date = datetime.now().strftime("%B %d, %Y")

    def make_cards(news):
        cards = ""
        for item in news:
            cards += f"""
            <div class="card">
                <h2><a href="{item['url']}" target="_blank">{item['headline']}</a></h2>
                <p>{item['summary']}</p>
                <span class="source">{item['source']} — {datetime.fromtimestamp(item['datetime']).strftime("%I:%M %p")}</span>
            </div>
            """
        return cards

    stock_sections = ""
    for ticker, news in stock_news.items():
        if news:
            stock_sections += f"<h2>🏢 {ticker}</h2>"
            stock_sections += make_cards(news)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Market News — {date}</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; background: #f5f5f5; }}
            h1 {{ color: #1a1a1a; }}
            h2 {{ color: #333; margin-top: 30px; }}
            .summary {{ background: #1a1a2e; color: #e0e0e0; padding: 20px; border-radius: 8px; margin-bottom: 30px; font-size: 15px; line-height: 1.6; }}
            .summary span {{ color: #4cc9f0; font-weight: bold; }}
            .card {{ background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .card h2 {{ font-size: 16px; margin: 0 0 10px 0; border: none; }}
            .card h2 a {{ color: #0066cc; text-decoration: none; }}
            .card p {{ color: #444; font-size: 14px; margin: 0 0 10px 0; }}
            .source {{ color: #999; font-size: 12px; }}
        </style>
    </head>
    <body>
        <h1>📈 Market News — {date}</h1>
        <div class="summary"><span>AI Summary:</span> {ai_summary}</div>

        <h2>🌍 General Market News</h2>
        {make_cards(general_news)}

        <h2>📊 Watchlist News</h2>
        {stock_sections}
    </body>
    </html>
    """
    return html

def run():
    print("Fetching general news...")
    general_news = get_general_news()

    print("Fetching stock news...")
    stock_news = {}
    for ticker in WATCHLIST:
        print(f"  {ticker}...")
        stock_news[ticker] = get_stock_news(ticker)

    print("Generating AI summary...")
    ai_summary = get_ai_summary(general_news)

    html = build_html(general_news, stock_news, ai_summary)

    filename = f"reports/market_news_{datetime.now().strftime('%Y-%m-%d')}.html"
    with open(filename, "w") as f:
        f.write(html)

    print(f"Report saved: {filename}")

run()