import finnhub
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))

def get_news():
    news = client.general_news('general', min_id=0)
    return news[:20]  # top 20 stories

def build_html(news):
    date = datetime.now().strftime("%B %d, %Y")
    
    cards = ""
    for item in news:
        cards += f"""
        <div class="card">
            <h2><a href="{item['url']}" target="_blank">{item['headline']}</a></h2>
            <p>{item['summary']}</p>
            <span class="source">{item['source']} — {datetime.fromtimestamp(item['datetime']).strftime("%I:%M %p")}</span>
        </div>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Market News — {date}</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; background: #f5f5f5; }}
            h1 {{ color: #1a1a1a; }}
            .card {{ background: white; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .card h2 {{ font-size: 16px; margin: 0 0 10px 0; }}
            .card h2 a {{ color: #0066cc; text-decoration: none; }}
            .card p {{ color: #444; font-size: 14px; margin: 0 0 10px 0; }}
            .source {{ color: #999; font-size: 12px; }}
        </style>
    </head>
    <body>
        <h1>📈 Market News — {date}</h1>
        {cards}
    </body>
    </html>
    """
    return html

def run():
    print("Fetching news...")
    news = get_news()
    html = build_html(news)
    
    filename = f"reports/market_news_{datetime.now().strftime('%Y-%m-%d')}.html"
    with open(filename, "w") as f:
        f.write(html)
    
    print(f"Report saved: {filename}")

run()