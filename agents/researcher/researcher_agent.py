import anthropic
import os
import json
import pdfplumber
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

PAPERS_DIR = "papers"
REPORTS_DIR = "reports"

def extract_text(pdf_path, max_pages=15):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    break
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def summarize_paper(pdf_path):
    filename = os.path.basename(pdf_path)
    print(f"Summarizing: {filename}")

    text = extract_text(pdf_path)

    if not text.strip():
        return {"file": filename, "summary": "Could not extract text from this PDF."}

    text = text[:8000]

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": f"""You are a research assistant for an AI investment fund building a LightGBM stock prediction model.

Summarize this academic paper. Cover:
1. Main research question
2. ML methods used and why
3. What stock characteristics/features were most predictive
4. Key findings on model performance
5. What we should steal for our LightGBM model

Skip anything about neural networks — we are not using them.
Keep it to 300 words max.

Paper text:
{text}"""
        }]
    )

    return {
        "file": filename,
        "summary": message.content[0].text
    }

def build_report(summaries):
    date = datetime.now().strftime("%B %d, %Y")

    cards = ""
    for s in summaries:
        cards += f"""
        <div class="paper">
            <h2>{s['file']}</h2>
            <div class="summary">{s['summary'].replace(chr(10), '<br>')}</div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Research Library — {date}</title>
    <style>
        body {{ font-family: Georgia, serif; max-width: 900px; margin: 40px auto; background: #f9f9f9; padding: 20px; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
        .paper {{ background: white; padding: 25px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .paper h2 {{ color: #1a1a2e; font-size: 16px; font-family: Arial, sans-serif; }}
        .summary {{ line-height: 1.8; color: #333; }}
    </style>
</head>
<body>
    <h1>Research Library — {date}</h1>
    <p>{len(summaries)} papers summarized.</p>
    {cards}
</body>
</html>"""

def run():
    pdfs = [f for f in os.listdir(PAPERS_DIR) if f.endswith('.pdf')]

    if not pdfs:
        print("No PDFs found in papers/ folder.")
        print("Download papers from SSRN and put them in papers/ first.")
        return

    print(f"Found {len(pdfs)} PDFs\n")

    summaries = []
    for pdf in pdfs:
        path = os.path.join(PAPERS_DIR, pdf)
        result = summarize_paper(path)
        summaries.append(result)

        with open(os.path.join(PAPERS_DIR, 'summaries.json'), 'w') as f:
            json.dump(summaries, f, indent=2)

    html = build_report(summaries)
    filename = os.path.join(REPORTS_DIR, f"research_library_{datetime.now().strftime('%Y-%m-%d')}.html")
    with open(filename, 'w') as f:
        f.write(html)

    print(f"\nDone! Report saved to: {filename}")

run()