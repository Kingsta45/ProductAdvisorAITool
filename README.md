# Product Fit Advisor

An AI tool that researches a prospective investor (family office, fund, wealth manager) from public websites and recommends which product to pitch them with a scored fit, talking points, objection handling, and data gaps to confirm in the meeting.

Built as a self-directed project to sharpen BD research for crypto fund allocators



## What it does

1. Scrape: Pulls data from different websites about the prospect.
2. Analyse: Sends the scraped content to DeepSeek alongside hypothetical product data, and returns a score across all four offerings:
• Market Neutral Fund (USD)
• Bitcoin Market Neutral Fund (BTC)
• Separately Managed Accounts (SMA)
• Vaults on Midas and/or Accountable
4. Report: Outputs a structured pre-meeting brief which includes the prospect profile, per-product fit scores with evidence, a primary recommendation, meeting talking points, likely objections, and open data gaps. Exportable as `.txt` or a formatted `.pdf`.



## Interface

Simple desktop GUI (`gui.py`): Enter a prospect name and one or more website URLs, click Run Analysis, review the report, save it.

## Tech stack
• Python, `requests` + `BeautifulSoup` for scraping, `selenium` as a fallback for JS-rendered sites

• DeepSeek API (OpenAI-compatible) for the analysis

• `customtkinter` for the GUI

• `fpdf2` for PDF export



## Setup

`pip install -r requirements.txt`

Create a .env file in the project root:

`DEEPSEEK_API_KEY=your-key-here`

`DEEPSEEK_MODEL=deepseek-v4-flash`

Run:
`python gui.py`



## Notes

• Scraping is limited to a prospect's own public website or websites. No LinkedIn scraping or login-gated data by design.

• This is a pre-meeting prep aid, not investment advice.
