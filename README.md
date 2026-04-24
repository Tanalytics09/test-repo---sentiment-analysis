# 🛡️ Risk Intelligence Terminal

A financial sentiment analysis app powered by **FinBERT** (ProsusAI) and **Streamlit**.

Enter any NASDAQ ticker and instantly get:
- Real-time headline sentiment (Bullish / Bearish / Neutral)
- Per-article FinBERT confidence scores
- Aggregated risk verdict

---

## 🚀 Deploy on Streamlit Cloud (Recommended — Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy** — done ✅

No environment variables needed.

---

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 File Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚠️ Notes

- **FinBERT model** (~500 MB) is downloaded automatically on first run and cached.
- Only **NASDAQ tickers** are supported (Google Finance scraper targets NASDAQ).
- Google Finance scraping may occasionally return no results if their HTML structure changes.
