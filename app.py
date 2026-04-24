import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Intelligence Terminal",
    page_icon="🛡️",
    layout="wide"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .verdict-bullish { color: #3fb950; font-weight: bold; font-size: 1.3em; }
    .verdict-bearish { color: #f85149; font-weight: bold; font-size: 1.3em; }
    .verdict-neutral { color: #d29922; font-weight: bold; font-size: 1.3em; }
</style>
""", unsafe_allow_html=True)


# ─── MODEL LOADER ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FinBERT model...")
def get_analyzer():
    """Load FinBERT – cached so it only downloads once per session."""
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        truncation=True,
        max_length=512
    )


# ─── NEWS SCRAPER ───────────────────────────────────────────────────────────
def get_google_finance_news(ticker: str) -> list[dict]:
    """
    Scrape recent headlines from Google Finance.
    Returns a list of {title, link} dicts (up to 6 items).
    Falls back to an empty list on any error.
    """
    url = f"https://www.google.com/finance/quote/{ticker}:NASDAQ"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        news_items = []
        links = soup.find_all(
            "a",
            href=re.compile(r"\./articles/|/finance/article/")
        )
        for link in links[:6]:
            title = link.get_text(strip=True)
            raw_href = link.get("href", "")
            href = raw_href.replace("./", "https://www.google.com/finance/")
            if len(title) > 20:
                news_items.append({"title": title, "link": href})

        return news_items

    except Exception:
        return []


# ─── SENTIMENT HELPERS ──────────────────────────────────────────────────────
def analyze_news(news_items: list[dict]) -> list[dict]:
    """Run FinBERT on each headline and attach the result."""
    analyzer = get_analyzer()
    enriched = []
    for item in news_items:
        try:
            sentiment = analyzer(item["title"])[0]
        except Exception:
            sentiment = {"label": "neutral", "score": 0.5}
        enriched.append({**item, "sentiment": sentiment})
    return enriched


def compute_sentiment_score(analyzed: list[dict]) -> float:
    """
    Returns a float in [-1, +1].
    +1 = fully positive, -1 = fully negative.
    """
    if not analyzed:
        return 0.0
    vals = []
    for item in analyzed:
        label = item["sentiment"]["label"]
        score = item["sentiment"]["score"]
        if label == "positive":
            vals.append(score)
        elif label == "negative":
            vals.append(-score)
        else:
            vals.append(0.0)
    return sum(vals) / len(vals)


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Risk Intelligence")
    st.caption("Powered by FinBERT · ProsusAI")
    ticker_input = st.text_input(
        "Enter Ticker (NASDAQ)",
        value="AAPL",
        placeholder="e.g. AAPL, TSLA, MSFT"
    ).strip().upper()

    st.divider()
    st.info(
        "**How it works**\n\n"
        "1. Fetches latest headlines from Google Finance\n"
        "2. Runs each title through FinBERT (financial NLP)\n"
        "3. Aggregates scores into a risk verdict"
    )


# ─── MAIN PANEL ─────────────────────────────────────────────────────────────
if not ticker_input:
    st.info("👈 Enter a ticker in the sidebar to begin analysis.")
    st.stop()

st.title(f"📈 Market Sentiment: **{ticker_input}**")

with st.spinner("Fetching headlines and running sentiment analysis…"):
    raw_news = get_google_finance_news(ticker_input)

if not raw_news:
    st.warning(
        f"⚠️ No recent news found for **{ticker_input}**. "
        "Make sure it's a valid NASDAQ ticker (e.g. AAPL, TSLA, NVDA)."
    )
    st.stop()

analyzed_news = analyze_news(raw_news)
avg_score = compute_sentiment_score(analyzed_news)

# ── Verdict banner ──────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Overall Verdict")
    if avg_score > 0.10:
        st.success(f"🟢 BULLISH  ({avg_score:+.2f})")
    elif avg_score < -0.10:
        st.error(f"🔴 BEARISH  ({avg_score:+.2f})")
    else:
        st.warning(f"🟡 NEUTRAL  ({avg_score:+.2f})")

    # Simple bar chart of pos/neg/neutral counts
    pos_count = sum(1 for i in analyzed_news if i["sentiment"]["label"] == "positive")
    neg_count = sum(1 for i in analyzed_news if i["sentiment"]["label"] == "negative")
    neu_count = len(analyzed_news) - pos_count - neg_count

    st.markdown("**Headline breakdown**")
    st.markdown(f"🟢 Positive: `{pos_count}`")
    st.markdown(f"🔴 Negative: `{neg_count}`")
    st.markdown(f"⚪ Neutral: `{neu_count}`")

with col2:
    st.markdown("### Sentiment Confidence Score")
    # Normalise avg_score from [-1,1] → [0,1] for the progress bar
    normalised = (avg_score + 1) / 2
    st.progress(normalised)
    st.caption(
        "0 % = fully bearish  ·  50 % = neutral  ·  100 % = fully bullish"
    )

st.divider()

# ── Individual headlines ────────────────────────────────────────────────────
st.subheader("📰 Latest Headlines")

for item in analyzed_news:
    label = item["sentiment"]["label"]
    score = item["sentiment"]["score"]
    color_map = {"positive": "green", "negative": "red", "neutral": "gray"}
    emoji_map = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
    color = color_map.get(label, "gray")
    emoji = emoji_map.get(label, "⚪")

    with st.container():
        st.markdown(f"**{item['title']}**")
        st.markdown(
            f":{color}[{emoji} Sentiment: **{label.upper()}** "
            f"(confidence: {score:.2%})]"
        )
        st.caption(f"[Read full article →]({item['link']})")
        st.divider()
