import streamlit as st
import requests
import xml.etree.ElementTree as ET
from transformers import pipeline
import html

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risk Intelligence Terminal",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
</style>
""", unsafe_allow_html=True)


# ─── MODEL LOADER ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FinBERT model...")
def get_analyzer():
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        truncation=True,
        max_length=512
    )


# ─── NEWS FETCHER (Yahoo Finance RSS — works on cloud) ──────────────────────
def get_yahoo_finance_news(ticker: str) -> list[dict]:
    """
    Fetch headlines from Yahoo Finance RSS feed.
    This is a public, unauthenticated endpoint that works reliably
    on Streamlit Cloud and other hosted environments.
    """
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            return []

        news_items = []
        for item in channel.findall("item")[:8]:
            title_el = item.find("title")
            link_el  = item.find("link")
            if title_el is None or title_el.text is None:
                continue
            title = html.unescape(title_el.text.strip())
            link  = link_el.text.strip() if link_el is not None and link_el.text else "#"
            if len(title) > 15:
                news_items.append({"title": title, "link": link})

        return news_items

    except ET.ParseError:
        st.error("Failed to parse news feed. Yahoo may have changed its format.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching news: {e}")
        return []


# ─── SENTIMENT HELPERS ──────────────────────────────────────────────────────
def analyze_news(news_items: list[dict]) -> list[dict]:
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
    """Returns float in [-1, +1]. +1 = fully positive."""
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
        "Enter Ticker",
        value="AAPL",
        placeholder="e.g. AAPL, TSLA, MSFT"
    ).strip().upper()

    st.divider()
    st.info(
        "**How it works**\n\n"
        "1. Fetches headlines from Yahoo Finance RSS\n"
        "2. Runs each title through FinBERT (financial NLP)\n"
        "3. Aggregates scores into a risk verdict"
    )


# ─── MAIN PANEL ─────────────────────────────────────────────────────────────
if not ticker_input:
    st.info("👈 Enter a ticker in the sidebar to begin analysis.")
    st.stop()

st.title(f"📈 Market Sentiment: **{ticker_input}**")

with st.spinner("Fetching headlines and running sentiment analysis…"):
    raw_news = get_yahoo_finance_news(ticker_input)

if not raw_news:
    st.warning(
        f"⚠️ No recent news found for **{ticker_input}**. "
        "Please check the ticker symbol and try again."
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

    pos_count = sum(1 for i in analyzed_news if i["sentiment"]["label"] == "positive")
    neg_count = sum(1 for i in analyzed_news if i["sentiment"]["label"] == "negative")
    neu_count = len(analyzed_news) - pos_count - neg_count

    st.markdown("**Headline breakdown**")
    st.markdown(f"🟢 Positive: `{pos_count}`")
    st.markdown(f"🔴 Negative: `{neg_count}`")
    st.markdown(f"⚪ Neutral:  `{neu_count}`")

with col2:
    st.markdown("### Sentiment Confidence Score")
    normalised = (avg_score + 1) / 2
    st.progress(normalised)
    st.caption("0% = fully bearish  ·  50% = neutral  ·  100% = fully bullish")

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
