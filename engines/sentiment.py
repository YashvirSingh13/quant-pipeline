"""
engines/sentiment.py — News Sentiment Engine.

Fetches recent news headlines from Yahoo Finance (via yfinance)
and scores sentiment using VADER — a lexicon-based model tuned
for financial/social text. No external API keys needed.

VADER compound score: -1.0 (very negative) to +1.0 (very positive)
Threshold:  > +0.05 → BULLISH, < -0.05 → BEARISH, else NEUTRAL

Falls back gracefully if vaderSentiment is not installed.
"""

import yfinance as yf

ENGINE_NAME = "sentiment"

# ── Sentiment thresholds ──────────────────────────────────────────────────────
BULL_THRESH  =  0.08   # aggregate compound score to trigger BUY
BEAR_THRESH  = -0.08
STRONG_BULL  =  0.25
STRONG_BEAR  = -0.25
MAX_ARTICLES =  15      # analyse latest N headlines


def _vader_scores(texts: list) -> list:
    """
    Score each text with VADER. Returns list of compound scores.
    Falls back to keyword scoring if vaderSentiment unavailable.
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return [sia.polarity_scores(t)["compound"] for t in texts]
    except ImportError:
        # Lightweight keyword fallback
        positive = {"surge","rally","growth","profit","gain","strong","beat",
                    "upgrade","breakout","record","bullish","positive","rise",
                    "outperform","boom","optimistic","expand","recovery"}
        negative = {"crash","loss","decline","fall","sell","downgrade","miss",
                    "bearish","weak","drop","concern","risk","cut","struggle",
                    "negative","warn","disappointing","debt","probe","penalty"}
        scores = []
        for text in texts:
            words = set(text.lower().split())
            p = len(words & positive)
            n = len(words & negative)
            scores.append((p - n) / max(p + n, 1))
        return scores


def run(ticker: str) -> dict:
    """
    Args:
        ticker : e.g. "WIPRO.NS"

    Returns dict:
        engine, signal, score, compound_score, news_count,
        sentiment_breakdown, headlines, detail
    """
    try:
        news = yf.Ticker(ticker).news or []

        if not news:
            return {
                "engine":    ENGINE_NAME,
                "signal":    "NEUTRAL",
                "score":     0.5,
                "compound_score": 0.0,
                "news_count": 0,
                "detail":    "No recent news found",
                "headlines": [],
            }

        # Extract titles + timestamps
        articles = [
            {
                "title": a.get("title", ""),
                "publisher": a.get("publisher", ""),
                "age_hrs": round((
                    __import__("time").time() - a.get("providerPublishTime", 0)
                ) / 3600, 1),
            }
            for a in news[:MAX_ARTICLES]
            if a.get("title")
        ]

        titles = [a["title"] for a in articles]
        if not titles:
            return {
                "engine": ENGINE_NAME, "signal": "NEUTRAL", "score": 0.5,
                "news_count": 0, "detail": "No valid titles", "headlines": [],
            }

        raw_scores = _vader_scores(titles)

        # Weight recent news more heavily
        weighted = []
        for i, (s, a) in enumerate(zip(raw_scores, articles)):
            age  = a.get("age_hrs", 24)
            w    = max(0.3, 1.0 - age / 72)    # decay over 72 hours
            weighted.append(s * w)

        compound  = sum(weighted) / len(weighted) if weighted else 0.0
        pos_count = sum(1 for s in raw_scores if s >  0.05)
        neg_count = sum(1 for s in raw_scores if s < -0.05)
        neu_count = len(raw_scores) - pos_count - neg_count

        # ── Signal ────────────────────────────────────────────────────────────────
        if compound >= STRONG_BULL:
            signal, score = "BUY",  min(0.92, 0.75 + compound * 0.5)
            detail = f"Strong positive sentiment ({pos_count} of {len(titles)} bullish)"
        elif compound >= BULL_THRESH:
            signal, score = "BUY",  0.63
            detail = f"Mildly positive sentiment ({pos_count} bullish)"
        elif compound <= STRONG_BEAR:
            signal, score = "SELL", max(0.08, 0.25 + compound * 0.5)
            detail = f"Strong negative sentiment ({neg_count} of {len(titles)} bearish)"
        elif compound <= BEAR_THRESH:
            signal, score = "SELL", 0.37
            detail = f"Mildly negative sentiment ({neg_count} bearish)"
        else:
            signal, score = "NEUTRAL", 0.50
            detail = f"Neutral sentiment (pos:{pos_count} neg:{neg_count} neu:{neu_count})"

        headlines_out = [
            {"title": a["title"], "score": round(s, 3), "age_hrs": a["age_hrs"]}
            for a, s in zip(articles[:6], raw_scores[:6])
        ]

        return {
            "engine":       ENGINE_NAME,
            "signal":       signal,
            "score":        round(score, 3),
            "compound_score": round(compound, 4),
            "news_count":   len(titles),
            "sentiment_breakdown": {
                "positive": pos_count,
                "negative": neg_count,
                "neutral":  neu_count,
            },
            "headlines":    headlines_out,
            "detail":       detail,
        }

    except Exception as exc:
        return {
            "engine":    ENGINE_NAME,
            "signal":    "NEUTRAL",
            "score":     0.5,
            "detail":    f"Error: {exc}",
            "headlines": [],
        }
