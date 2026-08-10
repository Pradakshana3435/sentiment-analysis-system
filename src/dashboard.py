"""
Customer Sentiment Analysis Dashboard
Run with: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Sentiment Dashboard",
    page_icon="\U0001F4AC",
    layout="wide"
)

MODEL_PATH = "Pradak/yelp-sentiment-distilbert"  # Hugging Face Hub repo (auto-downloads & caches)
DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "sentiment_predictions.csv"

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
COLORS = {"positive": "#2ecc71", "neutral": "#f1c40f", "negative": "#e74c3c"}


# ── Cached loaders (so the model/data only load once per session) ──
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device


@st.cache_data
def load_predictions():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


def predict_sentiment(text, model, tokenizer, device):
    inputs = tokenizer(
        text, truncation=True, padding=True, max_length=256, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred_id = int(probs.argmax())
    return {
        "sentiment": ID2LABEL[pred_id],
        "confidence": float(probs[pred_id]),
        "all_scores": {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))},
    }


# ── Load resources ──────────────────────────────────────────
st.title("Customer Sentiment Analysis Dashboard")
st.caption(
    "Built on DistilBERT (fine-tuned, class-weighted loss) \u2014 "
    "trained and evaluated on Yelp customer reviews."
)

try:
    model, tokenizer, device = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.warning(
        f"Live model could not be loaded ({e}). "
        "Live prediction tab will be unavailable, but the trends dashboard below still works."
    )

try:
    df = load_predictions()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Could not load predictions data: {e}")
    st.stop()


# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["\U0001F4CA Overview & Trends", "\u270D\uFE0F Try It Live"])

# ============================================================
# TAB 1: Overview & Trends
# ============================================================
with tab1:
    st.info(
        "Dates in this dataset are **simulated** for demonstration purposes "
        "(the underlying Yelp dataset has no real timestamps). "
        "In production, this view would use actual review dates and reflect genuine trends.",
        icon="\u2139\uFE0F",
    )

    # --- Top-line metrics ---
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    pos_pct = (df["predicted_sentiment"] == "positive").mean() * 100
    neg_pct = (df["predicted_sentiment"] == "negative").mean() * 100
    avg_conf = df["confidence"].mean()

    col1.metric("Total Reviews Analyzed", f"{total:,}")
    col2.metric("Positive Sentiment", f"{pos_pct:.1f}%")
    col3.metric("Negative Sentiment", f"{neg_pct:.1f}%")
    col4.metric("Avg. Model Confidence", f"{avg_conf:.1%}")

    st.divider()

    # --- Distribution + confidence side by side ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Sentiment Distribution")
        counts = df["predicted_sentiment"].value_counts().reindex(
            ["positive", "neutral", "negative"]
        )
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            color=counts.index,
            color_discrete_map=COLORS,
            labels={"x": "Sentiment", "y": "Number of Reviews"},
        )
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Prediction Confidence by Class")
        fig2 = px.box(
            df,
            x="predicted_sentiment",
            y="confidence",
            color="predicted_sentiment",
            color_discrete_map=COLORS,
            category_orders={"predicted_sentiment": ["negative", "neutral", "positive"]},
            labels={"predicted_sentiment": "Sentiment", "confidence": "Confidence Score"},
        )
        fig2.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Note the wider, lower confidence spread for **neutral** predictions \u2014 "
        "this reflects a known limitation: 3-star / mixed reviews are inherently "
        "harder to classify confidently, for the model and for human readers alike."
    )

    st.divider()

    # --- Trend over time ---
    st.subheader("Sentiment Trend Over Time")
    df["date_only"] = pd.to_datetime(df["date"]).dt.date
    trend = df.groupby(["date_only", "predicted_sentiment"]).size().unstack(fill_value=0)
    trend = trend.reindex(columns=["negative", "neutral", "positive"], fill_value=0)
    trend_smooth = trend.rolling(window=3, min_periods=1).mean()

    fig3 = go.Figure()
    for sentiment in ["negative", "neutral", "positive"]:
        fig3.add_trace(
            go.Scatter(
                x=trend_smooth.index,
                y=trend_smooth[sentiment],
                mode="lines+markers",
                name=sentiment,
                line=dict(color=COLORS[sentiment]),
            )
        )
    fig3.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Reviews (3-day rolling avg)",
        height=420,
        legend_title="Sentiment",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # --- Raw data explorer ---
    st.subheader("Explore Individual Reviews")
    sentiment_filter = st.multiselect(
        "Filter by predicted sentiment",
        options=["positive", "neutral", "negative"],
        default=["positive", "neutral", "negative"],
    )
    filtered = df[df["predicted_sentiment"].isin(sentiment_filter)]
    st.dataframe(
        filtered[["text", "predicted_sentiment", "confidence", "date"]]
        .sort_values("confidence", ascending=True)
        .reset_index(drop=True),
        use_container_width=True,
        height=350,
    )

# ============================================================
# TAB 2: Live prediction
# ============================================================
with tab2:
    st.subheader("Classify New Feedback")
    st.write(
        "Paste any customer review, comment, or feedback text below to see how the "
        "model classifies it in real time."
    )

    example_choice = st.selectbox(
        "Or try an example:",
        [
            "-- Type your own below --",
            "The food was absolutely amazing, best meal I've had all year!",
            "Terrible service, waited an hour and the order was wrong.",
            "It was okay, nothing special but not bad either.",
        ],
    )

    default_text = "" if example_choice.startswith("--") else example_choice
    user_text = st.text_area("Review text", value=default_text, height=120)

    if st.button("Analyze Sentiment", type="primary"):
        if not model_loaded:
            st.error("Model is not loaded, cannot run live prediction.")
        elif not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            result = predict_sentiment(user_text, model, tokenizer, device)
            sentiment = result["sentiment"]
            confidence = result["confidence"]

            st.markdown(
                f"### Predicted sentiment: "
                f"<span style='color:{COLORS[sentiment]}'>**{sentiment.upper()}**</span> "
                f"({confidence:.1%} confidence)",
                unsafe_allow_html=True,
            )

            scores_df = pd.DataFrame(
                {
                    "Sentiment": list(result["all_scores"].keys()),
                    "Score": list(result["all_scores"].values()),
                }
            )
            fig4 = px.bar(
                scores_df,
                x="Sentiment",
                y="Score",
                color="Sentiment",
                color_discrete_map=COLORS,
                range_y=[0, 1],
            )
            fig4.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig4, use_container_width=True)