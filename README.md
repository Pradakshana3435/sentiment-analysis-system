# Customer Sentiment Analysis System

An end-to-end NLP pipeline that classifies customer feedback as **positive**, **negative**, or **neutral**, with confidence scores and an interactive dashboard. Built on Yelp customer reviews.

## Live model
Fine-tuned DistilBERT weights are hosted on Hugging Face Hub (too large for GitHub):
**[Pradak/yelp-sentiment-distilbert](https://huggingface.co/Pradak/yelp-sentiment-distilbert)**

The dashboard downloads and caches this automatically on first run — no manual download needed.

## What this project does
1. **Preprocesses** raw customer review text (cleaning, stopword removal, tokenization)
2. **Trains and compares** multiple NLP approaches:
   - Traditional ML: Logistic Regression & Linear SVM on TF-IDF features
   - Transformer: fine-tuned DistilBERT, including a class-weighted variant to address class imbalance
3. **Outputs sentiment with confidence scores** for any new text
4. **Visualizes results** in an interactive Streamlit dashboard — sentiment distribution, confidence by class, trends over time, and live prediction on new text

## Results

| Model | Accuracy | Macro F1 | Neutral F1 |
|---|---|---|---|
| Logistic Regression (TF-IDF) | 73% | 0.70 | 0.51 |
| Linear SVM (TF-IDF) | 73% | 0.68 | 0.45 |
| DistilBERT (unweighted) | 77% | 0.71 | 0.46 |
| **DistilBERT (class-weighted) — final model** | 76% | **0.72** | 0.50 |

The transformer clearly outperforms traditional TF-IDF models on positive/negative classification. The **neutral** class (ambiguous, 3-star-equivalent reviews) remains the hardest to classify across every approach — a class-weighted loss function improved neutral recall (0.44 → 0.54) at a small cost to overall accuracy, a deliberate and documented trade-off in favor of balanced performance.

## Project structure
sentiment-analysis-system/
├── data/
│ └── processed/ # cleaned data, predictions (not committed — see .gitignore)
├── models/
│ └── final-sentiment-model/ # local model cache (not committed — see .gitignore)
├── notebooks/
│ └── 01-data-exploration.ipynb # full pipeline: preprocessing, training, evaluation
├── src/
│ └── dashboard.py # Streamlit dashboard (live predictions + trend visualization)
├── .gitignore
├── requirements.txt
└── README.md
## Setup

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-system.git
cd sentiment-analysis-system
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

## Running the dashboard

```bash
streamlit run src/dashboard.py
```

Opens at `http://localhost:8501`. First launch downloads the model from Hugging Face Hub (~268MB, cached afterward).

## Dataset
[Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full) (Hugging Face), 650,000 reviews. Star ratings mapped to sentiment: 1–2★ = negative, 3★ = neutral, 4–5★ = positive.

## Known limitations
- **Trend data uses simulated dates** — the Yelp dataset has no real timestamps; the dashboard's trend chart demonstrates the visualization capability, not a genuine temporal finding.
- **Single data source** — trained and evaluated on Yelp reviews only. The original brief referenced multiple feedback channels (website, social media, surveys); this project demonstrates the pipeline on one representative channel.
- **Neutral-class performance** remains the weakest area across all models tested, reflecting the inherent ambiguity of 3-star/mixed reviews rather than a specific model shortcoming.

## Tech stack
Python · scikit-learn · Hugging Face Transformers · PyTorch · Streamlit · Plotly · NLTK
