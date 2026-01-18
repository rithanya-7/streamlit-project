# =========================================================
# Smart Hostel Problem Detector — Production Streamlit App
# GitHub + HuggingFace + Gemini
# =========================================================

import os
import requests
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib
import folium
from streamlit_folium import st_folium
from transformers import DistilBertTokenizerFast, DistilBertModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import google.generativeai as genai

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="Smart Hostel Problem Detector", layout="wide")
st.title("🏨 Smart Hostel Problem Detector — AI-Based Issue Prioritization")

# -------------------------------
# Secrets (Streamlit Cloud)
# -------------------------------
HF_TOKEN = os.environ["HF_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# -------------------------------
# Constants
# -------------------------------
MODEL_DIR = "SmartCivicAI_Models"
BASE_URL = "https://huggingface.co/kruthi19/smartcivicai-models/resolve/main/SmartCivicAI_Models"
DATA_FILE = "complaints_preprocessed.csv"
FEEDBACK_FILE = "feedback.csv"
MAX_LEN = 128
FLAG_FILE = ".models_ready"

# -------------------------------
# Civic → Hostel Category Mapping
# -------------------------------
CIVIC_TO_HOSTEL = {
    # Hygiene related
    "Garbage": "Washroom Hygiene Issue",
    "Sewage": "Washroom Hygiene Issue",

    # Noise related
    "Noise": "Excessive Noise in Hostel",

    # Water related
    "Water Leakage": "Water Overflow / Leakage",

    # Electrical / Utilities
    "Streetlight": "Fan / AC Failure",
    "Traffic Signal": "Fan / AC Failure",

    # Infrastructure
    "Road Damage": "Hostel Infrastructure Issue",
    "Tree Fallen": "Hostel Infrastructure Issue",
}

def to_hostel_category(label: str) -> str:
    return CIVIC_TO_HOSTEL.get(label, "General Hostel Issue")


# -------------------------------
# Fix NLTK path for Streamlit Cloud
# -------------------------------
NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)

nltk.data.path.append(NLTK_DIR)
nltk.download("vader_lexicon", download_dir=NLTK_DIR, quiet=True)
sentiment_analyzer = SentimentIntensityAnalyzer()

# =========================================================
# HuggingFace Download Helpers (ONE TIME)
# =========================================================
REQUIRED_FILES = [
    "multitask_distilbert.pt",
    "sentiment_model.pt",
    "regression_model.pkl",
    "category_encoder.pkl",
    "urgency_encoder.pkl",
    "sentiment_encoder.pkl",
]

def download_file(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        return
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    r = requests.get(f"{BASE_URL}/{filename}", headers=headers)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)

# -------------------------------
# First Run: Download Models
# -------------------------------
if not os.path.exists(FLAG_FILE):
    st.info("⬇️ Downloading AI models from Hugging Face (first run only)...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    for f in REQUIRED_FILES:
        download_file(f)
    with open(FLAG_FILE, "w") as f:
        f.write("ok")
    st.success("✅ Models downloaded. Reloading app...")
    st.stop()

# =========================================================
# Model Definitions (UNCHANGED)
# =========================================================
class MultiTaskDistilBERT(nn.Module):
    def __init__(self, num_urgency, num_category):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        h = self.bert.config.hidden_size
        self.urgency_head = nn.Linear(h, num_urgency)
        self.category_head = nn.Linear(h, num_category)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return {
            "urgency_logits": self.urgency_head(cls),
            "category_logits": self.category_head(cls),
        }

# =========================================================
# Load Models (Cached)
# =========================================================
@st.cache_resource
def load_models():
    category_encoder = joblib.load(f"{MODEL_DIR}/category_encoder.pkl")
    urgency_encoder = joblib.load(f"{MODEL_DIR}/urgency_encoder.pkl")
    sentiment_encoder = joblib.load(f"{MODEL_DIR}/sentiment_encoder.pkl")
    regression_model = joblib.load(f"{MODEL_DIR}/regression_model.pkl")

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "kruthi19/smartcivicai-models",
        token=HF_TOKEN,
        subfolder="SmartCivicAI_Models/multitask_tokenizer",
    )

    model = MultiTaskDistilBERT(
        len(urgency_encoder.classes_),
        len(category_encoder.classes_),
    )
    model.load_state_dict(
        torch.load(f"{MODEL_DIR}/multitask_distilbert.pt", map_location="cpu"),
        strict=False,
    )
    model.eval()

    return category_encoder, urgency_encoder, regression_model, tokenizer, model

category_encoder, urgency_encoder, regression_model, tokenizer, multitask_model = load_models()

# -------------------------------------------------
# Initialize / Load complaints data
# -------------------------------------------------
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=[
        "complaint_id", "complaint_text", "category", "urgency_label",
        "final_score", "latitude", "longitude", "eta_hours", "status"
    ]).to_csv(DATA_FILE, index=False)

# Load data
df = pd.read_csv(DATA_FILE)



# =========================================================
# Submit Hostel Issue
# =========================================================
st.subheader("📝 Submit a Hostel Issue")

text = st.text_area("Describe the hostel problem")
lat = st.number_input("Latitude", value=28.6139)
lon = st.number_input("Longitude", value=77.2090)

if st.button("Analyze & Submit"):
    if not text.strip():
        st.warning("Please enter an issue description.")
        st.stop()

    enc = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=MAX_LEN, return_tensors="pt"
    )

    with torch.no_grad():
        out = multitask_model(enc["input_ids"], enc["attention_mask"])

    civic_cat = category_encoder.inverse_transform(
        [out["category_logits"].argmax(1).item()]
    )[0]
    # Always map from MODEL output, never from existing CSV value
    hostel_cat = to_hostel_category(civic_cat)

    urgency = urgency_encoder.inverse_transform(
        [out["urgency_logits"].argmax(1).item()]
    )[0]

    try:
        eta = regression_model.predict(
            [[lat, lon, category_encoder.transform([civic_cat])[0]]]
        )[0]
    except:
        eta = 24.0

    score = {"High":1.0,"Medium":0.5,"Low":0.2}.get(urgency,0.5)
    cid = int(df["complaint_id"].max()) + 1 if not df.empty else 1

    new_row = {
        "complaint_id": cid,
        "complaint_text": text,
        "category": hostel_cat,
        "urgency_label": urgency,
        "final_score": score,
        "latitude": lat,
        "longitude": lon,
        "eta_hours": round(float(eta), 2),
        "status": "Pending",
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

    st.success(f"✅ Hostel issue registered! ID: {cid}")

# =========================================================
# Map
# =========================================================
st.subheader("🗺 Hostel Issue Map")

if not df.empty:
    m = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=12)
    for _, r in df.iterrows():
        folium.CircleMarker(
            [r.latitude, r.longitude],
            popup=f"{r.category} ({r.urgency_label})",
            radius=6,
            color="red" if r.urgency_label == "High" else "blue",
            fill=True,
        ).add_to(m)
    st_folium(m, height=500)

# =========================================================
# Feedback + Gemini
# =========================================================
st.subheader("✅ Mark Issue as Resolved")

pending = df[df.status == "Pending"]

if not pending.empty:
    cid = st.selectbox("Issue ID", pending.complaint_id.tolist())
    feedback = st.text_area("Resident feedback")

    if st.button("Submit Feedback"):
        scores = sentiment_analyzer.polarity_scores(feedback)
        sentiment = (
            "Positive" if scores["compound"] > 0.05
            else "Negative" if scores["compound"] < -0.05
            else "Neutral"
        )

        model = genai.GenerativeModel("gemini-2.5-flash")
        summary = model.generate_content(
            f"Resident feedback: {feedback}. Explain sentiment empathetically."
        ).text

        fdf = pd.read_csv(FEEDBACK_FILE) if os.path.exists(FEEDBACK_FILE) else pd.DataFrame(
            columns=["complaint_id","feedback","sentiment","score","ai_summary"]
        )

        fdf = pd.concat([fdf, pd.DataFrame([{
            "complaint_id": cid,
            "feedback": feedback,
            "sentiment": sentiment,
            "score": scores["compound"],
            "ai_summary": summary,
        }])], ignore_index=True)

        fdf.to_csv(FEEDBACK_FILE, index=False)
        df.loc[df.complaint_id == cid, "status"] = "Completed"
        df.to_csv(DATA_FILE, index=False)

        st.success("✅ Feedback saved with AI summary.")
# =========================================================
# Feedback History & Sentiment Analytics (RESTORED UI)
# =========================================================
st.subheader("🗂 Feedback History")

if os.path.exists(FEEDBACK_FILE):
    fdf = pd.read_csv(FEEDBACK_FILE)

    # Ensure correct types
    fdf["complaint_id"] = fdf["complaint_id"].astype(str)
    df["complaint_id"] = df["complaint_id"].astype(str)

    # Merge feedback with complaint category
    merged = fdf.merge(
        df[["complaint_id", "category"]],
        on="complaint_id",
        how="left"
    )

    # -------------------------------
    # Feedback Table
    # -------------------------------
    st.markdown("### 📄 Feedback Records")

    display_cols = ["complaint_id", "category", "feedback", "sentiment", "score"]
    display_df = merged[display_cols].copy()

    def sentiment_style(val):
        if val == "Positive":
            return "background-color: #e8f5e9; color: #2e7d32; font-weight: bold;"
        if val == "Negative":
            return "background-color: #ffebee; color: #c62828; font-weight: bold;"
        if val == "Neutral":
            return "background-color: #fff8e1; color: #f9a825; font-weight: bold;"
        return ""

    st.dataframe(
        display_df.style.applymap(sentiment_style, subset=["sentiment"]),
        use_container_width=True
    )

    # -------------------------------
    # Sentiment Summary
    # -------------------------------
    st.markdown("### 📊 Sentiment Summary")

    counts = (
        merged["sentiment"]
        .value_counts()
        .reindex(["Positive", "Neutral", "Negative"])
        .fillna(0)
        .astype(int)
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("😊 Positive", counts.get("Positive", 0))
    c2.metric("😐 Neutral", counts.get("Neutral", 0))
    c3.metric("😠 Negative", counts.get("Negative", 0))

    # -------------------------------
    # Sentiment by Category
    # -------------------------------
    st.markdown("### 🧩 Sentiment by Category")

    by_category = (
        merged.groupby(["category", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Positive", "Neutral", "Negative"])
        .reset_index()
    )

    st.dataframe(by_category, use_container_width=True)

else:
    st.info("No feedback data available yet.")
