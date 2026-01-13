import os
import streamlit as st
st.title("My Streamlit Website")
import pandas as pd
import folium
from streamlit_folium import st_folium
import torch
import joblib
from transformers import DistilBertTokenizerFast
import torch.nn as nn
from transformers import DistilBertModel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import google.generativeai as genai

genai.configure(api_key="AIzaSyDF-N9DN6T-J5EwnoedO6H8Sh-Bnquah8s")

analyzer = SentimentIntensityAnalyzer()

os.environ["TORCH_COMPILE_DISABLE"] = "1"

MAX_LEN = 128


def encode_text(tok, text: str, max_len: int = MAX_LEN):
    enc = tok(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_len
    )
    if hasattr(tok, "vocab_size"):
        vmax = int(enc["input_ids"].max())
        if vmax >= tok.vocab_size:
            raise ValueError("Tokenizer produced id >= vocab_size. Load the exact tokenizer folder saved during training.")
    return enc

class MultiTaskDistilBERT(nn.Module):
    def __init__(self, num_urgency, num_category):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        hidden = self.bert.config.hidden_size
        self.urgency_head = nn.Linear(hidden, num_urgency)
        self.category_head = nn.Linear(hidden, num_category)
    def forward(self, input_ids=None, attention_mask=None, last_hidden_state=None):
        if last_hidden_state is None:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = out.last_hidden_state
        cls = last_hidden_state[:, 0]
        cls = self.dropout(cls)
        urgency_logits = self.urgency_head(cls)
        category_logits = self.category_head(cls)
        return {"urgency_logits": urgency_logits, "category_logits": category_logits}

class SentimentDistilBERT(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)
    def forward(self, input_ids=None, attention_mask=None, last_hidden_state=None):
        if last_hidden_state is None:
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = out.last_hidden_state
        cls = last_hidden_state[:, 0]
        cls = self.dropout(cls)
        logits = self.classifier(cls)
        return {"logits": logits}

def forward_bert_safe(distilbert: DistilBertModel, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    bs, seqlen = input_ids.shape
    device = input_ids.device
    position_ids = torch.arange(seqlen, dtype=torch.long, device=device).unsqueeze(0).expand(bs, -1)
    max_pos = distilbert.embeddings.position_embeddings.num_embeddings
    position_ids = torch.clamp(position_ids, max=max_pos - 1)
    inputs_embeds = distilbert.embeddings.word_embeddings(input_ids)
    pos_embeds = distilbert.embeddings.position_embeddings(position_ids)
    hidden_states = distilbert.embeddings.LayerNorm(inputs_embeds + pos_embeds)
    hidden_states = distilbert.embeddings.dropout(hidden_states)
    n_layers = distilbert.config.n_layers
    head_mask = [None] * n_layers
    attn_mask = attention_mask.to(torch.float32) if attention_mask.dtype != torch.float32 else attention_mask
    output = distilbert.transformer(
        x=hidden_states,
        attn_mask=attn_mask,
        head_mask=head_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return output.last_hidden_state

st.set_page_config(page_title="SmartCivicAI", layout="wide")
st.title("🌍 SmartCivicAI — Civic Complaint Prioritization")

MODEL_DIR = "SmartCivicAI_Models"
if not os.path.isdir(MODEL_DIR):
    st.error("Model directory not found. Please run the training notebook first.")
    st.stop()

for k, v in {
    "last_submitted": None,
    "focus_cid": None,
    "focus_lat": None,
    "focus_lon": None,
    "category_filter_override": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

category_encoder = joblib.load(os.path.join(MODEL_DIR, "category_encoder.pkl"))
urgency_encoder = joblib.load(os.path.join(MODEL_DIR, "urgency_encoder.pkl"))
sentiment_encoder = joblib.load(os.path.join(MODEL_DIR, "sentiment_encoder.pkl"))

mt_tok_path = os.path.join(MODEL_DIR, "multitask_tokenizer")
s_tok_path = os.path.join(MODEL_DIR, "sentiment_tokenizer")
if not (os.path.isdir(mt_tok_path) and os.path.isdir(s_tok_path)):
    st.error("Tokenizer folders not found. Ensure 'multitask_tokenizer' and 'sentiment_tokenizer' from training are present.")
    st.stop()

multitask_tokenizer = DistilBertTokenizerFast.from_pretrained(mt_tok_path)
sentiment_tokenizer = DistilBertTokenizerFast.from_pretrained(s_tok_path)
multitask_tokenizer.model_max_length = 512
sentiment_tokenizer.model_max_length = 512
if multitask_tokenizer.pad_token is None:
    multitask_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
if sentiment_tokenizer.pad_token is None:
    sentiment_tokenizer.add_special_tokens({"pad_token": "[PAD]"})

num_categories = len(category_encoder.classes_)
num_urgencies = len(urgency_encoder.classes_)
num_sentiments = len(sentiment_encoder.classes_)

multitask_model = MultiTaskDistilBERT(num_urgency=num_urgencies, num_category=num_categories)
sentiment_model = SentimentDistilBERT(num_labels=num_sentiments)

mt_vocab_full = len(multitask_tokenizer)
s_vocab_full = len(sentiment_tokenizer)
if mt_vocab_full != multitask_model.bert.embeddings.word_embeddings.num_embeddings:
    multitask_model.bert.resize_token_embeddings(mt_vocab_full)
if s_vocab_full != sentiment_model.bert.embeddings.word_embeddings.num_embeddings:
    sentiment_model.bert.resize_token_embeddings(s_vocab_full)

if any(p.is_meta for p in multitask_model.parameters()):
    multitask_model.to_empty(device="cpu")
if any(p.is_meta for p in sentiment_model.parameters()):
    sentiment_model.to_empty(device="cpu")

mt_state = torch.load(os.path.join(MODEL_DIR, "multitask_distilbert.pt"), map_location="cpu")
mt_missing, mt_unexpected = multitask_model.load_state_dict(mt_state, strict=False)
sent_state = torch.load(os.path.join(MODEL_DIR, "sentiment_model.pt"), map_location="cpu")
s_missing, s_unexpected = sentiment_model.load_state_dict(sent_state, strict=False)
multitask_model.eval()
sentiment_model.eval()

with torch.no_grad():
    bs = 1
    seq = min(8, MAX_LEN)
    dummy_ids = torch.zeros((bs, seq), dtype=torch.long)
    dummy_mask = torch.ones((bs, seq), dtype=torch.long)
    last_mt = forward_bert_safe(multitask_model.bert, dummy_ids, dummy_mask)
    last_s = forward_bert_safe(sentiment_model.bert, dummy_ids, dummy_mask)
    _ = multitask_model(last_hidden_state=last_mt)
    _ = sentiment_model(last_hidden_state=last_s)

regression_model = joblib.load(os.path.join(MODEL_DIR, "regression_model.pkl"))

try:
    _test = encode_text(multitask_tokenizer, "sanity test", MAX_LEN)
    with torch.no_grad():
        last = forward_bert_safe(multitask_model.bert, _test["input_ids"], _test["attention_mask"])
        _ = multitask_model(last_hidden_state=last)
except Exception as e:
    st.error(f"Startup check failed: {e}. Ensure tokenizer folders and MAX_LEN match training; verify weights match this backbone.")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv("complaints_preprocessed.csv")

def save_data(df_):
    df_.to_csv("complaints_preprocessed.csv", index=False)

df = load_data()
required_cols = {"complaint_id","complaint_text","category","urgency_label","final_score","latitude","longitude","eta_hours","status"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing columns in complaints_preprocessed.csv: {missing}")
    st.stop()

st.sidebar.header("🔎 Filters")
category_filter = st.sidebar.multiselect("Category", sorted(df["category"].dropna().unique()), default=sorted(df["category"].dropna().unique()))
urgency_filter = st.sidebar.multiselect("Urgency", sorted(df["urgency_label"].dropna().unique()), default=sorted(df["urgency_label"].dropna().unique()))
show_status = st.sidebar.multiselect("Status", ["Pending","Completed"], default=["Pending"])
top_n = st.sidebar.slider("Show Top N Complaints", 5, 50, 10)

if st.session_state.get("category_filter_override") and st.session_state.get("focus_cid"):
    focus_row = df[df["complaint_id"] == st.session_state["focus_cid"]]
    if not focus_row.empty:
        fc_cat = focus_row.iloc["category"]
        fc_urg = focus_row.iloc["urgency_label"]
        fc_status = focus_row.iloc["status"]
        if fc_cat not in category_filter:
            category_filter = list(sorted(set(category_filter) | {fc_cat}))
        if fc_urg not in urgency_filter:
            urgency_filter = list(sorted(set(urgency_filter) | {fc_urg}))
        if fc_status not in show_status:
            show_status = list(sorted(set(show_status) | {fc_status}))

filtered = df[df["category"].isin(category_filter) & df["urgency_label"].isin(urgency_filter) & df["status"].isin(show_status)]
ranked = filtered.sort_values("final_score", ascending=False).head(top_n)

if st.session_state.get("last_submitted"):
    new_id = st.session_state["last_submitted"]
    if new_id in filtered["complaint_id"].values:
        new_row = filtered[filtered["complaint_id"] == new_id]
        ranked = pd.concat([new_row, ranked.drop(new_row.index)]).drop_duplicates("complaint_id")
        ranked = ranked.drop_duplicates("complaint_id", keep="first")

st.markdown('<div id="complaints-table"></div>', unsafe_allow_html=True)
st.subheader(f"📌 Top {top_n} Pending Complaints")

def color_urg(val):
    if val == "High":
        return "background-color: #ffcccc"
    elif val == "Medium":
        return "background-color: #fff5cc"
    elif val == "Low":
        return "background-color: #ccffcc"
    return ""

focus_cid = st.session_state.get("focus_cid")

def highlight_focus(row):
    if focus_cid is not None and row["complaint_id"] == focus_cid:
        return ["background-color: #e3f2fd"] * len(row)
    return [""] * len(row)

pending_ranked = ranked[ranked["status"] == "Pending"]

view_cols = ["complaint_id","complaint_text","category","urgency_label","final_score","eta_hours","status"]  
styled_table = (
    pending_ranked[view_cols]
    .style.applymap(color_urg, subset=["urgency_label"])
    .apply(highlight_focus, axis=1)
)

st.dataframe(styled_table, use_container_width=True)

csv_bytes = pending_ranked.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download pending complaints CSV", data=csv_bytes, file_name="complaints_pending.csv", mime="text/csv")

st.markdown('<div id="complaints-map"></div>', unsafe_allow_html=True)
st.subheader("🗺 Complaint Hotspot Map")

center_override = None
if st.session_state.get("focus_lat") is not None and st.session_state.get("focus_lon") is not None:
    center_override = [st.session_state["focus_lat"], st.session_state["focus_lon"]]

if not ranked.empty:
    center = center_override or [ranked["latitude"].mean(), ranked["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=14 if center_override else 12)
    map_data = ranked.copy()
    if st.session_state.get("last_submitted"):
        new_id = st.session_state["last_submitted"]
        if new_id not in map_data["complaint_id"].values:
            extra_row = filtered[filtered["complaint_id"] == new_id]
            map_data = pd.concat([map_data, extra_row])
    for _, row in map_data.iterrows():
        is_focus = focus_cid is not None and row["complaint_id"] == focus_cid
        is_new = st.session_state.get("last_submitted") == row["complaint_id"]

        popup = f"<b>{row['category']}</b><br>Urgency: {row['urgency_label']}<br>ETA: {row['eta_hours']} hrs<br>Status: {row['status']}"

        if is_new:
            marker_color = "green"
        elif is_focus:
            marker_color = "orange"
        else:
            marker_color = "red" if row["urgency_label"] == "High" else "blue"

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=9 if is_new else (8 if is_focus else 6),
            popup=popup,
            color=marker_color,
            fill=True,
            fill_opacity=0.95 if is_new else (0.9 if is_focus else 0.7),
        ).add_to(m)

    st_folium(m, width=700, height=500)
else:
    st.warning("No complaints match your filters.")

st.subheader("📝 Submit a New Complaint")
new_complaint = st.text_input("Enter your complaint:")
latitude = st.number_input("Latitude", value=float(df['latitude'].mean()) if not df.empty else 28.6139)
longitude = st.number_input("Longitude", value=float(df['longitude'].mean()) if not df.empty else 77.2090)

c1, c2 = st.columns([1, 3])
with c2:
    st.markdown('[Scroll to table](#complaints-table) | [Scroll to map](#complaints-map)', unsafe_allow_html=True)

if st.button("Submit Complaint"):
    text_clean = new_complaint.strip()
    if not text_clean:
        st.warning("Please enter a complaint.")
    else:
        enc = encode_text(multitask_tokenizer, text_clean, MAX_LEN)
        with torch.no_grad():
            last = forward_bert_safe(multitask_model.bert, enc["input_ids"], enc["attention_mask"])
            outputs = multitask_model(last_hidden_state=last)
        category_idx = int(torch.argmax(outputs["category_logits"], dim=1).cpu().numpy())
        urgency_idx = int(torch.argmax(outputs["urgency_logits"], dim=1).cpu().numpy())

        # --- map model index to label (safe extraction) ---
        try:
            predicted_category = category_encoder.inverse_transform([category_idx])[0]
        except Exception:
            # fallback if encoder returns something unexpected
            predicted_category = str(category_idx)

        try:
            predicted_urgency = urgency_encoder.inverse_transform([urgency_idx])[0]
        except Exception:
            predicted_urgency = str(urgency_idx)

        # --- Keyword-based override for obvious domain mismatches (minimal, non-invasive) ---
        # Define keywords mapped to preferred category labels (tweak these strings to match your encoder's labels exactly)
        keyword_category_overrides = {
            "Sewage/Drainage": ["sewage", "sewer", "drain", "drainage", "blocked drain", "manhole", "septic", "overflow"],
            "Water": ["water", "water supply", "tap", "no water", "water leak", "leak", "burst pipe"],
            "Garbage": ["garbage", "trash", "rubbish", "waste", "bin", "dump", "overflowing garbage"],
            "Road/ Pothole": ["pothole", "road", "street", "traffic", "broken road", "uneven road"],
            "Lighting": ["light", "streetlight", "lamp", "pole light", "bulb", "light out"],
            "Noise": ["noise", "loud", "music", "party", "construction noise"],
            "Electricity": ["electric", "power", "electricity", "no power", "power outage"],
            # add more categories/keywords as needed
        }

        # lowercased complaint text for matching
        _text_for_override = text_clean.lower() if 'text_clean' in locals() else new_complaint.strip().lower()

        # Try to find a sensible override (first match wins)
        override_found = None
        for cat_label, keywords in keyword_category_overrides.items():
            for kw in keywords:
                if kw in _text_for_override:
                    # only apply override if the target label exists in your encoder classes
                    if cat_label in list(category_encoder.classes_):
                        override_found = cat_label
                    else:
                        # try to find a close existing label (case-insensitive match)
                        for existing in category_encoder.classes_:
                            if cat_label.lower() in existing.lower() or existing.lower() in cat_label.lower():
                                override_found = existing
                                break
                    if override_found:
                        break
            if override_found:
                break

        if override_found:
            predicted_category = override_found
            # optional: notify in Streamlit (comment out if you don't want messages)
            st.info(f"Category override applied based on keywords → {predicted_category}")

        # Ensure predicted_urgency is a pure string, consistent with earlier code
        predicted_urgency = str(predicted_urgency) if not isinstance(predicted_urgency, str) else predicted_urgency

        X = [[latitude, longitude, category_idx]]
        eta_pred = float(regression_model.predict(X))
        final_score = {"High": 1.0, "Medium": 0.5, "Low": 0.2}.get(predicted_urgency, 0.5)

        new_entry = {
            "complaint_id": int(df["complaint_id"].max()) + 1 if not df.empty else 1,
            "complaint_text": text_clean,
            "category": predicted_category[0] if hasattr(predicted_category, "__getitem__") else predicted_category,
            "urgency_label": predicted_urgency,
            "final_score": final_score,
            "latitude": latitude,
            "longitude": longitude,
            "eta_hours": round(eta_pred, 2),
            "status": "Pending",
        }
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        save_data(df)

        st.success(f"Complaint submitted! Complaint ID: {new_entry['complaint_id']}, Category: {new_entry['category']}, Urgency: {new_entry['urgency_label']}, ETA: {round(eta_pred,2)} hrs")
        st.session_state["last_submitted"] = new_entry["complaint_id"]
        st.rerun()

nltk.download("vader_lexicon", quiet=True)

st.subheader("✅ Mark Complaint as Complete")

pending = df[df["status"] == "Pending"]
if not pending.empty:
    complete_id = st.selectbox("Select Complaint ID to mark complete", pending["complaint_id"].tolist())

    @st.dialog("Provide completion feedback")
    def feedback_dialog(cid: int):
        st.write(f"Feedback for complaint ID: {cid}")
        feedback_text = st.text_area("Enter feedback:")
        if st.button("Submit Feedback"):
            text_fb = feedback_text.strip()
            if not text_fb:
                st.warning("Please enter some feedback.")
                st.stop()

            # VADER sentiment
            scores = analyzer.polarity_scores(feedback_text)
            compound = scores["compound"]
            if compound >= 0.05:
                sentiment = "Positive"
            elif compound <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            
            prompt = (
                f"Citizen feedback: '{feedback_text}'.\n\n"
                f"VADER analysis result: {sentiment} (score={compound:.2f}).\n\n"
                "Write a short, natural, and emotionally intelligent summary (1–2 sentences) "
                "explaining why the feedback shows this sentiment. "
                "Use a warm, human-like tone — sound like a civic officer reflecting on the citizen’s feelings. "
                "Avoid technical words or listing tone features; make it sound conversational and thoughtful."
            )

            try:
                model = genai.GenerativeModel("gemini-2.5-flash")  # Fast, free version
                response = model.generate_content(prompt)
                ai_summary = response.text.strip()
            except Exception as e:
                ai_summary = f"(AI summary unavailable: {e})"

            # --- Step 3: Save feedback with AI summary ---
            entry = {
                "complaint_id": cid,
                "feedback": text_fb,
                "sentiment": sentiment,
                "score": compound,
                "sentiment_score": round(compound, 3),
                "ai_summary": ai_summary,
            }

            if os.path.exists("feedback.csv"):
                fdf = pd.read_csv("feedback.csv")
                fdf = pd.concat([fdf, pd.DataFrame([entry])], ignore_index=True)
            else:
                fdf = pd.DataFrame([entry])
            fdf.to_csv("feedback.csv", index=False)

            # --- Step 4: Mark complaint as completed ---
            df.loc[df["complaint_id"] == cid, "status"] = "Completed"
            save_data(df)

            # --- Step 5: Show success message ---
            st.success(
                f"✅ Feedback saved! Sentiment: **{sentiment}** "
                f"(Score: {compound:.2f})\n\n🧠 **AI Reasoning:** {ai_summary}"
            )
            st.rerun()

    if st.button("Mark Selected Complaint as Completed"):
        feedback_dialog(int(complete_id))
else:
    st.info("No pending complaints to mark as complete.")

st.subheader("🗂 Feedback History")

def color_sentiment_cell(val: str):
    if val == "Positive":
        return "background-color: #e8f5e9; color: #2e7d32; font-weight: 600;"
    if val == "Neutral":
        return "background-color: #fff8e1; color: #b08900; font-weight: 600;"
    if val == "Negative":
        return "background-color: #ffebee; color: #c62828; font-weight: 600;"
    return ""

if os.path.exists("feedback.csv"):
    fdf = pd.read_csv("feedback.csv")

    # --- Ensure both complaint_id columns are of same type ---
    fdf["complaint_id"] = fdf["complaint_id"].astype(str)
    df["complaint_id"] = df["complaint_id"].astype(str)

    # Select columns to join from df (category + ward optional)
    join_cols = ["complaint_id", "category"]
    if "ward" in df.columns:
        join_cols.append("ward")

    # --- Merge feedback with complaint metadata ---
    merged = fdf.merge(df[join_cols], on="complaint_id", how="left")

    # --- Display feedback overview ---
    show_cols = ["complaint_id", "category", "feedback", "sentiment", "score", "sentiment_score", "ai_summary"]
    show_cols = [c for c in show_cols if c in merged.columns]
    display_df = merged[show_cols].copy()

    left, right = st.columns([3, 2])
    with left:
        styled = display_df.style.applymap(color_sentiment_cell, subset=["sentiment"])
        st.dataframe(styled, use_container_width=True)

    with right:
        st.markdown("#### Sentiment summary")
        counts = (
            merged["sentiment"]
            .value_counts()
            .reindex(["Positive", "Neutral", "Negative"])
            .fillna(0)
            .astype(int)
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", counts.get("Positive", 0))
        c2.metric("Neutral", counts.get("Neutral", 0))
        c3.metric("Negative", counts.get("Negative", 0))

        st.markdown("##### By category")
        if "category" in merged.columns:
            by_cat = (
                merged.groupby(["category", "sentiment"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=["Positive", "Neutral", "Negative"])
                .reset_index()
            )
            st.dataframe(by_cat, use_container_width=True)

        if "ward" in merged.columns:
            st.markdown("##### By ward")
            by_ward = (
                merged.groupby(["ward", "sentiment"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=["Positive", "Neutral", "Negative"])
                .reset_index()
            )
            st.dataframe(by_ward, use_container_width=True)

else:
    st.info("No feedback submitted yet. Once feedback is added, it will appear here with sentiment colors and aggregates.")

















