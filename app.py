import streamlit as st
import pandas as pd
import plotly.express as px
import spacy
import io
from collections import Counter
from itertools import zip_longest

# Load spaCy model (en_core_web_sm must be installed via setup.sh)
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Feedback Analyzer (No API)", layout="wide")
st.title("🧠 Feedback Analyzer (Offline, No API)")

# Utility to split list into chunks of N
def chunked(iterable, size):
    args = [iter(iterable)] * size
    return zip_longest(*args)

# Simple keyword extractor using spaCy noun chunks
def extract_keywords(responses):
    all_keywords = []
    for response in responses:
        doc = nlp(response)
        for chunk in doc.noun_chunks:
            all_keywords.append(chunk.text.lower())
    return all_keywords

# Basic sentiment keyword matching
def classify_sentiment(text):
    text = text.lower()
    positive_words = ["good", "great", "excellent", "helpful", "clear", "amazing", "awesome", "nice", "love", "understood"]
    negative_words = ["bad", "confusing", "poor", "difficult", "boring", "hate", "worst", "unclear", "not helpful"]

    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)

    if pos_count > neg_count:
        return "POSITIVE", "Detected more positive words"
    elif neg_count > pos_count:
        return "NEGATIVE", "Detected more negative words"
    else:
        return "NEUTRAL", "No clear sentiment words found"

uploaded_file = st.file_uploader("📂 Upload Feedback CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_cols = df.select_dtypes(include="object").columns.tolist()
    ignore = ["timestamp", "email", "name", "id"]
    questions = [col for col in text_cols if col.lower() not in ignore]

    st.markdown("### 📊 Question-wise Analysis")
    summary_data = []
    response_data = []

    for group in chunked(questions, 3):
        cols = st.columns(3)
        for i, question in enumerate(group):
            if question is None:
                continue
            with cols[i]:
                st.subheader(f"❓ {question}")
                responses = df[question].dropna().astype(str).tolist()

                sentiments = [classify_sentiment(r) for r in responses]
                labels = [label for label, _ in sentiments]
                reasons = [reason for _, reason in sentiments]

                count = Counter(labels)
                total = len(responses)
                pos = count.get("POSITIVE", 0)
                neg = count.get("NEGATIVE", 0)
                neu = count.get("NEUTRAL", 0)

                percentages = {
                    "Positive": round((pos / total) * 100, 1),
                    "Negative": round((neg / total) * 100, 1),
                    "Neutral": round((neu / total) * 100, 1)
                }

                pie_df = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative", "Neutral"],
                    "Count": [pos, neg, neu]
                })

                pie = px.pie(pie_df, values="Count", names="Sentiment", hole=0.3,
                             title=None, color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(pie, use_container_width=True)

                bar = px.bar(pie_df, x="Sentiment", y="Count", text="Count",
                             color="Sentiment", color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(bar, use_container_width=True)

                summary_data.append({
                    "Question": question,
                    "Total": total,
                    "Positive %": percentages["Positive"],
                    "Negative %": percentages["Negative"],
                    "Neutral %": percentages["Neutral"]
                })

                for r, l, rsn in zip(responses, labels, reasons):
                    response_data.append({
                        "Question": question,
                        "Response": r,
                        "Sentiment": l,
                        "Reason": rsn
                    })

                with st.expander("📋 Sample Responses"):
                    st.dataframe(pd.DataFrame({
                        "Response": responses,
                        "Sentiment": labels,
                        "Reason": reasons
                    }).head(10), use_container_width=True)

    # Keyword Extraction Summary
    st.markdown("## 🗝️ Common Keywords from All Responses")
    all_responses = df[questions].astype(str).fillna("").values.flatten().tolist()
    keywords = extract_keywords(all_responses)
    keyword_counts = Counter(keywords)
    keyword_df = pd.DataFrame(keyword_counts.items(), columns=["Keyword", "Count"]).sort_values(by="Count", ascending=False)
    st.dataframe(keyword_df.head(20), use_container_width=True)

    # Download section
    st.markdown("## 📥 Download Complete Report")
    summary_df = pd.DataFrame(summary_data)
    response_df = pd.DataFrame(response_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        response_df.to_excel(writer, sheet_name="Responses", index=False)
        keyword_df.to_excel(writer, sheet_name="Keywords", index=False)
    output.seek(0)
    st.download_button("📥 Download Excel Report", data=output,
                       file_name="feedback_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Please upload a CSV file to analyze.")
