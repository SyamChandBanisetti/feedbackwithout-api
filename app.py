import os
import io
import pandas as pd
import plotly.express as px
import streamlit as st
from collections import Counter
from rake_nltk import Rake
from itertools import zip_longest

st.set_page_config(page_title="Feedback Analyzer", layout="wide")
st.title("🧠 Offline Feedback Analyzer (No API Required)")

def chunked(iterable, size):
    args = [iter(iterable)] * size
    return zip_longest(*args)

@st.cache_resource
def get_rake():
    return Rake()

def classify_sentiments(texts):
    results = []
    for text in texts:
        text = text.strip().lower()
        if not text:
            results.append(("NEUTRAL", "Empty response"))
            continue
        # Simple rule-based sentiment
        if any(w in text for w in ["good", "great", "excellent", "loved", "amazing", "fantastic"]):
            results.append(("POSITIVE", "Positive keywords found"))
        elif any(w in text for w in ["bad", "poor", "terrible", "worst", "boring", "hate"]):
            results.append(("NEGATIVE", "Negative keywords found"))
        else:
            results.append(("NEUTRAL", "No strong sentiment keywords"))
    return results

def extract_keywords(responses):
    r = get_rake()
    all_keywords = []
    for response in responses:
        r.extract_keywords_from_text(response)
        all_keywords.extend(r.get_ranked_phrases()[:2])  # top 2 per response
    return all_keywords

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
                sentiments = classify_sentiments(responses)
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

                st.markdown(f"📝 **Summary**: {percentages}")
                summary_data.append({
                    "Question": question,
                    "Total": total,
                    "Positive %": percentages["Positive"],
                    "Negative %": percentages["Negative"],
                    "Neutral %": percentages["Neutral"],
                })

                for r, l, rsn in zip(responses, labels, reasons):
                    response_data.append({
                        "Question": question,
                        "Response": r,
                        "Sentiment": l,
                        "Reason": rsn
                    })

                with st.expander("📋 View Sample Responses"):
                    st.dataframe(pd.DataFrame({
                        "Response": responses,
                        "Sentiment": labels,
                        "Reason": reasons
                    }).head(10), use_container_width=True)

    # Global Keyword Analysis
    st.markdown("### 🔑 Keyword Analysis (All Responses Combined)")
    all_text = df[questions].astype(str).fillna("").values.flatten().tolist()
    keywords = extract_keywords(all_text)
    if keywords:
        keyword_counts = Counter(keywords).most_common(20)
        keyword_df = pd.DataFrame(keyword_counts, columns=["Keyword", "Frequency"])
        fig = px.bar(keyword_df, x="Keyword", y="Frequency", title="Top Keywords", color="Frequency",
                     color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    # Final Report Download
    st.markdown("## 📥 Download Complete Report")
    summary_df = pd.DataFrame(summary_data)
    response_df = pd.DataFrame(response_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        response_df.to_excel(writer, sheet_name="Responses", index=False)
    output.seek(0)
    st.download_button("📥 Download Excel Report", data=output,
                       file_name="feedback_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Please upload a CSV file to analyze.")
