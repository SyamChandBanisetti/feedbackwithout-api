import os
import io
import pandas as pd
import plotly.express as px
import streamlit as st
from collections import Counter
from itertools import zip_longest
from textblob import TextBlob
import nltk

# Download necessary corpora (runs only once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Streamlit UI setup
st.set_page_config(page_title="Offline Feedback Analyzer", layout="wide")
st.title("🧠 Offline Feedback Analyzer (No API Required)")

# Helper: Chunking for layout
def chunked(iterable, size):
    args = [iter(iterable)] * size
    return zip_longest(*args)

# Sentiment classifier using TextBlob
def classify_sentiments(texts):
    results = []
    for text in texts:
        text = text.strip()
        if not text:
            results.append(("NEUTRAL", "Empty response"))
            continue
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            label = "POSITIVE"
        elif polarity < -0.1:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        reason = f"Polarity={polarity:.2f}"
        results.append((label, reason))
    return results

# Extract top keywords
def extract_keywords(texts, top_n=5):
    all_keywords = []
    for text in texts:
        blob = TextBlob(text)
        all_keywords.extend(blob.noun_phrases)
    keywords_freq = Counter(all_keywords)
    return keywords_freq.most_common(top_n)

# Summarize based on sentiment distribution
def summarize_sentiments(percentages):
    pos = percentages["Positive"]
    neg = percentages["Negative"]
    neu = percentages["Neutral"]

    if pos > 60:
        summary = "Majority feedback is positive."
        insight = "Students appreciated the content or teaching."
        reco = "Continue current efforts; consider gathering more specific praise."
    elif neg > 40:
        summary = "Significant negative sentiment."
        insight = "There may be dissatisfaction in delivery, pace, or content."
        reco = "Review complaints or open text feedback to identify key concerns."
    elif neu > 50:
        summary = "Feedback is mostly neutral."
        insight = "Students may not be highly engaged or responses are vague."
        reco = "Encourage more detailed and expressive feedback."
    else:
        summary = "Mixed feedback."
        insight = "Varied experiences among students."
        reco = "Investigate both positive and negative patterns for deeper insight."

    return summary, insight, reco

# File upload
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
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(pie, use_container_width=True)

                bar = px.bar(pie_df, x="Sentiment", y="Count", text="Count",
                             color="Sentiment", color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(bar, use_container_width=True)

                keywords = extract_keywords(responses)
                keywords_list = ", ".join([kw for kw, _ in keywords])
                summary, insight, reco = summarize_sentiments(percentages)

                st.markdown(f"🔑 **Top Keywords**: {keywords_list}")
                st.markdown(f"📝 **Summary**: {summary}")
                st.markdown(f"🔎 **Insights**: {insight}")
                st.markdown(f"✅ **Recommendations**: {reco}")

                summary_data.append({
                    "Question": question,
                    "Total": total,
                    "Positive %": percentages["Positive"],
                    "Negative %": percentages["Negative"],
                    "Neutral %": percentages["Neutral"],
                    "Top Keywords": keywords_list,
                    "Summary": summary,
                    "Insights": insight,
                    "Recommendations": reco
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

    # Final download section
    st.markdown("## 📥 Download Complete Report")
    summary_df = pd.DataFrame(summary_data)
    response_df = pd.DataFrame(response_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        response_df.to_excel(writer, sheet_name="Responses", index=False)
    output.seek(0)
    st.download_button("📥 Download Excel Report", data=output,
                       file_name="offline_feedback_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Please upload a CSV file to analyze.")
