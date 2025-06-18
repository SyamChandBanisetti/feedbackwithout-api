import os
import io
import pandas as pd
import plotly.express as px
import streamlit as st
from collections import Counter
import spacy
from itertools import zip_longest

# Load spaCy small English model
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Feedback Analyzer", layout="wide")
st.title("🧠 Offline Feedback Analyzer (No API Needed)")

def chunked(iterable, size):
    args = [iter(iterable)] * size
    return zip_longest(*args)

def classify_sentiments(texts):
    results = []
    for text in texts:
        text = text.strip()
        if not text:
            results.append(("NEUTRAL", "Empty response"))
            continue

        text_lower = text.lower()
        if any(x in text_lower for x in ["good", "excellent", "nice", "love", "helpful"]):
            results.append(("POSITIVE", "Contains positive words"))
        elif any(x in text_lower for x in ["bad", "poor", "hate", "difficult", "worst"]):
            results.append(("NEGATIVE", "Contains negative words"))
        else:
            results.append(("NEUTRAL", "No clear sentiment"))
    return results

def extract_keywords(texts):
    all_keywords = []
    for text in texts:
        doc = nlp(text)
        for chunk in doc.noun_chunks:
            all_keywords.append(chunk.text.lower())
    return Counter(all_keywords).most_common(10)

def summarize_sentiments(question, percentages):
    pos = percentages["Positive"]
    neg = percentages["Negative"]
    neu = percentages["Neutral"]

    summary = f"The feedback for '{question}' includes {pos}% positive, {neg}% negative, and {neu}% neutral sentiments."
    insights = "Majority seem satisfied." if pos > max(neg, neu) else (
        "There is room for improvement." if neg > pos else "Mixed opinions observed.")
    reco = "Continue current efforts." if pos > 60 else (
        "Investigate complaints." if neg > 40 else "Seek more detailed feedback.")
    return summary, insights, reco

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

                st.plotly_chart(px.pie(pie_df, values="Count", names="Sentiment", hole=0.3,
                                       color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)

                st.plotly_chart(px.bar(pie_df, x="Sentiment", y="Count", text="Count",
                                       color="Sentiment", color_discrete_sequence=px.colors.qualitative.Set2), use_container_width=True)

                summary, insight, reco = summarize_sentiments(question, percentages)
                st.markdown(f"📝 **Summary**: {summary}")
                st.markdown(f"🔎 **Insights**: {insight}")
                st.markdown(f"✅ **Recommendations**: {reco}")

                summary_data.append({
                    "Question": question,
                    "Total": total,
                    "Positive %": percentages["Positive"],
                    "Negative %": percentages["Negative"],
                    "Neutral %": percentages["Neutral"],
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

    # Keywords section
    st.markdown("## 🔑 Top Keywords from All Responses")
    all_texts = df[questions].astype(str).values.flatten().tolist()
    keywords = extract_keywords(all_texts)
    st.dataframe(pd.DataFrame(keywords, columns=["Keyword", "Frequency"]))

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
                       file_name="feedback_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Please upload a CSV file to analyze.")
