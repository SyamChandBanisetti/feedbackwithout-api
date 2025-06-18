import io
import pandas as pd
import plotly.express as px
import streamlit as st
from collections import Counter
from textblob import TextBlob
from itertools import zip_longest
from wordcloud import STOPWORDS

st.set_page_config(page_title="📋 Advanced Feedback Analyzer", layout="wide")
st.title("🧠 Advanced Offline Feedback Analyzer")
st.markdown("Analyze textual feedback with **sentiment, keyword, and response pattern insights** — no internet or API required.")

# Utility functions
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
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.2:
            label = "POSITIVE"
            reason = "Clearly positive tone"
        elif polarity < -0.2:
            label = "NEGATIVE"
            reason = "Clearly negative tone"
        else:
            label = "NEUTRAL"
            reason = "Mixed or unclear tone"
        results.append((label, reason))
    return results

def summarize_sentiments(question, percentages):
    pos, neg, neu = percentages["Positive"], percentages["Negative"], percentages["Neutral"]
    summary = f"Among all feedback for **'{question}'**, {pos}% were positive, {neg}% negative, and {neu}% neutral."
    if pos >= 60:
        insight = "This aspect is highly appreciated by users."
        reco = "Continue reinforcing this strength and promote it more."
    elif neg >= 40:
        insight = "There is significant dissatisfaction."
        reco = "Investigate root causes and take corrective actions."
    elif neu >= 50:
        insight = "Many responses are ambiguous or neutral."
        reco = "Consider rewording the question for clearer answers."
    else:
        insight = "Feedback shows a mixed sentiment distribution."
        reco = "Further qualitative analysis might be beneficial."
    return summary, insight, reco

def extract_keywords(texts, stopwords=set(STOPWORDS)):
    phrases = []
    for text in texts:
        blob = TextBlob(text)
        for phrase in blob.noun_phrases:
            if phrase.lower() not in stopwords:
                phrases.append(phrase.lower())
    return Counter(phrases).most_common(10)

def get_response_lengths(texts):
    lengths = [len(text.split()) for text in texts]
    return {
        "Average": round(sum(lengths)/len(lengths), 1),
        "Min": min(lengths),
        "Max": max(lengths)
    }

# File uploader
uploaded_file = st.file_uploader("📂 Upload your feedback CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_cols = df.select_dtypes(include="object").columns.tolist()
    ignore_cols = ["timestamp", "email", "name", "id"]
    questions = [col for col in text_cols if col.lower() not in ignore_cols]

    st.markdown("### 📊 Question-wise Feedback Analysis")
    summary_data, response_data = [], []

    for group in chunked(questions, 2):
        cols = st.columns(2)
        for i, question in enumerate(group):
            if not question: continue
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

                # Horizontal ratio bar
                ratio_chart = px.bar(pd.DataFrame({
                    "Sentiment": ["Positive", "Negative", "Neutral"],
                    "Value": [pos, neg, neu]
                }), x="Value", y=["Sentiment"], orientation="h",
                   color="Sentiment", text="Value",
                   color_discrete_map={"Positive": "#66c2a5", "Negative": "#fc8d62", "Neutral": "#8da0cb"})
                st.plotly_chart(ratio_chart, use_container_width=True)

                # Pie chart
                pie_df = pd.DataFrame({
                    "Sentiment": ["Positive", "Negative", "Neutral"],
                    "Count": [pos, neg, neu]
                })
                pie = px.pie(pie_df, values="Count", names="Sentiment", hole=0.4)
                st.plotly_chart(pie, use_container_width=True)

                # Summary, insights, and recommendations
                summary, insight, reco = summarize_sentiments(question, percentages)
                st.success(f"📝 **Summary**: {summary}")
                st.info(f"🔍 **Insights**: {insight}")
                st.warning(f"✅ **Recommendations**: {reco}")

                # Additional metrics
                with st.expander("📐 Response Length Stats & Keywords"):
                    lengths = get_response_lengths(responses)
                    keywords = extract_keywords(responses)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg Words", lengths["Average"])
                        st.metric("Min Words", lengths["Min"])
                        st.metric("Max Words", lengths["Max"])
                    with col2:
                        st.markdown("**Top Keywords**:")
                        for kw, freq in keywords:
                            st.markdown(f"• {kw} ({freq}x)")

                # Response samples
                with st.expander("📋 Sample Responses (first 10)"):
                    emoji = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "⚪"}
                    st.dataframe(pd.DataFrame({
                        "Sentiment": [f"{emoji[l]} {l}" for l in labels[:10]],
                        "Response": responses[:10],
                        "Reason": reasons[:10]
                    }), use_container_width=True)

                summary_data.append({
                    "Question": question,
                    "Total Responses": total,
                    "Positive %": percentages["Positive"],
                    "Negative %": percentages["Negative"],
                    "Neutral %": percentages["Neutral"],
                    "Avg Words": lengths["Average"],
                    "Top Keywords": ", ".join([kw for kw, _ in keywords]),
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

    # Download Excel report
    st.markdown("---")
    st.markdown("### 📥 Download Full Report")
    summary_df = pd.DataFrame(summary_data)
    response_df = pd.DataFrame(response_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        response_df.to_excel(writer, sheet_name="Responses", index=False)
    output.seek(0)
    st.download_button("📤 Download Excel Report", data=output,
                       file_name="enhanced_feedback_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("👆 Upload a CSV file with open-ended textual feedback to begin.")
