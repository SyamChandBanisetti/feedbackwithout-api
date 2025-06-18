import io
import pandas as pd
import streamlit as st
import plotly.express as px
from collections import Counter
from itertools import zip_longest
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Feedback Analyzer", layout="wide")
st.title("🧠 Feedback Analyzer (No API Needed)")

def chunked(iterable, size):
    args = [iter(iterable)] * size
    return zip_longest(*args)

@st.cache_data
def extract_keywords(text_list, top_n=10):
    vectorizer = CountVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform(text_list)
    keywords = vectorizer.get_feature_names_out()
    return list(keywords)

@st.cache_data
def classify_sentiment(text):
    text = text.lower()
    if any(word in text for word in ["good", "great", "excellent", "love", "awesome"]):
        return "POSITIVE", "Positive words detected"
    elif any(word in text for word in ["bad", "poor", "terrible", "hate", "worst"]):
        return "NEGATIVE", "Negative words detected"
    elif text.strip() == "":
        return "NEUTRAL", "Empty response"
    else:
        return "NEUTRAL", "No strong sentiment detected"

uploaded_file = st.file_uploader("📂 Upload Feedback CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    text_cols = df.select_dtypes(include="object").columns.tolist()
    ignore_cols = ["timestamp", "email", "name", "id"]
    questions = [col for col in text_cols if col.lower() not in ignore_cols]

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
                labels = [s[0] for s in sentiments]
                reasons = [s[1] for s in sentiments]

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

                with st.container():
                    pie = px.pie(pie_df, values="Count", names="Sentiment", hole=0.3,
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(pie, use_container_width=True, key=f"pie_{question}")

                    bar = px.bar(pie_df, x="Sentiment", y="Count", text="Count",
                                 color="Sentiment", color_discrete_sequence=px.colors.qualitative.Set2)
                    st.plotly_chart(bar, use_container_width=True, key=f"bar_{question}")

                top_keywords = extract_keywords(responses)
                st.markdown("🔑 **Top Keywords:**")
                st.markdown(", ".join(top_keywords))

                st.markdown("📝 **Summary**:")
                st.markdown(f"Total: {total} responses")
                st.markdown(f"👍 Positive: {percentages['Positive']}%")
                st.markdown(f"👎 Negative: {percentages['Negative']}%")
                st.markdown(f"😐 Neutral: {percentages['Neutral']}%")

                summary_data.append({
                    "Question": question,
                    "Total": total,
                    "Positive %": percentages["Positive"],
                    "Negative %": percentages["Negative"],
                    "Neutral %": percentages["Neutral"],
                    "Top Keywords": ", ".join(top_keywords)
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
                       file_name="feedback_report.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Please upload a CSV file to analyze.")
