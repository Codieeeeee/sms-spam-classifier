import streamlit as st
import pickle
import nltk
import os
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# nltk.data.path.append(os.path.expanduser("~/nltk_data"))
# nltk.download('punkt', download_dir=os.path.expanduser("~/nltk_data"))

# nltk.data.path.append('/home/siddhu/nltk_data')
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# For preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for t in text:
        if t.isalnum():
            y.append(t)
    text = y[:]
    y.clear()

    for t in text:
        if t not in stopwords.words('english') and t not in string.punctuation:
            y.append(t)
    text = y[:]
    y.clear()

    for t in text:
        y.append(ps.stem(t))
    return " ".join(y)

# Load the saved model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .stTextArea textarea { font-size: 16px; }
    .stButton button { font-size: 18px; font-weight: bold; padding: 10px; width: 100%; }
    .stTitle { margin-top: -50px; padding-bottom: 10px; text-align: center; }
    .stMarkdown { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='stTitle'>📩 SMS Spam Detector</h1>", unsafe_allow_html=True)
st.markdown("### 🚀 Detect whether a message is **Spam** or **Not Spam**")

# User input section
st.markdown("#### 📜 Enter your message below:")
message = st.text_area(" ", height=125)

# Button
if st.button("🔍 Classify Message"):
    # Preprocess the input text
    preprocessed_text = transform_text(message)

    # Vectorization
    transformed_text = vectorizer.transform([preprocessed_text])

    # Make prediction
    prediction = model.predict(transformed_text)[0]

    # Display result
    st.markdown("---")
    if prediction == 1:
        st.error("🚨 **Spam Message Detected!**")
    else:
        st.success("✅ **This is not a spam Message!**")


# Footer
st.markdown("---")
st.markdown("📌 **Built with Machine Learning & Streamlit** | 🚀 *Fast & Secure*")
