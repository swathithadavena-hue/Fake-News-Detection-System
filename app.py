import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------------------------
# DOWNLOAD NLTK STOPWORDS
# ---------------------------------
nltk.download('stopwords')

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# ---------------------------------
# CUSTOM CSS
# ---------------------------------
st.markdown("""
<style>

.stApp{
    background-color:#f4f7fb;
}

.main-title{
    font-size:58px;
    font-weight:800;
    color:#1e293b;
    line-height:1.1;
}

.subtitle{
    font-size:22px;
    color:#475569;
            
    margin-top:10px;
}

.card{
    background:white;
    padding:30px;
    border-radius:22px;
    box-shadow:0px 5px 18px rgba(0,0,0,0.08);
    margin-bottom:20px;
}

.result-card{
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
}

.stButton>button{
    background:linear-gradient(to right,#2563eb,#3b82f6);
    color:white;
    border:none;
    border-radius:12px;
    height:50px;
    width:220px;
    font-size:20px;
    font-weight:600;
}

.stTextArea textarea{
    border-radius:15px;
    border:2px solid #cbd5e1;
    font-size:18px;
}

.metric-box{
    background:#eff6ff;
    padding:20px;
    border-radius:15px;
    text-align:center;
}

.metric-value{
    font-size:35px;
    font-weight:bold;
    color:#2563eb;
}

.metric-label{
    font-size:18px;
    color:#475569;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# LOAD MODEL
# ---------------------------------
model = pickle.load(open('fake_news_model.pkl', 'rb'))

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# ---------------------------------
# STEMMER
# ---------------------------------
ps = PorterStemmer()

# ---------------------------------
# TEXT PREPROCESSING
# ---------------------------------
def stemming(content):

    stemmed_content = re.sub('[^a-zA-Z]', ' ', str(content))

    stemmed_content = stemmed_content.lower()

    stemmed_content = stemmed_content.split()

    stemmed_content = [
        ps.stem(word)
        for word in stemmed_content
        if word not in stopwords.words('english')
    ]

    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content

# ---------------------------------
# HEADER
# ---------------------------------
header1, header2 = st.columns([1, 1.6])

with header1:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2965/2965879.png",
        width=320
    )

with header2:

    st.markdown("""
    <div class='main-title'>
    Intelligence Fake News<br>
    Detection System
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='subtitle'>
    Machine Learning & NLP powered system for accurate Fake and Real News Detection
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------
# MAIN LAYOUT
# ---------------------------------
left_col, center_col, right_col = st.columns([1,2,1])

# ---------------------------------
# ABOUT PROJECT
# ---------------------------------
with left_col:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2103/2103832.png",
        width=100
    )

    st.subheader("About Project")

    st.write("""
    This system detects whether a news article is REAL or FAKE using:
    
    ✔ Machine Learning  
    ✔ Natural Language Processing  
    ✔ TF-IDF Vectorization  
    ✔ Logistic Regression  
    ✔ Streamlit Interface
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# NEWS INPUT
# ---------------------------------
with center_col:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📰 Enter News Content")

    news_text = st.text_area(
        "Paste News Article Here",
        height=320,
        placeholder="Enter complete news article..."
    )

    predict_button = st.button("🔍 Analyze News")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# DETECTION OVERVIEW
# ---------------------------------
with right_col:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.image(
        "https://cdn-icons-png.flaticon.com/512/4149/4149670.png",
        width=100
    )

    st.subheader("Detection Overview")

    st.write("""
    The system analyzes:
    
    ✔ News Content  
    ✔ Language Patterns  
    ✔ NLP Features  
    ✔ TF-IDF Scores  
    ✔ Prediction Probability
    """)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# PREDICTION
# ---------------------------------
if predict_button:

    if news_text.strip() == "":

        st.warning("Please enter news content.")

    else:

        # Preprocess
        processed_news = stemming(news_text)

        # Vectorize
        transformed_text = vectorizer.transform([processed_news])

        # Prediction
        prediction = model.predict(transformed_text)

        # Probability
        probability = model.predict_proba(transformed_text)

        fake_score = round(probability[0][0] * 100, 2)

        real_score = round(probability[0][1] * 100, 2)

        # ---------------------------------
        # RESULT SECTION
        # ---------------------------------
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        st.subheader("📌 Detection Result")

        if prediction[0] == 1:

            st.success("✅ THIS NEWS IS REAL")

        else:

            st.error("❌ THIS NEWS IS FAKE")

        # ---------------------------------
        # METRICS
        # ---------------------------------
        col1, col2 = st.columns(2)

        with col1:

            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{real_score}%</div>
                <div class='metric-label'>REAL Probability</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{fake_score}%</div>
                <div class='metric-label'>FAKE Probability</div>
            </div>
            """, unsafe_allow_html=True)

        # ---------------------------------
        # REALISTIC GRAPH
        # ---------------------------------
        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("📊 News Analysis Graph")

        categories = ['REAL', 'FAKE']

        values = [real_score, fake_score]

        fig, ax = plt.subplots(figsize=(7,4))

        bars = ax.bar(categories, values)

        ax.set_ylim(0,100)

        ax.set_ylabel("Confidence Percentage")

        ax.set_title("Prediction Confidence Analysis")

        for bar in bars:

            yval = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width()/2,
                yval + 1,
                f'{yval}%',
                ha='center',
                fontsize=12
            )

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("---")

st.caption(
    "Developed using Machine Learning and NLP for Fake News Detection"
)