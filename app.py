import streamlit as st
import joblib

# Load trained NLP model (TF-IDF + Logistic Regression pipeline)
MODEL_PATH = "model.joblib"
model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Netflix Review Rating Predictor",
    page_icon="🎬",
    layout="centered"
)

# Header
st.markdown("""
### CIS 9665 – Applied Natural Language Processing
**Instructor:** Chaoqun Deng  
**Student:** Apu Datta  

##### An NLP Analysis of Linguistic Patterns in Netflix User Reviews
""")

# Tabs
tab1, tab2 = st.tabs(["⭐ Review Predictor", "📊 Findings & Charts"])

# -------------------------
# TAB 1: Predictor + Intro
# -------------------------
with tab1:
    st.title("🎬 Netflix Review Rating Predictor")
    st.write("Type a Netflix app review and get a predicted star rating (1–5 ⭐).")

    with st.expander("📌 Introduction & Background", expanded=False):
        st.write("""
Natural Language Processing (NLP) provides powerful techniques for analyzing large volumes of unstructured text, 
enabling organizations to extract meaningful insights from user-generated content at scale. 
For streaming platforms such as Netflix, mobile app reviews serve as an important source of feedback, reflecting user 
satisfaction, usability concerns, and perceptions of content quality. Because manually reviewing tens of thousands of 
reviews is impractical, NLP methods allow for systematic identification of sentiment patterns, recurring themes, and 
linguistic signals that would otherwise remain hidden.
""")

    with st.expander("🎯 Motivation", expanded=False):
        st.write("""
Applying NLP to Netflix user reviews helps uncover the underlying factors driving user satisfaction or frustration, 
including content availability, app performance, and technical issues—details that are not fully captured by numerical star 
ratings alone. By examining linguistic differences between positive and negative reviews, this analysis supports data-driven 
decision-making, enables early detection of emerging issues, and contributes to improving overall user experience and platform 
quality.
""")

    with st.expander("📂 Dataset Description", expanded=False):
        st.write("""
This study uses the Kaggle dataset “Netflix Reviews with NLP,” which contains 113,068 mobile app reviews along with corresponding
1–5 star ratings and metadata. During preprocessing, duplicate review IDs were removed to ensure data quality. The dataset 
exhibits a strong class imbalance, with one star reviews accounting for approximately 39% of all entries and five star reviews 
representing about 28%. This polarization indicates that users are more likely to leave reviews when experiencing extreme 
dissatisfaction or satisfaction, motivating a deeper linguistic analysis across rating levels and presenting meaningful 
challenges for multi-class prediction.            
""")

    # Text area (inside Tab 1)
    review_text = st.text_area(
        "Enter your review text:",
        value=st.session_state.get("review", ""),
        placeholder="Example: The app keeps crashing after the update. Very frustrating...",
        height=180
    )

    # Sample buttons (inside Tab 1)
    col1, col2, col3 = st.columns(3)

    if col1.button("Sample 1★"):
        st.session_state["review"] = "This app is terrible. It crashes all the time and nothing works."
        st.rerun()

    if col2.button("Sample 3★"):
        st.session_state["review"] = "It is okay, but sometimes it buffers and the app feels slow."
        st.rerun()

    if col3.button("Sample 5★"):
        st.session_state["review"] = "Love Netflix! Great shows, smooth streaming, and very easy to use."
        st.rerun()

    # Predict button (inside Tab 1)
    if st.button("Predict Rating ⭐", use_container_width=True):
        if not review_text.strip():
            st.warning("Please type a review first.")
        else:
            prediction = model.predict([review_text])[0]
            st.success(f"⭐ Predicted Rating: {prediction} / 5")
            st.info("Tip: The model is usually most accurate for clearly negative (1★) or clearly positive (5★) reviews.")

    # Disclaimer (inside Tab 1)
    st.markdown("""
#### **Model Disclaimer & Interpretation**

This prediction is based solely on the linguistic patterns present in the review text.

Project results show that clearly positive language is strongly associated with higher star ratings,
while expressions of frustration or dissatisfaction typically align with lower ratings.

However, mixed/neutral reviews (2★–4★) are harder to classify, and the dataset is imbalanced (more 1★ and 5★).
Predictions are probabilistic estimates, not final judgments of content quality.
""")

# -------------------------
# TAB 2: Findings + PNGs
# -------------------------
with tab2:
    st.header("📊 Findings & Visual Results (Report)")

    st.write("""
This section summarizes the key EDA, sentiment, and model evaluation results from the Netflix review dataset.
Each figure is followed by a short interpretation.
""")

    # 1) Ratings Distribution
    st.subheader("1) Ratings Distribution (1–5 Stars)")
    st.image("images/ratings_distribution_plots.png", use_container_width=True)
    st.write("""
**Finding:** The dataset is highly imbalanced.  
1★ reviews are the largest group and 5★ reviews are the second largest.  
Mid-range ratings (2★–4★) are much smaller, which makes it harder for models to learn and predict those middle classes accurately.
""")
    st.markdown("---")

    # 2) Text Preprocessing Example
    st.subheader("2) Text Preprocessing (Original vs Cleaned)")
    st.image("images/original_vs_cleaned_side-by-side.png", use_container_width=True)
    st.write("""
**Finding:** Cleaning makes text more consistent for modeling.  
After cleaning (lowercase, removing punctuation/stopwords, lemmatization), the review becomes shorter and focuses on meaningful words.
""")
    st.markdown("---")

    # 3) Sentiment Summary Table (VADER)
    st.subheader("3) Sentiment Analysis Summary (VADER Results)")
    st.image("images/sentiment_analysis.png", use_container_width=True)
    st.write("""
**Finding:** Sentiment matches rating very clearly.  
- Review counts: Positive ≈ 69,270, Negative ≈ 30,018, Neutral ≈ 13,780  
- Average rating by sentiment: Negative ≈ 1.661, Neutral ≈ 2.085, Positive ≈ 3.456  
Also, the average sentiment score increases steadily from 1★ to 5★ (more positive sentiment → higher rating).
""")
    st.markdown("---")

    # 4) Sentiment vs Rating Count Chart
    st.subheader("4) Sentiment Label Distribution Across Ratings")
    st.image("images/Sentiment_vs_rating_count.png", use_container_width=True)
    st.write("""
**Finding:**  
- 1★ reviews contain a lot of **negative** sentiment.  
- 5★ reviews are mostly **positive** sentiment.  
- Mid-range ratings (2★–4★) show more **mixed/neutral** sentiment, which is one reason these classes are harder to predict.
""")
    st.markdown("---")

    # 5) Correlation Heatmap
    st.subheader("5) Correlation Heatmap: Rating, Sentiment & Text Length")
    st.image("images/correlation_heatmap_rating,_sentiment.png", use_container_width=True)
    st.write("""
**Finding (from the numbers in the heatmap):**  
- Rating vs Sentiment score has a **moderate positive correlation (~0.56)** → more positive sentiment tends to mean higher rating.  
- Word count and character count are **almost perfectly correlated (~0.99)** → they measure the same “length” signal.  
- Rating vs length is **slightly negative** (about -0.17 to -0.19), meaning longer reviews do not necessarily mean higher ratings.
""")
    st.markdown("---")

    # 6) Top 20 Most Common Words
    st.subheader("6) Top 20 Most Common Words (All Reviews)")
    st.image("images/top_20_most_common_word.png", use_container_width=True)
    st.write("""
**Finding:** Most frequent words are common stopwords (the, i, to, and, it…).  
This shows why stopword removal is important—these words appear a lot but do not help predict ratings.
""")
    st.markdown("---")

    # 7) Top Words in 1★ vs 5★ Reviews
    st.subheader("7) Top Words: 1★ Reviews vs 5★ Reviews")
    st.image("images/top_words_1_&_5_star_reviews.png", use_container_width=True)
    st.write("""
**Finding:** Word choices differ strongly between negative and positive reviews.  
- 1★ reviews contain more frustration/negation words (e.g., **not**, app, netflix, etc.).  
- 5★ reviews contain more positive words (e.g., **love**, watch, movies, best, etc.).  
This supports the idea that language clearly reflects user sentiment.
""")
    st.markdown("---")

    # 8) Model Performance Summary (table image)
    st.subheader("8) Final Model Performance Summary (Classical ML + Deep Learning)")
    st.image("images/model_performamce_summary.png", use_container_width=True)
    st.write("""
**Finding (from your table):**  
- Best results are around **~0.645 accuracy** (Bi-LSTM ≈ 0.645, Logistic Regression ≈ 0.645).  
- Linear SVM / Multinomial NB / XGBoost are slightly lower.  
- Balanced Logistic Regression shows **lower overall accuracy** (because it reduces bias toward the big classes).  
- BERT was **not completed** due to environment/framework constraints.
""")
    st.markdown("---")

    # 9) Logistic Regression Classification Report
    st.subheader("9) Logistic Regression (TF-IDF) – Classification Report")
    st.image("images/training_model_logistic_regression.png", use_container_width=True)
    st.write("""
**Finding:** The model is strong on extreme ratings but weak on mid-range.  
- 1★ recall ≈ **0.92**, 5★ recall ≈ **0.83** (very good)  
- 2★ recall ≈ **0.04** (very hard)  
This confirms: text predicts polarized sentiment well, but mixed mid-range classes are difficult.
""")
    st.markdown("---")

    # 10) Confusion Matrix (Logistic Regression)
    st.subheader("10) Confusion Matrix — Logistic Regression (TF-IDF)")
    st.image("images/confusion_matrix_logistic_regression.png", use_container_width=True)
    st.write("""
**Finding:** The confusion matrix shows strong accuracy for 1★ and 5★.  
Mid-range ratings (2★–4★) are often misclassified toward 1★ or 5★, because their language is more mixed/neutral and the dataset is imbalanced.
""")
    st.markdown("---")

    # 11) Confusion Matrix (Bi-LSTM)
    st.subheader("11) Confusion Matrix — Bi-LSTM Model")
    st.image("images/confusion_matrix_bi-LSTM.png", use_container_width=True)
    st.write("""
**Finding:** The Bi-LSTM model shows performance patterns very similar to Logistic Regression.  
- Extreme ratings (1★ and 5★) are predicted very well.  
- 1★ reviews have very high correct classification counts, and 5★ reviews are also strongly identified.  
- Mid-range ratings (2★–4★) are frequently misclassified, often shifting toward extreme categories.  

This confirms that even deep learning models struggle with subtle or mixed sentiment, while polarized sentiment is easier to predict from text alone.
""")
    st.markdown("---")

    st.subheader("✅ Final Takeaways")
    st.write("""
- The dataset is imbalanced (many 1★ and 5★ reviews), which makes 2★–4★ harder to predict.
- Sentiment strongly increases with rating (more positive sentiment → higher stars).
- Logistic Regression (TF-IDF) and Bi-LSTM perform best overall (~0.645 accuracy).
- Both models predict extreme ratings well, but struggle with mid-range ratings due to mixed language.
""")
