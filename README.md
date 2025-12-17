## An NLP Analysis of Linguistic Patterns in Netflix User Reviews

**Course: CIS 9665 – Applied Natural Language Processing**
Instructor: Chaoqun Deng
Author: Apu Datta

## 📌 Project Overview

This project applies Natural Language Processing (NLP) techniques to analyze large-scale Netflix app reviews and understand how linguistic patterns reflect user sentiment and perceived content quality. Using over 113,000 Netflix mobile app reviews, the analysis goes beyond star ratings to examine what users say, how they say it, and whether review text alone can predict numerical ratings.

The work was completed as a group project, combining exploratory linguistic analysis, topic modeling, classical machine learning, and deep learning models to provide both interpretability and predictive insight.
My individual contribution focuses primarily on predictive modeling and evaluation, addressing whether user ratings can be inferred directly from written review text.

## 🎯 Research Questions 

**How Do Linguistic Patterns in Netflix Reviews Reflect the Overall Sentiment and Perceived Quality of the Content Being Reviewed?**

The project is guided by the following research questions:

> Can we accurately predict a reviewer’s rating based solely on their written text? My primary focus. I build and evaluate supervised machine learning and deep learning models to assess how well review text alone predicts 1–5 star ratings.

> What themes or topics are most associated with positive versus negative reviews? Explored at the group level using linguistic analysis and topic modeling.

> Do reviewers focus more on story, acting, or production quality when leaving extreme ratings? Examined through word frequency patterns and linguistic differences between 1-star and 5-star reviews.

## 📂 Dataset

> Source: Kaggle – Netflix Reviews with NLP
- Website: https://www.kaggle.com/code/darrylljk/netflix-reviews-with-nlp/input
- Size: 113,068 Netflix mobile app reviews

> Key Fields
- content – free-text review
- score – star rating (1–5)
- metadata (user, timestamp, app version)

## Data Characteristics

> Ratings are highly imbalanced, with most reviews at 1★ and 5★
> Mid-range ratings (2–4★) are relatively rare
> Review length ranges from very short comments to detailed narratives
> This imbalance has important implications for modeling and evaluation

** 🤖 Predictive Modeling: Supervised Learning Setup

- Input (X): Cleaned review text
- Target (y): Star rating (1–5)
- Split: 80% train / 20% test (stratified)

## Models Implemented

**Classical Machine Learning (TF-IDF based):**

> Logistic Regression (best baseline)
> Linear SVM
> Multinomial Naive Bayes
> XGBoost

**Deep Learning:**

> Bi-LSTM (tokenized sequences)
> BERT (attempted; limited by environment constraints)

## 📊 Model Performance Summary

- Logistic Regression (TF-IDF) performs best among classical models (~65% accuracy)
- Models perform very well on extreme ratings (1★ and 5★)
- Mid-range ratings (2★–4★) are consistently difficult to classify
- Random oversampling improves class fairness but lowers overall accuracy
- Bi-LSTM matches classical performance but does not fully resolve mixed-sentiment ambiguity

These results confirm that text alone reliably captures polarized sentiment, while subtle or mixed evaluations remain challenging.

## 📈 Evaluation & Interpretation

- Confusion matrices show strong diagonal performance for 1★ and 5★
- Mid-range ratings are often misclassified toward extreme classes
- Linguistic ambiguity and class imbalance are the primary causes

Overall, the findings demonstrate a strong alignment between linguistic expression and numerical ratings, validating the use of NLP for large-scale sentiment inference.

## 🧪 Interactive Review Prediction Demo

- An interactive demo allows users to input arbitrary review text and receive:
- Predicted rating from Logistic Regression
- Predicted rating from Bi-LSTM

This component demonstrates real-world applicability and model behavior using unseen text.

```text 
📁 Repository Structure
├── Data/
│   └── netflix_reviews.csv
├── nlp_project_working/
│   └── nlp_report_individual_analysis
	└──	nlp_Term_Project _netflix_review.ipynb
	└──	Term_Project_Codeing_Apu_Datta
	└──	README
```

## 🚀 How to Run the Project

- pip install pandas numpy scikit-learn nltk spacy tensorflow transformers xgboost
- python -m nltk.downloader stopwords
- python -m spacy download en_core_web_sm

## 🧠 Key Takeaways

- Linguistic patterns strongly mirror user sentiment
- Text alone is sufficient to predict extreme satisfaction or dissatisfaction
- Mixed and neutral reviews remain an open challenge in NLP
- Classical linear models remain highly competitive for sparse text features

## 📄 References

- Kaggle Netflix Reviews Dataset
- NLTK, spaCy, scikit-learn, TensorFlow, Hugging Face Transformers