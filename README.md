# Cyber Risk Sentiment Analysis using Natural Language Processing

## Project Overview
This project performs sentiment analysis on Reddit posts from cyber risk-related subreddits to gauge public sentiment on cyber risks, vulnerabilities, and incidents. Leveraging Natural Language Processing (NLP) techniques, it analyzes large volumes of unstructured data and provides insights into trending cyber threats, user sentiments, and potential areas of cyber risk. The results aim to support cybersecurity decision-making, threat intelligence, and risk management for organizations like Marsh.

---

## Objectives
- **Extract and analyze cybersecurity discussions** from Reddit to assess public sentiment on cyber risk topics.
- **Use NLP techniques** to process, analyze, and visualize unstructured data on cyber threats.
- **Generate actionable insights** on emerging cyber risks, such as ransomware, data breaches, and vulnerabilities, to aid in cyber intelligence and risk assessment.

---

## Methodology

1. **Data Collection**  
   Using the Reddit API (PRAW), data is collected from selected subreddits focused on cybersecurity, including:
   - `cybersecurity`
   - `netsec`
   - `infosec`
   - `AskNetsec`
   - `ThreatIntel`

2. **Data Preprocessing**  
   Text data undergoes preprocessing steps:
   - Removing stop words
   - Lemmatization to normalize words
   - Text cleaning with regular expressions

3. **Sentiment Analysis**  
   Two sentiment analysis methods are used:
   - **VADER**: A rule-based model optimized for social media text, assessing sentiment polarity.
   - **BERT (DistilBERT)**: A transformer model fine-tuned for sentiment analysis, providing more nuanced sentiment classification.

4. **Feature Extraction and Vectorization**  
   - **TF-IDF Vectorization** for feature extraction.
   - **Count Vectorizer** to analyze word frequency and uncover common themes in cybersecurity discussions.

5. **Visualization**  
   - Word clouds to visualize commonly discussed topics.
   - Bar charts and sentiment distribution graphs to present insights into sentiment trends.

---

## Technology Stack
- **Programming Language**: Python
- **Data Collection**: PRAW (Python Reddit API Wrapper)
- **Data Processing and NLP**: NLTK, TextBlob, WordNetLemmatizer, Stopwords, re (regex)
- **Sentiment Analysis**: VADER, Hugging Face Transformers (DistilBERT)
- **Feature Extraction**: Scikit-learn (TF-IDF Vectorizer, Count Vectorizer)
- **Visualization**: Matplotlib, WordCloud

---

## Algorithms and Libraries Used

- **VADER Sentiment Analysis**: A lexicon and rule-based model designed for social media sentiment analysis.
- **DistilBERT Sentiment Analysis**: A transformer-based model fine-tuned on sentiment data (SST-2).
- **TF-IDF Vectorization**: Calculates term frequency-inverse document frequency for feature extraction.
- **Count Vectorization**: Extracts features based on word frequency, useful for identifying prominent themes in cybersecurity discussions.

---

## Insights and Findings

- **Sentiment Trends**: Analysis reveals predominant sentiment trends regarding cyber threats. For instance, spikes in negative sentiment may indicate heightened concerns about specific risks like ransomware or phishing.
- **Common Cyber Risks**: Word cloud visualizations and vectorized features highlight common topics in cybersecurity, such as vulnerabilities, breaches, and malware, giving insight into which areas are of greatest concern.
- **Comparison of Sentiment Models**: VADER provides quick sentiment scoring for social media text, while BERT offers a more context-aware sentiment classification, useful for understanding complex discussions.

---

## Conclusion
This project demonstrates the use of NLP in understanding public sentiment on cybersecurity risks. By identifying sentiment patterns and commonly discussed threats, organizations can gain insights that support proactive risk management and inform cybersecurity strategies. Future work may explore the integration of additional NLP models, such as BERT-based topic modeling, and expanded data sources for a more comprehensive risk intelligence assessment.

---
