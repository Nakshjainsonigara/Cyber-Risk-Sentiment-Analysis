# Cyber Risk Sentiment Analysis using Natural Language Processing

## Project Overview
This project performs sentiment analysis on Reddit posts from cyber risk-related subreddits to gauge public sentiment on cyber risks, vulnerabilities, and incidents. Leveraging Natural Language Processing (NLP) techniques, it analyzes large volumes of unstructured data and provides insights into trending cyber threats, user sentiments, and potential areas of cyber risk. The results aim to support cybersecurity decision-making, threat intelligence, and risk management for organizations.

View the [Interactive Kaggle Notebook](https://www.kaggle.com/code/nakshjain11/cyber-risk-sentiment-analysis-using-nlp)
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

Sentiment analysis of **cyber risk** discussions on social media or forums can provide valuable insights for organizations operating in the cybersecurity domain. By analyzing how people talk about **cybersecurity** and **cyber risk**, organization can gain an understanding of current trends, emerging threats, and public perception, helping them make informed decisions.

- **Positive Sentiment**: Positive posts highlight confidence in **cybersecurity** practices and solutions. These discussions often focus on successful implementations of security measures, the adoption of new technologies, or effective responses to cyber incidents. Organizations can track these posts to identify new tools, strategies, or practices that are helping reduce **cyber risk**. This helps companies stay updated on the latest advancements and share best practices with clients or stakeholders.

- **Negative Sentiment**: Negative posts often reflect concerns or frustrations related to **cybersecurity** challenges, such as data breaches, increasing cyberattacks, or new vulnerabilities. By analyzing these posts, companies can stay informed about potential threats and risks in the industry. This information can be used to adjust security strategies, anticipate new attack vectors, and identify areas that require immediate attention. Negative sentiment also helps organizations understand what concerns the public or professionals, allowing them to address these issues proactively.

- **Neutral Sentiment**: Neutral posts provide objective, fact-based information without strong emotions. These could include news articles, updates on **cybersecurity** policies, or technical discussions about vulnerabilities and solutions. Although neutral posts may not indicate any urgent problems, they offer valuable data about ongoing trends, research, or industry standards. Tracking neutral sentiment helps organizations keep track of non-urgent but important developments that may impact **cyber risk** in the long run.

From this Analysis, We infer that:

- **VADER** tends to categorize a larger portion of posts as **Neutral**, which might indicate that it is more conservative or cautious in assigning clear sentiment, especially with text that has mixed or subtle tones.
  
- **BERT**, with its deep learning capabilities, provides more **decisive** sentiment labels, leaning towards **Positive** and **Negative** classifications, which suggests that it is better at capturing stronger sentiments or more polarized opinions in the data.

- This difference highlights that VADER is better suited for general sentiment detection in informal contexts, while BERT’s ability to understand context in greater depth allows it to more confidently classify posts into distinct sentiment categories.

---

## Conclusion
This project demonstrates the use of NLP in understanding public sentiment on cybersecurity risks. By identifying sentiment patterns and commonly discussed threats, organizations can gain insights that support proactive risk management and inform cybersecurity strategies. Future work may explore the integration of additional NLP models, such as BERT-based topic modeling, and expanded data sources for a more comprehensive risk intelligence assessment.

---
