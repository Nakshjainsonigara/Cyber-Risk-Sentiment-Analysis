# Cyber Risk Sentiment Analysis using Natural Language Processing

## Project Overview
This project performs sentiment analysis on Reddit posts from cyber risk-related subreddits to gauge public sentiment on cyber risks, vulnerabilities, and incidents. Leveraging Natural Language Processing (NLP) techniques, including entity extraction to identify key cyber threats, it analyzes large volumes of unstructured data and provides insights into trending cyber threats, user sentiments, and potential areas of cyber risk. The results aim to support cybersecurity decision-making, threat intelligence, and risk management for organizations.

View the [Interactive Kaggle Notebook](https://www.kaggle.com/code/nakshjain11/cyber-risk-sentiment-analysis)
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
   - **Fine-tuned BERT**: A transformer model specifically fine-tuned on cybersecurity domain data for enhanced accuracy and domain adaptation, providing more nuanced sentiment classification tailored to cyber risk discussions.

4. **Feature Extraction and Vectorization**  
   - **TF-IDF Vectorization** for feature extraction.
   - **Count Vectorizer** to analyze word frequency and uncover common themes in cybersecurity discussions.

5. **Visualization**  
   - Word clouds to visualize commonly discussed topics.
   - Bar charts and sentiment distribution graphs to present insights into sentiment trends.

6. **Entity Extraction and Analysis**
   - **Entity Extraction**: Named entities (such as organizations, technologies, and cyber threats) are extracted from the cleaned text using spaCy. This process identifies and isolates key terms relevant to cybersecurity.

   - **Entity Analysis**: After extraction, entities are categorized, filtered, and analyzed for frequency and associated sentiment (positive, negative, or neutral). 
---

## Technology Stack
- **Programming Language**: Python
- **Data Collection**: PRAW (Python Reddit API Wrapper)
- **Data Processing and NLP**: NLTK, TextBlob, WordNetLemmatizer, Stopwords, re (regex),spaCy
- **Sentiment Analysis**: VADER, Hugging Face Transformers (DistilBERT)
- **Feature Extraction**: Scikit-learn (TF-IDF Vectorizer, Count Vectorizer)
- **Visualization**: Matplotlib, WordCloud

---

## Algorithms and Libraries Used

- **VADER Sentiment Analysis**: A lexicon and rule-based model designed for social media sentiment analysis.
- **Fine-tuned BERT Sentiment Analysis**: A transformer-based model fine-tuned specifically on cybersecurity domain data for enhanced accuracy and better adaptation to cyber risk vocabulary and context.
- **TF-IDF Vectorization**: Calculates term frequency-inverse document frequency for feature extraction.
- **Count Vectorization**: Extracts features based on word frequency, useful for identifying prominent themes in cybersecurity discussions.
- **Named Entity Recognition**: Extracts relevant entities (e.g., threats, technologies) for focused analysis.
---

## Insights and Findings

Sentiment analysis of **cyber risk** discussions on social media or forums can provide valuable insights for organizations operating in the cybersecurity domain. By analyzing how people talk about **cybersecurity** and **cyber risk**, organization can gain an understanding of current trends, emerging threats, and public perception, helping them make informed decisions.

- **Positive Sentiment**: Positive posts highlight confidence in **cybersecurity** practices and solutions. These discussions often focus on successful implementations of security measures, the adoption of new technologies, or effective responses to cyber incidents. Organizations can track these posts to identify new tools, strategies, or practices that are helping reduce **cyber risk**. This helps companies stay updated on the latest advancements and share best practices with clients or stakeholders.

- **Negative Sentiment**: Negative posts often reflect concerns or frustrations related to **cybersecurity** challenges, such as data breaches, increasing cyberattacks, or new vulnerabilities. By analyzing these posts, companies can stay informed about potential threats and risks in the industry. This information can be used to adjust security strategies, anticipate new attack vectors, and identify areas that require immediate attention. Negative sentiment also helps organizations understand what concerns the public or professionals, allowing them to address these issues proactively.

- **Neutral Sentiment**: Neutral posts provide objective, fact-based information without strong emotions. These could include news articles, updates on **cybersecurity** policies, or technical discussions about vulnerabilities and solutions. Although neutral posts may not indicate any urgent problems, they offer valuable data about ongoing trends, research, or industry standards. Tracking neutral sentiment helps organizations keep track of non-urgent but important developments that may impact **cyber risk** in the long run.

From Model Comparision, We infer that:

- **VADER** tends to categorize a larger portion of posts as **Neutral**, which might indicate that it is more conservative or cautious in assigning clear sentiment, especially with text that has mixed or subtle tones.
  
- **Fine-tuned BERT**, with its domain-specific training on cybersecurity data, provides more **accurate** sentiment classification tailored to cyber risk contexts, better understanding cybersecurity-specific vocabulary and sentiment expressions compared to generic pre-trained models.

- This difference highlights that VADER is better suited for general sentiment detection in informal contexts, while the fine-tuned BERT's domain-specific training allows it to more accurately classify sentiment in cybersecurity discussions, understanding context and terminology specific to cyber risk scenarios.

<div align="center">
  
| **Model** | **Positive (%)** | **Neutral (%)** | **Negative (%)** |
|-----------|------------------|-----------------|------------------|
| **VADER** | 43.3%            | 42.8%           | 14.0%            |
| **BERT**  | 48.2%            | N/A             | 51.8%            |

</div>


> **Note** : BERT does not have a Neutral classification by default because it is typically fine-tuned for binary sentiment analysis (Positive/Negative). However, Neutral can be incorporated by defining a probability threshold where the model classifies results as Neutral when its confidence in both Positive and Negative is below a certain level.

The top entities reveal focused areas in cyber risk discussions:

1. **Pentagon** : The negative sentiment classification from BERT suggests that discussions about the Pentagon are likely focused on potential cybersecurity risks or vulnerabilities.

2. **Cyber Threat Intelligence (CTI)** : CTI is primarily discussed with negative sentiment, underscoring industry concerns related to intelligence on potential threats and vulnerabilities.

3. **RTM Locker Ransomware** : This specific ransomware variant is mentioned with consistent negative sentiment, highlighting ongoing risks associated with ransomware attacks.
   
---

## Conclusion
This project demonstrates the use of NLP in understanding public sentiment on cybersecurity risks. By identifying sentiment patterns and commonly discussed threats, organizations can gain insights that support proactive risk management and inform cybersecurity strategies. The implementation of fine-tuned BERT models specifically trained on cybersecurity domain data represents a significant improvement over generic sentiment analysis approaches, providing more accurate and context-aware sentiment classification for cyber risk discussions. Future work may explore expanded datasets, multi-label classification for different types of cyber threats, and integration with real-time threat intelligence feeds for comprehensive risk assessment.

---
