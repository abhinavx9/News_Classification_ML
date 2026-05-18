# 📰 NewsLens — News Classifier & Summarizer
### Traditional ML Only (No Deep Learning)

---

## 🧠 Model Architecture

| Component | Method | Description |
|-----------|--------|-------------|
| Preprocessing | Rule-based NLP | Lowercasing, URL removal, stopword removal, custom stemmer |
| Feature Extraction | **TF-IDF** (custom) | Unigrams + Bigrams, top 3000 features |
| Classification | **Naive Bayes** | Multinomial with Laplace smoothing |
| Summarization | **TextRank** | Graph-based PageRank on sentence similarity |
| Keywords | TF-IDF scoring | Top-5 informative terms per article |
| Sentiment | Lexicon-based | Positive/Negative/Neutral word matching |

**Zero deep learning. Zero transformers. Zero neural networks.**

---

## 📁 Project Structure

```
news_classifier/
│
├── model_training/
│   └── train_model.py        ← Train & save the ML model
│
├── backend/
│   ├── app.py                ← Flask REST API
│   ├── models/               ← Created automatically after training
│   │   └── news_classifier.pkl
│   └── (requirements in root)
│
├── frontend/
│   └── index.html            ← Complete web app (open directly in browser)
│
├── requirements.txt
└── README.md
```

---

## 🚀 STEP-BY-STEP SETUP

### ─── OPTION A: Browser Only (Fastest — 2 minutes) ───

1. Open `frontend/index.html` in any browser
2. The JS ML engine runs completely in the browser
3. No Python, no server, no installation needed!
4. Click example pills to test different categories

---

### ─── OPTION B: Full Python Backend ───

#### Step 1 — Install dependencies
```bash
pip install numpy flask flask-cors
```

#### Step 2 — Train the model
```bash
cd model_training
python train_model.py
```
Expected output:
```
[0/5] Preprocessing training data... Processed 93 training samples
[1/5] Building TF-IDF vocabulary... Vocabulary size: 847 terms
[2/5] Training Naive Bayes classifier... Classes trained: sports, politics, ...
[3/5] Evaluating model... Training Accuracy: 95.70%
[4/5] Model saved to: ../backend/models/news_classifier.pkl
[5/5] Testing with sample headlines...
✅ Training Complete!
```

#### Step 3 — Start the API server
```bash
cd ../backend
python app.py
```
Server runs at: `http://localhost:5000`

#### Step 4 — Open the frontend
Open `frontend/index.html` in browser — it works in demo mode by default.

To connect to backend, open `frontend/index.html` and in the `<script>` section:
1. Find the line: `const { category, confidence, probs } = classifyText(text);`
2. Replace the classify block with the API call (see commented code at bottom of JS)

---

## 🔌 API Endpoints

### Classify Single Article
```
POST http://localhost:5000/classify
Content-Type: application/json

{
  "text": "India won the cricket match against Australia..."
}
```

Response:
```json
{
  "success": true,
  "category": "sports",
  "icon": "⚽",
  "confidence": 89.3,
  "all_probabilities": {
    "sports": 89.3, "politics": 2.1, "technology": 1.8, ...
  },
  "summary": "India won the cricket match... team celebrates victory...",
  "keywords": ["cricket", "india", "australia", "match", "victory"],
  "reading_time": 2,
  "word_count": 145,
  "sentiment": "Positive"
}
```

### Batch Classification
```
POST http://localhost:5000/batch
{
  "articles": ["text1", "text2", "text3"]
}
```

### Get All Categories
```
GET http://localhost:5000/categories
```

---

## 📊 Supported Categories

| # | Category | Icon | Example Topics |
|---|----------|------|----------------|
| 1 | Sports | ⚽ | Cricket, Football, Olympics, IPL, Tennis |
| 2 | Politics | 🏛️ | Parliament, Elections, Government, Policy |
| 3 | Technology | 💻 | AI, Startups, Space, Gadgets, Cybersecurity |
| 4 | Business | 📈 | Stock Market, RBI, Economy, Companies |
| 5 | Health | 🏥 | Medicine, Vaccines, Diseases, Surgery |
| 6 | Environment | 🌿 | Climate, Wildlife, Pollution, Renewable Energy |
| 7 | Education | 📚 | JEE, Schools, Universities, Scholarships |

---

## 🔧 Adding More Training Data

In `train_model.py`, add to the `TRAINING_DATA` list:

```python
TRAINING_DATA = [
    # existing data...
    ("Your new training text here", "category_name"),
    ("Another example article", "sports"),
    # ...
]
```

Or load from CSV:
```python
import csv
with open('your_newspaper_data.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        TRAINING_DATA.append((row[0], row[1]))  # text, label
```

---

## 🧪 Traditional ML Algorithms Used

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)
```
TF(t,d)  = count(t in d) / total_words(d)
IDF(t)   = log((N+1)/(df(t)+1)) + 1
TF-IDF   = TF × IDF
```

### 2. Multinomial Naive Bayes
```
P(class|doc) ∝ P(class) × ∏ P(word|class)
Log form used for numerical stability
Laplace smoothing (α=0.5) prevents zero probabilities
```

### 3. TextRank (Graph-based Summarization)
```
Score(s) = (1-d)/N + d × Σ [sim(si,sj)/Σsim(sj,sk)] × Score(sj)
Sentence similarity = Jaccard coefficient on word sets
PageRank iterations = 30, damping = 0.85
```

---

## 🎯 Exam/Submission Points to Highlight

1. **No sklearn used** — All algorithms implemented from scratch
2. **Custom TF-IDF** with bigram support for better feature coverage
3. **TextRank summarizer** uses PageRank on sentence similarity graph
4. **Rule-based stemmer** without NLTK dependency
5. **Lexicon-based sentiment** analysis (traditional NLP)
6. **Handcrafted features**: word count, unique ratio, number density
7. **Laplace smoothing** prevents zero-probability issues in Bayes
8. **Softmax normalization** for probability distribution output
9. Model is fully serializable with Python `pickle`
10. REST API separates model from frontend (production-ready pattern)

---

## 🐛 Troubleshooting

**"Model not found" error:**
→ Run `python model_training/train_model.py` first

**"Module not found" error:**
→ Run `pip install numpy flask flask-cors`

**Low accuracy on your data:**
→ Add more training examples in `TRAINING_DATA`
→ Adjust `max_features` in TFIDFVectorizer (try 5000)
→ Adjust Laplace smoothing `alpha` (try 0.1 to 1.0)

**Frontend not calling backend:**
→ Make sure Flask is running on port 5000
→ Check browser console for CORS errors
→ The frontend works standalone in demo mode without backend
