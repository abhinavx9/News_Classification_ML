"""
=============================================================
  NEWS CLASSIFIER + SUMMARIZER - Multi-Algorithm Comparison
  Classifiers: Naive Bayes | Logistic Regression | KNN
               Decision Tree | Linear SVM | Random Forest
  Feature Extraction: Custom TF-IDF (unigrams + bigrams)
  Summarization: TextRank (PageRank on sentence graph)
  Zero sklearn. Zero deep learning. All from scratch.
=============================================================
  STEPS TO RUN:
  1. pip install numpy flask flask-cors
  2. Place bbc_data.csv next to this file
  3. python train_model.py
  4. Model saved as: ../backend/models/news_classifier.pkl
=============================================================
"""

import os, re, math, pickle, warnings, sys, time
import numpy as np
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  STEP 1 — HANDCRAFTED TRAINING DATA (India-specific)
# ══════════════════════════════════════════════════════════════
TRAINING_DATA = [
    ("India won the cricket match against Australia by 5 wickets in a thrilling finale", "sports"),
    ("Virat Kohli scored a brilliant century in the test series against England", "sports"),
    ("Manchester United defeated Chelsea 3-1 in the Premier League clash", "sports"),
    ("Olympics 2024 saw India win 6 medals including gold in javelin throw", "sports"),
    ("Rohit Sharma to lead Indian team in T20 World Cup tournament", "sports"),
    ("Barcelona signs new striker for record 200 million euros transfer fee", "sports"),
    ("Federer announces retirement from professional tennis after brilliant career", "sports"),
    ("Indian hockey team qualifies for Olympic Games after winning Asia Cup", "sports"),
    ("IPL 2024 auction sees players go for record breaking amounts", "sports"),
    ("Neeraj Chopra breaks world record in javelin throw at international championship", "sports"),
    ("FIFA World Cup 2026 preparations underway as teams begin qualification matches", "sports"),
    ("Serena Williams wins her 23rd Grand Slam title at Wimbledon", "sports"),
    ("Lewis Hamilton secures his eighth Formula One world championship title", "sports"),
    ("PV Sindhu wins gold medal at Commonwealth Games badminton tournament", "sports"),
    ("Mumbai Indians beat Chennai Super Kings in thrilling IPL final match", "sports"),
    ("Lionel Messi scores hat trick in Argentina victory over Brazil", "sports"),
    ("Indian sprinter wins silver medal at World Athletics Championships", "sports"),
    ("Saina Nehwal makes comeback after injury in national badminton championship", "sports"),
    ("Prime Minister Modi inaugurated the new parliament building in New Delhi", "politics"),
    ("Election Commission announced dates for upcoming state assembly elections", "politics"),
    ("Opposition parties staged protest against new government policies", "politics"),
    ("President Biden signed new legislation on climate change and infrastructure", "politics"),
    ("United Nations Security Council discussed the conflict in Eastern Europe", "politics"),
    ("Chief Minister announced new welfare scheme for farmers in the state", "politics"),
    ("Parliament session debate on new education policy amendments", "politics"),
    ("G20 summit held in India with leaders discussing global economic issues", "politics"),
    ("Supreme Court passed landmark judgment on electoral bonds controversy", "politics"),
    ("Budget session of parliament began with Finance Minister annual speech", "politics"),
    ("Home Minister addressed the security situation at the border regions", "politics"),
    ("Political party alliance formed ahead of general elections next year", "politics"),
    ("Governor dissolved state assembly and called for fresh elections", "politics"),
    ("Opposition demanded resignation of minister over corruption allegations", "politics"),
    ("Rajya Sabha passed the constitutional amendment bill after heated debate", "politics"),
    ("Diplomatic talks between India and China over border dispute resumed", "politics"),
    ("Municipal corporation elections held peacefully across the state", "politics"),
    ("Apple launched its new iPhone with revolutionary AI camera features", "technology"),
    ("Google released an update to its search algorithm affecting millions of websites", "technology"),
    ("Microsoft acquired gaming company for 70 billion dollars deal", "technology"),
    ("Elon Musk SpaceX successfully launched reusable rocket into orbit", "technology"),
    ("Artificial intelligence startup raised 100 million funding in series B round", "technology"),
    ("New chip by NVIDIA achieves breakthrough in machine learning performance", "technology"),
    ("Indian startup builds drone delivery system for remote rural healthcare", "technology"),
    ("Cybersecurity breach affects 500 million users data across social network", "technology"),
    ("5G networks rolled out in major cities across India by telecom companies", "technology"),
    ("Electric vehicle startup announces battery with 1000 km range per charge", "technology"),
    ("Meta released new virtual reality headset with advanced eye tracking", "technology"),
    ("Amazon Web Services launches new cloud computing region in India", "technology"),
    ("Scientists develop quantum computer that solves problem in seconds", "technology"),
    ("Twitter introduces new features for subscription users on platform", "technology"),
    ("Robot performs complex surgery autonomously for first time in hospital", "technology"),
    ("Samsung unveiled foldable smartphone with new display technology", "technology"),
    ("Stock market reached all time high as Sensex crossed 80000 mark", "business"),
    ("Reserve Bank of India kept interest rates unchanged at monetary meeting", "business"),
    ("Tata Group announced merger of two subsidiary companies this quarter", "business"),
    ("Inflation rate drops to 4 percent lowest in past 18 months reported", "business"),
    ("Reliance Industries reported record quarterly profit of 15000 crore rupees", "business"),
    ("Foreign investment in India reached 84 billion dollars this fiscal year", "business"),
    ("Startup ecosystem in India sees 100 new unicorn companies this decade", "business"),
    ("Adani Group shares saw massive rally after report cleared company name", "business"),
    ("India GDP growth rate projected at 7.4 percent for current fiscal year", "business"),
    ("Bank of India announced merger with smaller regional cooperative banks", "business"),
    ("Gold prices hit record high as investors flee to safe haven assets", "business"),
    ("SEBI imposed fine on brokerage firm for market manipulation charges", "business"),
    ("Real estate sector sees revival with housing sales up 30 percent", "business"),
    ("Petrol diesel prices revised after crude oil falls in global markets", "business"),
    ("Insurance company IPO oversubscribed 50 times on stock exchange listing", "business"),
    ("New vaccine developed against malaria shows 90 percent efficacy in trials", "health"),
    ("Health ministry launched national campaign against tuberculosis disease", "health"),
    ("Scientists discovered new cancer treatment using targeted gene therapy", "health"),
    ("WHO reported increase in cases of dengue fever across tropical regions", "health"),
    ("AIIMS doctors performed successful heart transplant surgery on patient", "health"),
    ("Mental health awareness campaign launched by government for youth", "health"),
    ("New antibiotic resistant bacteria strain discovered in hospital patients", "health"),
    ("Yoga and meditation reduces stress levels by 40 percent study finds", "health"),
    ("Government approved new drug for diabetes treatment at affordable price", "health"),
    ("Ayushman Bharat scheme expanded to cover 10 crore more beneficiaries", "health"),
    ("COVID booster dose recommended for elderly and immunocompromised people", "health"),
    ("Medical research team develops blood test to detect Alzheimer early", "health"),
    ("Nutrition deficiency found in 60 percent of children under five years", "health"),
    ("National cancer registry shows decline in tobacco related cancer cases", "health"),
    ("India pledged to achieve net zero carbon emissions by year 2070", "environment"),
    ("Massive flooding in Kerala caused by heavy monsoon rainfall this season", "environment"),
    ("Tiger population in India increases to 3000 according to latest census", "environment"),
    ("Solar power generation in India crossed 70 gigawatt milestone achieved", "environment"),
    ("Plastic ban extended to cover single use items in all states nationwide", "environment"),
    ("Arctic ice sheet melting at fastest rate in recorded history scientists warn", "environment"),
    ("Forest fires destroy thousands of hectares in Amazon rainforest region", "environment"),
    ("New species of fish discovered in deep ocean trench by researchers", "environment"),
    ("Air quality index in Delhi reached severe levels during winter months", "environment"),
    ("Ganga cleaning project shows improvement with dolphin population rising", "environment"),
    ("Climate change summit in Paris agreed on new emission reduction targets", "environment"),
    ("Renewable energy capacity of India set to double by year 2030", "environment"),
    ("IIT entrance exam JEE results declared with girl topping the merit list", "education"),
    ("New national education policy implemented in schools from this academic year", "education"),
    ("Scholarship program launched to help meritorious students from rural areas", "education"),
    ("India ranks higher in global education index improving from previous year", "education"),
    ("Online learning platform sees 50 million new users after pandemic period", "education"),
    ("Teacher shortage crisis affects government schools in rural districts badly", "education"),
    ("University grants commission introduced new undergraduate course structure", "education"),
    ("IIM Ahmedabad ranked among top business schools in Asia Pacific region", "education"),
    ("Digital literacy program reaches 1 crore students in government schools", "education"),
    ("CBSE board exam results show improvement in pass percentage this year", "education"),
    ("Government announces free coaching for competitive exams for SC ST students", "education"),
]

print("=" * 65)
print("   NEWS CLASSIFIER - Multi-Algorithm Comparison Pipeline")
print("=" * 65)


# ══════════════════════════════════════════════════════════════
#  BBC DATASET LOADER
# ══════════════════════════════════════════════════════════════
def load_bbc_dataset(csv_path=None):
    import csv
    BBC_MAP = {
        "sport": "sports", "tech": "technology",
        "business": "business", "politics": "politics",
        "entertainment": "entertainment",
    }
    TEXT_NAMES  = ("data", "text", "article", "content", "body")
    LABEL_NAMES = ("labels", "label", "category", "class", "type")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    SEARCH_PATHS = [p for p in [
        csv_path, "bbc_data.csv",
        os.path.join(script_dir, "bbc_data.csv"),
        "bbc-news-data.csv", "bbc_news.csv",
        "../bbc_data.csv", "../bbc_news.csv",
    ] if p]
    data = []
    for path in SEARCH_PATHS:
        if not os.path.exists(path):
            continue
        print(f"    Found: {path}")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                text_col = next((c for c in fieldnames if c.lower().strip() in TEXT_NAMES), None)
                cat_col  = next((c for c in fieldnames if c.lower().strip() in LABEL_NAMES), None)
                if not text_col or not cat_col:
                    if len(fieldnames) >= 2:
                        text_col, cat_col = fieldnames[0], fieldnames[1]
                print(f"    Columns: text={text_col!r}  label={cat_col!r}")
                for row in reader:
                    txt = (row.get(text_col) or "").strip()
                    cat = (row.get(cat_col)  or "").strip().lower()
                    mapped = BBC_MAP.get(cat)
                    if txt and mapped:
                        data.append((txt, mapped))
            if data:
                dist = Counter(lbl for _, lbl in data)
                print(f"    Loaded {len(data)} samples")
                for c, n in sorted(dist.items()):
                    print(f"      {c:<15} {n:4d}")
                return data
        except Exception as e:
            print(f"    Error: {e}")
    return data


# ══════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════
class TextPreprocessor:
    STOPWORDS = set([
        "a","an","the","and","or","but","in","on","at","to","for","of","with",
        "by","from","is","are","was","were","be","been","being","have","has",
        "had","do","does","did","will","would","could","should","may","might",
        "shall","can","this","that","these","those","it","its","he","she","they",
        "we","you","i","me","him","her","us","them","my","your","his","their",
        "our","as","if","so","not","no","nor","up","out","about","into","after",
        "before","between","during","through","also","than","then","when","where",
        "which","who","whom","what","how","all","each","every","both","few","more",
        "most","other","some","such","over","under","again","further","too","very",
        "just","new","said","says","say",
    ])
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\d+", " NUM ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = [t for t in text.split() if t not in self.STOPWORDS and len(t) > 2]
        return " ".join([self._stem(t) for t in tokens])
    def _stem(self, word):
        for sfx in ["ing","tion","ness","ment","ity","ies","ers","est","ed","ly","es","s"]:
            if word.endswith(sfx) and len(word)-len(sfx) >= 3:
                return word[:-len(sfx)]
        return word


# ══════════════════════════════════════════════════════════════
#  TF-IDF VECTORIZER
# ══════════════════════════════════════════════════════════════
class TFIDFVectorizer:
    def __init__(self, max_features=3000, ngram_range=(1,2)):
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self.vocabulary_  = {}
        self.idf_         = {}

    def _ngrams(self, tokens):
        ng = list(tokens)
        if self.ngram_range[1] >= 2:
            ng += [tokens[i]+"_"+tokens[i+1] for i in range(len(tokens)-1)]
        return ng

    def fit(self, corpus):
        tokenized = [doc.split() for doc in corpus]
        ngrammed  = [self._ngrams(t) for t in tokenized]
        df = defaultdict(int)
        for doc in ngrammed:
            for term in set(doc): df[term] += 1
        top = sorted(df.items(), key=lambda x: -x[1])[:self.max_features]
        self.vocabulary_ = {t: i for i,(t,_) in enumerate(top)}
        N = len(corpus)
        self.idf_ = {t: math.log((N+1)/(f+1))+1 for t,f in df.items() if t in self.vocabulary_}
        return self

    def transform(self, corpus):
        M = np.zeros((len(corpus), len(self.vocabulary_)))
        for i, doc in enumerate(corpus):
            tokens = self._ngrams(doc.split())
            tf = Counter(tokens)
            total = sum(tf.values()) or 1
            for term, cnt in tf.items():
                if term in self.vocabulary_:
                    M[i, self.vocabulary_[term]] = (cnt/total) * self.idf_.get(term, 1.0)
        return M

    def fit_transform(self, corpus):
        return self.fit(corpus).transform(corpus)


# ══════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════
def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    n = len(y)
    idx = np.random.permutation(n)
    split = int(n*(1-test_size))
    tr, te = idx[:split], idx[split:]
    return X[tr], X[te], [y[i] for i in tr], [y[i] for i in te]

def accuracy(y_true, y_pred):
    return sum(a==b for a,b in zip(y_true,y_pred))/len(y_true)

def f1_macro(y_true, y_pred):
    classes = list(set(y_true))
    f1s = []
    for c in classes:
        tp = sum(p==c and t==c for p,t in zip(y_pred,y_true))
        fp = sum(p==c and t!=c for p,t in zip(y_pred,y_true))
        fn = sum(p!=c and t==c for p,t in zip(y_pred,y_true))
        pr = tp/(tp+fp) if (tp+fp) else 0
        rc = tp/(tp+fn) if (tp+fn) else 0
        f1s.append(2*pr*rc/(pr+rc) if (pr+rc) else 0)
    return sum(f1s)/len(f1s)


# ══════════════════════════════════════════════════════════════
#  CLASSIFIER 1 - Multinomial Naive Bayes
# ══════════════════════════════════════════════════════════════
class NaiveBayesClassifier:
    """Multinomial Naive Bayes with Laplace smoothing"""
    name = "Naive Bayes"
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.priors_ = {}
        self.feat_log_prob_ = {}
        self.classes_ = []

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        n = len(y)
        for c in self.classes_:
            mask = np.array([1 if lbl==c else 0 for lbl in y])
            self.priors_[c] = math.log(mask.sum()/n)
            feat = X[mask==1].clip(0).sum(axis=0) + self.alpha
            self.feat_log_prob_[c] = np.log(feat/feat.sum())
        return self

    def _scores(self, x):
        return {c: self.priors_[c]+np.dot(x.clip(0), self.feat_log_prob_[c]) for c in self.classes_}

    def predict(self, X):
        return [max(self._scores(x), key=self._scores(x).get) for x in X]

    def predict_proba(self, X):
        out = []
        for x in X:
            sc = self._scores(x)
            mx = max(sc.values())
            exp = {c: math.exp(v-mx) for c,v in sc.items()}
            tot = sum(exp.values())
            out.append({c: v/tot for c,v in exp.items()})
        return out


# ══════════════════════════════════════════════════════════════
#  CLASSIFIER 2 - Logistic Regression (One-vs-Rest, SGD)
# ══════════════════════════════════════════════════════════════
class LogisticRegressionClassifier:
    """One-vs-Rest Logistic Regression with mini-batch SGD"""
    name = "Logistic Regression"
    def __init__(self, lr=0.1, epochs=25, C=1.0, batch_size=64):
        self.lr = lr; self.epochs = epochs; self.C = C; self.batch_size = batch_size
        self.weights_ = {}; self.bias_ = {}; self.classes_ = []

    @staticmethod
    def _sigmoid(z):
        return np.where(z>=0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        n, d = X.shape
        for c in self.classes_:
            w = np.zeros(d); b = 0.0
            yb = np.array([1.0 if lbl==c else 0.0 for lbl in y])
            for _ in range(self.epochs):
                idx = np.random.permutation(n)
                for start in range(0, n, self.batch_size):
                    bi = idx[start:start+self.batch_size]
                    xb, tb = X[bi], yb[bi]
                    pred = self._sigmoid(xb@w+b)
                    err = pred-tb
                    w -= self.lr*(xb.T@err/len(bi)+w/(self.C*n))
                    b -= self.lr*err.mean()
            self.weights_[c] = w; self.bias_[c] = b
        return self

    def predict(self, X):
        mat = np.column_stack([self._sigmoid(X@self.weights_[c]+self.bias_[c]) for c in self.classes_])
        return [self.classes_[i] for i in np.argmax(mat, axis=1)]

    def predict_proba(self, X):
        mat = np.column_stack([self._sigmoid(X@self.weights_[c]+self.bias_[c]) for c in self.classes_])
        mat = mat/mat.sum(axis=1,keepdims=True)
        return [{c: float(mat[i,j]) for j,c in enumerate(self.classes_)} for i in range(len(X))]


# ══════════════════════════════════════════════════════════════
#  CLASSIFIER 3 - K-Nearest Neighbours (cosine similarity)
# ══════════════════════════════════════════════════════════════
class KNNClassifier:
    """KNN with cosine similarity for sparse TF-IDF vectors"""
    name = "K-Nearest Neighbours"
    def __init__(self, k=7):
        self.k = k; self.X_train = None; self.y_train = []; self.classes_ = []

    def fit(self, X, y):
        norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms==0] = 1
        self.X_train = X/norms; self.y_train = list(y); self.classes_ = sorted(set(y))
        return self

    def predict(self, X):
        preds = []
        for x in X:
            norm = np.linalg.norm(x); xn = x/norm if norm else x
            sims = self.X_train@xn
            top  = np.argsort(-sims)[:self.k]
            vote = Counter([self.y_train[i] for i in top])
            preds.append(vote.most_common(1)[0][0])
        return preds

    def predict_proba(self, X):
        out = []
        for x in X:
            norm = np.linalg.norm(x); xn = x/norm if norm else x
            sims = self.X_train@xn
            top  = np.argsort(-sims)[:self.k]
            vote = Counter([self.y_train[i] for i in top])
            tot  = sum(vote.values())
            out.append({c: vote.get(c,0)/tot for c in self.classes_})
        return out


# ══════════════════════════════════════════════════════════════
#  CLASSIFIER 4 - Decision Tree (Information Gain)
# ══════════════════════════════════════════════════════════════
class DecisionTreeClassifier:
    """Decision Tree using Information Gain with random feature subsampling"""
    name = "Decision Tree"
    def __init__(self, max_depth=12, min_samples_split=5):
        self.max_depth = max_depth; self.min_samples_split = min_samples_split
        self.tree_ = None; self.classes_ = []

    @staticmethod
    def _entropy(labels):
        n = len(labels)
        if n == 0: return 0.0
        return -sum((c/n)*math.log2(c/n+1e-12) for c in Counter(labels).values())

    def _best_split(self, X, y):
        best_gain, best_feat, best_thr = -1, 0, 0.0
        base_ent = self._entropy(y); n = len(y)
        feat_idx = np.random.choice(X.shape[1], min(200, X.shape[1]), replace=False)
        for f in feat_idx:
            vals = X[:,f]; thr = np.median(vals)
            lm = vals <= thr
            if lm.sum()==0 or (~lm).sum()==0: continue
            yl = [y[i] for i in range(n) if lm[i]]
            yr = [y[i] for i in range(n) if not lm[i]]
            gain = base_ent-(len(yl)/n)*self._entropy(yl)-(len(yr)/n)*self._entropy(yr)
            if gain > best_gain: best_gain,best_feat,best_thr = gain,f,thr
        return best_feat, best_thr

    def _build(self, X, y, depth):
        if depth>=self.max_depth or len(y)<self.min_samples_split or len(set(y))==1:
            return Counter(y).most_common(1)[0][0]
        f, thr = self._best_split(X, y)
        mask = X[:,f] <= thr
        if mask.sum()==0 or (~mask).sum()==0: return Counter(y).most_common(1)[0][0]
        return {"feat":f,"thr":thr,
                "left": self._build(X[mask],  [y[i] for i in range(len(y)) if mask[i]],  depth+1),
                "right":self._build(X[~mask], [y[i] for i in range(len(y)) if not mask[i]],depth+1)}

    def fit(self, X, y):
        self.classes_ = sorted(set(y)); self.tree_ = self._build(X, list(y), 0); return self

    def _traverse(self, node, x):
        if isinstance(node, str): return node
        return self._traverse(node["left"] if x[node["feat"]]<=node["thr"] else node["right"], x)

    def predict(self, X):
        return [self._traverse(self.tree_, x) for x in X]

    def predict_proba(self, X):
        preds = self.predict(X)
        return [{c:(1.0 if p==c else 0.0) for c in self.classes_} for p in preds]


# ══════════════════════════════════════════════════════════════
#  CLASSIFIER 5 - Linear SVM (Hinge loss + SGD)
# ══════════════════════════════════════════════════════════════
class LinearSVMClassifier:
    """One-vs-Rest Linear SVM (Pegasos SGD on hinge loss)"""
    name = "Linear SVM"
    def __init__(self, C=1.0, epochs=15, batch_size=64):
        self.C = C; self.epochs = epochs; self.batch_size = batch_size
        self.weights_ = {}; self.bias_ = {}; self.classes_ = []

    def fit(self, X, y):
        self.classes_ = sorted(set(y)); n, d = X.shape; t = 1
        for c in self.classes_:
            w = np.zeros(d); b = 0.0
            yb = np.array([1.0 if lbl==c else -1.0 for lbl in y])
            for _ in range(self.epochs):
                idx = np.random.permutation(n)
                for start in range(0, n, self.batch_size):
                    bi = idx[start:start+self.batch_size]
                    xb, tb = X[bi], yb[bi]
                    lr = 1.0/(0.01*t); t += 1
                    margin = tb*(xb@w+b); mask = margin<1
                    grad_w = w-self.C*(xb[mask].T@tb[mask])/len(bi)
                    grad_b = -self.C*tb[mask].mean() if mask.any() else 0.0
                    w -= lr*grad_w; b -= lr*grad_b
            self.weights_[c] = w; self.bias_[c] = b
        return self

    def predict(self, X):
        mat = np.column_stack([X@self.weights_[c]+self.bias_[c] for c in self.classes_])
        return [self.classes_[i] for i in np.argmax(mat, axis=1)]

    def predict_proba(self, X):
        mat = np.column_stack([X@self.weights_[c]+self.bias_[c] for c in self.classes_])
        mat = mat-mat.max(axis=1,keepdims=True)
        exp = np.exp(mat); probs = exp/exp.sum(axis=1,keepdims=True)
        return [{c: float(probs[i,j]) for j,c in enumerate(self.classes_)} for i in range(len(X))]


# ══════════════════════════════════════════════════════════════
#  CLASSIFIER 6 - Random Forest (ensemble of Decision Trees)
# ══════════════════════════════════════════════════════════════
class RandomForestClassifier:
    """Random Forest: bagging of Decision Trees"""
    name = "Random Forest"
    def __init__(self, n_trees=20, max_depth=10, min_samples_split=5):
        self.n_trees = n_trees; self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees_ = []; self.classes_ = []

    def fit(self, X, y):
        self.classes_ = sorted(set(y)); n = len(y)
        for _ in range(self.n_trees):
            idx = np.random.choice(n, n, replace=True)
            Xb, yb = X[idx], [y[i] for i in idx]
            tree = DecisionTreeClassifier(self.max_depth, self.min_samples_split)
            tree.fit(Xb, yb); self.trees_.append(tree)
        return self

    def predict(self, X):
        votes = np.array([t.predict(X) for t in self.trees_])
        return [Counter(votes[:,i]).most_common(1)[0][0] for i in range(X.shape[0])]

    def predict_proba(self, X):
        out = [{c: 0.0 for c in self.classes_} for _ in range(len(X))]
        for tree in self.trees_:
            for i, p in enumerate(tree.predict_proba(X)):
                for c,v in p.items(): out[i][c] += v/self.n_trees
        return out


# ══════════════════════════════════════════════════════════════
#  TEXTRANK SUMMARIZER
# ══════════════════════════════════════════════════════════════
class TextRankSummarizer:
    def __init__(self, damping=0.85, iterations=30):
        self.damping = damping; self.iterations = iterations

    def _sim(self, s1, s2):
        w1, w2 = set(s1.lower().split()), set(s2.lower().split())
        if not w1 or not w2: return 0.0
        inter = w1&w2
        return (len(inter)/len(w1|w2) + len(inter)/(math.log(len(w1)+1)+math.log(len(w2)+1)))/2

    def summarize(self, text, num_sentences=2):
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.split())>4]
        if len(sents)<=2: return " ".join(sents)
        n = len(sents)
        sim = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i!=j: sim[i][j] = self._sim(sents[i], sents[j])
        rs = sim.sum(axis=1,keepdims=True); rs[rs==0]=1; sim/=rs
        sc = np.ones(n)/n
        for _ in range(self.iterations):
            nsc = (1-self.damping)/n + self.damping*sim.T.dot(sc)
            if np.allclose(sc,nsc,atol=1e-6): break
            sc = nsc
        idx = sorted(np.argsort(-sc)[:num_sentences])
        return " ".join(sents[i] for i in idx)


# ══════════════════════════════════════════════════════════════
#  KEYWORD EXTRACTOR
# ══════════════════════════════════════════════════════════════
class KeywordExtractor:
    def __init__(self): self.pre = TextPreprocessor()
    def extract(self, text, top_k=5):
        original = re.findall(r"\b[A-Za-z]{4,}\b", text)
        return [w for w,_ in Counter(w.lower() for w in original).most_common(top_k)]


# ══════════════════════════════════════════════════════════════
#  BENCHMARK
# ══════════════════════════════════════════════════════════════
def run_benchmark(X_train, X_test, y_train, y_test):
    classifiers = [
        NaiveBayesClassifier(alpha=0.5),
        LogisticRegressionClassifier(lr=0.1, epochs=25, C=1.0),
        KNNClassifier(k=7),
        DecisionTreeClassifier(max_depth=12),
        LinearSVMClassifier(C=1.0, epochs=15),
        RandomForestClassifier(n_trees=20, max_depth=10),
    ]

    print("\n" + "="*65)
    print("  CLASSIFIER BENCHMARK  (80% train / 20% test split)")
    print("="*65)
    print(f"  {'Model':<24} {'Accuracy':>10} {'F1-Macro':>10} {'Time':>8}")
    print("  " + "-"*55)

    results = []
    for clf in classifiers:
        t0 = time.time()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        elapsed = time.time()-t0
        acc = accuracy(y_test, preds)
        f1  = f1_macro(y_test, preds)
        results.append((clf, acc, f1, elapsed, preds))
        print(f"  {clf.name:<24} {acc:>9.2%} {f1:>10.4f} {elapsed:>7.2f}s")

    print("  " + "-"*55)

    # Per-category F1
    print("\n  Per-Category F1 Scores")
    print("  " + "-"*65)
    classes = sorted(set(y_test))
    header  = f"  {'Category':<15}" + "".join(f"{r[0].name[:11]:>13}" for r in results)
    print(header)
    print("  " + "-"*65)
    for c in classes:
        row = f"  {c:<15}"
        for clf,_,_,_,preds in results:
            tp = sum(p==c and t==c for p,t in zip(preds,y_test))
            fp = sum(p==c and t!=c for p,t in zip(preds,y_test))
            fn = sum(p!=c and t==c for p,t in zip(preds,y_test))
            pr = tp/(tp+fp) if (tp+fp) else 0
            rc = tp/(tp+fn) if (tp+fn) else 0
            f  = 2*pr*rc/(pr+rc) if (pr+rc) else 0
            row += f"{f:>13.3f}"
        print(row)

    best = max(results, key=lambda r: r[2])
    print("\n" + "="*65)
    print(f"  BEST MODEL : {best[0].name}")
    print(f"  Accuracy   : {best[1]:.2%}")
    print(f"  F1-Macro   : {best[2]:.4f}")
    print("="*65)
    return best[0]


# ══════════════════════════════════════════════════════════════
#  FULL NEWS PIPELINE
# ══════════════════════════════════════════════════════════════
class NewsPipeline:
    ICONS = {
        "sports":"⚽","politics":"🏛️","technology":"💻","business":"📈",
        "health":"🏥","environment":"🌿","education":"📚",
        "entertainment":"🎬","unknown":"📰"
    }
    POS = {"win","won","success","growth","record","improve","achievement","gold","best","top","rise","increase","award","victory"}
    NEG = {"loss","defeat","crisis","fail","problem","issue","decline","fire","flood","attack","controversy","corruption","death","ban"}

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer   = TFIDFVectorizer(max_features=3000, ngram_range=(1,2))
        self.classifier   = None
        self.summarizer   = TextRankSummarizer()
        self.keywords     = KeywordExtractor()

    def train_and_select(self, data):
        texts, labels = zip(*data)
        print("\n[1/4] Preprocessing...")
        cleaned = [self.preprocessor.clean_text(t) for t in texts]
        print(f"    {len(cleaned)} samples preprocessed")
        print("\n[2/4] Building TF-IDF features (3000 features, unigrams+bigrams)...")
        X = self.vectorizer.fit_transform(cleaned)
        print(f"    Matrix: {X.shape[0]} x {X.shape[1]}")
        print("\n[3/4] Benchmarking all 6 classifiers...")
        X_tr, X_te, y_tr, y_te = train_test_split(X, list(labels), test_size=0.2)
        best_clf = run_benchmark(X_tr, X_te, y_tr, y_te)
        print(f"\n[4/4] Re-training {best_clf.name} on full dataset...")
        best_clf.fit(X, list(labels))
        self.classifier = best_clf
        print(f"    Pipeline ready  |  Classifier: {best_clf.name}")
        return self

    def predict(self, text):
        cleaned  = self.preprocessor.clean_text(text)
        X        = self.vectorizer.transform([cleaned])
        category = self.classifier.predict(X)[0]
        proba    = self.classifier.predict_proba(X)[0]
        conf     = proba.get(category, 0)*100
        summary  = self.summarizer.summarize(text, num_sentences=2)
        if len(summary)<30: summary = text[:200]+"..."
        kws      = self.keywords.extract(text, top_k=5)
        wc       = len(text.split())
        words    = set(text.lower().split())
        pos      = len(words&self.POS); neg = len(words&self.NEG)
        sent     = "Positive" if pos>neg else ("Negative" if neg>pos else "Neutral")
        return {
            "category": category, "icon": self.ICONS.get(category,"📰"),
            "confidence": round(conf,1),
            "all_probs": {k: round(v*100,1) for k,v in proba.items()},
            "summary": summary, "keywords": kws,
            "reading_time": max(1,wc//200), "word_count": wc, "sentiment": sent,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,"wb") as f: pickle.dump(self, f)
        print(f"\n  Model saved -> {path}")

    @staticmethod
    def load(path):
        with open(path,"rb") as f: return pickle.load(f)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cli_path = sys.argv[1] if len(sys.argv) > 1 else None

    print("\n  Loading BBC News dataset...")
    bbc = load_bbc_dataset(csv_path=cli_path)

    if bbc:
        ALL_DATA = bbc + TRAINING_DATA
        dist = Counter(lbl for _,lbl in ALL_DATA)
        print(f"\n  Combined dataset: {len(ALL_DATA)} samples")
        for c,n in sorted(dist.items()):
            print(f"    {c:<15} {n:4d}  {chr(9608)*(n//30)}")
    else:
        print("  BBC file not found - using built-in data only.")
        print("  Place bbc_data.csv next to train_model.py and re-run.")
        ALL_DATA = TRAINING_DATA

    pipeline = NewsPipeline()
    pipeline.train_and_select(ALL_DATA)

    MODEL_PATH = "../backend/models/news_classifier.pkl"
    pipeline.save(MODEL_PATH)

    TEST_CASES = [
        "India defeated Australia in the cricket World Cup final with a stunning performance",
        "Parliament passed new bill on data protection and digital privacy rights",
        "New AI model from Google can generate code better than human programmers",
        "RBI raised interest rates to control inflation in the economy this quarter",
        "Doctors discover new treatment for diabetes using plant-based compounds",
        "Oscar winning actress stars in new Hollywood blockbuster film this week",
    ]
    print("\n" + "="*65)
    print("  SAMPLE PREDICTIONS  (using best model: " + pipeline.classifier.name + ")")
    print("="*65)
    for news in TEST_CASES:
        r = pipeline.predict(news)
        print(f"\n  {news[:62]}...")
        print(f"  {r['icon']} {r['category'].upper():<14} {r['confidence']:.1f}% confidence")
        print(f"  Keywords : {', '.join(r['keywords'])}")
        print(f"  Sentiment: {r['sentiment']}")

    print("\n" + "="*65)
    print(f"  Done! Best classifier: {pipeline.classifier.name}")
    print("="*65)