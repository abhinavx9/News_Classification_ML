"""
=============================================================
  NEWS CLASSIFIER - Flask Backend API
=============================================================
  STEPS TO RUN:
  1. pip install numpy flask flask-cors
  2. Make sure model is trained first (run train_model.py)
  3. python app.py
  4. API runs at: http://localhost:5000
=============================================================
"""

import os
import sys
import json
import pickle
import traceback
from collections import Counter          # ✅ FIX 3: import at top level, not inside function
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend to call API

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────

# ✅ FIX 1: Resolve path relative to this file so it works from any working directory.
#    train_model.py saves to "../backend/models/news_classifier.pkl"
#    so when app.py IS the backend, the model lives in a "models/" subfolder next to it.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "news_classifier.pkl")
pipeline = None

def load_model():
    global pipeline
    # ✅ FIX 2: Raise RuntimeError instead of sys.exit() so Flask can still start
    #    and return a proper 503 error rather than crashing the whole process.
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at: {MODEL_PATH}\n"
            "Please run:  python train_model.py  first to generate the model file."
        )
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    print(f"✅ Model loaded from: {MODEL_PATH}")

# ─── API ROUTES ───────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "running",
        "message": "News Classifier API",
        "endpoints": {
            "POST /classify": "Classify a single news article",
            "POST /batch": "Classify multiple articles",
            "GET /categories": "List all categories",
            "GET /health": "Health check"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": pipeline is not None})

@app.route('/categories', methods=['GET'])
def categories():
    """Return all supported categories"""
    cats = [
        {"id": "sports",      "label": "Sports",      "icon": "⚽", "color": "#ef4444"},
        {"id": "politics",    "label": "Politics",    "icon": "🏛️", "color": "#8b5cf6"},
        {"id": "technology",  "label": "Technology",  "icon": "💻", "color": "#3b82f6"},
        {"id": "business",    "label": "Business",    "icon": "📈", "color": "#10b981"},
        {"id": "health",      "label": "Health",      "icon": "🏥", "color": "#f59e0b"},
        {"id": "environment", "label": "Environment", "icon": "🌿", "color": "#22c55e"},
        {"id": "education",   "label": "Education",   "icon": "📚", "color": "#ec4899"},
    ]
    return jsonify({"categories": cats})

@app.route('/classify', methods=['POST'])
def classify():
    """
    POST /classify
    Body: { "text": "your news article here..." }
    Returns: classification + summary + keywords + sentiment
    """
    # ✅ FIX 2 (continued): Return 503 if model failed to load instead of crashing
    if pipeline is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request body"}), 400

        text = data['text'].strip()

        if len(text) < 10:
            return jsonify({"error": "Text too short. Provide at least 10 characters."}), 400

        if len(text) > 10000:
            text = text[:10000]  # Truncate very long articles

        result = pipeline.predict(text)

        return jsonify({
            "success": True,
            "input_text": text[:200] + "..." if len(text) > 200 else text,
            "category": result['category'],
            "icon": result['icon'],
            "confidence": result['confidence'],
            "all_probabilities": result['all_probs'],
            "summary": result['summary'],
            "keywords": result['keywords'],
            "reading_time": result['reading_time'],
            "word_count": result['word_count'],
            "sentiment": result['sentiment']
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/batch', methods=['POST'])
def batch_classify():
    """
    POST /batch
    Body: { "articles": ["text1", "text2", ...] }
    Returns: list of classifications
    """
    # ✅ FIX 2 (continued): Guard against unloaded model
    if pipeline is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    try:
        data = request.get_json()

        if not data or 'articles' not in data:
            return jsonify({"error": "Missing 'articles' field"}), 400

        articles = data['articles']

        if not isinstance(articles, list) or len(articles) == 0:
            return jsonify({"error": "'articles' must be a non-empty list"}), 400

        if len(articles) > 20:
            return jsonify({"error": "Maximum 20 articles per batch request"}), 400

        results = []
        for i, text in enumerate(articles):
            try:
                result = pipeline.predict(str(text))
                results.append({
                    "index": i,
                    "category": result['category'],
                    "icon": result['icon'],
                    "confidence": result['confidence'],
                    "summary": result['summary'],
                    "keywords": result['keywords'],
                    "sentiment": result['sentiment'],
                    "word_count": result['word_count']
                })
            except Exception as e:
                results.append({"index": i, "error": str(e)})

        # Stats — ✅ FIX 3: Counter now imported at top, not here
        categories_list = [r.get('category') for r in results if 'category' in r]
        category_counts = dict(Counter(categories_list))

        return jsonify({
            "success": True,
            "total": len(results),
            "results": results,
            "category_distribution": category_counts
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        load_model()
    except RuntimeError as e:
        # ✅ FIX 2: Print warning but don't exit — API still starts,
        #    endpoints will return 503 until model is trained and server restarted.
        print(f"\n⚠️  WARNING: {e}")
        print("   The server will start, but /classify and /batch will return 503.")
        print("   Train the model first, then restart this server.\n")

    print("\n🚀 Starting News Classifier API...")
    print(f"   Model path : {MODEL_PATH}")
    print("   URL        : http://localhost:5000")
    print("   Press CTRL+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
