# Mtindo Fashion FAQ Chatbot 👗
**Bilingual (English + Swahili) FAQ Chatbot using NLP**

A full NLP-powered FAQ chatbot for a fashion e-commerce shop.
Implements the complete text processing pipeline: tokenization, stopword removal,
stemming, TF-IDF vectorization, and cosine similarity matching.

---

## Project Structure

```
mtindo_bot/
├── faqs.py            ← FAQ knowledge base (EN + SW questions & answers)
├── nlp_engine.py      ← NLP pipeline: preprocess → TF-IDF → cosine similarity
├── app.py             ← Flask REST API backend
├── requirements.txt   ← Python dependencies
├── README.md          ← This file
└── static/
    └── index.html     ← Chat UI (HTML/CSS/JS)
```

---

## NLP Pipeline (nlp_engine.py)

### Step 1 — Text Preprocessing
| Operation | Implementation |
|---|---|
| Lowercase | `text.lower()` |
| Remove URLs/emails/phones | `re.sub(...)` |
| Remove punctuation & digits | `re.sub(r"[^a-z\s]", ...)` |
| Tokenization | Regex word tokenizer `[a-z]+` |
| Stopword removal | English + Swahili stopword sets |
| Stemming | Custom Porter Stemmer (pure Python) |

### Step 2 — TF-IDF Vectorization
- `TfidfVectorizer` from scikit-learn
- `ngram_range=(1,2)` — unigrams + bigrams
- `sublinear_tf=True` — log(1+tf) dampens common terms
- Fitted on all 160+ FAQ question variants

### Step 3 — Cosine Similarity Matching
```
similarity = (query_vector · faq_vector) / (|query| × |faq|)
```
- Returns score 0.0–1.0
- Score < 0.08 → no match (fallback message)
- Score 0.08–0.22 → medium confidence
- Score ≥ 0.22 → high confidence

### Step 4 — Language Detection
- Swahili indicator vocabulary (morphologically distinct words)
- Prefix detection (`mnaf-`, `ninaw-`, `nawez-`, etc.)
- Returns `'sw'` or `'en'` → selects correct answer language

---

## Setup & Run

### 1. Install dependencies
```bash
pip install flask scikit-learn numpy
```

### 2. Test the NLP engine
```bash
python nlp_engine.py
```

### 3. Start the server
```bash
python app.py
```

### 4. Open the chat UI
Visit: **http://localhost:5000**

---

## API Endpoint

### POST /chat
**Request:**
```json
{ "message": "What sizes do you have?" }
```
**Response:**
```json
{
  "answer":     "We stock a full range of sizes...",
  "category":   "Sizes",
  "language":   "en",
  "confidence": "high",
  "score":      1.0,
  "tokens":     "size stock rang",
  "elapsed_ms": 4.2
}
```

---

## FAQ Categories
| # | Category | Questions |
|---|---|---|
| 1 | Products | What do you sell? / Mnatoa bidhaa gani? |
| 2 | Sizes | What sizes? / Vipimo gani? |
| 3 | Delivery | Do you deliver? / Je, mnafanya delivery? |
| 4 | Returns | Return policy / Kurudisha bidhaa |
| 5 | Payment | M-Pesa, cards / Jinsi ya kulipa |
| 6 | Discounts | Offers, promo codes / Punguzo |
| 7 | Custom Orders | Tailoring / Kushona |
| 8 | Contact | Phone, address / Mawasiliano |
| 9 | Care | Wash instructions / Jinsi ya kuosha |
| 10 | Order Tracking | Track package / Fuatilia oda |
| 11 | Greeting | Hello / Habari |
| 12 | Hours | Opening times / Masaa ya kufungua |

---

## Technologies
- **Python 3.11+**
- **scikit-learn** — TF-IDF, cosine similarity
- **numpy** — matrix operations
- **Flask** — REST API server
- **HTML/CSS/JS** — Chat UI (no framework needed)
