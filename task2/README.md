# Mtindo Fashion FAQ Chatbot

**Bilingual English and Swahili FAQ chatbot**

A customer support chatbot for a fashion e-commerce shop. It answers common
questions about products, sizes, delivery, returns, payments, discounts, custom
orders, contact details, care instructions, tracking, greetings, and opening
hours.

---

## Project Structure

```text
mtindo_bot/
+-- faqs.py
+-- matcher_engine.py
+-- app.py
+-- requirements.txt
+-- README.md
+-- static/
    +-- index.html
```

---

## Message Matching

### Step 1 - Text Preprocessing

| Operation | Implementation |
|---|---|
| Lowercase | `text.lower()` |
| Remove URLs/emails/phones | `re.sub(...)` |
| Remove punctuation and digits | `re.sub(r"[^a-z\s]", ...)` |
| Tokenization | Regex word tokenizer `[a-z]+` |
| Stopword removal | English and Swahili stopword sets |
| Stemming | Custom pure-Python stemmer |

### Step 2 - Query Matching

- Builds a searchable index from FAQ question variants
- Compares incoming questions with known FAQ entries
- Selects the closest matching category

### Step 3 - Confidence Scoring

- Returns score 0.0-1.0
- Score < 0.08: no match
- Score 0.08-0.22: medium confidence
- Score >= 0.22: high confidence

### Step 4 - Language Detection

- Checks Swahili indicator vocabulary
- Checks common Swahili prefixes
- Returns `sw` or `en` to choose the answer language

---

## Setup & Run

### 1. Install dependencies

```bash
pip install flask scikit-learn numpy
```

### 2. Test the matching engine

```bash
python matcher_engine.py
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
  "answer": "We stock a full range of sizes...",
  "category": "Sizes",
  "language": "en",
  "confidence": "high",
  "score": 1.0,
  "tokens": "size stock rang",
  "elapsed_ms": 4.2
}
```

---

## FAQ Categories

| # | Category | Example Questions |
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

- Python 3.11+
- Flask
- scikit-learn
- numpy
- HTML/CSS/JS
