import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faqs import FAQS

_WORD_RE = re.compile(r"[a-z]+")

def tokenize(text):
    return _WORD_RE.findall(text.lower())

ENGLISH_SW = {
    "a","an","the","and","or","but","if","in","on","at","to","for","of","with",
    "by","from","as","is","was","are","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might","can",
    "it","its","this","that","these","those","i","me","my","we","our","you",
    "your","he","him","his","she","her","they","them","their","what","which",
    "who","how","all","both","each","more","most","other","some","no","not",
    "only","so","than","too","very","just","about","after","again","here",
    "into","now","off","out","over","up","while","there","then","s","t",
}
SWAHILI_SW = {
    "na","ya","wa","la","za","kwa","ni","au","si","pia","yote","hii","hizi",
    "hilo","hayo","ile","zile","huyu","hao","wake","yake","lake","mimi",
    "wewe","yeye","sisi","ninyi","wao","nini","gani","vipi","wapi","lini",
    "nani","je","ama","lakini","hata","kama","kwamba","sana","tu","tayari",
    "bado","zaidi","kabla","baada","juu","chini","ndani","nje","mbele",
    "nyuma","mbali","mara","wakati","siku","wiki","mwezi","mwaka","ndiyo",
    "hapana","ndio","pamoja",
}
ALL_SW = ENGLISH_SW | SWAHILI_SW

class PorterStemmer:
    V = set("aeiou")
    def _has_vowel(self, w): return any(c in self.V for c in w)
    def _ends(self, w, s): return w.endswith(s)
    def stem(self, word):
        w = word
        if len(w) < 4: return w
        if self._ends(w,"sses"): w = w[:-2]
        elif self._ends(w,"ies"): w = w[:-2]
        elif self._ends(w,"ss"): pass
        elif self._ends(w,"s") and self._has_vowel(w[:-1]): w = w[:-1]
        if self._ends(w,"eed"):
            if len(w) > 4: w = w[:-1]
        elif self._ends(w,"ed") and self._has_vowel(w[:-2]):
            w = self._fix(w[:-2])
        elif self._ends(w,"ing") and self._has_vowel(w[:-3]):
            w = self._fix(w[:-3])
        if self._ends(w,"y") and self._has_vowel(w[:-1]):
            w = w[:-1] + "i"
        return w
    def _fix(self, w):
        if w.endswith(("at","bl","iz")): return w + "e"
        if len(w) >= 2 and w[-1] == w[-2] and w[-1] not in "lsz": return w[:-1]
        return w

_stemmer = PorterStemmer()

def preprocess(text, stem=True):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+|\S+@\S+|\+?\d[\d\s\-]{6,}", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in tokenize(text) if t not in ALL_SW and len(t) > 2]
    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

_SW_VOCAB = {
    "habari","karibu","asante","tafadhali","samahani","pamoja","bidhaa",
    "nguo","gauni","vipimo","uwasilishaji","malipo","kurudisha","punguzo",
    "duka","asubuhi","mchana","usiku","jioni","haraka","polepole",
    "mnafanya","mnatoa","mnauza","mnavyo","mnakubali","mnafungua",
    "naweza","ninaweza","nataka","ninataka","unapatikana","inapatikana",
}
_SW_PREFIXES = ("mnaf","mnau","ninaw","nawez","unapatik","inapatik")

def detect_language(text):
    tokens = set(text.lower().split())
    if len(tokens & _SW_VOCAB) >= 2: return "sw"
    if any(t.startswith(p) for t in tokens for p in _SW_PREFIXES): return "sw"
    return "en"

class FAQMatcher:
    THRESH_LOW  = 0.08
    THRESH_HIGH = 0.22

    def __init__(self, faqs):
        self.faqs = faqs
        self._corpus  = []
        self._faq_map = []
        self._vec     = None
        self._mat     = None
        self._build()

    def _build(self):
        print("  Building message index...")
        for i, faq in enumerate(self.faqs):
            for q in faq["questions"]:
                self._corpus.append(preprocess(q))
                self._faq_map.append(i)
        self._vec = TfidfVectorizer(ngram_range=(1,2), min_df=1,
                                    sublinear_tf=True, max_features=8000)
        self._mat = self._vec.fit_transform(self._corpus)
        print(f"  Ready: {len(self._corpus)} vectors, "
              f"{self._mat.shape[1]} features")

    def match(self, query):
        lang   = detect_language(query)
        tokens = preprocess(query)
        if not tokens.strip():
            return dict(faq=None,score=0.0,confidence="low",
                        language=lang,preprocessed="",matched_q="")
        try:
            qv   = self._vec.transform([tokens])
            sims = cosine_similarity(qv, self._mat).flatten()
            bi   = int(np.argmax(sims))
            bs   = float(sims[bi])
        except Exception:
            return dict(faq=None,score=0.0,confidence="low",
                        language=lang,preprocessed=tokens,matched_q="")
        if bs < self.THRESH_LOW:
            return dict(faq=None,score=round(bs,4),confidence="low",
                        language=lang,preprocessed=tokens,matched_q="")
        return dict(
            faq        = self.faqs[self._faq_map[bi]],
            score      = round(bs,4),
            confidence = "high" if bs >= self.THRESH_HIGH else "medium",
            language   = lang,
            preprocessed = tokens,
            matched_q  = self._corpus[bi],
        )

    def get_answer(self, result):
        faq  = result.get("faq")
        lang = result.get("language","en")
        if faq is None:
            if lang == "sw":
                return ("Samahani, sikuelewa swali lako. 🙏\n\n"
                        "Tafadhali jaribu tena, au:\n"
                        "📱 WhatsApp: +255 700 123 456\n"
                        "📧 habari@mtindofashion.co.tz")
            return ("Sorry, I didn't understand that. 🙏\n\n"
                    "Please rephrase or contact us:\n"
                    "📱 WhatsApp: +255 700 123 456\n"
                    "📧 habari@mtindofashion.co.tz")
        ans = faq.get("answer_sw") if lang == "sw" else faq.get("answer_en")
        return ans or faq.get("answer_en") or faq.get("answer_sw","")

_instance = None
def get_matcher():
    global _instance
    if _instance is None:
        _instance = FAQMatcher(FAQS)
    return _instance

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Mtindo Message Matcher - Self Test")
    print("="*60)
    m = get_matcher()
    tests = [
        "What sizes do you have?",
        "Je, mnafanya delivery Arusha?",
        "Can I pay with M-Pesa?",
        "Ninaweza kurudisha bidhaa zangu?",
        "Do you have plus sizes?",
        "Habari",
        "How do I track my order?",
        "Ninaweza kuagiza custom design?",
        "What are your opening hours?",
        "random gibberish xyz",
        "care instructions for kitenge",
        "Je, kuna discounts au offers?",
    ]
    for q in tests:
        r = m.match(q)
        a = m.get_answer(r)
        cat = r["faq"]["category"] if r["faq"] else "NO MATCH"
        print(f"\nQ  : {q}")
        print(f"   Lang={r['language'].upper()} | Score={r['score']:.4f} | Conf={r['confidence']} | Cat={cat}")
        print(f"   {a[:75]}...")
    print("\n" + "="*60)
