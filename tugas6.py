import io
import re
import json
import requests
from collections import Counter

import streamlit as st

# Optional deps
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import docx2txt
except ImportError:
    docx2txt = None

# OCR stack (optional: needs system Tesseract)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
except Exception:
    convert_from_bytes = None
    pytesseract = None
    Image = None

# NER (optional)
try:
    import spacy
except Exception:
    spacy = None

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Document Analyzer+", page_icon="📄", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def read_txt(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return data.decode("latin-1", errors="ignore")

def read_pdf_bytes(data: bytes):
    """Return (text, tables). If pdfplumber absent, fallback to OCR if enabled later."""
    if pdfplumber is None:
        return "", []
    text_parts = []
    tables = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
            try:
                page_tables = page.extract_tables()
                for t in page_tables or []:
                    tables.append(t)
            except Exception:
                pass
    return "\n".join(text_parts), tables

def ocr_pdf_bytes(data: bytes, lang: str = "eng") -> str:
    if convert_from_bytes is None or pytesseract is None:
        return ""
    images = convert_from_bytes(data, dpi=300)
    ocr_texts = []
    for img in images:
        ocr_texts.append(pytesseract.image_to_string(img, lang=lang) or "")
    return "\n".join(ocr_texts)

def read_docx_bytes(data: bytes) -> str:
    if docx2txt is None:
        return ""
    return docx2txt.process(io.BytesIO(data))

def basic_stats(text):
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len([s for s in sentences if s.strip()]),
        "avg_words_per_sentence": (len(words) / max(1, len(sentences)))
    }

def find_patterns(text):
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phones = re.findall(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}", text)
    urls = re.findall(r"https?://[^\s)>\]]+", text)
    return {"emails": sorted(set(emails)), "phones": sorted(set(phones)), "urls": sorted(set(urls))}

def top_ngrams(text, n=1, topk=20, stopwords=None):
    if stopwords is None:
        stopwords = set("""a an the and or is are was were be been being of to in for on with as by at from that this it
        i you he she they we not but if then else when where who whom whose which into out up down over under again
        very can will just also di ke dari untuk yang dan atau tidak""".split())
    tokens = [t for t in re.findall(r"\b[a-zA-Z0-9']+\b", text.lower()) if t not in stopwords]
    grams = tokens if n == 1 else [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    counts = Counter(grams)
    return counts.most_common(topk)

def freq_summarize(text, max_sentences=5):
    sentences = re.split(r"(?<=[.!?])\s+", normalize_ws(text))
    words = re.findall(r"\b\w+\b", text.lower())
    stop = set("""a an the and or is are was were be been being of to in for on with as by at from that this it
        i you he she they we not but if then else when where who whom whose which into out up down over under again
        very can will just also di ke dari untuk yang dan atau tidak""".split())
    freqs = Counter([w for w in words if w not in stop])
    if not freqs:
        return ""
    scores = []
    for idx, s in enumerate(sentences):
        sw = re.findall(r"\b\w+\b", s.lower())
        score = sum(freqs.get(w, 0) for w in sw) / max(1, len(sw))
        scores.append((score, idx, s))
    top = sorted(scores, reverse=True)[:max_sentences]
    top_sorted = [t[2] for t in sorted(top, key=lambda x: x[1])]
    return " ".join(top_sorted)

def make_tableframes(tables):
    frames = []
    for t in tables:
        max_cols = max(len(r) for r in t)
        frames.append([r + [""]*(max_cols - len(r)) for r in t])
    return frames

def run_ner(text, model_name: str):
    if spacy is None:
        return None, "spaCy belum terpasang."
    try:
        nlp = spacy.load(model_name)
    except Exception as e:
        return None, f"Gagal load model '{model_name}'. Pastikan sudah di-download. Error: {e}"
    doc = nlp(text[:200000])  # cap
    ents = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return ents, None

# -----------------------------
# UI
# -----------------------------
st.title("📄 Document Analyzer+ (NER • OCR • n8n)")
st.caption("Upload dokumen (.pdf, .docx, .txt), ringkas & analisa, lalu kirim hasil ke n8n webhook.")

with st.sidebar:
    st.header("⚙️ Opsi")
    use_ocr = st.checkbox("Gunakan OCR untuk PDF (butuh Tesseract)", value=True)
    ocr_lang = st.text_input("Bahasa OCR (kode Tesseract)", value="eng+ind")
    enable_ner = st.checkbox("Jalankan NER (spaCy)", value=False)
    ner_model = st.text_input("spaCy model", value="en_core_web_sm")
    st.markdown("---")
    n8n_url = st.text_input("n8n Webhook URL (opsional)", value="", help="Contoh: https://host/webhook/document_analysis")
    n8n_token = st.text_input("Auth Bearer (opsional)", value="", type="password")

uploaded = st.file_uploader("Upload satu file dokumen", type=["pdf", "docx", "txt"])

if uploaded:
    name = uploaded.name
    data = uploaded.read()
    text, extracted_tables = "", []

    if name.lower().endswith(".txt"):
        text = read_txt(data)
    elif name.lower().endswith(".pdf"):
        text, extracted_tables = read_pdf_bytes(data)
        # Fallback to OCR if empty or user forced OCR
        if use_ocr and ((not text.strip()) or st.checkbox("Paksakan OCR untuk PDF ini", value=False)):
            if convert_from_bytes is None or pytesseract is None:
                st.warning("OCR tidak tersedia: instal pdf2image, pytesseract, dan Tesseract biner sistem.")
            else:
                with st.spinner("Melakukan OCR pada halaman PDF..."):
                    text = ocr_pdf_bytes(data, lang=ocr_lang) or text
    elif name.lower().endswith(".docx"):
        text = read_docx_bytes(data)
    else:
        st.error("Format tidak didukung.")
        st.stop()

    text = (text or "").strip()
    st.write(f"**File:** {name}")
    if not text:
        st.warning("Tidak ada teks yang berhasil diekstrak.")
        st.stop()

    with st.expander("Preview Teks", expanded=False):
        st.text_area("Teks", text, height=220)

    cols = st.columns(3)
    stats = basic_stats(text)
    cols[0].metric("Kata", f"{stats['words']:,}")
    cols[1].metric("Kalimat", f"{stats['sentences']:,}")
    cols[2].metric("Rata-rata kata/kalimat", f"{stats['avg_words_per_sentence']:.2f}")

    patterns = find_patterns(text)
    with st.expander("Deteksi Pola (Email/Telepon/URL)"):
        st.write({"emails": patterns["emails"][:50], "phones": patterns["phones"][:50], "urls": patterns["urls"][:50]})

    st.subheader("🔑 Keywords / N-grams")
    c1, c2, c3 = st.columns(3)
    top1 = top_ngrams(text, n=1, topk=20)
    top2 = top_ngrams(text, n=2, topk=20)
    top3 = top_ngrams(text, n=3, topk=20)
    with c1: st.write("Top Unigram"); st.table(top1)
    with c2: st.write("Top Bigram"); st.table(top2)
    with c3: st.write("Top Trigram"); st.table(top3)

    st.subheader("📝 Ringkasan Otomatis")
    n_sent = st.slider("Jumlah kalimat ringkasan", 2, 12, 5)
    summary = freq_summarize(text, max_sentences=n_sent)
    st.write(summary if summary else "_Tidak cukup teks untuk diringkas._")

    st.subheader("🙂 Analisis Sentimen (VADER)")
    analyzer = SentimentIntensityAnalyzer()
    senti = analyzer.polarity_scores(text[:20000])
    st.write(senti)

    ner_results = None
    if enable_ner:
        st.subheader("🏷️ Named Entity Recognition (spaCy)")
        if spacy is None:
            st.warning("spaCy belum terpasang.")
        else:
            with st.spinner("Menjalankan NER..."):
                ents, err = run_ner(text, ner_model)
            if err:
                st.error(err)
            else:
                ner_results = ents
                if ents:
                    st.table(ents[:200])
                else:
                    st.info("Tidak ada entitas terdeteksi oleh model ini.")

    if extracted_tables:
        st.subheader("📊 Tabel Terdeteksi (PDF)")
        for i, tbl in enumerate(make_tableframes(extracted_tables), start=1):
            st.write(f"Tabel {i}")
            st.table(tbl[:30])

    # Export data
    export = {
        "file_name": name,
        "stats": stats,
        "patterns": patterns,
        "summary": summary,
        "top_unigram": top1,
        "top_bigram": top2,
        "top_trigram": top3,
        "sentiment": senti,
        "ner": ner_results,
        "ocr_used": bool(use_ocr),
    }

    st.subheader("⬇️ Ekspor & Kirim")
    st.download_button("Unduh Hasil (JSON)", data=json.dumps(export, ensure_ascii=False, indent=2), file_name="analysis.json", mime="application/json")

    if n8n_url:
        if st.button("Kirim ke n8n Webhook"):
            headers = {"Content-Type": "application/json"}
            if n8n_token:
                headers["Authorization"] = f"Bearer {n8n_token}"
            try:
                resp = requests.post(n8n_url, data=json.dumps(export), headers=headers, timeout=30)
                st.success(f"Dikirim ke n8n. Status: {resp.status_code}")
                st.caption(resp.text[:3000])
            except Exception as e:
                st.error(f"Gagal kirim ke n8n: {e}")
else:
    st.write("⬆️ Mulai dengan mengupload dokumen.")

