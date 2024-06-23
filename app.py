from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from keybert import KeyBERT
from summa import keywords as summa_keywords
import nltk
import fitz  # PyMuPDF
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import logging
import re

nltk.download('stopwords')
stop_words = list(set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS)))

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

class TextRequest(BaseModel):
    text: str = None
    pdf_path: str = None

default_pdf_path = '1910.14296v2.pdf'

def pdf_to_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        logging.debug(f"PDF에서 추출된 텍스트 길이: {len(text)}")
        return text
    except Exception as e:
        logging.error(f"PDF에서 텍스트를 추출하는 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail="PDF 파일에서 텍스트를 추출하는 데 실패했습니다.")

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip().lower()
    return text

def extract_keywords_keybert(text, top_n=5, keyphrase_ngram_range=(1, 3), use_maxsum=False, nr_candidates=20, diversity=0.5):
    text = preprocess_text(text)
    if len(text.split()) < 10:
        logging.debug("텍스트가 너무 짧아 키워드 추출이 불가능합니다.")
        return []
    
    model = KeyBERT('all-MiniLM-L6-v2')
    try:
        keywords = model.extract_keywords(text, keyphrase_ngram_range=keyphrase_ngram_range, top_n=top_n, use_maxsum=use_maxsum, diversity=diversity)
        logging.debug(f"KeyBERT에서 추출된 키워드: {keywords}")
        return keywords
    except Exception as e:
        logging.error(f"KeyBERT 키워드 추출 중 오류 발생: {e}")
        return []

def extract_keywords_textrank(text, top_n=5):
    text = preprocess_text(text)
    try:
        textrank_keywords = summa_keywords.keywords(text, words=top_n, scores=True)
        return textrank_keywords
    except Exception as e:
        logging.error(f"TextRank 키워드 추출 중 오류 발생: {e}")
        return []

def extract_keywords_tfidf(text, top_n=5):
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = tfidf_matrix.toarray().flatten().argsort()[-top_n:]
    top_keywords = [(feature_array[i], tfidf_matrix[0, i]) for i in tfidf_sorting]
    return top_keywords

def extract_keywords_lda(text, top_n=5, num_topics=1):
    processed_text = preprocess_text(text)
    words = processed_text.split()
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow(words)]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    topics = lda_model.show_topics(formatted=False)
    top_keywords = [(word, float(prob)) for word, prob in topics[0][1]][:top_n]
    return top_keywords

@app.post("/Bert_Keyword")
async def bert_keyword(request: TextRequest = Body(None), top_n: int = 5):
    if request and (request.text or request.pdf_path):
        text = request.text if request.text else pdf_to_text(request.pdf_path)
    else:
        text = pdf_to_text(default_pdf_path)
    keybert_keywords = extract_keywords_keybert(text, top_n=top_n)
    return {"keybert_keywords": keybert_keywords}

@app.post("/TextRank_Keyword")
async def textrank_keyword(request: TextRequest = Body(None), top_n: int = 5):
    if request and (request.text or request.pdf_path):
        text = request.text if request.text else pdf_to_text(request.pdf_path)
    else:
        text = pdf_to_text(default_pdf_path)
    textrank_keywords = extract_keywords_textrank(text, top_n=top_n)
    return {"textrank_keywords": textrank_keywords}

@app.post("/TFIDF_Keyword")
async def tfidf_keyword(request: TextRequest = Body(None), top_n: int = 5):
    if request and (request.text or request.pdf_path):
        text = request.text if request.text else pdf_to_text(request.pdf_path)
    else:
        text = pdf_to_text(default_pdf_path)
    tfidf_keywords = extract_keywords_tfidf(text, top_n=top_n)
    return {"tfidf_keywords": tfidf_keywords}

@app.post("/LDA_Keyword")
async def lda_keyword(request: TextRequest = Body(None), top_n: int = 5):
    if request and (request.text or request.pdf_path):
        text = request.text if request.text else pdf_to_text(request.pdf_path)
    else:
        text = pdf_to_text(default_pdf_path)
    lda_keywords = extract_keywords_lda(text, top_n=top_n)
    return {"lda_keywords": lda_keywords}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
