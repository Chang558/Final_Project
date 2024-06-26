from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from keybert import KeyBERT
from summa import keywords as summa_keywords
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import fitz  # PyMuPDF
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re
import os
import requests
import spacy

# 외부 API URL
FASTAPI_URL3 = os.getenv('FASTAPI_URL3')
FASTAPI_URL4 = os.getenv('FASTAPI_URL4')

# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))
stop_words.update(['et', 'al', 'nn', 'task', 'tasks', 'model', 'models', 'modeling'])  # 추가 불용어
logging.basicConfig(level=logging.DEBUG)
nlp = spacy.load("en_core_web_sm")
app = FastAPI()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class TextRequest(BaseModel):
    text: str = None
    pdf_path: str = None

def get_wordnet_pos(word):
    """단어의 품사 태그를 얻어 WordNet의 품사로 변환합니다."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text, additional_stopwords=None):
    if additional_stopwords:
        stop_words.update(additional_stopwords.split())
    text = text.replace('\n\n', '  ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.strip().lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def extract_keywords_keybert(text, top_n=10, keyphrase_ngram_range=(1, 3), additional_stopwords=None, use_maxsum=True, nr_candidates=20, diversity=0.3):
    text = preprocess_text(text, additional_stopwords=additional_stopwords)
    
    # 문서에서 명사, 동사 필터링합니다.

    model = KeyBERT('all-MiniLM-L12-v2')
    keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=keyphrase_ngram_range, 
        top_n=top_n, 
        use_maxsum=use_maxsum, 
        diversity=diversity,
        use_mmr=True
    )
    return keywords

def remove_similar_keywords(keywords, threshold=0.7):
    if not keywords:
        return []
    
    vectorizer = TfidfVectorizer().fit_transform([kw[0] for kw in keywords])
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    to_remove = set()
    
    for i in range(len(cosine_matrix)):
        for j in range(i + 1, len(cosine_matrix)):
            if cosine_matrix[i][j] > threshold:
                to_remove.add(j)
                
    filtered_keywords = [kw for i, kw in enumerate(keywords) if i not in to_remove]
    return filtered_keywords

def ngrams(input_list, n):
    return [' '.join(input_list[i:i+n]) for i in range(len(input_list)-n+1)]


@app.get('/Bert_Keyword')
async def test_bert(title: str):
    try:
        text_response = requests.get(f"{FASTAPI_URL3}/getFullText?title={title}")
        context = text_response.json().get("data", "")
        print(f"context : {context}")
        
        if context:
            keybert_keywords = extract_keywords_keybert(context, top_n=10)
            word_list = [(word[0], word[1]) for word in keybert_keywords]
            print(word_list)
            
            # 키워드와 점수를 분리
            sorted_keywords = sorted(word_list, key=lambda x: x[1], reverse=True)
            print('sorted_keywords:', sorted_keywords)
            
            # 중복을 제거하면서 순서 유지
            split_keywords = []
            seen = set()
            for word, score in sorted_keywords:
                for k in word.split(" "):
                    if k not in seen:
                        seen.add(k)
                        split_keywords.append(k)
                    if len(split_keywords) >= 10:
                        break
                if len(split_keywords) >= 10:
                    break
                    
            unique_keywords = list(set(split_keywords))[:10]
            
            print('unique_keywords:', unique_keywords)
            save_payload = {"title": title, "keyword": unique_keywords}
            save = requests.post(f'{FASTAPI_URL3}/savebertKeyword', json=save_payload)
            print(save.text)
            getbert = requests.get(f'{FASTAPI_URL3}/getbertKeyword?title={title}')
            print(getbert.text)
            
            return {"data": unique_keywords, "result_code": 200}
            
    except Exception as e:
        logging.error(f"Error during keyword extraction: {e}")
        return {"data": [], "result_code": 500}
        
# 키워드를 레마타이제이션하고 조건에 따라 스테밍을 적용하는 함수
def process_rank_keywords(keywords):
    # 레마타이제이션 결과를 저장
    lemmatized_words = {}
    for word in keywords:
        lemma = lemmatizer.lemmatize(word, get_wordnet_pos(word))
        if lemma not in lemmatized_words:
            lemmatized_words[lemma] = []
        lemmatized_words[lemma].append(word)

    # 동일한 레마타이즈 결과를 가진 단어들에 대해 스테밍 적용
    final_words = []
    
    for lemma, words in lemmatized_words.items():
        final_words.append(sorted(words)[0])
    return final_words

def extract_keywords_textrank(text, ratio=0.2, words=30, additional_stopwords=None, use_ngrams=False, threshold=0.7):
    text = preprocess_text(text, additional_stopwords=additional_stopwords)
    
    # 사전 처리된 텍스트를 spaCy NLP 파이프라인으로 처리합니다.
    doc = nlp(text)
    
    # 문서에서 명사, 동사 필터링합니다.
    filtered_tokens = [token.text for token in doc if token.pos_ == 'VERB' or token.pos_ == 'NOUN']
    filtered_text = ' '.join(filtered_tokens)
    
    try:
        # summa의 keywords 함수를 사용해 TextRank 기반 키워드 추출을 시도합니다.
        textrank_keywords = summa_keywords.keywords(
            filtered_text, 
            ratio=ratio, 
            words=words, 
            split=True, 
            scores=True
        )

        # n-grams을 사용하는 경우 추가 처리를 수행합니다.
        if use_ngrams:
            ngram_keywords = []
            for word, score in textrank_keywords:
                ngram_keywords.extend([(phrase, score) for phrase in ngrams(word.split(), 2)])
            # 유사한 n-gram 키워드를 제거합니다.
            ngram_keywords = remove_similar_keywords(ngram_keywords, threshold)
            return ngram_keywords
        else:
            # 유사한 단일 키워드를 제거합니다.
            textrank_keywords = remove_similar_keywords(textrank_keywords, threshold)
            return textrank_keywords
    
    except Exception as e:
        # 오류 발생 시 로깅합니다.
        logging.error(f"TextRank 키워드 추출 중 오류 발생: {e}")
        return []

def ngrams(input_list, n):
    return [' '.join(input_list[i:i+n]) for i in range(len(input_list)-n+1)]

@app.get('/TextRank_Keyword')
async def text_rank(title: str):
    try:
        text_response = requests.get(f"{FASTAPI_URL3}/getFullText?title={title}")
        context = text_response.json().get("data", "")
        print(f"context : {context}")
        
        if context:
            textrank_keywords = extract_keywords_textrank(context)
            word_list = [word[0] for word in textrank_keywords[:30]]
            print('word_list : ',word_list)
            
            split_key = [k for word in word_list if len(word) > 2 for k in word.split(" ")]
            print('split_key :',split_key)
            
            processed_keywords = list(set(process_rank_keywords(split_key)))[:10]
            print('processed_keywords :', processed_keywords)
            
            save_payload = {"title": title, "keyword": processed_keywords}
            save = requests.post(f'{FASTAPI_URL3}/saverankKeyword', json=save_payload)
            print(save.text)
            
            getbert = requests.get(f'{FASTAPI_URL3}/getrankKeyword?title={title}')
            print(getbert.text)

            return {"data": processed_keywords, "result_code": 200}

    except Exception as e:
        logging.error(f"Error during keyword extraction: {e}")
        return {"data": [], "result_code": 500}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
