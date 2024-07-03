import fitz
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
import textrazor
import os
import time
from transformers.pipelines import pipeline
from transformers import AutoTokenizer
from fastapi import FastAPI
import uvicorn
from fastapi.responses import JSONResponse
app = FastAPI()

# 텍스트를 토큰 단위로 나누는 함수
def chunk_text(text, max_length=512):
    words = text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return chunks

# Tokenizer 초기화
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

print('키워드 추출 처리 중')

def highlighted_text_to_html(highlighted_text: str):
    """하이라이트된 텍스트를 HTML 파일로 저장"""
    html_content = f"""
    <html>
        <head>
            <title>Highlighted Text</title>
        </head>
        <body>
            {highlighted_text}
        </body>
    </html>
    """
    return html_content
#--------------------------------------------TextRazor---------------------------------------------
def textrazor_key_extraction(text):
    print('textrazor_key_extraction 시작')
    wiki = []
    start = time.time()
    data = {}
    link = {}
    keyword = {}
    # TextRazor를 사용한 키워드 추출
    textrazor.api_key = os.getenv('TEXTRAZOR_API_KEY')
    client = textrazor.TextRazor(extractors=["entities", "keywords"])
    response = client.analyze(text).json['response']
    entities = response['entities']

    # 엔티티에서 키워드와 점수 추출 
    for entity in entities:
        word = entity['entityId']
        score = entity['relevanceScore']
        wikiLink = entity['wikiLink']
        # data = entity['data']
        keyword[word] = score
        link[word]=wikiLink
        
    # 키워드를 점수 기준으로 정렬하여 상위 10개 선택
    top_10_keywords = dict(sorted(keyword.items(), key=lambda item: item[1], reverse=True)[:10])
    top_10_wikiLink = dict(sorted(link.items(), key=lambda item: item[1], reverse=True)[:10])
    end = time.time()
    print("\nTextRazor 키워드:", top_10_keywords)
    print('\ntop_10_wikiLink:', top_10_wikiLink)
    print(f"{end - start:.5f} 초")
    print('TextRazor_extraction 종료')
    
    
    return top_10_keywords, top_10_wikiLink
# --------------------------------------------허깅페이스 모델---------------------------------------------

def hugging_face_key_extraction(text):
    print('hugging_face_key_extraction 시작')
    chunks = chunk_text(text, max_length=512)

    hf_model = pipeline("feature-extraction", model="distilbert-base-cased")
    hf_kw_model = KeyBERT(model=hf_model)
    start = time.time()

    hf_key = []
    for chunk in chunks:
        hf_key.extend(hf_kw_model.extract_keywords(
            chunk,
            keyphrase_ngram_range=(1, 3), 
            top_n=5, 
            use_maxsum=True, 
            diversity=0.5,
            use_mmr=True,
            nr_candidates=20,
            highlight=highlight
        ))
    end = time.time()
    print("\nKeyBERT (hf_model) 키워드:", hf_key[:10])
    print(f"{end - start:.5f} 초")
    print('hugging_face_key_extraction 종료')
    return  hf_key[:10]

# --------------------------------------------기본 모델---------------------------------------------
def default_key_extraction(text: str, highlight: bool): 
    print('default_key_extraction 시작')

    start_time = time.time()
    chunks = chunk_text(text, max_length=512)

    model = KeyBERT('all-MiniLM-L12-v2')
    default_key = []

    for chunk in chunks:
        keywords_with_highlight = model.extract_keywords(
            chunk,
            keyphrase_ngram_range=(1, 3), 
            top_n=5, 
            use_maxsum=True, 
            diversity=0.5,
            use_mmr=True,
            nr_candidates=20,
            highlight=False  # highlight를 False로 설정하여 키워드만 추출
        )
        default_key.extend(keywords_with_highlight)

    replace_text = text
    if highlight:
        for keyword, _ in default_key:
            replace_text = replace_text.replace(keyword, f'<mark style="background-color: yellow;">{keyword}</mark>')

    end_time = time.time()
    print(f"{end_time - start_time:.5f} 초")
    print('Highlight background End')
    print('replace_text :', replace_text)

    if highlight:
        highlighted_text_to_html(replace_text)

    return default_key[:10], replace_text

# --------------------------------------------roberta 모델---------------------------------------------
def roberta_key_extraction(text):
    print('roberta_key_extraction 시작')

    start = time.time()
    chunks = chunk_text(text, max_length=512)
    print('roberta_key_extraction 시작')

    # KeyBERT를 사용한 키워드 추출
    roberta = TransformerDocumentEmbeddings('roberta-base')
    kw_model = KeyBERT(model=roberta)

    roberta_key = []
    for chunk in chunks:
        roberta_key.extend(kw_model.extract_keywords(
            chunk,
            keyphrase_ngram_range=(1,3), 
            top_n=5, 
            use_maxsum=True, 
            diversity=0.5,
            use_mmr=True,
            nr_candidates=20,
            highlight=highlight
        ))
    end = time.time()


    print("\nKeyBERT (roberta) 키워드:", roberta_key[:10])
    print(f"{end - start:.5f} 초")
    print('roberta_key_extraction 종료')
    return roberta_key[:10]

@app.get('/textrazor_key_extraction')
async def textrazor_key(text: str):
    result = textrazor_key_extraction(text)
    return result

@app.get('/hugging_face_key_extraction')
async def hugging_face_key(text: str, highlight: bool):
    result = hugging_face_key_extraction(text, highlight)
    return result


@app.get('/default_key_extraction')
async def default_key(text: str, highlight: bool):
    result, highlighted_text = default_key_extraction(text, highlight)  # 동기 함수 호출
    return JSONResponse(content={"keywords": result, "highlighted_text": highlighted_text})

@app.get('/roberta_key_extraction')
async def roberta_key(text: str, highlight: bool):
    result = roberta_key_extraction(text,highlight)
    return result
