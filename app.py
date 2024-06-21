from fastapi import FastAPI, Query, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any
from keybert import KeyBERT
import fitz
from summa import keywords as summa_keywords
import nltk
from nltk.corpus import stopwords
import requests
import feedparser

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = FastAPI()

# PDF를 텍스트로 변환
def pdf_to_text(pdf_path: str) -> str:
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# KeyBERT 키워드 추출
def extract_keywords_keybert(text: str, top_n: int) -> list:
    model = KeyBERT()
    kb_keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=top_n)
    return kb_keywords

# Summa 라이브러리를 사용한 TextRank 키워드 추출
def extract_keywords_textrank_summa(text: str, top_n: int) -> list:
    tr_keywords = summa_keywords.keywords(text, words=top_n, scores=True)
    return [(kw, round(score, 4)) for kw, score in tr_keywords][:top_n]

# 논문 데이터 모델 정의
class Paper(BaseModel):
    title: str
    authors: List[str]
    summary: str
    published: str
    direct_link: str
    pdf_link: str
    category: str

class MetaResponse(BaseModel):
    resultCode: int
    data: List[Paper]

class SaveWeaResponse(BaseModel):
    resultCode: int
    data: Dict[str, str]

# 논문 데이터 불러오는 API 엔드포인트
@app.get("/getMeta")
async def get_meta(searchword: str = Query(..., description="Search term for arXiv API")) -> Dict[str, Any]:
    text = searchword.replace(" ", "+")
    base_url = f"http://export.arxiv.org/api/query?search_query=ti:{text}+OR+abs:{text}&sortBy=relevance&sortOrder=descending&start=0&max_results=15"

    response = requests.get(base_url)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch data from arXiv API")
    
    feed = feedparser.parse(response.content)
    papers = []

    for entry in feed.entries:
        link = entry.links[0]['href'] if entry.links else None
        pdf_link = entry.links[1]['href'] if len(entry.links) > 1 else None
        category = entry.arxiv_primary_category['term'] if 'arxiv_primary_category' in entry else None

        paper = {
            "title": entry.title,
            "authors": [author.name for author in entry.authors],
            "summary": entry.summary,
            "published": entry.published,
            "direct_link": link,
            "pdf_link": pdf_link,
            "category": category
        }
        papers.append(paper)

    return {
        "resultCode": 200,
        "data": papers
    }

# 논문 데이터 저장하는 API 엔드포인트
@app.post("/saveWea", response_model=SaveWeaResponse)
async def save_wea(meta_response: MetaResponse = Body(...)) -> SaveWeaResponse:
    papers = meta_response.data

    try:
        with collection.batch.fixed_size(5) as batch:
            for paper in papers:
                # title 중복 확인
                response = collection.query.fetch_objects(
                    filters=Filter.by_property("title").equal(paper.title),
                    limit=1
                )
                # object가 있으면 건너뛰기
                if response.objects:
                    continue
                
                properties = {
                    "title": paper.title,
                    "authors": paper.authors,
                    "summary": paper.summary,
                    "published": paper.published,
                    "direct_link": paper.direct_link,
                    "pdf_link": paper.pdf_link,
                    "category": paper.category,
                }

                batch.add_object(
                    properties=properties,
                )
        return SaveWeaResponse(resultCode=200, data={"message": "데이터 저장이 완료되었습니다."})
    except Exception as e:
        return SaveWeaResponse(resultCode=500, data={"message": str(e)})

def fetch_pdf_text(pdf_url: str) -> str:
    response = requests.get(pdf_url)
    with open("/tmp/paper.pdf", "wb") as f:
        f.write(response.content)
    return pdf_to_text("/tmp/paper.pdf")

# 논문 데이터 중 하나를 선택하여 키워드 추출하는 API 엔드포인트
@app.get("/extractKeywords")
async def extract_keywords_from_paper(searchword: str, title: str, top_n: int = 20):
    meta_response = await get_meta(searchword)
    if meta_response["resultCode"] != 200 or not meta_response["data"]:
        raise HTTPException(status_code=500, detail="Failed to fetch data from arXiv API")

    # 논문 목록 출력 (디버깅 용도)
    for p in meta_response["data"]:
        print(f"Title: {p['title']}")

    paper = next((p for p in meta_response["data"] if p["title"] == title), None)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    text = paper["summary"]

    # PDF 내용 추출
    if paper["pdf_link"]:
        text = fetch_pdf_text(paper["pdf_link"])

    kb_keywords = extract_keywords_keybert(text, top_n=top_n)
    tr_keywords_summa = extract_keywords_textrank_summa(text, top_n=top_n)

    return {
        "title": paper["title"],
        "summary": paper["summary"],
        "keybert_keywords": {keyword: score for keyword, score in kb_keywords},
        "textrank_keywords_summa": {keyword: score for keyword, score in tr_keywords_summa}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
