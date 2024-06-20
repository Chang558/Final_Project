import fitz  # PyMuPDF
from summa import keywords
from keybert import KeyBERT

def pdf_to_text(pdf_path):
    # PDF 파일 열기
    pdf_document = fitz.open(pdf_path)
    
    # 모든 페이지의 텍스트를 추출하여 하나의 문자열로 결합
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    
    return text

# PDF 파일 경로
pdf_path = "President_Obamas_Farewell_Address_영어_원본.pdf"

# PDF 파일을 텍스트로 변환
text = pdf_to_text(pdf_path)

def extract_keywords_keybert(text, top_n=20):
    kw_model = KeyBERT()
    keybert_keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=top_n)
    return [keyword[0] for keyword in keybert_keywords]


keybert_keywords = extract_keywords_keybert(text)

print(keybert_keywords)