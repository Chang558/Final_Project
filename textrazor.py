import textrazor
import os
import fitz

# Textrazor API 키 설정
textrazor.api_key = os.getenv('TEXTRAZOR_API_KEY')
client = textrazor.TextRazor(extractors=["entities", "keywords"])

# PDF에서 텍스트 추출
path = "./1910.14296v2.pdf"
doc = fitz.open(path)
text = ""
for page in doc:
    text += page.get_text("text")

# Textrazor로 텍스트 분석
response = client.analyze(text).json['response']
entities = response['entities']
keyword = {}

# 엔티티에서 키워드와 점수 추출
for entity in entities:
    word = entity['entityId']
    score = entity['relevanceScore']
    keyword[word] = score

# 키워드를 점수 기준으로 정렬하여 상위 10개 선택
top_10_keywords = dict(sorted(keyword.items(), key=lambda item: item[1], reverse=True)[:10])

# 상위 10개 키워드 출력
print("상위 10개 키워드:", top_10_keywords)
