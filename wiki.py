import wikipedia
import wikipediaapi

# 검색 키워드 설정
search_query = "Python programming"

# 검색 수행
search_results = wikipedia.search(search_query)

# 검색 결과 출력
print("검색 결과:")
for i, result in enumerate(search_results):
    print(f"{i + 1}. {result}")

# 첫 번째 결과 가져오기
if search_results:
    first_result = search_results[0]
    print(f"\n첫 번째 결과: {first_result}")

    # wikipediaapi를 사용하여 첫 번째 결과의 페이지 내용 가져오기
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent='MyProjectName (merlin@example.com)',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    page = wiki_wiki.page(first_result)
    print("\n페이지 내용 (텍스트 형식):")
    print(page.text)

    wiki_html = wikipediaapi.Wikipedia(
        user_agent='MyProjectName (merlin@example.com)',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.HTML
    )
    page_html = wiki_html.page(first_result)
    print("\n페이지 내용 (HTML 형식):")
    print(page_html.text)
else:
    print("검색 결과가 없습니다.")
