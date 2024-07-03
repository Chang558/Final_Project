import streamlit as st
import os
import tiktoken
import warnings
import requests
import textract
import re
import tempfile
from langchain.text_splitter import CharacterTextSplitter
import concurrent.futures
import time
# from bar import single_bar_all, single_bar_time, single_bar_accuracy

warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub')

FASTAPI_URL1 = os.getenv('FASTAPI_URL1')
FASTAPI_URL2 = os.getenv('FASTAPI_URL2')
# FASTAPI_URL3 = os.getenv('FASTAPI_URL3')
FASTAPI_URL3 = 'http://54.180.183.205:3500/'
# FASTAPI_URL4 = os.getenv('FASTAPI_URL4')
FASTAPI_URL4 = 'http://3.38.197.61:8000/'
def summarize_PDF_file(pdf_file, model, input_title):

    if (pdf_file is not None):

        st.write("PDF 문서를 요약 중입니다. 잠시만 기다려 주세요.")

        # PDF 파일을 바이트 스트림으로 읽기
        file_bytes = pdf_file.read()
        url = f"{FASTAPI_URL3}/upload"
        headers = {"Content-Type": "application/pdf", "titles": input_title}
        response = requests.post(url, data=file_bytes, headers=headers)
        print("pdf sended")

        if response.status_code == 200:
            texts = response.json().get("texts", "")
            st.write("파일이 성공적으로 전송되었습니다.")

            # 스플리터 지정
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                separator="\\n\\n",  # 분할 기준
                chunk_size=2000,   # 청크 사이즈
                chunk_overlap=100, # 중첩 사이즈
            )

            split_texts = text_splitter.split_text(texts)
            st.write(len(split_texts), "단락이 있음.")
            
            # # 결과 출력
            # st.write("단락별 텍스트:")
            # for i, paragraph in enumerate(split_texts):
            #     st.write(f"단락 {i+1}:")
            #     st.write(paragraph)
            #     st.write("---")
        if model == "bart":
            url2 = f"{FASTAPI_URL2}/summarize"
            headers2 = {"Content-Type": "application/json", "titles": input_title}
            summaries = requests.post(url2, json={"title": input_title,"texts": split_texts}, headers=headers2)
            # st.write(f"요약 시간: {int(bart_time // 60)}분 {(bart_time % 60):.2f}초")
            sum_list = summaries.json().get("data", "")
            st.write("요약된 단락:")
            for i, summary in enumerate(sum_list):
                st.write(f"{summary}")
                st.write("--")
        elif model == "flan":
            print("구현중")
            
        elif model == "mean":
            print("구현중")
    else:
        st.write("PDF 파일를 업로드하세요.")
        

#키워드 추출
def keyword_extraction(input_title):
    if input_title:
        url = f"{FASTAPI_URL4}/keyword"
        data = {"title": input_title}
        response = requests.post(url, json=data)
        if response.status_code == 200:
            keywords = response.json().get("keywords", "")
            st.write("추출된 키워드:", keywords)
        else:
            st.write("키워드 추출 실패:", response.json().get("detail", "Unknown error occurred"))
    else:
        st.write("제목을 입력하세요.")


# # ------------- 사이드바 화면 구성 -----------------------
# with st.sidebar:
#     st.title('Summarization')
#     sum_model = ["bart-large-cnn","flan-t5-3b-summarizer", "의미론적 방법론"]
#     selected_sum_model = st.selectbox("Please select the PDF summary model.", sum_model)

#     if selected_sum_model == "bart-large-cnn":
#         st.caption('https://huggingface.co/facebook/bart-large-cnn \
#                    bart-large-cnn 허깅페이스 모델을 사용하여 텍스트 요약을 수행합니다. \
#                    이 모델의 차이점은 모델의 크기가 500MB로 적은 disk로도 요약이 가능합니다. \
#                    그러나 모델의 크기가 작은만큼 AI 학습이 효과가 다른 모델에 비해 떨어질 수 있습니다.')
#     elif selected_sum_model == "flan-t5-3b-summarizer":
#         st.caption('https://huggingface.co/jordiclive/flan-t5-3b-summarizer')
#         st.caption('bart-large-cnn 허깅페이스 모델을 사용하여 텍스트 요약을 수행합니다. \
#                    이 모델의 차이점은 모델의 크기가 500MB로 적은 disk로도 요약이 가능합니다. \
#                    그러나 모델의 크기가 작은만큼 AI 학습이 효과가 다른 모델에 비해 떨어질 수 있습니다.')

#     elif selected_sum_model == "의미론적 방법론":
#         st.caption('https://huggingface.co/jordiclive/flan-t5-3b-summarizer')
#         st.caption('bart-large-cnn 허깅페이스 모델을 사용하여 텍스트 요약을 수행합니다. \
#                    이 모델의 차이점은 모델의 크기가 500MB로 적은 disk로도 요약이 가능합니다. \
#                    그러나 모델의 크기가 작은만큼 AI 학습이 효과가 다른 모델에 비해 떨어질 수 있습니다.')

#     chart = ["Choose a chart","All", "Time", "Accuracy"]
#     selected_chart = st.selectbox("Please select the chart.", chart)

#     st.markdown("---")

#     st.markdown(
#             '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" \
#             alt="Streamlit logo" height="16">&nbsp by',unsafe_allow_html=True)

#     st.markdown(
#             '<h6> \
#             <a href="https://github.com/raminicano">@raminicano</a> \
#             <a href="https://github.com/sinzng">@sinzng</a> \
#             <a href="https://github.com/wjdguswn1203">@wjdguswn1203</a> \
#             <a href="https://github.com/chang558">@chang558</a> \
#             </h6>',
#             unsafe_allow_html=True,
#         )

#     st.markdown('<h5></h5>', unsafe_allow_html=True)



# # ------------- 메인 화면 구성 --------------------------  

# if selected_sum_model == "bart-large-cnn":
#     st.header('Summary with bart-large-cnn')
#     model = "bart"
# elif selected_sum_model == "flan-t5-3b-summarizer":
#     st.header('Summary with flan-t5-3b-summarizer')
#     model = "flan"
# elif selected_sum_model == "의미론적 방법론":
#     st.header('Summary with 의미론적 방법론')
#     model = "mean"

# input_title = st.text_input("논문의 제목을 입력하세요.")

# upload_file = st.file_uploader("PDF 파일를 업로드하세요.", type="pdf")

# # 컬럼을 사용하여 버튼을 오른쪽에 배치
# col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
# with col3:
#     clicked_sum_model = st.button('PDF 문서 요약')

# with col3:
#     clicked_keyword_model = st.button('키워드 추출')

# if clicked_sum_model:
#     summarize_PDF_file(upload_file, model, input_title)
    
# if clicked_keyword_model:
#     keyword_extraction(input_title)
    
# 선택한 유형에 따른 그래프 표시 

# if selected_chart == "All":
#     single_bar_all()
# elif selected_chart == "Time":
#     single_bar_time()
# elif selected_chart == "Accuracy":
#     single_bar_accuracy()
    
def test():
    title = requests.get(FASTAPI_URL3 + 'getFullText', params={'title': 'bert paper3'})
    context = title.json()
    # print('title : ', context)
    keyword = requests.post(FASTAPI_URL4 + 'Bert_Keyword', json={'text':context['data']})
    print('keyword : ', keyword.json())
    
test()