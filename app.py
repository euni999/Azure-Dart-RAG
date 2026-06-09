import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from sentence_transformers import CrossEncoder

load_dotenv()

SEARCH_ENDPOINT = os.getenv('AZURE_SEARCH_ENDPOINT')
SEARCH_API_KEY = os.getenv('AZURE_SEARCH_API_KEY')
SEARCH_INDEX_NAME = os.getenv('AZURE_SEARCH_INDEX_NAME')
OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
EMBEDDING_DEPLOYMENT = os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')
CHAT_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')

openai_client = OpenAI(base_url=OPENAI_ENDPOINT + '/openai/v1', api_key=OPENAI_API_KEY)
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

COMPANIES = ['삼성전자', 'SK하이닉스', '현대자동차', 'NAVER', '카카오']

@st.cache_resource
def load_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def hybrid_search(query: str, company: str, top_k: int = 20) -> list[dict]:
    query_embedding = openai_client.embeddings.create(
        input=query,
        model=EMBEDDING_DEPLOYMENT
    ).data[0].embedding

    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=top_k,
        fields='embedding'
    )

    # 키워드 기반 섹션 라우팅: 리스크 관련 질문은 위험관리 섹션으로
    if any(k in query for k in ['리스크', '위험', '위협', '불확실']):
        section_filter = "section eq '위험관리'"
    else:
        section_filter = "section eq '사업의 내용'"

    filter_expr = f"company eq '{company}' and {section_filter}"

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        filter=filter_expr,
        select=['id', 'company', 'section', 'text', 'chunk_id'],
        top=top_k
    )

    return [
        {
            'id': r['id'],
            'company': r['company'],
            'section': r.get('section', ''),
            'text': r['text'],
            'score': r['@search.score'],
        }
        for r in results
    ]


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    reranker = load_reranker()
    pairs = [(query, c['text']) for c in chunks]
    scores = reranker.predict(pairs)
    for i, c in enumerate(chunks):
        c['rerank_score'] = float(scores[i])
    return sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)[:top_k]


def rag_answer(query: str, company: str) -> tuple[str, list[dict]]:
    chunks = hybrid_search(query, company=company, top_k=20)

    if not chunks:
        return '관련 정보를 찾을 수 없습니다.', []

    chunks = rerank(query, chunks, top_k=5)

    context = '\n\n'.join([
        f"[{c['company']} - {c['section']}]\n{c['text']}"
        for c in chunks
    ])

    system_prompt = """당신은 한국 기업의 사업보고서를 분석하는 전문가입니다.
주어진 컨텍스트를 바탕으로 질문에 정확하고 간결하게 답변하세요.
컨텍스트에 없는 내용은 '보고서에서 확인되지 않습니다'라고 답하세요."""

    response = openai_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f'컨텍스트:\n{context}\n\n질문: {query}'},
        ],
        temperature=0,
    )

    return response.choices[0].message.content, chunks


# UI
st.set_page_config(page_title='DART 사업보고서 RAG', page_icon='📊', layout='wide')
st.title('📊 DART 사업보고서 RAG')
st.caption('2023년 사업보고서 기반 질의응답')

col1, col2 = st.columns([1, 2])

with col1:
    company = st.selectbox('기업 선택', COMPANIES)
    query = st.text_area('질문 입력', height=120, placeholder='예) 주요 사업 리스크가 뭐야?')
    submit = st.button('검색', use_container_width=True, type='primary')

with col2:
    if submit and query:
        with st.spinner('검색 중...'):
            answer, chunks = rag_answer(query, company=company)

        st.subheader('답변')
        st.write(answer)

        st.divider()
        st.subheader('출처')
        for i, chunk in enumerate(chunks, 1):
            with st.expander(f'{i}. [{chunk["section"]}] rerank_score={chunk["rerank_score"]:.4f}'):
                st.text(chunk['text'])

    elif submit and not query:
        st.warning('질문을 입력해주세요.')