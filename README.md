# DART 사업보고서 RAG

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Azure](https://img.shields.io/badge/Azure_AI_Search-0078D4?logo=microsoftazure&logoColor=white)
![OpenAI](https://img.shields.io/badge/Azure_OpenAI-412991?logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![DART](https://img.shields.io/badge/DART_OpenAPI-0B6E4F?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyek0xMSAxN0g5VjdoMnYxMHptNCAwaC0yVjdoMnYxMHoiLz48L3N2Zz4=&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Azure 기반 한국 상장사 사업보고서 질의응답 시스템. DART Open API로 수집한 2023년 사업보고서를 청킹 후 Azure AI Search에 인덱싱하고, 하이브리드 검색 + Cross-encoder Reranking + GPT-4o-mini로 답변을 생성한다.

## 아키텍처

```
DART Open API
→ Azure Blob Storage (원문 txt + 청크 JSON)
→ Azure OpenAI text-embedding-3-small (임베딩)
→ Azure AI Search (HNSW 벡터 + ko.microsoft 하이브리드 검색)
→ Cross-encoder Reranking
→ GPT-4o-mini (답변 생성)
→ Streamlit UI
```

## 대상 기업

삼성전자, SK하이닉스, 현대자동차, NAVER, 카카오 (2023년 사업보고서)

## 파일 구조

```
dart-rag/
├── .env
├── .env.example
├── requirements.txt
├── app.py                          # Streamlit UI
└── notebooks/
    ├── 1_collect.ipynb             # DART API → Blob Storage
    ├── 2_chunk.ipynb               # Fixed-size / Section 청킹
    ├── 3_index.ipynb               # 임베딩 + AI Search 인덱싱
    └── 4_query.ipynb               # 검색 + Reranking + 답변
```

## 설치 및 실행

```bash
pip install -r requirements.txt
cp .env.example .env   # .env에 키 입력
streamlit run app.py
```

## 환경변수 (.env)

| 변수명 | 설명 |
|---|---|
| `DART_API_KEY` | DART Open API 인증키 (금융감독원 OpenDART에서 발급) |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage 연결 문자열 (Storage Account → Access keys) |
| `AZURE_STORAGE_CONTAINER_NAME` | 원문 및 청크 JSON을 저장할 컨테이너 이름 |
| `AZURE_SEARCH_ENDPOINT` | Azure AI Search 서비스 엔드포인트 URL |
| `AZURE_SEARCH_API_KEY` | Azure AI Search Admin API 키 |
| `AZURE_SEARCH_INDEX_NAME` | 생성할 인덱스 이름 (예: `dart-index`) |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI 서비스 엔드포인트 URL |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API 키 |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | 임베딩 모델 배포 이름 (예: `text-embedding-3-small`) |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | 채팅 모델 배포 이름 (예: `gpt-4o-mini`) |

## 청킹 전략

사업보고서는 수만 자의 연속 텍스트라 청킹 전략이 검색 품질에 직접 영향을 준다.

fixed-size는 500글자마다 자르는 베이스라인이다. 구현이 단순하고 청크 크기가 일정하지만 문장 중간에 잘릴 수 있다.

section-based는 "사업의 내용", "위험관리" 같은 섹션 헤더 기준으로 분할하고, 섹션이 너무 길면 fixed-size로 2차 분할한다. 섹션명이 메타데이터로 남아 필터링에 활용 가능하고 의미 단위가 보존된다.

두 전략의 답변 품질 차이는 크지 않다. 지금 section 청킹도 긴 섹션은 fixed-size로 2차 분할하는 하이브리드 방식이라 pure fixed와 실질적 차이가 적고, 청킹 방식보다 섹션 필터링이 검색 품질에 더 큰 영향을 주기 때문이다. section 기반을 채택한 이유는 섹션명을 메타데이터로 활용해 질문 유형에 따라 검색 범위를 좁힐 수 있기 때문이다.

재무에 관한 사항 섹션은 숫자 노이즈가 많아 인덱싱에서 제외했다.

## Reranking

Azure AI Search 하이브리드 검색은 키워드 점수와 벡터 점수를 따로 계산한 뒤 합산한다. 쿼리와 청크를 직접 비교하지 않아 관련성 낮은 청크가 상위에 잡히는 경우가 생긴다.

Cross-encoder reranker는 쿼리와 각 청크를 쌍으로 묶어 직접 관련도 점수를 계산한다. 검색 결과 top 20을 뽑은 뒤 reranker로 상위 5개를 추려 GPT에 전달한다.

사용 모델: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Reranking 전후 비교

질문: "삼성전자의 DS 부문 사업 현황은?"

| 순위 | hybrid score | rerank score | 청크 내용 |
|------|-------------|--------------|-----------|
| 1 | 0.0331 | 8.4856 (→ 2위) | 사업의 개요 - 본사 거점, 지역총괄 설명 |
| 2 | 0.0323 | 8.6614 (→ 1위) | DX/DS 부문 구성 설명 (핵심 내용) |
| 3 | 0.0318 | 8.4386 | 가동률 테이블 |
| 4 | 0.0313 | 8.0165 | 매출 및 법인 현황 |
| 5 | 0.0311 | 7.9670 | China Company 법인 목록 → 계약 관련 청크로 교체 |

2위였던 DS 부문 직접 설명 청크가 1위로 올라오고, 노이즈 청크가 탈락했다. 1위와 5위 점수 차이가 8.66 vs 7.97로 명확해져 GPT에 더 관련성 높은 컨텍스트가 전달됐다.

## 한계

연혁 청크 노이즈: 사업의 내용 섹션에 이사 선임, 공장 준공 등 연혁 데이터가 섞여 있다. 연도 패턴 감지로 필터링 가능하지만 미적용 상태다.

섹션 필터링 한계: 키워드 기반 섹션 분류가 질문 의도를 항상 정확히 반영하지 못한다. 의도 분류 모델 도입을 고려할 수 있다.

쿼리 확장 미적용: 보고서에 없는 표현으로 질문하면 검색이 실패한다. GPT로 쿼리를 확장한 뒤 검색하는 방식으로 개선 가능하다.
