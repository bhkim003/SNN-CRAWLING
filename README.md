# ⚡ SNN Paper Tracker

> **Spiking Neural Network(SNN)** 논문을 자동 수집·분류해 GitHub Pages로 제공하는 다크 테마 논문 대시보드

🌐 라이브 사이트: [bhkim003.github.io/SNN-CRAWLING](https://bhkim003.github.io/SNN-CRAWLING/)

---

## 한눈에 보기

| 항목 | 내용 |
|---|---|
| 수집 소스 | Semantic Scholar + arXiv |
| 검색 키워드 | `spiking neural network`, `SNN`, `spike-based` |
| 분류 방식 | 제목 + 초록 기반 분류, 새 토픽 자동 생성 |
| UI | 좌측 세로 주제 목록 + 우측 논문 테이블, 다크 테마, 고대비 표 |
| 부가 기능 | 제목 hover 시 abstract 툴팁(큰 글씨) 표시 |
| 자동 갱신 | 매일 오전 09:00 KST (GitHub Actions) |

---

## 처음 설정

이미 완료했다면 건너뛰어도 됩니다.

1. `Settings` → `Pages` → `Source`를 `GitHub Actions`로 변경
2. `Actions` 탭 → `pages build and deployment` → `Run workflow`

수동 실행 화면에서는 `s2_max`, `arxiv_max`를 조절할 수 있습니다.

---

## 로컬 실행

```bash
python3 scripts/build_site.py
```

기본값은 다음과 같습니다.

- Semantic Scholar: 10,000건
- arXiv: 500건

대량 재색인 예시:

```bash
SNN_MAX_S2=100000 SNN_MAX_ARXIV=100000 python3 scripts/build_site.py
```

더 큰 재색인(요청 상한):

```bash
SNN_MAX_S2=500000 SNN_MAX_ARXIV=500000 python3 scripts/build_site.py
```

> API 제한과 응답 속도에 따라 실제 수집량은 더 적을 수 있습니다. Semantic Scholar API Key를 등록하면 안정성이 올라갑니다.

---

## 동작 방식

- Semantic Scholar와 arXiv를 함께 조회합니다.
- 제목과 초록을 함께 읽어서 분류합니다.
- 기존 카테고리에 맞지 않으면 새 주제를 자동 생성하고, 생성 신뢰도가 낮으면 `Etc`로 배치합니다.
- 중복 논문은 제목 기준으로 한 번만 저장합니다.
- 사이트에서는 각 논문의 제목에 마우스를 올리면 abstract 툴팁을 볼 수 있습니다.
- 검색창에서 제목/저자/학회뿐 아니라 주제명으로도 필터링할 수 있습니다.

---

## Semantic Scholar API Key

키 없이도 동작하지만, 무료 키를 등록하면 rate limit이 완화됩니다.

- 키 발급: [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)
- 등록 위치: `Settings` → `Secrets and variables` → `Actions` → `S2_API_KEY`

## OpenAlex polite pool / API Key

OpenAlex는 `mailto`와 `api_key`를 함께 사용하면 더 안정적인 처리(우선순위 라우팅)에 도움이 됩니다.

- 워크플로 기본 `mailto`: `bhkim003@snu.ac.kr`
- 등록 위치: `Settings` → `Secrets and variables` → `Actions` → `OPENALEX_API_KEY`
- 로컬 실행 시:

```bash
SNN_OPENALEX_CONTACT=bhkim003@snu.ac.kr \
SNN_OPENALEX_API_KEY=YOUR_OPENALEX_KEY \
python3 scripts/build_site.py
```

---

## 파일 구조

```text
.github/workflows/update-snn-papers.yml  # pages build + data update 통합 워크플로
scripts/build_site.py                    # 크롤러 + HTML 생성
docs/index.html                          # 생성된 페이지
docs/papers.json                         # 수집 데이터(JSON)
```

---

## 참고

- Google Scholar는 공식 API가 없어 CI 환경에서 안정적으로 쓸 수 없습니다.
- 그래서 Semantic Scholar를 기본 소스로 사용하고, arXiv로 최신 preprint를 보완합니다.
- 새 논문이 기존 주제에 맞지 않으면 `Topic: ...` 형식의 새 카테고리가 생깁니다.
