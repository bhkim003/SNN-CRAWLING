# ⚡ SNN Paper Tracker

> **Spiking Neural Network(SNN)** 논문을 자동 수집·분류해 GitHub Pages로 제공하는 다크 테마 논문 대시보드

🌐 라이브 사이트: [bhkim003.github.io/SNN-CRAWLING](https://bhkim003.github.io/SNN-CRAWLING/)

---

## 한눈에 보기

| 항목 | 내용 |
|---|---|
| 수집 소스 | OpenAlex + arXiv |
| 검색 키워드 | `spiking neural network`, `SNN`, `spike-based` |
| 분류 방식 | 제목 + 초록 기반 분류, 새 토픽 자동 생성 |
| UI | 좌측 세로 주제 목록 + 우측 논문 테이블, 다크 테마, 고대비 표 |
| 부가 기능 | 제목 hover abstract 툴팁, 200개 단위 페이지네이션, arXiv 제외 토글 |
| 자동 갱신 | 매일 오전 09:00 KST (GitHub Actions) |

---

## 처음 설정

이미 완료했다면 건너뛰어도 됩니다.

1. `Settings` → `Pages` → `Source`를 `GitHub Actions`로 변경
2. `Actions` 탭 → `pages build and deployment` → `Run workflow`

이 워크플로는 `daily` 갱신 + GitHub Pages 배포만 수행합니다.
초기 대량 크롤링은 로컬에서 1회 실행한 뒤 결과(`docs/papers.json`, `docs/index.html`)를 반영하세요.
크롤링 범위는 2017년 이후 논문만 대상으로 하고, 초기 전체 수집 상한은 OpenAlex/arXiv 각각 30,000건입니다.

---

## 로컬 실행

```bash
python3 scripts/build_site.py
```

기본값은 다음과 같습니다.

- OpenAlex: 30,000건
- arXiv: 30,000건

OpenAlex 수집은 다음 순서로 진행됩니다.

- 우선 지정 저널/학회(NeurIPS, ICLR, ICCV, JMLR, ICML, CVPR, ISSCC, VLSI, CICC, TCAS-I/II, JSSC 등)에서 먼저 수집
- 이후 일반 OpenAlex 검색으로 부족분 보충
- 최종 수집량은 `SNN_MAX_OPENALEX` 상한(예: 30000) 안에서 맞춤

대량 색인 예시:

```bash
SNN_MAX_OPENALEX=30000 SNN_MAX_ARXIV=30000 python3 scripts/build_site.py
```

필요하면 환경변수로 더 작은 범위로 조절할 수 있습니다.

```bash
SNN_MAX_OPENALEX=15000 SNN_MAX_ARXIV=15000 python3 scripts/build_site.py
```

일간(daily) 실행 정책:

- 최근 30일 내 신규 논문만 수집
- OpenAlex 최대 1,000건
- arXiv 최대 1,000건
- 2017년 1월 1일 이전 논문은 수집하지 않음

> API 제한과 응답 속도에 따라 실제 수집량은 더 적을 수 있습니다. OpenAlex API Key를 등록하면 안정성이 올라갑니다.

---

## 동작 방식

- OpenAlex와 arXiv를 함께 조회합니다.
- OpenAlex는 우선 저널/학회 목록을 먼저 조회하고, 그다음 일반 검색으로 채웁니다.
- 제목과 초록을 함께 읽어서 분류합니다.
- 기존 카테고리에 맞지 않으면 새 주제를 자동 생성하고, 생성 신뢰도가 낮으면 `Etc`로 배치합니다.
- 중복 논문은 제목 기준으로 한 번만 저장합니다.
- 시간 가중 인용수 정책을 적용합니다.
- 최근 6개월: 인용수 제한 없음
- 6~24개월: 인용수 1회 이상
- 24개월 초과: 인용수 3회 이상
- 사이트에서는 각 논문의 제목에 마우스를 올리면 abstract 툴팁을 볼 수 있습니다.
- 논문 목록은 200개 단위 페이지네이션(`Prev / 1 2 3 ... / Next`)으로 표시합니다.
- `Exclude arXiv papers` 토글로 arXiv 소스 논문을 숨길 수 있습니다.
- 우선 저널/학회에서 수집된 논문은 표에서 하이라이트됩니다.
- 논문 테이블은 `Published Date`가 `1st Author`보다 왼쪽 열에 고정됩니다.
- 검색창에서 제목/저자/학회뿐 아니라 주제명으로도 필터링할 수 있습니다.
- 페이지 상단 `Updated` 시각은 KST로 표기됩니다.

---

## OpenAlex mailto / API Key

OpenAlex는 `mailto`와 `api_key`를 함께 쓰면 우선순위 라우팅(politeness pool)에 유리해서, 대량 수집 시 더 안정적으로 동작합니다.

- 등록 위치: `Settings` → `Secrets and variables` → `Actions`
- Secrets: `OPENALEX_API_KEY`
- Variables: `OPENALEX_CONTACT` (예: `your-email@example.com`)

로컬 실행 예시:

```bash
SNN_OPENALEX_CONTACT=your-email@example.com \
OPENALEX_API_KEY=YOUR_OPENALEX_KEY \
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
- 그래서 OpenAlex를 기본 소스로 사용하고, arXiv로 최신 preprint를 보완합니다.
- 새 논문이 기존 주제에 맞지 않으면 `Topic: ...` 형식의 새 카테고리가 생깁니다.
