# ⚡ SNN Paper Tracker

**Spiking Neural Network(SNN)** 관련 논문을 자동 수집·분류하여 GitHub Pages로 정리하는 저장소입니다.

🌐 **사이트**: [bhkim003.github.io/SNN-CRAWLING](https://bhkim003.github.io/SNN-CRAWLING/)

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **다중 소스 크롤링** | Semantic Scholar (IEEE, ACM, NeurIPS, Nature 등 포함) + arXiv 병행 수집 |
| **주제별 분류** | LLM, Object Detection, Drone 포함 16개 카테고리 자동 분류 |
| **최신순 정렬** | 각 카테고리 내 발행일 기준 내림차순 |
| **논문 메타데이터** | 1저자 · 제목(링크) · 저널/학회 · 날짜 |
| **전문 검색** | 제목/저자/학회로 실시간 필터 |
| **Etc 카테고리** | 위 분류에 해당하지 않는 논문 자동 수집 |
| **자동 갱신** | GitHub Actions로 **매일 오전 9시(KST)** 실행 |

## 카테고리

LLM · Object Detection · Drone · Event-based Vision · Neuromorphic Hardware ·
ANN-to-SNN Conversion · Learning Rules · Reinforcement Learning · Robotics & Control ·
Segmentation · Medical & BCI · Speech & Audio · Image Classification · NLP ·
Time Series · **Etc**

---

## 크롤링 소스

Google Scholar는 공식 API가 없고 CI 환경에서 즉시 차단됩니다.
대신 **Semantic Scholar**(무료 공개 API)를 주 소스로 사용합니다.
Semantic Scholar는 Google Scholar와 동일한 논문 데이터베이스를 커버하며
arXiv, IEEE Xplore, ACM DL, Nature, PubMed 등을 포함합니다.
arXiv는 보조 소스로 최신 preprint를 보완합니다.

---

## GitHub Pages 설정 (최초 1회 필요)

워크플로가 GitHub Actions 방식으로 배포하므로 Pages 소스를 바꿔야 합니다.

1. 저장소 **Settings → Pages** 이동
2. **Source** 를 `GitHub Actions` 로 변경 (기존 "Deploy from a branch"에서 변경)
3. Actions 탭에서 `Update SNN Papers` 워크플로를 수동 실행 (`Run workflow`)

---

## 로컬 실행

```bash
python3 scripts/build_site.py
```

수집량 조정 (기본값: Semantic Scholar 10,000 / arXiv 500):

```bash
SNN_MAX_S2=2000 SNN_MAX_ARXIV=200 python3 scripts/build_site.py
```

Semantic Scholar 무료 API key 사용 (요청 속도 향상):

```bash
S2_API_KEY=your_key python3 scripts/build_site.py
```

---

## API Key (선택 사항)

| 서비스 | 필요 여부 | 비고 |
|--------|----------|------|
| Semantic Scholar | 선택 | 무료 키 발급 시 rate limit 향상 |
| arXiv | 불필요 | 공개 API |

Semantic Scholar API key를 사용하려면 저장소 **Settings → Secrets and variables → Actions → `S2_API_KEY`** 에 등록하세요. 워크플로에서 자동으로 사용됩니다.

---

## 자동 갱신 스케줄

- 워크플로: [`.github/workflows/update-snn-papers.yml`](.github/workflows/update-snn-papers.yml)
- 실행 시각: **매일 00:00 UTC = 오전 09:00 KST**
- 수동 실행: Actions 탭 → `Run workflow`

## 출력물

- `docs/index.html` — GitHub Pages 메인 페이지
- `docs/papers.json` — 수집·분류된 데이터 (JSON)
