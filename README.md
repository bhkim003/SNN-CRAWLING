# ⚡ SNN Paper Tracker

> **Spiking Neural Network(SNN)** 관련 논문을 자동 수집·분류하여 GitHub Pages로 제공하는 저장소

🌐 **라이브 사이트** → [bhkim003.github.io/SNN-CRAWLING](https://bhkim003.github.io/SNN-CRAWLING/)

---

## 처음 설정 (딱 두 단계)

> 이미 완료됐다면 이 섹션은 건너뛰어도 됩니다.

**Step 1 — GitHub Pages 소스 변경**

`Settings` → `Pages` → **Source** 를 `GitHub Actions` 으로 변경

> 기본값이 "Deploy from a branch"로 되어 있으면 워크플로 배포가 안 됩니다.

**Step 2 — 첫 실행**

`Actions` 탭 → `Update SNN Papers` → **Run workflow** 클릭

> 약 5~10분 후 사이트가 갱신됩니다. 이후엔 매일 오전 9시(KST)에 자동 실행됩니다.

---

## 기능

| | |
|---|---|
| **크롤링 소스** | Semantic Scholar (IEEE Xplore, ACM DL, NeurIPS/ICLR/ICML, Nature, PubMed 등) + arXiv |
| **카테고리 수** | 16개 — LLM, Object Detection, Drone 포함, 미분류는 Etc |
| **정렬** | 각 카테고리 내 최신순 |
| **논문 정보** | 1저자 · 제목(링크) · 저널/학회 · 날짜 |
| **검색** | 제목·저자·학회 실시간 필터 |
| **자동 갱신** | 매일 오전 09:00 KST (GitHub Actions) |

### 카테고리

`LLM` `Object Detection` `Drone` `Event-based Vision` `Neuromorphic Hardware`
`ANN-to-SNN Conversion` `Learning Rules` `Reinforcement Learning` `Robotics & Control`
`Segmentation` `Medical & BCI` `Speech & Audio` `Image Classification` `NLP`
`Time Series` `Etc`

---

## 크롤링 소스에 대해

Google Scholar는 공식 API가 없고 CI 환경에서 즉시 IP가 차단됩니다.
**Semantic Scholar**는 Google Scholar와 동일한 데이터베이스를 커버하는 무료 공개 API입니다.
arXiv는 Semantic Scholar가 미처 인덱싱하기 전의 최신 preprint를 보완합니다.

---

## 로컬 실행

```bash
python3 scripts/build_site.py
```

수집량 조정 (기본: Semantic Scholar 10,000건 / arXiv 500건):

```bash
SNN_MAX_S2=3000 SNN_MAX_ARXIV=200 python3 scripts/build_site.py
```

---

## Semantic Scholar API Key (선택)

키 없이도 동작하지만, 무료 키를 등록하면 rate limit이 올라가 수집 속도가 빨라집니다.

- 키 발급: [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)
- 등록 위치: `Settings` → `Secrets and variables` → `Actions` → New secret → **`S2_API_KEY`**

---

## 파일 구조

```
.github/workflows/update-snn-papers.yml  # 스케줄 워크플로
scripts/build_site.py                    # 크롤러 + HTML 생성
docs/index.html                          # 생성된 페이지
docs/papers.json                         # 수집 데이터 (JSON)
```
