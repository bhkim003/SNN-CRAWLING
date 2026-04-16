# SNN-CRAWLING

Spiking Neural Network(SNN) 관련 논문을 자동 수집/분류하여 GitHub Pages로 정리하는 저장소입니다.

## 주요 기능

- arXiv API 기반으로 SNN 관련 논문 수집
- 주제별 분류 (필수 포함: **LLM**, **Object Detection**, **Drone**)
- 각 주제 내 최신순 정렬
- 논문별 메타데이터 표기
  - 1저자
  - 제목
  - 저널/학회(가능한 경우 원문 메타데이터 기반, 없으면 arXiv)
  - 링크
- 누락 가능 논문을 위한 **Etc** 카테고리 자동 배치
- GitHub Actions로 매일 오전 9시(KST) 자동 갱신

## 출력물

- `/docs/index.html`: GitHub Pages에서 보여줄 메인 페이지
- `/docs/papers.json`: 수집/분류된 데이터
- 사이트 주소: **https://bhkim003.github.io/SNN-CRAWLING/**

## 로컬 실행

```bash
python3 scripts/build_site.py
```

필요 시 수집량 조정:

```bash
SNN_MAX_RESULTS=500 python3 scripts/build_site.py
```

## GitHub Pages 설정

1. GitHub 저장소 설정(Settings) > Pages 이동
2. Source를 **Deploy from a branch**로 설정
3. Branch를 작업 브랜치(또는 main), 폴더를 `/docs`로 지정

## 자동 갱신 스케줄

- 워크플로 파일: `.github/workflows/update-snn-papers.yml`
- 실행 시각: **매일 00:00 UTC = 오전 09:00 KST**
- 수동 실행: `workflow_dispatch` 지원

## API Key 관련

현재 구현은 공개 arXiv API만 사용하므로 **필수 API key가 없습니다**.
