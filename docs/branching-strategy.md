# Branching Strategy

This document defines the branching model for this project.  
The goal is to maintain clean and stable development workflows while supporting
modular feature development.

---

# 1. Overview of Branch Structure

main
dev
feature/data
feature/model
feature/app


### main  
- Stable production-ready branch.  
- All completed features are merged here.  
- This branch should always remain clean and functional.  
- Tagged versions (v0.1, v0.2 ...) originate from here.

### dev  
- Integration branch for ongoing development.  
- All feature branches merge into this branch before going to `main`.  
- May contain in-progress code but should stay stable enough for daily work.

### feature/data  
- Data-related development: preprocessing, dataset loaders, augmentation, collators, etc.  
- Merges into `dev` after completion.

### feature/model  
- Model development branch:  
  - LLM fine-tuning  
  - Whisper / WavLM / TTS  
  - YOLO / CLIP / Diffusion  
  - LoRA experimentation  
  - Training pipelines  
- All training-related code is developed here, then merged into `dev`.

### feature/app  
- Application-level development:  
  - Streamlit / Gradio UI  
  - FastAPI backend  
  - CLI tools  
  - inference pipelines  
- Responsible for connecting data + model into a usable application.

---

# 2. Git Workflow Summary (How to Use These Branches)

이 섹션은 실제로 **어떤 브랜치에서 어떻게 작업하고, 어떤 순서로 merge/push 하는지**를 정리한다.

## 2.1 dev 브랜치에서 파일 작성 -> main에 반영하는 절차 (docs 예시)

### 1. dev에서 문서/코드 수정

```python
# dev로 이동
git checkout dev

# 작업 후 변경 사항 확인
git status

# 변경 파일 스테이징
git add docs/branching-strategy.md   # 혹은 git add docs/ 또는 git add .

# 커밋
git commit -m "Update: branching strategy documentation"

# 원격 dev에 푸시
git push origin dev
```

### 2. dev에서 문제가 없다고 판단되면 main으로 병합

```python
# main으로 이동
git checkout main

# 원격 main 최신 상태 반영
git pull origin main

# dev 내용 병합
git merge dev

# 병합 결과 푸시
git push origin main
```

이 흐름이 **“dev에서 작성 -> 검증 -> main에 반영”**의 기본 패턴이다.

## 2.2 새 기능 개발 (feature 브랜치 사용)

### 1. feature 브랜치 생성 및 작업

```python
# 항상 dev에서 새 feature 브랜치를 만든다
git checkout dev
git pull origin dev   # dev를 최신으로 맞추기

# 예: 모델 관련 작업
git checkout -b feature/model

# 코드 수정 후 상태 확인
git status

# 변경 사항 스테이징
git add src/ configs/   # 필요에 따라 경로 지정

# 커밋
git commit -m "Add: base training pipeline for Qwen model"

# 원격 feature 브랜치 푸시
git push -u origin feature/model
```

### 2. feature 브랜치를 dev에 병합
기능이 어느 정도 안정되면:

```python
# dev로 이동
git checkout dev
git pull origin dev

# feature 브랜치 병합
git merge feature/model

# dev 푸시
git push origin dev
```

그 다음 위 2.1 절차를 따라 dev -> main 병합을 진행한다.

## 2.3 dev의 docs만 main으로 따로 가져오고 싶을 때

코드 변경은 아직 main에 넣고 싶지 않고,
docs 폴더만 main에 반영하고 싶을 때 사용하는 패턴:

```python
# main으로 이동 후 최신 상태 반영
git checkout main
git pull origin main

# dev에서 docs 폴더만 가져오기
git checkout dev -- docs/

# 스테이징 및 커밋
git add docs
git commit -m "Merge docs folder from dev into main"

# 푸시
git push origin main
```

이 방법은 dev 전체를 merge하지 않고, docs만 선택적으로 main에 반영할 때 사용한다.

## 2.4 이미 존재하는 feature 브랜치에서 이어서 작업할 때

```python
# 이어서 작업할 feature 브랜치로 이동
git checkout feature/model

# 필요하면 dev의 최신 변경 사항을 feature에 가져옴
git checkout dev
git pull origin dev
git checkout feature/model
git merge dev   # 또는 git rebase dev

# 이후 평소처럼 작업
git status
git add .
git commit -m "Refactor: training loop for stability"
git push
```

# 3. Rules

### 1. Never commit directly to main
모든 변경 사항은 반드시 dev 또는 feature/*에서 작업한 뒤
dev -> main 흐름으로 반영한다.

### 2. dev remains stable
dev는 통합 브랜치지만, 가능한 한 “깨지지 않는 상태”를 유지한다.
크게 망가질 수 있는 작업은 feature/*에서 먼저 진행한다.

### 3. Small, atomic commits
커밋은 작고 의미 있게 나눈다. 예:

Add: dataset collator

Fix: WavLM padding issue

Refactor: inference pipeline

Update: branching strategy docs

### 4. Delete merged branches
기능이 dev/main에 모두 반영되면 해당 feature/* 브랜치는 삭제해서
리포지토리를 깔끔하게 유지한다.

```python
# 로컬 브랜치 삭제
git branch -d feature/model

# 원격 브랜치 삭제
git push origin --delete feature/model
```

# 4. Summary
This branching model offers:

- Clean separation of concerns (data / model / app)
- A stable main and an integration-focused dev
- Clear workflows for:

1. 작업용 브랜치 선택
2. dev에서 작업 후 main에 반영하는 절차
3. docs만 선택적으로 main으로 가져오는 절차

- Scalable structure that works well for solo development and can be extended to a team later

All branches should exist for a clear purpose and be merged or deleted when their lifecycle ends.