# 과제 안내

## 과제 내용

> 멘토가 과제마다 이 칸에 추가합니다.

## 제출 흐름 한눈에 보기

1. 내 브랜치를 만든다
2. HW 또는 SW 폴더에 코드를 올린다
3. PR을 열고 멘토에게 리뷰를 요청한다
4. 리뷰를 반영한다
5. 승인(Approve)을 받으면 merge한다

**main에는 직접 올리지 않습니다.** 모든 코드는 PR과 리뷰를 거쳐 main에 들어갑니다.

## 처음 보는 용어

- **저장소(repository)**: 프로젝트 파일을 모아 두는 곳. 지금 보고 있는 SentiFrame 페이지
- **main**: 저장소의 기준 버전. 리뷰를 통과한 코드만 들어간다
- **브랜치(branch)**: main을 복사해서 만든 내 작업 공간. 여기서 무엇을 바꿔도 main에는 영향이 없다
- **커밋(commit)**: 파일 변경을 저장하는 단위. 무엇을 바꿨는지 한 줄 메시지를 함께 남긴다
- **PR(Pull Request)**: "내 브랜치의 변경을 main에 합쳐 주세요"라는 요청. 멘토가 여기서 코드를 읽고 의견을 남긴다
- **리뷰(review)·승인(Approve)**: 멘토가 PR을 읽고 수정 요청이나 승인을 남기는 것
- **merge**: 승인된 PR의 변경을 main에 합치는 것

---

## 1. 내 브랜치 만들기

작업마다 새 브랜치를 하나 만든다.

1. 저장소 첫 화면 왼쪽 위에서 `main`이라고 적힌 **브랜치 버튼**을 누른다
2. 입력칸에 브랜치 이름을 적는다
   - 형식: hw/[수행내용], sw/[수행내용]
3. 아래에 나오는 **Create branch: [branch_name] from main**을 누른다
4. 브랜치 버튼에 내 브랜치 이름이 보이면 성공. **이후 작업은 항상 이 상태에서 한다**

## 2. 코드 올리기

1. 저장소에서 **브랜치 버튼이 내 브랜치인지 다시 확인**하고, `HW` 또는 `SW` 폴더를 누른다
2. 오른쪽 위 **Add file** → **Upload files**
3. 올릴 파일을 화면에 끌어다 놓는다. 여러 개를 한 번에 놓아도 되고, 폴더째 놓으면 폴더 구조 그대로 올라간다 (크롬이나 엣지 권장)
4. 아래 **Commit changes** 칸에 무엇을 올렸는지 한 줄 적는다. 예: `센서 데이터 불러오기 코드 추가`
5. **Commit directly to the `[branch_name]` branch**가 선택돼 있는지 확인하고 **Commit changes**를 누른다

### 올리지 말 것

- **데이터 파일**(원본 센서 데이터, 큰 csv 등)과 **25MB가 넘는 파일**: 저장소가 무거워지고, 웹 업로드는 25MB까지만 된다
- **실행하면 생기는 파일**: `__pycache__`, `.DS_Store`, 학습된 모델 파일
- **비밀번호, API 키**

## 3. PR 열고 리뷰 요청하기

1. 커밋하고 나면 화면 위에 노란 줄로 **Compare & pull request**가 보인다. 누른다
   - 안 보이면: **Pull requests** 탭 → **New pull request** → `base: main`, `compare: 내 브랜치` 선택
2. 제목은 `[SW] - {수행 내용}` 형식으로 쓴다
3. 본문에 세 가지를 적는다
   - 무엇을 했나
   - 어떻게 확인했나 (실행 결과, 출력된 숫자)
   - 궁금한 점
4. 오른쪽 **Reviewers** 옆 톱니바퀴를 눌러 `shjoo0407`을 고른다 (이미 들어가 있으면 그대로 둔다)
5. **Create pull request**를 누른다

## 4. 리뷰 반영하기

- **고친 파일은 같은 브랜치에 다시 올린다.** 2번과 같은 방법으로, 같은 위치에 같은 이름으로 올리면 덮어쓴다. 파일 하나를 조금만 고칠 때는 파일을 열고 연필 아이콘(Edit)으로 바로 고쳐도 된다
- **PR은 새로 만들지 않는다.** 같은 브랜치에 커밋하면 열려 있는 PR에 자동으로 반영된다
- 코멘트마다 무엇을 어떻게 고쳤는지 답글을 달고, 해결됐으면 **Resolve conversation**을 누른다
- 다 고쳤으면 PR 대화창에 "리뷰 반영했습니다"라고 남긴다

## 5. 승인과 merge
1. 멘토가 **Approve**하면 PR 아래 **Merge pull request** → **Confirm merge**를 누른다. **승인 전에는 merge하지 않는다**
2. merge가 끝나면 **Delete branch**를 눌러 다 쓴 브랜치를 지운다
---

## 자주 막히는 곳
- **저장소가 404로 보인다**: 초대를 아직 수락하지 않았거나, 다른 계정으로 로그인한 상태다
- **파일이 main에 올라갔다**: 브랜치 버튼을 확인하지 않고 올린 것이다. 바로 멘토에게 알린다
- **끌어다 놓기가 안 된다**: Upload files 화면의 **choose your files**를 눌러 파일을 직접 고른다
- **Upload files나 Commit 버튼이 없다**: 쓰기 권한이 없는 상태다. 멘토에게 문의한다

## (선택) 터미널로 올리기

git 명령에 익숙해지면 이 방법이 더 빠르다. 웹 방법과 섞어 써도 된다.

```bash
# 처음 한 번
#   macOS: brew install git gh
#   Windows: git-scm.com 에서 Git, cli.github.com 에서 GitHub CLI 설치
gh auth login            # GitHub.com → HTTPS → Yes → Login with a web browser
git clone https://github.com/bing58qiz-hash/SentiFrame.git
cd SentiFrame

# 작업마다 (예: SW 멘티가 센서 데이터 불러오기 코드를 올릴 때)
git switch main && git pull           # 최신 main 받기
git switch -c sw/data-loader          # 내 브랜치 만들기: sw/[수행내용]
#   SW 폴더에 파일을 넣은 뒤
git add SW
git commit -m "센서 데이터 불러오기 코드 추가"
git push -u origin sw/data-loader
gh pr create --base main --reviewer shjoo0407 --fill   # PR 열기 (웹에서 열어도 된다)
```

리뷰를 반영할 때는 같은 브랜치에서 고친 뒤 `git add` → `git commit` → `git push`를 하면 PR에 반영된다.
