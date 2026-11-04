# 위 줄(%%writefile)은 이 셀의 내용을 실행하지 않고 app/model_service.py 라는 파일로 저장하라는 주피터 명령이다
"""
모델 부품 + 전체 조립: 모델 로드, 모델 탐지, 녹취록 마스킹(가명 처리 포함).
"""
# os(운영체제와 대화하는 모듈)를 가져온다 — 환경 변수에서 모델 이름을 읽으려고
import os
# time(시간 모듈)을 가져온다 — 처리 시간을 재려고
import time
# Counter(개수를 세는 사전)를 가져온다 — 유형별 개수와 가명 순번을 세려고
from collections import Counter

# torch를 가져온다 — GPU 사용 가능 여부를 묻기 위해
import torch
# pipeline을 가져온다 — 토크나이저+모델+후처리를 한 번에 묶어 주는 도구
from transformers import pipeline

# 4-A에서 만든 규칙 부품들을 가져온다
from app.detectors import (
    MODEL_PRIORITY,   # 모델 탐지의 우선순위 값 (100)
    find_regex,       # 정규식 탐지
    is_timestamp,     # 타임스탬프 줄 판별
    merge_spans,      # 조각 합치기·겹침 정리
    model_label,      # 모델 영어 라벨 → 한국어 유형
)

# 모델 이름(또는 로컬 경로)을 환경 변수 PII_MODEL에서 읽고, 없으면 기본값을 쓴다
#   나중에 오프라인 실행으로 바꿀 때 코드는 그대로 두고 이 환경 변수만 로컬 경로로 바꾸면 된다
MODEL_NAME = os.environ.get("PII_MODEL", "FrameByFrame/korean-pii-e5-base")
# 한 번에 모델에 넣을 줄 수
BATCH_SIZE = 16


# 모델을 불러와 돌려주는 함수 — 서버가 켜질 때 한 번만 부른다 (요청마다 부르면 매번 수백 MB를 다시 읽는다)
def load_model():
    # GPU가 있으면 0번 GPU를, 없으면 -1(CPU)을 쓰도록 정한다
    device = 0 if torch.cuda.is_available() else -1
    # 토큰 분류 pipeline을 만들어 돌려준다 (Step 2 셀 3에서 검증한 설정 그대로)
    return pipeline(
        "token-classification",          # 태스크: 토큰(단어 조각)마다 라벨을 붙이는 분류
        model=MODEL_NAME,                # 모델 이름 또는 로컬 경로
        device=device,                   # GPU 또는 CPU
        aggregation_strategy="simple",   # 토큰 라벨을 덩어리로 묶는다 (이 모델은 조각으로 나오지만 merge_spans가 마저 합친다)
    )


# 여러 줄을 한 번에 모델에 넣고, 줄마다 탐지 목록을 돌려주는 함수
#   반환 모양: [[줄1의 탐지들], [줄2의 탐지들], ...] — 넣은 줄 순서와 같다
def find_model(ner, lines: list[str], threshold: float) -> list[list[dict]]:
    # 넣을 줄이 하나도 없으면 모델을 부르지 않고 빈 목록을 돌려준다 (빈 목록을 모델에 넣으면 에러가 날 수 있어서)
    if not lines:
        return []
    # 줄 목록을 통째로 넣는다 → 모델이 BATCH_SIZE개씩 묶어 처리한다 (한 줄씩 부르는 것보다 GPU를 덜 놀린다)
    raw_results = ner(lines, batch_size=BATCH_SIZE)
    # 줄마다의 결과를 담을 빈 리스트
    all_spans = []
    # 줄 하나의 결과(raw)를 하나씩 꺼낸다
    for raw in raw_results:
        # 이 줄의 탐지를 담을 빈 리스트
        spans = []
        # 탐지 하나(r)를 하나씩 꺼낸다
        for r in raw:
            # 확신도가 기준보다 낮으면 버린다
            if r["score"] < threshold:
                continue
            # 4-A의 정규식 결과와 똑같은 모양의 사전으로 바꾼다 — 같은 모양이어야 merge_spans에 한데 넣을 수 있다
            spans.append({
                "start": int(r["start"]),            # int() = 모델이 numpy 숫자로 줄 수 있어서 일반 정수로 바꾼다
                "end": int(r["end"]),
                "type": model_label(r["entity_group"]),  # 영어 라벨 → 한국어 유형 (모르는 라벨은 '기타개인정보')
                "source": "model",
                "priority": MODEL_PRIORITY,
            })
        # 이 줄의 결과를 전체 목록에 넣는다
        all_spans.append(spans)
    # 줄 순서대로 정리된 결과를 돌려준다
    return all_spans


# 순번(0, 1, 2...)을 가명 글자(A, B, C...)로 바꾸는 작은 도우미 함수
def _letters(n: int) -> str:
    # 0~25는 A~Z, 26부터는 27, 28처럼 숫자로 (chr = 번호 → 글자, ord("A") = 'A'의 번호 65)
    return chr(ord("A") + n) if n < 26 else str(n + 1)


# 녹취록 전체를 받아 가린 결과와 요약을 돌려주는 함수 — 서버의 /predict가 부를 핵심 함수
def mask_transcript(ner, text: str, threshold: float = 0.5,
                    pseudonymize: bool = True) -> dict:
    # 처리 시작 시각을 잰다 (perf_counter = 짧은 시간 측정에 쓰는 정밀 시계)
    t0 = time.perf_counter()
    # 줄 단위로 나눈다. splitlines() 대신 split("\n")을 쓰는 이유: 마지막 빈 줄까지 정확히 보존해서
    #   나중에 "\n".join()으로 다시 합쳤을 때 원본과 형식이 완전히 같게 하려고
    lines = text.split("\n")

    # 발언 줄의 '줄 번호'와 '타임스탬프'를 따로 모은다
    speech_idx, speech_ts = [], []
    # 바로 앞에서 본 타임스탬프 (처음엔 없음)
    current_ts = ""
    # enumerate = 줄 번호(i)와 줄 내용(line)을 함께 꺼낸다
    for i, line in enumerate(lines):
        # 빈 줄은 건너뛴다 (그대로 남겨 둔다)
        if not line.strip():
            continue
        # 타임스탬프 줄이면 기억만 하고 건너뛴다 (가리지 않고 그대로 남겨 둔다)
        if is_timestamp(line):
            current_ts = line.strip()
            continue
        # 여기까지 왔으면 발언 줄 — 줄 번호와 그 줄의 타임스탬프를 기록한다
        speech_idx.append(i)
        speech_ts.append(current_ts)

    # 발언 줄만 골라 한 번에 모델에 넣는다 — [lines[i] for i in speech_idx] = 발언 줄 번호에 해당하는 줄들의 목록
    model_spans = find_model(ner, [lines[i] for i in speech_idx], threshold)

    # 가명 대응표: (유형, 원문) → 가명. 함수 안에서만 쓰고 응답·로그에 내보내지 않는다 (원본 복원 열쇠라서)
    pseudonyms = {}
    # 유형별로 가명을 몇 개 만들었는지 센다 (다음 가명 글자를 정하려고)
    type_counters = Counter()
    # 유형별 탐지 개수 (응답의 summary)
    summary = Counter()
    # 탐지 목록 (응답의 entities — 원문 글자는 넣지 않는다)
    entities = []

    # k = 발언 줄 중 몇 번째인지, i = 전체 줄 중 몇 번째인지
    for k, i in enumerate(speech_idx):
        # 이번 발언 줄
        line = lines[i]
        # 정규식 결과 + 모델 결과를 합쳐 조각·겹침을 정리한다
        spans = merge_spans(line, find_regex(line) + model_spans[k])

        # 탐지 하나씩: 가명(tag)을 정하고, 요약·목록에 기록한다
        for sp in spans:
            # 가릴 원문 조각 (대응표의 열쇠로만 쓰고 밖으로 내보내지 않는다)
            original = line[sp["start"]:sp["end"]]
            # 가명 처리를 켰으면
            if pseudonymize:
                # 같은 유형 + 같은 글자면 같은 사람으로 보고 같은 가명을 쓴다
                key = (sp["type"], original)
                # 처음 나온 것이면 새 가명을 만든다
                if key not in pseudonyms:
                    # 예: 이름 유형의 첫 번째 → [이름_A], 두 번째 → [이름_B]
                    pseudonyms[key] = f"[{sp['type']}_{_letters(type_counters[sp['type']])}]"
                    # 이 유형의 가명 순번을 하나 올린다
                    type_counters[sp["type"]] += 1
                # 이 탐지에 붙일 가명
                sp["tag"] = pseudonyms[key]
            # 가명 처리를 껐으면 유형 이름만으로 가린다
            else:
                sp["tag"] = f"[{sp['type']}]"
            # 유형별 개수를 하나 올린다
            summary[sp["type"]] += 1
            # 목록에 유형·타임스탬프·출처만 기록한다 (원문 글자는 넣지 않음 → 응답 전체에 개인정보가 없다)
            entities.append({"type": sp["type"], "timestamp": speech_ts[k], "source": sp["source"]})

        # 뒤에서부터(start가 큰 것부터) 바꿔 끼운다
        #   앞에서부터 바꾸면 '김서연'(3글자) → '[이름_D]'(6글자)처럼 길이가 달라져,
        #   그 뒤에 있는 탐지들의 start·end 위치가 전부 밀려 엉뚱한 곳을 가리키게 되기 때문
        for sp in sorted(spans, key=lambda s: s["start"], reverse=True):
            # 앞부분 + 가명 + 뒷부분으로 줄을 다시 만든다 (문자열은 직접 고칠 수 없어 새로 만들어 끼운다)
            line = line[:sp["start"]] + sp["tag"] + line[sp["end"]:]
        # 가린 줄을 원래 자리에 넣는다
        lines[i] = line

    # 결과를 사전으로 돌려준다 (schemas.py의 MaskResponse에서 success만 빠진 모양)
    return {
        "masked_text": "\n".join(lines),                        # 줄들을 다시 하나로 — 타임스탬프·빈 줄도 원래 자리 그대로
        "summary": dict(summary),                               # Counter → 일반 사전 (JSON으로 바꿀 수 있게)
        "entities": entities,
        "processing_ms": round((time.perf_counter() - t0) * 1000, 1),  # 걸린 초 × 1000 = 밀리초, 소수 첫째 자리까지
    }
