# 위 줄(%%writefile)은 이 셀의 내용을 실행하지 않고 frontend/app.py 라는 파일로 저장하라는 주피터 명령이다
"""
녹취록 개인정보 마스킹 — Streamlit 화면.
실행: streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1 --browser.gatherUsageStats false
"""
# hashlib(해시 도구)을 가져온다 — 입력+설정의 '지문'을 만들어, 결과가 지금 입력의 것인지 확인하려고
import hashlib
# os(운영체제와 대화하는 모듈)를 가져온다 — 서버 주소를 환경 변수에서 읽으려고
import os

# requests(HTTP 요청 도구)를 가져온다 — FastAPI 서버로 요청을 보내려고
import requests
# streamlit을 st라는 짧은 이름으로 가져온다
import streamlit as st

# 서버 주소. 환경 변수 PII_API_BASE가 있으면 그 값, 없으면 이 PC의 8000번 (localhost 대신 127.0.0.1 — 2초 지연 방지)
API_BASE = os.environ.get("PII_API_BASE", "http://127.0.0.1:8000")
# 서버 응답을 기다릴 최대 시간(초)
TIMEOUT = 60

# 상태 코드 → 사람이 읽을 안내 문구
ERROR_GUIDE = {
    400: "파일 형식을 확인해 주세요.",
    401: "API Key를 확인해 주세요.",
    413: "파일이 너무 큽니다.",
    422: "입력 내용을 확인해 주세요.",
    500: "서버 처리 중 오류가 났습니다. 잠시 후 다시 시도해 주세요.",
    503: "모델이 아직 준비되지 않았습니다. 잠시 후 다시 시도해 주세요.",
}

# 브라우저 탭 제목·아이콘, 화면을 넓게 쓰기(원문·결과를 좌우로 나란히 보려고)
st.set_page_config(page_title="녹취록 개인정보 마스킹", page_icon="🔒", layout="wide")


# 서버 상태를 물어보는 함수 — 재실행될 때마다 한 번씩 불린다
def check_health():
    # 서버가 꺼져 있어도 화면이 죽지 않게 감싼다
    try:
        # /health에 GET 요청 (5초 안에 답이 없으면 포기)
        r = requests.get(f"{API_BASE}/health", timeout=5)
        # 응답 JSON에서 status 값을 꺼낸다 (없으면 'unknown')
        return r.json().get("status", "unknown")
    # 연결 실패·시간 초과 등 requests의 모든 에러
    except requests.exceptions.RequestException:
        # 서버가 꺼져 있다고 표시
        return "offline"


# 서버에 마스킹을 요청하고, 성공하면 결과 사전을, 실패하면 화면에 안내를 띄우고 None을 돌려주는 함수
def call_api(text, file_name, file_bytes, api_key, threshold, pseudonymize):
    # 키를 입력했으면 X-API-Key 헤더에 담고, 아니면 빈 헤더 (서버가 401을 돌려주게)
    headers = {"X-API-Key": api_key} if api_key else {}
    # 에러를 종류별로 잡기 위해 감싼다
    try:
        # 파일을 올렸으면 /predict/file로
        if file_bytes is not None:
            r = requests.post(
                f"{API_BASE}/predict/file",
                headers=headers,
                # 파일 업로드 형식: (파일 이름, 바이트, 종류)
                files={"file": (file_name, file_bytes, "text/plain")},
                # 옵션은 Form 칸으로 — Form은 글자로만 보낼 수 있어 str()로 바꾼다. True → "true"
                data={"threshold": str(threshold), "pseudonymize": str(pseudonymize).lower()},
                timeout=TIMEOUT,
            )
        # 직접 붙여 넣었으면 /predict로 JSON 요청
        else:
            r = requests.post(
                f"{API_BASE}/predict",
                headers=headers,
                json={"text": text, "threshold": threshold, "pseudonymize": pseudonymize},
                timeout=TIMEOUT,
            )
        # 4xx·5xx면 HTTPError를 일으킨다 — 이 줄이 없으면 requests는 401·422도 성공처럼 넘긴다
        r.raise_for_status()
        # 성공이면 응답 JSON을 사전으로 돌려준다
        return r.json()
    # 서버가 꺼져 있을 때 (except는 위에서부터 첫 일치 하나만 실행 → 구체적인 것부터)
    except requests.exceptions.ConnectionError:
        st.error(f"서버에 연결할 수 없습니다. FastAPI 서버({API_BASE})가 켜져 있는지 확인해 주세요.")
    # 응답이 너무 늦을 때
    except requests.exceptions.Timeout:
        st.error(f"{TIMEOUT}초 안에 응답이 없습니다.")
    # 서버가 4xx·5xx로 답했을 때
    except requests.exceptions.HTTPError:
        # 상태 코드
        code = r.status_code
        # 서버가 보낸 이유(detail)를 꺼내 본다
        try:
            detail = r.json().get("detail")
        # 응답이 JSON이 아니면
        except ValueError:
            detail = None
        # 422처럼 detail이 목록이면 각 항목의 msg만 이어 붙인다
        if isinstance(detail, list):
            detail = " / ".join(d.get("msg", "") for d in detail)
        # 상태 코드 + 안내 문구 + 서버가 보낸 이유를 빨간 상자로 보여 준다 (끝의 공백 두 칸 + \n = 줄바꿈)
        st.error(f"[{code}] {ERROR_GUIDE.get(code, '요청이 실패했습니다.')}  \n{detail or ''}")
    # 실패한 경우 모두 None
    return None


# ── 사이드바 ─────────────────────────────────
# with st.sidebar: 이 블록 안에서 그리는 것은 왼쪽 사이드바에 그려진다
with st.sidebar:
    # 사이드바 제목
    st.header("설정")
    # 서버 상태를 물어본다
    status = check_health()
    # 상태 글자 → 화면 표시
    badge = {"healthy": "🟢 정상", "loading": "🟡 모델 로딩 중", "offline": "🔴 서버 꺼짐"}
    # 표에 없는 상태(예: error)는 빨간 동그라미 + 상태 글자
    st.write(f"서버 상태: {badge.get(status, '🔴 ' + status)}")
    # API Key 입력칸. type="password" = 글자를 ●로 가린다. placeholder = 빈칸일 때 회색 안내 글자
    api_key = st.text_input("API Key", type="password", placeholder="test-key-001")
    # 확신도 기준 슬라이더 (최소, 최대, 기본값, 간격)
    threshold = st.slider("모델 확신도 기준", 0.0, 1.0, 0.5, 0.05,
                          help="낮을수록 더 많이 가립니다(놓침↓, 과잉 마스킹↑)")
    # 가명 처리 체크박스
    pseudonymize = st.checkbox("가명 처리 ([이름_A]처럼 같은 사람은 같은 가명)", value=True)

# ── 본문: 입력 ───────────────────────────────
# 큰 제목
st.title("🔒 녹취록 개인정보 마스킹")
# 제목 아래 작은 설명
st.caption("갤럭시 녹음 앱에서 내보낸 녹취록(.txt)의 이름·연락처·주소 등을 가명으로 바꿉니다. "
           "모든 처리는 이 PC 안의 서버에서만 이루어집니다.")

# 파일 업로드칸 (.txt만 고를 수 있게)
uploaded = st.file_uploader("녹취록 파일 (.txt)", type=["txt"])
# 직접 붙여 넣는 칸. 파일을 올렸으면 비활성화 (둘 중 무엇을 보낼지 헷갈리지 않게)
typed = st.text_area("또는 직접 붙여 넣기", height=150,
                     placeholder="00:00\n안녕하세요 저는 ...", disabled=uploaded is not None)

# 보낼 파일 바이트(없으면 None)와 화면에 보여 줄 원문 글자
file_bytes, text = None, typed
# 파일을 올렸으면
if uploaded is not None:
    # 파일 전체를 바이트로 꺼낸다
    file_bytes = uploaded.getvalue()
    # 원문 표시용으로 UTF-8 글자로 풀어 본다
    try:
        text = file_bytes.decode("utf-8-sig")
    # 풀리지 않으면 표시용 원문은 비워 둔다 (서버로는 그대로 보내 서버가 400으로 판정하게)
    except UnicodeDecodeError:
        text = ""

# 지금 입력+설정의 지문(SHA-256) — 결과가 '지금 입력'의 것인지 확인하는 데 쓴다
input_key = hashlib.sha256(
    (text + f"|{threshold}|{pseudonymize}").encode("utf-8", "ignore") + (file_bytes or b"")
).hexdigest()

# 실행 버튼. 입력이 비어 있으면 누를 수 없게. 누른 그 재실행에서만 True
if st.button("마스킹 실행", type="primary", disabled=not (text.strip() or file_bytes)):
    # 처리 중 빙글빙글 표시
    with st.spinner("개인정보를 찾는 중..."):
        result = call_api(text, uploaded.name if uploaded else None, file_bytes,
                          api_key, threshold, pseudonymize)
    # 성공했으면 결과와 원문·지문·파일 이름을 보관함에 저장 — 다음 재실행(다운로드 클릭 등)에도 남도록
    if result:
        st.session_state["result"] = result
        st.session_state["original"] = text
        st.session_state["result_key"] = input_key
        st.session_state["source_name"] = uploaded.name if uploaded else "pasted.txt"

# ── 본문: 결과 ───────────────────────────────
# 보관함에 결과가 있으면 보여 준다 (버튼을 방금 누르지 않았어도)
if "result" in st.session_state:
    # 보관된 결과
    result = st.session_state["result"]
    # 결과를 만든 입력의 지문이 지금 입력의 지문과 다르면 안내 (이전 결과를 지금 것으로 착각하지 않게)
    if st.session_state["result_key"] != input_key:
        st.info("입력이나 설정이 바뀌었습니다. 아래는 이전 실행 결과입니다. 다시 실행해 주세요.")

    # 결과 소제목
    st.subheader("결과")
    # 유형별 개수
    summary = result["summary"]
    # 숫자 칸 4개를 가로로
    cols = st.columns(4)
    # 가린 총 건수 (사전 값들의 합)
    cols[0].metric("가린 개인정보", f"{sum(summary.values())}건")
    # 유형 수
    cols[1].metric("유형 수", f"{len(summary)}종")
    # 처리 시간 (소수점 없이)
    cols[2].metric("처리 시간", f"{result['processing_ms']:.0f} ms")
    # 정규식이 찾은 것 / 모델이 찾은 것 (True는 1로 세어진다)
    cols[3].metric("정규식 / 모델", "{} / {}".format(
        sum(e["source"] == "regex" for e in result["entities"]),
        sum(e["source"] == "model" for e in result["entities"])))

    # 유형별 개수 가로 막대 그래프
    st.bar_chart(summary, horizontal=True)

    # 좌우 두 칸
    left, right = st.columns(2)
    # 왼쪽: 원문 (읽기 전용)
    left.text_area("원문", st.session_state["original"], height=450, disabled=True)
    # 오른쪽: 마스킹 결과 (복사하기 쉽게 편집 가능 칸으로)
    right.text_area("마스킹 결과", result["masked_text"], height=450)

    # 마스킹 결과를 .txt로 내려받는 버튼 — 누르면 재실행되지만 결과는 보관함에 있어 사라지지 않는다
    st.download_button(
        "마스킹된 텍스트 다운로드",
        data=result["masked_text"].encode("utf-8"),
        file_name="masked_" + st.session_state["source_name"],
        mime="text/plain",
    )

    # 접었다 펴는 상자 안에 탐지 목록 표 (유형·타임스탬프·출처만, 원문 없음)
    with st.expander("탐지 목록 (유형·위치·출처만, 원문 없음)"):
        st.dataframe(result["entities"], width="stretch")
