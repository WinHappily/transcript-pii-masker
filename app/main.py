# 위 줄(%%writefile)은 이 셀의 내용을 실행하지 않고 app/main.py 라는 파일로 저장하라는 주피터 명령이다
"""
녹취록 개인정보 마스킹 API 서버.
흐름: 인증(401) → 입력 검증(422/400/413) → 모델 준비 확인(503) → 전용 스레드에서 마스킹(500) → 응답(200)
"""
# asyncio(이벤트 루프 도구)를 가져온다 — run_in_executor, create_task를 쓰려고
import asyncio
# logging(기록 도구)을 가져온다 — 처리 기록을 남기려고 (원문은 절대 남기지 않음)
import logging
# uuid(겹치지 않는 번호 생성기)를 가져온다 — 요청마다 추적 번호(request_id)를 붙이려고
import uuid
# ThreadPoolExecutor(일꾼 스레드 대기조)를 가져온다 — 오래 걸리는 모델 작업을 이벤트 루프 밖 스레드에 맡기려고
from concurrent.futures import ThreadPoolExecutor
# asynccontextmanager(비동기 '준비-정리' 틀을 만드는 표시)를 가져온다 — lifespan을 만들려고
from contextlib import asynccontextmanager

# FastAPI의 부품들을 가져온다
#   Depends(먼저 실행할 함수 연결), File·Form·UploadFile(파일 업로드 받기), HTTPException(에러 응답), Request(요청 정보)
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
# 입력 검증 실패 에러의 종류를 가져온다 — 이 에러가 났을 때 우리가 직접 응답을 만들려고
from fastapi.exceptions import RequestValidationError
# JSON 응답을 직접 만드는 도구
from fastapi.responses import JSONResponse
# Pydantic의 검증 실패 에러 — 파일 내용을 MaskRequest로 검사할 때 잡으려고
from pydantic import ValidationError

# 우리가 만든 부품들을 가져온다
from app.auth import verify_api_key                               # 5A-1: API Key 인증
from app.model_service import MODEL_NAME, load_model, mask_transcript  # 4-B: 모델 로드와 마스킹
from app.schemas import MaskRequest, MaskResponse                 # 3: 입력·출력 설계도

# 로그 기본 설정: INFO 등급 이상을 '시각 등급 메시지' 모양으로 화면에 찍는다
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# 이 서버의 로거 (이름이 같으면 프로그램 어디서든 같은 로거)
logger = logging.getLogger("pii_api")

# 업로드 파일 최대 크기 (200KB)
MAX_FILE_BYTES = 200_000
# 모델 전용 일꾼 대기조. GPU 모델은 한 번에 하나씩 → 일꾼 1명
executor = ThreadPoolExecutor(max_workers=1)
# 서버 상태 보관함: 모델 객체, 상태 글자(loading/healthy/error), 로딩 작업 손잡이
state = {"ner": None, "status": "loading", "task": None}


# 모델을 뒤에서 불러오는 함수 — 서버는 바로 요청을 받기 시작하고, 그동안 /health는 loading을 답한다
async def _load_model_in_background():
    # 지금 돌고 있는 이벤트 루프를 가져온다
    loop = asyncio.get_running_loop()
    # 로드가 실패해도 서버 전체가 죽지 않도록 감싼다
    try:
        # load_model은 오래 걸리는 동기 함수 → 일꾼 스레드에 맡기고 await로 양보 (그동안 /health가 답할 수 있다)
        #   load_model 뒤에 괄호를 붙이지 않는다 — 붙이면 이 자리(루프 스레드)에서 바로 실행돼 루프가 막힌다
        state["ner"] = await loop.run_in_executor(executor, load_model)
        # 성공하면 상태를 healthy로
        state["status"] = "healthy"
        # 성공 기록 (모델 이름만)
        logger.info("모델 로드 완료: %s", MODEL_NAME)
    # 실패하면
    except Exception as e:
        # 상태를 error로 → 이후 /predict는 503
        state["status"] = "error"
        # 에러 종류 이름만 기록한다
        logger.error("모델 로드 실패: %s", type(e).__name__)


# 서버가 켜질 때와 꺼질 때 할 일 (yield 앞 = 켜질 때, 뒤 = 꺼질 때)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델 로딩을 '뒤에서 도는 작업'으로 띄운다. 손잡이를 state에 보관 — 보관하지 않으면 도중에 정리(삭제)될 수 있어서
    state["task"] = asyncio.create_task(_load_model_in_background())
    # 여기서 서버가 요청을 받기 시작한다
    yield
    # 서버가 꺼질 때 일꾼 대기조를 정리한다
    executor.shutdown(wait=False)


# FastAPI 앱을 만든다. title·description·version은 Swagger 화면 맨 위에 보인다
app = FastAPI(
    title="녹취록 개인정보 마스킹 API",
    description="갤럭시 녹음 앱에서 내보낸 녹취록의 개인정보를 가명으로 바꿔 돌려줍니다.",
    version="1.0.0",
    lifespan=lifespan,
)


# 검증 에러 목록에서 '위치(loc)·이유(msg)·종류(type)'만 남기는 함수 — 입력 원문(input)은 뺀다 (개인정보라서)
def _safe_errors(errors) -> list[dict]:
    # 에러 하나(e)마다 세 가지만 골라 새 사전을 만든다. .get(열쇠, 기본값) = 없으면 기본값
    return [
        {"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")}
        for e in errors
    ]


# 입력 검증 실패(422)가 나면 FastAPI 기본 처리 대신 이 함수가 응답을 만든다
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    # 추적 번호를 만든다 (uuid 앞 8글자)
    request_id = uuid.uuid4().hex[:8]
    # 원문을 뺀 에러 목록
    safe = _safe_errors(exc.errors())
    # 로그에는 위치와 종류만 남긴다
    logger.warning("입력 검증 실패 request_id=%s errors=%s", request_id,
                   [(e["loc"], e["type"]) for e in safe])
    # 손님에게는 원문 없는 에러 목록 + 추적 번호를 돌려준다
    return JSONResponse(status_code=422, content={"detail": safe, "request_id": request_id})


# /predict와 /predict/file이 함께 쓰는 공통 처리 함수 (같은 코드를 두 번 쓰지 않으려고)
async def _run_mask(req: MaskRequest, user: str) -> MaskResponse:
    # 모델이 아직 로딩 중이면 503 (서버 탓이지만 시간이 지나면 해결됨 → 다시 시도하라고 안내)
    if state["status"] == "loading":
        raise HTTPException(status_code=503, detail="모델을 불러오는 중입니다. 잠시 후 다시 시도해 주세요.")
    # 로딩에 실패한 상태면 503
    if state["status"] != "healthy":
        raise HTTPException(status_code=503, detail="모델을 사용할 수 없습니다. 관리자에게 문의해 주세요.")
    # 이 요청의 추적 번호
    request_id = uuid.uuid4().hex[:8]
    # 지금 이벤트 루프
    loop = asyncio.get_running_loop()
    # 마스킹 중 에러가 나면 잡아서 500으로 바꾼다
    try:
        # mask_transcript(모델, 글자, 기준값, 가명여부)를 일꾼 스레드에 맡긴다
        #   run_in_executor는 인자를 이름 없이 순서대로만 받는다 → 4-B 함수의 매개변수 순서와 같게 넣는다
        result = await loop.run_in_executor(
            executor, mask_transcript, state["ner"], req.text, req.threshold, req.pseudonymize
        )
    # 어떤 에러든
    except Exception as e:
        # 로그에는 추적 번호·사용자·에러 종류 이름만 남긴다 (에러 메시지에 원문 조각이 섞일 수 있어서)
        logger.error("마스킹 실패 request_id=%s user=%s error=%s", request_id, user, type(e).__name__)
        # 반쯤 가린 결과를 돌려주지 않고 실패로 끝낸다 (fail-closed). 손님에게는 고정 문구 + 추적 번호
        raise HTTPException(
            status_code=500,
            detail=f"마스킹 처리 중 오류가 발생했습니다. (request_id={request_id})",
        )
    # 성공 기록: 원문 대신 글자 수·탐지 건수·처리 시간만
    logger.info("마스킹 완료 request_id=%s user=%s 글자수=%d 탐지=%d건 %.1fms",
                request_id, user, len(req.text), len(result["entities"]), result["processing_ms"])
    # 결과 사전을 펼쳐(**) 응답 설계도에 담는다 + success=True
    return MaskResponse(success=True, **result)


# 상태 확인 — 인증을 걸지 않는다 (모니터링·게이트웨이가 키 없이 물어볼 수 있어야 해서)
@app.get("/health")
async def health():
    # 상태 글자, 모델 이름, 모델이 실제로 올라왔는지
    return {"status": state["status"], "model": MODEL_NAME, "model_loaded": state["ner"] is not None}


# 텍스트를 JSON으로 받아 마스킹 (강의안 필수 엔드포인트)
#   req: MaskRequest → 본문 JSON을 설계도로 검사 (실패하면 422, 이 함수는 실행되지 않음)
#   user: Depends(verify_api_key) → 이 함수보다 인증이 먼저 실행 (실패하면 401)
@app.post("/predict", response_model=MaskResponse)
async def predict(req: MaskRequest, user: str = Depends(verify_api_key)):
    # 공통 처리 함수에 넘긴다
    return await _run_mask(req, user)


# 갤럭시 녹음 앱의 .txt 파일을 받아 마스킹
#   파일과 옵션을 함께 보내야 해서 JSON 대신 multipart 형식 → 옵션은 Form으로 받는다
@app.post("/predict/file", response_model=MaskResponse)
async def predict_file(
    file: UploadFile = File(..., description="갤럭시 녹음 앱에서 내보낸 .txt 파일"),
    threshold: float = Form(0.5, ge=0.0, le=1.0),
    pseudonymize: bool = Form(True),
    user: str = Depends(verify_api_key),
):
    # ① 확장자 검사 (가장 싼 검사라 먼저). file.filename이 None일 수 있어 or ""로 대비, .lower()로 대소문자 무시
    if not (file.filename or "").lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail=".txt 파일만 업로드할 수 있습니다.")
    # ② 크기 검사: 최대 크기보다 '1바이트만 더' 읽는다
    #   다 읽고 나서 재면 거대한 파일이 이미 메모리를 다 차지한 뒤라 보호가 안 된다 (D49 실습의 빈틈)
    data = await file.read(MAX_FILE_BYTES + 1)
    # 1바이트 더 읽혔다면 최대 크기를 넘은 것 → 413 (본문이 너무 큼)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"파일이 너무 큽니다. 최대 {MAX_FILE_BYTES // 1000}KB까지 가능합니다.")
    # ③ 실제로 UTF-8 글자로 풀어 본다 — 확장자는 손님이 붙인 꼬리표일 뿐, 진짜 판정은 열어 보는 것
    try:
        # utf-8-sig = 파일 맨 앞에 숨은 표식(BOM)이 있으면 떼고 읽는다 (윈도우 메모장 저장 파일 대비)
        text = data.decode("utf-8-sig")
    # UTF-8로 풀리지 않으면 (예: CP949로 저장된 파일, 이미지를 .txt로 이름만 바꾼 파일)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="UTF-8 텍스트 파일이 아닙니다. 갤럭시 녹음 앱에서 내보낸 .txt 파일인지 확인해 주세요.")
    # ④ /predict와 같은 규칙(길이·공백)으로 검사한다 — 파일 경로는 FastAPI 자동 검사를 거치지 않으므로 직접
    try:
        req = MaskRequest(text=text, threshold=threshold, pseudonymize=pseudonymize)
    # 규칙 위반이면 422, 원문은 빼고
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=_safe_errors(e.errors()))
    # 공통 처리 함수에 넘긴다
    return await _run_mask(req, user)
