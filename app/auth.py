# 위 줄(%%writefile)은 이 셀의 내용을 실행하지 않고 app/auth.py 라는 파일로 저장하라는 주피터 명령이다
"""
API Key 인증 모듈 (D52 Own Project 강의안 3.2절 그대로 재사용).
"""
# HTTPException(에러 응답을 만드는 도구), Header(요청 헤더에서 값을 꺼내는 표시)를 가져온다
from fastapi import HTTPException, Header

# 허용된 키 → 사용자 이름 대응표 (강의안의 시험용 키. 실제 운영이라면 코드 밖 환경 변수·해시 목록으로 — README 보완점)
VALID_API_KEYS = {
    "test-key-001": "사용자A",
    "test-key-002": "사용자B",
}


# 엔드포인트보다 먼저 실행되는 인증 함수 (Depends로 연결된다)
#   x_api_key: 헤더 'X-API-Key'의 값. FastAPI가 밑줄(_)을 하이픈(-)으로 바꿔 찾아 준다. 없으면 None
async def verify_api_key(x_api_key: str = Header(None)) -> str:
    # 헤더가 아예 없으면 401 (누군지 모름 = 인증 실패)
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API Key가 필요합니다. X-API-Key 헤더를 포함해 주세요.",
        )
    # 헤더는 있는데 표에 없는 키면 401
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="유효하지 않은 API Key입니다.",
        )
    # 통과하면 사용자 이름을 돌려준다 → 엔드포인트의 user 매개변수로 들어간다 (로그에 '누가' 썼는지 남기려고)
    return VALID_API_KEYS[x_api_key]
