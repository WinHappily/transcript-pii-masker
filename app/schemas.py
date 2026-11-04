# 위 줄(%%writefile)은 이 셀의 내용을 실행하지 않고 app/schemas.py 라는 파일로 저장하라는 주피터 명령이다

# Field(필드의 규칙을 적는 도구), BaseModel(데이터 설계도의 부모 클래스), field_validator(직접 검사 함수 표시)를 가져온다
from pydantic import BaseModel, Field, field_validator
# Literal(정해진 값 몇 개만 허용하는 타입)을 가져온다 — 출처를 "regex" 또는 "model" 둘 중 하나로만 제한하려고
from typing import Literal


# ── 입력 설계도 ──────────────────────────────
# BaseModel을 상속한 클래스 = "이런 모양의 JSON만 받는다"는 설계도
class MaskRequest(BaseModel):
    # 마스킹할 녹취록 텍스트. ... = 기본값 없음 = 필수
    # min_length=1     : 한 글자도 없는 입력을 막는다
    # max_length=50_000: 29분 녹취록이 약 1만 자 → 2시간 반 분량까지 허용 (_는 읽기 쉽게 넣은 자릿수 구분, 50000과 같음)
    text: str = Field(..., min_length=1, max_length=50_000,
                      description="갤럭시 녹음 앱에서 내보낸 녹취록 텍스트")
    # 모델 확신도 기준. 선택 항목이라 기본값 0.5를 준다
    # ge=0.0 : greater or equal = 0.0 '이상' (0.0 자체도 허용)
    # le=1.0 : less or equal    = 1.0 '이하' (1.0 자체도 허용) — 확신도는 0~1 사이 값이므로
    threshold: float = Field(0.5, ge=0.0, le=1.0,
                             description="이 값 이상인 모델 탐지만 가린다")
    # 가명 처리 여부. 선택 항목이라 기본값 True
    pseudonymize: bool = Field(True, description="True면 [이름_A]처럼 같은 사람에게 같은 가명")

    # text 필드를 직접 검사하는 함수를 붙인다 — Field로는 '공백만 있는 입력'을 못 막아서
    @field_validator("text")
    # classmethod = 개별 객체가 아니라 설계도(클래스)에 붙는 함수라는 표시 (field_validator가 요구하는 형식)
    @classmethod
    # v = 사용자가 보낸 text 값
    def not_blank(cls, v: str) -> str:
        # v.strip() = 앞뒤 공백·줄바꿈을 지운 결과. 그게 빈 글자("")면 거짓이므로 not이 붙어 참이 된다
        if not v.strip():
            # ValueError를 던지면 Pydantic이 422 응답으로 바꿔 준다
            raise ValueError("공백만 있는 텍스트는 처리할 수 없습니다")
        # 통과하면 원래 값을 그대로 돌려준다 (공백을 지운 값이 아니라 원본 — 녹취록 형식을 보존하려고)
        return v


# ── 출력 설계도 ──────────────────────────────
# 탐지 한 건의 모양. 원래 개인정보 글자는 일부러 넣지 않는다 (응답 전체에 개인정보가 없게)
class Entity(BaseModel):
    # 유형 (예: "이름", "전화번호")
    type: str
    # 그 발언의 타임스탬프 (예: "00:48") — 사용자가 원본에서 위치를 찾아 검토할 수 있게
    timestamp: str
    # 누가 찾았는지 — 정규식이면 "regex", 모델이면 "model"만 허용
    source: Literal["regex", "model"]


# 응답 전체의 모양
class MaskResponse(BaseModel):
    # 성공 여부
    success: bool
    # 가린 결과 전체 (타임스탬프 줄 포함, 원래 형식 그대로)
    masked_text: str
    # 유형별 개수 — dict[str, int] = "글자 열쇠 → 정수 값" 사전
    summary: dict[str, int]
    # 탐지 목록 — list[Entity] = Entity 설계도를 따르는 항목들의 목록
    entities: list[Entity]
    # 처리 시간(밀리초)
    processing_ms: float
