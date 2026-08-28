from dotenv import load_dotenv
from langchain.agents import create_agent

from middleware import (
    SkillMiddleware,
    auto_backup_middleware,
    request_safety_middleware,
    response_pii_masking_middleware,
    workspace_index_middleware,
)
from tools import TOOLS


load_dotenv()


def create_skill_agent():
    """Open LLM 모델 비교 Skill Agent를 생성합니다."""

    system_prompt = """당신은 여러 Open LLM의 실제 답변을 비교하고
평가 결과를 Markdown 보고서로 저장하는 모델 비교 에이전트입니다.

규칙:
- 모델 비교, 모델 평가, 모델 순위 또는 비교 보고서 요청에는
  먼저 load_skill 도구로 advanced-evaluation Skill을 로드하세요.
- Skill을 로드한 뒤에는 Skill에 정의된 평가 프로세스를 따라야 합니다.
- 모델 성능을 추측하지 말고 반드시 실제 모델 응답을 비교하세요.
- Open LLM 비교 요청에는 save_open_llm_report 도구를 정확히 한 번 사용하세요.
- save_open_llm_report의 request는 다음 형식으로 전달하세요.

  질문: <모든 모델에 동일하게 전달할 질문>
  선택: random:<2~4> 또는 models:<쉼표로 구분한 모델명이나 모델 ID>

- 사용자가 모델 수와 모델명을 모두 생략하면 선택: auto를 사용하세요.
- 사용자가 모델을 지정하면 지정한 모델만 요청하세요.
- 한 번에 비교할 수 있는 모델은 최대 4개입니다.
- save_open_llm_report가 반환한 report_path를 사용해 read_file로 보고서를 읽으세요.
- 모델별 실제 답변 원문을 수정하거나 삭제하지 마세요.
- loaded Skill의 지침에 따라 평가 결과를 작성하고 같은 report_path에 저장하세요.
- 답변 품질과 응답 속도를 별도로 평가하세요.
- 지원하지 않거나 실패한 모델을 다른 모델로 임의 교체하지 마세요.
- 모든 모델 답변과 Markdown 보고서는 한국어로 작성하세요.
- 작업 완료 후 최종 1위 또는 동률, 판정 신뢰도, 보고서 경로만 간결하게 알려주세요.
"""

    agent_executor = create_agent(
        model="gpt-5.4-mini",
        tools=TOOLS,
        system_prompt=system_prompt,
        middleware=[
            # 1. 빈 입력, 프롬프트 인젝션, 개인정보 입력 차단
            request_safety_middleware,

            # 2. Workspace 문서 목록을 실행 상태에 추가
            workspace_index_middleware,

            # 3. 관련 Skill 목록을 모델 시스템 프롬프트에 추가
            SkillMiddleware(),

            # 4. 모델 응답의 개인정보 형식을 마스킹
            response_pii_masking_middleware,

            # 5. 파일 수정 전에 자동 백업하고 민감 경로 변경 차단
            auto_backup_middleware,
        ],
    )

    return agent_executor


agent = create_skill_agent()
