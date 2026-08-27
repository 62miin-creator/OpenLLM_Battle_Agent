import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools import TOOLS
from middleware import (
    auto_backup_middleware,
    request_safety_middleware,
    response_pii_masking_middleware,
    workspace_index_middleware,
)


load_dotenv()


def create_middleware_agent():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(".env에서 GROQ_API_KEY를 찾을 수 없습니다.")

    system_prompt = """문서 Workspace 관리 에이전트입니다.

규칙:
- Open LLM/모델 답변 비교/평가 보고서는 반드시 save_open_llm_report 도구를 사용하세요.
- 보고서는 웹 검색 기반 예상 분석이 아니라 실제 모델 호출 답변 원문과 응답 시간을 포함해야 합니다.
- 모델 답변은 반드시 한국어로 생성하고 보고서에도 한국어 답변만 저장하세요.
- 사용자가 모델 개수만 말하면 현재 Groq 지원 모델 중 서로 다른 기업/계열을 우선해 무작위 선택합니다.
- 사용자가 모델명이나 모델 ID를 지정하면 지정한 모델만 선택합니다. 최대 4개입니다.
- save_open_llm_report의 request는 다음 형식으로 전달하세요.
  질문: <모델들에게 실제로 물을 질문>
  선택: random:<2~4> 또는 models:<쉼표로 구분한 모델명/ID>
- 사용자가 개수와 모델을 모두 생략하면 선택: auto 로 전달하세요.
- 보고서는 workspace/OpenLLM_Agent 폴더에 저장하세요.
- 문서 요청은 read_file, search_workspace, write_file, edit_file 도구만 사용하세요.
"""

    model = ChatOpenAI(
        model="qwen/qwen3.6-27b",
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
        temperature=0,
        max_tokens=512,
    )

    # 미들웨어가 적용된 에이전트 생성
    agent = create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=system_prompt,
        middleware=[
            request_safety_middleware,        # 위험 입력 차단
            workspace_index_middleware,       # Workspace 인덱싱
            response_pii_masking_middleware,  # 응답 개인정보 마스킹
            auto_backup_middleware,           # 파일 변경 전 자동 백업
        ],
    )

    return agent


# 에이전트 생성
agent = create_middleware_agent()
