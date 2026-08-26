"""LangGraph Studio entry point for OpenRouter-based LLM Battle Agent."""

import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from tools import CUSTOM_TOOLS


load_dotenv()


def create_open_llm_battle_agent():
    """오픈 LLM 검색, 실행, 평가와 보고서 생성을 담당합니다."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError(".env에서 OPENROUTER_API_KEY를 찾을 수 없습니다.")

    system_prompt = """당신은 Open LLM Battle Agent입니다.

사용자의 목적에 맞는 무료 오픈 LLM을 OpenRouter에서 검색하고, 같은 질문과 생성 조건으로 실제 실행한 뒤 결과를 비교합니다.

도구 사용 순서:
1. 모델 ID가 없으면 search_openrouter_llms로 무료 후보를 찾습니다.
2. 검색된 실제 model_id를 run_openrouter_battle에 전달합니다.
3. 질문과 배틀 JSON 원문을 evaluate_llm_responses에 전달합니다.
4. 사용자가 보고서를 요청한 경우 generate_report_markdown을 호출합니다.

규칙:
- 모든 모델에 같은 질문, max_tokens, temperature를 적용하세요.
- 별도 조건이 없으면 max_tokens는 1024, temperature는 0.2를 사용하세요.
- 사용자가 특정 모델명을 말하지 않고 여러 오픈 LLM 비교를 요청하면 search_query를 "all"로 설정해 한 번만 검색하세요.
- 사용자가 특정 모델 계열을 말하면 Gemma, MiniMax, Nemotron, GLM처럼 짧은 계열명만 search_query로 전달하세요.
- "multilingual instruct open LLM"처럼 설명 문장 전체를 검색어로 전달하지 마세요.
- 도구 결과를 임의로 만들거나 JSON 내용을 변경하지 마세요.
- 일부 모델이 실패해도 성공한 결과는 계속 비교하고 실패 원인을 알려주세요.
- 보고서는 사용자가 요청했을 때만 Markdown 파일로 생성하세요.
- 자동 점수는 정량 휴리스틱이며 사실 정확성을 확정하지 못한다는 한계를 표시하세요.
- 최종 답변에는 실제 답변 요약, 응답 시간, 세부 항목별 점수, 모델별 강점과 약점, 추천 용도, 종합 우승 모델을 한국어로 정리하세요.
- 반드시 가격이 0인 OpenRouter 무료 모델만 사용하세요.
- API 오류, 무료 요청 제한 또는 모델 이용 불가 상태를 숨기지 마세요.
"""
    # Tool Calling을 지원하는 무료 모델로 고정해 도구 실행 흐름을 안정화합니다.
    model = ChatOpenAI(
        model="google/gemma-4-31b-it:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        temperature=0,
        default_headers={"X-Title": "Open LLM Battle Agent"},
    )

    return create_agent(
        model=model,
        tools=CUSTOM_TOOLS,
        system_prompt=system_prompt,
    )


agent = create_open_llm_battle_agent()
