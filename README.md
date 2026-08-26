# Open LLM Battle Agent

OpenRouter API를 활용해 여러 무료 오픈 LLM에 동일한 질문을 전달하고, 실제 답변 품질과 응답 성능을 비교하여 적합한 모델을 추천하는 AI 에이전트입니다.

---

## 주요 기능

- OpenRouter의 무료 텍스트 LLM 검색
- 서로 다른 개발사의 모델 자동 선택
- 동일한 질문으로 최대 5개 모델 병렬 실행
- 모델별 실제 답변과 응답 시간 측정
- 관련성, 완성도, 명확성, 실용성, 속도, 안정성 평가
- 모델별 강점·약점과 추천 용도 제공
- 최종 순위와 종합 우승 모델 선정
- 비교 결과 Markdown 파일 생성

---

## 사용 도구

1. `search_openrouter_llms`
   - 무료로 실행 가능한 OpenRouter 모델을 검색합니다.

2. `run_openrouter_battle`
   - 여러 모델에 같은 질문을 동시에 전달하고 결과를 측정합니다.

3. `evaluate_llm_responses`
   - 모델별 답변과 실행 성능을 동일한 기준으로 평가합니다.

4. `generate_report_Markdown`
   - 실제 답변, 평가 결과, 순위와 추천 내용을 Markdown 파일로 생성합니다.
