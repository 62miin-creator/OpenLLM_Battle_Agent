"""Open LLM Battle Agent custom tools."""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1"


def _headers() -> dict:
    if not OPENROUTER_API_KEY:
        raise ValueError(".env에서 OPENROUTER_API_KEY를 찾을 수 없습니다.")
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Open LLM Battle Agent",
    }


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


@tool(parse_docstring=True)
def search_openrouter_llms(
    search_query: str, language: str = "multilingual", model_count: int = 3
) -> str:
    """OpenRouter에서 현재 무료로 실행 가능한 오픈 LLM을 검색합니다.

    Args:
        search_query: 모델명 또는 개발사 검색어입니다. 예: Qwen, Gemma.
        language: 원하는 지원 언어 조건입니다.
        model_count: 반환할 모델 수이며 1 이상 5 이하입니다.

    Returns:
        모델 ID와 Provider 및 모델 정보를 JSON으로 반환합니다.
    """
    try:
        if not search_query.strip():
            return "실패: 검색어가 비어 있습니다."
        if not 1 <= model_count <= 5:
            return "실패: model_count는 1 이상 5 이하이어야 합니다."

        response = requests.get(
            f"{OPENROUTER_API_URL}/models", headers=_headers(), timeout=30
        )
        response.raise_for_status()
        router_models = response.json().get("data", [])
        free_models = [
            model
            for model in router_models
            if float((model.get("pricing") or {}).get("prompt", 1)) == 0
            and float((model.get("pricing") or {}).get("completion", 1)) == 0
            and "text" in (model.get("architecture") or {}).get("output_modalities", [])
            and model.get("id") != "openrouter/free"
            and not any(
                word in model.get("id", "").lower()
                for word in ("lyria", "content-safety", "code", "note")
            )
        ]
        searchable_families = (
            "gemma", "minimax", "nemotron", "glm", "liquid", "inkling",
            "laguna", "cohere", "qwen", "llama", "mistral", "deepseek",
            "phi", "gpt-oss",
        )
        query_words = set(re.findall(r"[a-z0-9.-]+", search_query.lower()))
        requested_families = [
            family for family in searchable_families if family in query_words
        ]

        # 모델 계열이 명시되면 해당 계열을 찾고, 일반 문장이면 다양한 개발사의
        # 실행 가능한 모델을 자동 선택합니다.
        if requested_families:
            candidates = [
                model
                for model in free_models
                if any(
                    family in (
                        f"{model.get('id', '')} {model.get('owned_by', '')}"
                    ).lower()
                    for family in requested_families
                )
            ]
        else:
            candidates = free_models

        # 일반 검색에서는 같은 개발사의 모델이 중복되지 않도록 선택합니다.
        if not requested_families:
            preferred_families = (
                "gemma", "minimax", "glm", "nemotron", "liquid",
                "inkling", "laguna",
            )
            candidates.sort(
                key=lambda model: next(
                    (
                        index
                        for index, family in enumerate(preferred_families)
                        if family in model.get("id", "").lower()
                    ),
                    len(preferred_families),
                )
            )
            diverse_candidates = []
            seen_owners = set()
            for model in candidates:
                owner = str(model.get("owned_by") or model.get("id", "").split("/", 1)[0])
                if owner.lower() in seen_owners:
                    continue
                seen_owners.add(owner.lower())
                diverse_candidates.append(model)
            candidates = diverse_candidates

        if not candidates:
            return _json({
                "status": "failed",
                "message": "현재 실행 가능한 조건 일치 모델을 찾지 못했습니다.",
                "search_query": search_query,
            })

        models = []
        for candidate in candidates[:model_count]:
            model_id = candidate["id"]
            models.append({
                "model_id": model_id,
                "name": candidate.get("name"),
                "owned_by": candidate.get("id", "").split("/", 1)[0],
                "language_condition": language,
                "description": candidate.get("description"),
                "context_length": candidate.get("context_length"),
                "pricing": candidate.get("pricing"),
                "supported_parameters": candidate.get("supported_parameters", []),
                "model_url": f"https://openrouter.ai/{model_id}",
            })
        return _json({
            "status": "success", "search_query": search_query,
            "language": language, "model_count": len(models), "models": models,
        })
    except requests.RequestException as error:
        return f"실패: OpenRouter 모델 검색 오류 - {error}"
    except Exception as error:
        return f"실패: {error}"


@tool(parse_docstring=True)
def run_openrouter_battle(
    model_ids: list[str], question: str, max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """여러 오픈 LLM에 같은 질문을 전달하여 답변과 실행 성능을 측정합니다.

    Args:
        model_ids: 비교할 OpenRouter 무료 모델 ID 목록입니다.
        question: 모든 모델에 동일하게 전달할 질문입니다.
        max_tokens: 최대 출력 토큰 수이며 16 이상 1024 이하입니다.
        temperature: 답변 무작위성 값이며 0 이상 2 이하입니다.

    Returns:
        모델별 답변, 응답 시간, 성공 여부와 오류를 JSON으로 반환합니다.
    """
    try:
        if not model_ids:
            return "실패: 배틀에 사용할 모델 ID가 없습니다."
        if len(model_ids) > 5:
            return "실패: 한 번에 비교할 모델은 최대 5개입니다."
        if not question.strip():
            return "실패: 사용자 질문이 비어 있습니다."
        if not 16 <= max_tokens <= 1024:
            return "실패: max_tokens는 16 이상 1024 이하이어야 합니다."
        if not 0 <= temperature <= 2:
            return "실패: temperature는 0 이상 2 이하이어야 합니다."

        def run_model(model_id: str) -> dict:
            started = time.perf_counter()
            try:
                response = requests.post(
                    f"{OPENROUTER_API_URL}/chat/completions",
                    headers=_headers(), timeout=60,
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": question}],
                        "max_tokens": max_tokens, "temperature": temperature,
                    },
                )
                elapsed = round(time.perf_counter() - started, 3)
                if not response.ok:
                    return {"model_id": model_id, "status": "failed", "answer": None,
                            "response_time_seconds": elapsed, "error": response.text[:500]}
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    return {"model_id": model_id, "status": "failed", "answer": None,
                            "response_time_seconds": elapsed, "error": "생성된 답변이 없습니다."}
                message = choices[0].get("message") or {}
                answer = message.get("content")
                if not answer or not str(answer).strip():
                    return {
                        "model_id": model_id,
                        "status": "failed",
                        "answer": None,
                        "response_time_seconds": elapsed,
                        "finish_reason": choices[0].get("finish_reason"),
                        "error": (
                            "API 호출은 성공했지만 답변 내용이 비어 있습니다. "
                            "무료 Provider의 출력 제한 또는 모델 응답 오류일 수 있습니다."
                        ),
                    }
                return {
                    "model_id": model_id, "status": "success",
                    "answer": answer,
                    "response_time_seconds": elapsed,
                    "finish_reason": choices[0].get("finish_reason"),
                    "usage": data.get("usage") or {},
                }
            except requests.RequestException as error:
                return {"model_id": model_id, "status": "failed", "answer": None,
                        "response_time_seconds": round(time.perf_counter() - started, 3),
                        "error": str(error)}

        # 모델을 동시에 실행해 전체 대기 시간을 가장 느린 모델 수준으로 줄입니다.
        indexed_results = {}
        with ThreadPoolExecutor(max_workers=len(model_ids)) as executor:
            futures = {
                executor.submit(run_model, model_id): index
                for index, model_id in enumerate(model_ids)
            }
            for future in as_completed(futures):
                indexed_results[futures[future]] = future.result()
        results = [indexed_results[index] for index in range(len(model_ids))]
        success_count = sum(item["status"] == "success" for item in results)
        return _json({
            "status": "success" if success_count else "failed", "question": question,
            "generation_options": {"max_tokens": max_tokens, "temperature": temperature},
            "success_count": success_count, "failure_count": len(results) - success_count,
            "results": results,
        })
    except Exception as error:
        return f"실패: {error}"


@tool(parse_docstring=True)
def evaluate_llm_responses(question: str, battle_results_json: str) -> str:
    """모델 답변을 세부 기준으로 비교하고 선택에 필요한 추천을 만듭니다.

    Args:
        question: 모델들에게 전달한 원본 질문입니다.
        battle_results_json: run_openrouter_battle이 반환한 JSON 문자열입니다.

    Returns:
        관련성, 완성도, 명확성, 실용성, 속도, 안정성 점수와 용도별 추천을 반환합니다.
    """
    try:
        results = json.loads(battle_results_json).get("results") or []
        successful = [r for r in results if r.get("status") == "success" and r.get("answer")]
        if not results:
            return "실패: 평가할 모델 실행 결과가 없습니다."
        if not successful:
            return "실패: 정상적으로 답변을 생성한 모델이 없습니다."

        fastest = min(r["response_time_seconds"] for r in successful)
        keywords = set(re.findall(r"[A-Za-z0-9가-힣]{2,}", question.lower()))
        evaluations = []
        for result in results:
            if result.get("status") != "success" or not result.get("answer"):
                evaluations.append({"model_id": result.get("model_id"), "status": "failed",
                                    "total_score": 0, "reason": result.get("error", "실행 실패")})
                continue
            answer = str(result["answer"])
            matched = [word for word in keywords if word in answer.lower()]
            relevance = round(len(matched) / len(keywords) * 25, 2) if keywords else 12.5
            completeness = round(min(len(answer) / 700, 1) * 20, 2)
            sentences = [s.strip() for s in re.split(r"[.!?。！？\n]+", answer) if s.strip()]
            average = sum(map(len, sentences)) / len(sentences) if sentences else len(answer)
            clarity = 15 if 15 <= average <= 100 else 11 if average <= 160 else 7
            structure_markers = len(re.findall(r"(?:^|\n)\s*(?:[-*]|\d+[.)]|#{1,3})", answer))
            action_words = ("하세요", "확인", "설정", "사용", "피하세요", "주의", "방법", "단계")
            action_hits = sum(word in answer for word in action_words)
            actionability = round(min(structure_markers / 5, 1) * 7.5 + min(action_hits / 5, 1) * 7.5, 2)
            speed = round(min(fastest / max(result["response_time_seconds"], 0.001), 1) * 15, 2)
            stability = 10 if result.get("finish_reason") != "length" else 5
            total = round(relevance + completeness + clarity + actionability + speed + stability, 2)

            strengths = []
            weaknesses = []
            if relevance >= 17.5:
                strengths.append("질문의 핵심 내용을 잘 반영함")
            else:
                weaknesses.append("질문 핵심어 반영이 상대적으로 부족함")
            if completeness >= 16:
                strengths.append("설명이 충분하고 구체적임")
            else:
                weaknesses.append("답변 분량이나 세부 설명이 부족함")
            if clarity >= 13:
                strengths.append("문장이 읽기 쉽고 구조가 명확함")
            else:
                weaknesses.append("문장이 길거나 구조가 불명확함")
            if actionability >= 11:
                strengths.append("바로 적용할 행동 방법을 제시함")
            else:
                weaknesses.append("실행 가능한 조치가 충분하지 않음")
            if speed >= 12:
                strengths.append("응답 속도가 빠름")
            elif result["response_time_seconds"] > fastest * 3:
                weaknesses.append("다른 모델보다 응답이 느림")
            if result.get("finish_reason") == "length":
                weaknesses.append("출력 한도에 도달해 답변이 잘렸을 가능성이 있음")
            evaluations.append({
                "model_id": result.get("model_id"), "status": "success", "total_score": total,
                "scores": {"relevance": relevance, "completeness": completeness,
                           "clarity": clarity, "actionability": actionability,
                           "speed": speed, "stability": stability},
                "response_time_seconds": result["response_time_seconds"],
                "answer_length": len(answer), "finish_reason": result.get("finish_reason"),
                "strengths": strengths, "weaknesses": weaknesses,
                "recommended_for": (
                    "빠른 실시간 응답" if speed >= 12
                    else "상세한 설명과 문서 작성" if completeness >= 16
                    else "일반적인 질의응답"
                ),
            })
        ranking = sorted(evaluations, key=lambda item: item.get("total_score", 0), reverse=True)
        for index, item in enumerate(ranking, 1):
            item["rank"] = index
        winner = ranking[0]
        return _json({
            "status": "success", "question": question,
            "evaluation_method": {"relevance": 25, "completeness": 20, "clarity": 15,
                                  "actionability": 15, "speed": 15, "stability": 10},
            "ranking": ranking,
            "winner": {"model_id": winner["model_id"], "total_score": winner["total_score"],
                       "reason": "정량 평가 점수가 가장 높은 모델입니다."},
            "selection_guide": {
                "best_overall": winner["model_id"],
                "fastest": min(successful, key=lambda item: item["response_time_seconds"])["model_id"],
                "most_detailed": max(
                    (item for item in ranking if item.get("status") == "success"),
                    key=lambda item: item.get("answer_length", 0),
                )["model_id"],
            },
            "important_notice": (
                "이 평가는 답변 구조와 실행 성능을 비교한 자동 평가입니다. "
                "정답 자료가 없으므로 사실 정확성은 사용자가 실제 답변을 확인해야 합니다."
            ),
        })
    except json.JSONDecodeError:
        return "실패: battle_results_json이 올바른 JSON 형식이 아닙니다."
    except Exception as error:
        return f"실패: {error}"


@tool(parse_docstring=True)
def generate_report_markdown(
    battle_results_json: str, evaluation_results_json: str,
    report_filename: str = "open_llm_battle_report.md",
) -> str:
    """배틀 결과와 비교 평가를 읽기 쉬운 Markdown 보고서로 생성합니다.

    Args:
        battle_results_json: run_openrouter_battle이 반환한 JSON 문자열입니다.
        evaluation_results_json: evaluate_llm_responses가 반환한 JSON 문자열입니다.
        report_filename: 경로를 제외한 Markdown 파일명입니다.

    Returns:
        생성된 Markdown 보고서의 절대 경로 또는 오류 메시지를 반환합니다.
    """
    try:
        battle = json.loads(battle_results_json)
        evaluation = json.loads(evaluation_results_json)
        filename = Path(report_filename).name
        if not filename.lower().endswith(".md"):
            filename += ".md"
        output_dir = Path(__file__).resolve().parent / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        lines = [
            "# Open LLM Battle Agent 결과 보고서",
            "",
            "## 사용자 질문",
            "",
            str(battle.get("question", "-")),
            "",
            "## 1. 실행 결과 요약",
            "",
            "| 모델 | 상태 | 응답 시간 | 답변 길이 | 종료 사유 |",
            "|---|---:|---:|---:|---|",
        ]
        for result in battle.get("results") or []:
            answer = result.get("answer") or result.get("error") or "-"
            model_id = str(result.get("model_id", "-")).replace("|", "\\|")
            lines.append(
                f"| `{model_id}` | {result.get('status', '-')} | "
                f"{result.get('response_time_seconds', '-')}초 | {len(str(answer))}자 | "
                f"{result.get('finish_reason', '-')} |"
            )

        lines.extend(["", "## 2. 모델별 전체 답변", ""])
        for index, result in enumerate(battle.get("results") or [], 1):
            answer = result.get("answer") or result.get("error") or "답변 없음"
            lines.extend([
                f"### 2-{index}. `{result.get('model_id', '-')}`",
                "",
                f"- 상태: **{result.get('status', '-')}**",
                f"- 응답 시간: **{result.get('response_time_seconds', '-')}초**",
                f"- 종료 사유: `{result.get('finish_reason', '-')}`",
                "",
                str(answer),
                "",
            ])

        lines.extend([
            "## 3. 세부 비교 평가",
            "",
            "| 순위 | 모델 | 총점 | 관련성 | 완성도 | 명확성 | 실용성 | 속도 | 안정성 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for item in evaluation.get("ranking") or []:
            scores = item.get("scores") or {}
            lines.append(
                f"| {item.get('rank', '-')} | `{item.get('model_id', '-')}` | "
                f"{item.get('total_score', 0)} | {scores.get('relevance', 0)} | "
                f"{scores.get('completeness', 0)} | {scores.get('clarity', 0)} | "
                f"{scores.get('actionability', 0)} | {scores.get('speed', 0)} | "
                f"{scores.get('stability', 0)} |"
            )

        lines.extend(["", "## 4. 모델별 장단점과 추천 용도", ""])
        for item in evaluation.get("ranking") or []:
            if item.get("status") != "success":
                lines.extend([
                    f"### `{item.get('model_id', '-')}`",
                    "",
                    f"- 실행 실패: {item.get('reason', '원인 확인 필요')}",
                    "",
                ])
                continue
            lines.extend([
                f"### `{item.get('model_id', '-')}`",
                "",
                f"- 강점: {', '.join(item.get('strengths') or ['확인 필요'])}",
                f"- 약점: {', '.join(item.get('weaknesses') or ['뚜렷한 약점 없음'])}",
                f"- 추천 용도: **{item.get('recommended_for', '일반적인 질의응답')}**",
                "",
            ])

        winner = evaluation.get("winner") or {}
        guide = evaluation.get("selection_guide") or {}
        lines.extend([
            "## 5. 최종 선택 가이드",
            "",
            f"- 종합 추천: **`{winner.get('model_id', '선정 불가')}`** "
            f"({winner.get('total_score', '-')}점)",
            f"- 속도 우선: **`{guide.get('fastest', '-')}`**",
            f"- 상세 설명 우선: **`{guide.get('most_detailed', '-')}`**",
            "",
            "> " + str(evaluation.get("important_notice", "평가 한계를 확인하세요.")),
            "",
        ])
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return f"성공: Markdown 보고서가 생성되었습니다.\n파일 경로: {output_path.resolve()}"
    except json.JSONDecodeError:
        return "실패: 입력된 결과가 올바른 JSON 형식이 아닙니다."
    except Exception as error:
        return f"실패: Markdown 보고서 생성 오류 - {error}"


CUSTOM_TOOLS = [
    search_openrouter_llms,
    run_openrouter_battle,
    evaluate_llm_responses,
    generate_report_markdown,
]
