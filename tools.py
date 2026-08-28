from langchain.tools import tool
from dotenv import load_dotenv
import json
import os
import random
import re
import time
from pathlib import Path

import requests


load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1"
MAX_BATTLE_MODELS = 4
NON_CHAT_MODEL_KEYWORDS = (
    "whisper",
    "guard",
    "safeguard",
    "prompt-guard",
    "tts",
    "orpheus",
)


# ============================================
# 파일 시스템 도구
# ============================================

@tool(parse_docstring=True)
def read_file(file_path: str) -> str:
    """파일의 내용을 읽어서 반환합니다.

    Args:
        file_path: 읽을 파일의 경로

    Returns:
        파일 내용 또는 오류 메시지
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return f"파일: {file_path}\n\n{content}"
    except FileNotFoundError:
        return f"오류: 파일을 찾을 수 없습니다: {file_path}"
    except PermissionError:
        return f"오류: 파일에 대한 읽기 권한이 없습니다: {file_path}"
    except Exception as e:
        return f"오류: {str(e)}"


@tool(parse_docstring=True)
def write_file(file_path: str, content: str) -> str:
    """파일에 내용을 작성합니다. 파일이 없으면 생성하고, 있으면 덮어씁니다.

    Args:
        file_path: 작성할 파일의 경로
        content: 파일에 쓸 내용

    Returns:
        성공 메시지 또는 오류 메시지
    """
    try:
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"성공: 파일이 작성되었습니다: {file_path}"
    except PermissionError:
        return f"오류: 파일에 대한 쓰기 권한이 없습니다: {file_path}"
    except Exception as e:
        return f"오류: {str(e)}"


@tool(parse_docstring=True)
def delete_file(file_path: str) -> str:
    """파일을 삭제합니다.

    Args:
        file_path: 삭제할 파일의 경로

    Returns:
        성공 메시지 또는 오류 메시지
    """
    try:
        if not os.path.isfile(file_path):
            return f"오류: 파일을 찾을 수 없습니다: {file_path}"

        os.remove(file_path)
        return f"성공: 파일이 삭제되었습니다: {file_path}"
    except PermissionError:
        return f"오류: 파일에 대한 삭제 권한이 없습니다: {file_path}"
    except Exception as e:
        return f"오류: {str(e)}"


@tool(parse_docstring=True)
def create_directory(dir_path: str) -> str:
    """새로운 디렉터리를 생성합니다.

    Args:
        dir_path: 생성할 디렉터리의 경로

    Returns:
        성공 메시지 또는 오류 메시지
    """
    try:
        os.makedirs(dir_path, exist_ok=True)
        return f"성공: 디렉터리가 생성되었습니다: {dir_path}"
    except PermissionError:
        return f"오류: 디렉터리 생성 권한이 없습니다: {dir_path}"
    except Exception as e:
        return f"오류: {str(e)}"


@tool(parse_docstring=True)
def list_directory(dir_path: str = ".") -> str:
    """디렉터리의 파일과 폴더 목록을 반환합니다.

    Args:
        dir_path: 조회할 디렉터리 경로

    Returns:
        파일 및 폴더 목록 또는 오류 메시지
    """
    try:
        if not os.path.exists(dir_path):
            return f"오류: 디렉터리를 찾을 수 없습니다: {dir_path}"

        if not os.path.isdir(dir_path):
            return f"오류: 디렉터리가 아닙니다: {dir_path}"

        items = os.listdir(dir_path)
        if not items:
            return f"디렉터리가 비어있습니다: {dir_path}"

        folders = []
        files = []

        for item in sorted(items):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                folders.append(f"[폴더] {item}/")
            else:
                files.append(f"[파일] {item}")

        result = [f"디렉터리: {dir_path}", ""]

        if folders:
            result.append("폴더:")
            result.extend(folders)
            result.append("")

        if files:
            result.append("파일:")
            result.extend(files)

        return "\n".join(result)
    except PermissionError:
        return f"오류: 디렉터리에 대한 읽기 권한이 없습니다: {dir_path}"
    except Exception as e:
        return f"오류: {str(e)}"


# ============================================
# 문서 읽기 도구
# ============================================

@tool(parse_docstring=True)
def read_csv(file_path: str, max_rows: int = 50) -> str:
    """CSV 파일의 데이터를 읽습니다.

    Args:
        file_path: CSV 파일 경로
        max_rows: 읽을 최대 행 수

    Returns:
        파일 내용 또는 오류 메시지
    """
    try:
        if not os.path.exists(file_path):
            return f"오류: 파일을 찾을 수 없습니다: {file_path}"

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        displayed_lines = min(total_lines, max_rows)
        content = "".join(lines[:displayed_lines])

        result = (
            f"파일: {file_path}\n"
            f"타입: CSV\n"
            f"총 {total_lines}줄 (표시: {displayed_lines}줄)\n\n"
            f"{content}"
        )

        if total_lines > max_rows:
            result += f"\n\n... ({total_lines - max_rows}줄 생략)"

        return result
    except PermissionError:
        return f"오류: 파일에 대한 읽기 권한이 없습니다: {file_path}"
    except Exception as e:
        return f"오류: {str(e)}"


# ============================================
# Open LLM 비교 보고서 내부 함수
# ============================================

def _groq_headers() -> dict:
    """Groq API 요청 헤더를 반환합니다."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(".env에서 GROQ_API_KEY를 찾을 수 없습니다.")

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _safe_report_filename(report_filename: str) -> str:
    """보고서 파일명을 Markdown 확장자로 정규화합니다."""
    filename = os.path.basename(
        report_filename.strip() or "open_llm_battle_report.md"
    )

    if not filename.lower().endswith(".md"):
        filename += ".md"

    return filename


def _question_report_filename(question: str) -> str:
    """질문을 알아보기 쉬운 고유 Markdown 보고서 파일명으로 변환합니다."""
    title = re.sub(
        r"(?:쉽게|간단히|자세히|이해하기 쉽게|이해할 수 있게)\s*",
        "",
        question.strip(),
    )
    title = re.sub(
        r"(?:설명|작성|정리|추천|계산|비교|알려)해\s*줘[.!?]?$",
        "",
        title,
    ).strip()
    slug = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", title)
    slug = re.sub(r"_+", "_", slug).strip("_-")
    slug = slug[:24].rstrip("_-") or "open_llm"

    output_dir = (
        Path(__file__).resolve().parent
        / "workspace"
        / "OpenLLM_Agent"
    )
    base_name = f"LLM_{slug}"
    filename = f"{base_name}.md"
    sequence = 2

    while (output_dir / filename).exists():
        filename = f"{base_name}_{sequence}.md"
        sequence += 1

    return filename


def _clean_korean_answer(answer: str) -> str:
    """노출된 내부 추론을 제거하고 한국어 최종 답변만 허용합니다."""
    cleaned = re.sub(
        r"<think>.*?(?:</think>|$)",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    hangul_count = len(re.findall(r"[가-힣]", cleaned))
    latin_count = len(re.findall(r"[A-Za-z]", cleaned))
    language_char_count = hangul_count + latin_count
    korean_ratio = (
        hangul_count / language_char_count
        if language_char_count
        else 0
    )

    if hangul_count > 0 and korean_ratio >= 0.25:
        return cleaned

    return ""


def _model_family(model_id: str, owned_by: str = "") -> str:
    """Groq 모델 ID를 제공 기업 또는 모델 계열로 정규화합니다."""
    normalized = model_id.lower()

    family_rules = (
        (("openai/", "gpt-"), "OpenAI"),
        (("qwen/",), "Alibaba Qwen"),
        (("meta-llama/", "llama-"), "Meta Llama"),
        (("moonshotai/", "kimi"), "Moonshot Kimi"),
        (("google/", "gemma"), "Google Gemma"),
        (("minimax/", "minimax"), "MiniMax"),
        (("mistral",), "Mistral AI"),
        (("groq/",), "Groq"),
    )

    for prefixes, family in family_rules:
        if any(normalized.startswith(prefix) for prefix in prefixes):
            return family

    return owned_by.strip() or model_id.split("/", 1)[0]


def _get_supported_chat_models() -> list[dict]:
    """현재 Groq 계정에 노출된 채팅 모델을 조회합니다."""
    response = requests.get(
        f"{GROQ_API_URL}/models",
        headers=_groq_headers(),
        timeout=30,
    )
    response.raise_for_status()

    supported = []

    for model in response.json().get("data", []):
        model_id = str(model.get("id") or "").strip()

        if not model_id:
            continue

        if any(
            keyword in model_id.lower()
            for keyword in NON_CHAT_MODEL_KEYWORDS
        ):
            continue

        supported.append({
            "model_id": model_id,
            "family": _model_family(
                model_id,
                str(model.get("owned_by") or ""),
            ),
        })

    return supported


def _parse_report_request(request: str) -> tuple[str, str]:
    """단일 문자열에서 질문과 모델 선택 조건을 분리합니다."""
    question_match = re.search(
        r"(?:^|\n)\s*질문\s*:\s*(.+?)(?=\n\s*선택\s*:|$)",
        request,
        flags=re.DOTALL,
    )
    selection_match = re.search(
        r"(?:^|\n)\s*선택\s*:\s*(.+)$",
        request,
        flags=re.DOTALL,
    )

    if question_match:
        question = question_match.group(1).strip()
        selection = (
            selection_match.group(1).strip()
            if selection_match
            else "auto"
        )
        return question, selection

    quoted = re.search(
        r'["“](.+?)["”]',
        request,
        flags=re.DOTALL,
    )
    question = quoted.group(1).strip() if quoted else request.strip()

    explicit_ids = re.findall(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
        request,
    )
    if explicit_ids:
        return question, "models:" + ",".join(explicit_ids)

    count_match = re.search(r"([2-4])\s*개", request)
    if count_match:
        return question, f"random:{count_match.group(1)}"

    return question, "auto"


def _resolve_requested_models(
    selection: str,
    supported: list[dict],
) -> tuple[list[dict], list[str]]:
    """지정 모델을 해석하거나 서로 다른 모델 계열을 우선 선택합니다."""
    normalized_selection = selection.strip()

    if normalized_selection.lower().startswith("models:"):
        names = [
            name.strip()
            for name in normalized_selection
            .split(":", 1)[1]
            .split(",")
            if name.strip()
        ]

        if len(names) > MAX_BATTLE_MODELS:
            raise ValueError(
                f"한 번에 비교할 모델은 최대 "
                f"{MAX_BATTLE_MODELS}개입니다."
            )

        aliases = {
            "gpt": "OpenAI",
            "openai": "OpenAI",
            "qwen": "Alibaba Qwen",
            "알리바바": "Alibaba Qwen",
            "llama": "Meta Llama",
            "라마": "Meta Llama",
            "meta": "Meta Llama",
            "kimi": "Moonshot Kimi",
            "키미": "Moonshot Kimi",
            "gemma": "Google Gemma",
            "젬마": "Google Gemma",
            "google": "Google Gemma",
            "minimax": "MiniMax",
            "groq": "Groq",
        }

        selected = []
        missing = []

        for name in names:
            exact = next(
                (
                    model
                    for model in supported
                    if model["model_id"].lower() == name.lower()
                ),
                None,
            )

            family_name = aliases.get(name.lower())
            candidates = (
                [
                    model
                    for model in supported
                    if model["family"] == family_name
                ]
                if family_name
                else []
            )

            match = exact or (
                random.choice(candidates)
                if candidates
                else None
            )

            if match and match not in selected:
                selected.append(match)
            elif not match:
                missing.append(name)

        return selected, missing

    count_match = re.search(
        r"random\s*:\s*([2-4])",
        normalized_selection,
        flags=re.IGNORECASE,
    )
    requested_count = (
        int(count_match.group(1))
        if count_match
        else min(MAX_BATTLE_MODELS, len(supported))
    )

    # Compound는 도구 실행이 포함된 시스템이므로 일반 LLM 랜덤 비교에서
    # 제외합니다. 사용자가 Groq를 명시한 경우에는 위 분기에서 선택됩니다.
    random_candidates = [
        model
        for model in supported
        if not model["model_id"].lower().startswith("groq/compound")
    ]

    by_family: dict[str, list[dict]] = {}

    for model in random_candidates:
        by_family.setdefault(model["family"], []).append(model)

    families = list(by_family)
    random.shuffle(families)

    selected = [
        random.choice(by_family[family])
        for family in families[:requested_count]
    ]

    if len(selected) < requested_count:
        remaining = [
            model
            for model in supported
            if model not in selected
        ]
        random.shuffle(remaining)
        selected.extend(
            remaining[:requested_count - len(selected)]
        )

    return selected, []


# ============================================
# Open LLM 비교 보고서 도구
# ============================================

@tool(parse_docstring=True)
def compare_open_llm_models_and_save_report(
    question: str,
    selection: str = "auto",
    report_filename: str = "open_llm_battle_report.md",
) -> str:
    """Groq 모델을 실제 호출해 원문 답변을 Markdown 보고서로 저장합니다.

    Args:
        question: 모든 모델에 동일하게 전달할 비교 질문입니다.
        selection: random:2~4 또는 models:모델명,모델ID 형식입니다.
        report_filename: 저장할 Markdown 보고서 파일명입니다.

    Returns:
        보고서 절대 경로와 모델별 실행 요약을 JSON 문자열로 반환합니다.
    """
    try:
        if not question.strip():
            return "오류: 비교 질문이 비어 있습니다."

        supported = _get_supported_chat_models()
        if not supported:
            return (
                "오류: 현재 Groq 계정에서 사용할 수 있는 "
                "채팅 모델을 찾지 못했습니다."
            )

        selected, missing = _resolve_requested_models(
            selection,
            supported,
        )

        if not selected:
            return (
                "오류: 요청한 모델을 현재 Groq 계정에서 "
                f"찾지 못했습니다: {', '.join(missing)}"
            )

        results = []

        for target in selected:
            model_id = target["model_id"]
            started = time.perf_counter()

            try:
                request_body = {
                    "model": model_id,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "최종 답변은 반드시 한국어로만 작성하세요. "
                                "모델명, 코드, 고유명사를 제외한 설명은 "
                                "모두 한국어로 작성하고 내부 추론 과정이나 "
                                "<think> 태그는 출력하지 마세요. "
                                "답변은 한국어 500~700자 이내에서 "
                                "핵심 개념과 실행 구조가 완결되도록 작성하세요."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "다음 질문에 한국어로만 답변하세요."
                                f"\n\n질문: {question}"
                            ),
                        },
                    ],
                    "max_completion_tokens": 1024,
                    "temperature": 0.2,
                }

                if model_id.lower().startswith("openai/gpt-oss"):
                    request_body["reasoning_effort"] = "low"
                    request_body["reasoning_format"] = "hidden"
                elif model_id.lower().startswith("qwen/"):
                    request_body["reasoning_effort"] = "none"
                    request_body["reasoning_format"] = "hidden"

                response = requests.post(
                    f"{GROQ_API_URL}/chat/completions",
                    headers=_groq_headers(),
                    timeout=60,
                    json=request_body,
                )

                elapsed = round(
                    time.perf_counter() - started,
                    3,
                )

                if not response.ok:
                    results.append({
                        "model_id": model_id,
                        "family": target["family"],
                        "status": "failed",
                        "response_time_seconds": elapsed,
                        "finish_reason": None,
                        "answer": "",
                        "error": response.text[:500],
                    })
                    continue

                data = response.json()
                choices = data.get("choices") or []
                message = (
                    choices[0].get("message")
                    if choices
                    else {}
                )
                raw_answer = (message or {}).get("content") or ""
                answer = _clean_korean_answer(raw_answer)
                finish_reason = (
                    choices[0].get("finish_reason")
                    if choices
                    else None
                )
                is_truncated = finish_reason == "length"

                results.append({
                    "model_id": model_id,
                    "family": target["family"],
                    "status": (
                        "success"
                        if answer.strip() and not is_truncated
                        else "failed"
                    ),
                    "response_time_seconds": elapsed,
                    "finish_reason": finish_reason,
                    "answer": answer.strip(),
                    "error": (
                        ""
                        if answer.strip() and not is_truncated
                        else (
                            "출력 토큰 한도에 도달해 답변이 중간에 "
                            "종료되었습니다. 평가에서 제외합니다."
                            if is_truncated
                            else (
                                "모델이 저장 가능한 한국어 답변을 "
                                "생성하지 못했습니다."
                            )
                        )
                    ),
                })

            except Exception as error:
                results.append({
                    "model_id": model_id,
                    "family": target["family"],
                    "status": "failed",
                    "response_time_seconds": round(
                        time.perf_counter() - started,
                        3,
                    ),
                    "finish_reason": None,
                    "answer": "",
                    "error": str(error),
                })

        for model_name in missing:
            results.append({
                "model_id": model_name,
                "family": "요청 모델",
                "status": "failed",
                "response_time_seconds": 0,
                "finish_reason": None,
                "answer": "",
                "error": (
                    "현재 Groq 계정의 지원 모델 목록에서 "
                    "찾지 못했습니다."
                ),
            })

        output_dir = (
            Path(__file__).resolve().parent
            / "workspace"
            / "OpenLLM_Agent"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = (
            output_dir
            / _safe_report_filename(report_filename)
        )

        successful = [
            result
            for result in results
            if result["status"] == "success"
        ]

        fastest = (
            min(
                successful,
                key=lambda item: item[
                    "response_time_seconds"
                ],
            )
            if successful
            else None
        )

        longest = (
            max(
                successful,
                key=lambda item: len(item["answer"]),
            )
            if successful
            else None
        )

        lines = [
            "# Open LLM 모델 실제 답변 비교 보고서",
            "",
            "## 1. 비교 질문",
            "",
            question,
            "",
            "## 2. 실행 요약",
            "",
            (
                "| 기업/계열 | 모델 | 상태 | 응답 시간 | "
                "답변 길이 | 종료 사유 |"
            ),
            "|---|---|---:|---:|---:|---|",
        ]

        for result in results:
            answer_length = len(
                result.get("answer") or ""
            )
            lines.append(
                f"| {result['family']} "
                f"| `{result['model_id']}` "
                f"| {result['status']} "
                f"| {result['response_time_seconds']}초 "
                f"| {answer_length}자 "
                f"| {result.get('finish_reason') or '-'} |"
            )

        lines.extend([
            "",
            "## 3. 모델별 실제 답변 원문",
            "",
        ])

        for index, result in enumerate(results, 1):
            lines.extend([
                f"### 3-{index}. `{result['model_id']}`",
                "",
                f"- 기업/계열: **{result['family']}**",
                f"- 상태: **{result['status']}**",
                (
                    "- 응답 시간: "
                    f"**{result['response_time_seconds']}초**"
                ),
                "",
            ])

            if result["status"] == "success":
                lines.extend([
                    result["answer"],
                    "",
                ])
            else:
                if result.get("answer"):
                    lines.extend([
                        "> 아래 답변은 중간에 종료되어 평가에서 제외되었습니다.",
                        "",
                        result["answer"],
                        "",
                    ])
                lines.extend([
                    f"실패 원인: {result['error']}",
                    "",
                ])

        lines.extend([
            "## 4. 간단 평가",
            "",
            (
                "- 정상 응답 모델 수: "
                f"**{len(successful)} / {len(results)}**"
            ),
            (
                "- 가장 빠른 모델: "
                f"**`{fastest['model_id'] if fastest else '없음'}`**"
            ),
            (
                "- 가장 긴 답변 모델: "
                f"**`{longest['model_id'] if longest else '없음'}`**"
            ),
            "",
            (
                "> 이 보고서는 각 모델을 실제 Groq API로 호출한 "
                "결과를 저장한 것입니다. 답변의 사실 정확성은 "
                "별도 검증이 필요합니다."
            ),
            "",
        ])

        output_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        summary = {
            "status": "success",
            "report_path": str(output_path.resolve()),
            "success_count": len(successful),
            "failure_count": len(results) - len(successful),
            "models": [
                {
                    "family": result["family"],
                    "model_id": result["model_id"],
                    "status": result["status"],
                    "response_time_seconds": result[
                        "response_time_seconds"
                    ],
                    "answer_length": len(
                        result.get("answer") or ""
                    ),
                    "error": result.get("error") or "",
                }
                for result in results
            ],
            "unsupported_requested_models": missing,
        }

        return json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        return f"오류: {str(e)}"


@tool(parse_docstring=True)
def save_open_llm_report(request: str) -> str:
    """요청 조건에 맞는 Groq 모델들을 비교해 보고서를 저장합니다.

    Args:
        request: 질문과 선택 조건입니다.
            예:
            질문: 프롬프트 인젝션을 한 문장으로 설명해줘.
            선택: random:2

    Returns:
        저장된 Markdown 보고서 경로와 실행 요약을 반환합니다.
    """
    question, selection = _parse_report_request(request)

    return compare_open_llm_models_and_save_report.invoke({
        "question": question,
        "selection": selection,
        "report_filename": _question_report_filename(question),
    })


# ============================================
# 스킬 로드 도구
# ============================================

@tool(parse_docstring=True)
def load_skill(skill_name: str) -> str:
    """필요한 전문 Skill의 SKILL.md 전체 내용을 로드합니다.

    Progressive Disclosure 방식으로 시스템 프롬프트에 등록된 Skill 중
    현재 작업에 필요한 Skill만 선택하여 상세 실행 지침을 읽습니다.

    로드된 Skill은 단순 참고 자료가 아니라 작업 실행 지침입니다.
    Skill을 로드한 뒤에는 해당 Skill에 정의된 프로세스를 따라야 합니다.

    Args:
        skill_name: 로드할 Skill 폴더명입니다.
            예: langgraph-docs, minimalist-ui, open-llm-evaluator

    Returns:
        Skill의 전체 SKILL.md 내용 또는 오류 메시지를 반환합니다.
    """
    try:
        normalized_name = skill_name.strip()

        if not normalized_name:
            return "오류: Skill 이름이 비어 있습니다."

        # skills 디렉터리 밖의 파일을 읽지 못하도록 폴더명만 허용합니다.
        if (
            normalized_name != os.path.basename(normalized_name)
            or "/" in normalized_name
            or "\\" in normalized_name
            or normalized_name in {".", ".."}
        ):
            return f"오류: 올바르지 않은 Skill 이름입니다: {skill_name}"

        skills_dir = (
            Path(__file__).resolve().parent
            / "skills"
        )
        skill_path = (
            skills_dir
            / normalized_name
            / "SKILL.md"
        )

        if not skill_path.exists():
            available_skills = []

            if skills_dir.exists():
                available_skills = sorted(
                    item.name
                    for item in skills_dir.iterdir()
                    if item.is_dir()
                    and (item / "SKILL.md").exists()
                )

            error_message = (
                f"오류: '{normalized_name}' Skill을 "
                "찾을 수 없습니다."
            )

            if available_skills:
                skill_list = "\n".join(
                    f"- {name}"
                    for name in available_skills
                )
                error_message += (
                    "\n\n사용 가능한 Skills:\n"
                    f"{skill_list}"
                )
            else:
                error_message += (
                    "\n\n현재 등록된 Skill이 없습니다."
                )

            return error_message

        skill_content = skill_path.read_text(
            encoding="utf-8",
        )

        return (
            f"[Skill 로드 완료: {normalized_name}]\n\n"
            f"{'=' * 70}\n"
            f"{skill_content}\n"
            f"{'=' * 70}\n\n"
            "다음 단계: 위 Skill에 정의된 프로세스를 "
            "순서대로 따라 실행하세요."
        )

    except PermissionError:
        return (
            "오류: Skill 파일에 대한 읽기 권한이 없습니다: "
            f"{skill_name}"
        )
    except Exception as e:
        return f"오류: Skill 로드 중 문제가 발생했습니다: {str(e)}"


# ============================================
# LangGraph Agent 등록 도구 목록
# ============================================

TOOLS = [
    read_file,
    write_file,
    save_open_llm_report,
    load_skill,
]
