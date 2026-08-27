"""Document workspace agent middleware."""

import asyncio
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentState, after_model, before_agent, wrap_tool_call
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime


DOCUMENT_EXTENSIONS = {".md", ".csv", ".txt"}
MAX_INDEX_FILES = 40
MAX_INDEX_CHARS = 2500
SENSITIVE_PATH_PARTS = {".env", ".git", ".venv", "venv", "__pycache__"}
PII_PATTERNS = {
    "주민등록번호": re.compile(r"\d{6}-\d{7}"),
    "전화번호": re.compile(r"01[0-9]-\d{3,4}-\d{4}"),
    "이메일": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "신용카드": re.compile(r"\d{4}-\d{4}-\d{4}-\d{4}"),
}
INJECTION_PATTERNS = (
    "system prompt",
    "시스템 프롬프트",
    "이전 지시",
    "ignore previous",
    "developer mode",
    "관리자 모드",
)


def _last_user_text(state: AgentState) -> str:
    messages = state.get("messages") or []
    user_messages = [
        message
        for message in messages
        if getattr(message, "type", None) == "human"
    ]
    return str(user_messages[-1].content) if user_messages else ""


def _is_sensitive_path(file_path: str) -> bool:
    parts = {part.lower() for part in Path(file_path).parts}
    return any(part in parts for part in SENSITIVE_PATH_PARTS)


def _mask_pii(content: str) -> str:
    masked = content
    masked = PII_PATTERNS["주민등록번호"].sub("******-*******", masked)
    masked = PII_PATTERNS["전화번호"].sub("***-****-****", masked)
    masked = PII_PATTERNS["이메일"].sub("***@***.***", masked)
    masked = PII_PATTERNS["신용카드"].sub("****-****-****-****", masked)
    return masked


def scan_workspace() -> tuple[str, list[str]]:
    """Workspace 문서 파일 목록을 동기 방식으로 스캔합니다."""
    cwd = os.getcwd()
    file_list = []

    for root, dirs, files in os.walk(cwd):
        dirs[:] = [
            directory
            for directory in dirs
            if not directory.startswith(".")
            and directory not in {"node_modules", "venv", ".venv", ".cache", "backup"}
        ]

        level = root.replace(cwd, "").count(os.sep)
        if level > 3:
            dirs[:] = []
            continue

        for file_name in files:
            if file_name.startswith("."):
                continue

            extension = os.path.splitext(file_name)[1].lower()
            if extension in DOCUMENT_EXTENSIONS:
                file_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(file_path, cwd)
                file_list.append(f"- {rel_path}")

    return cwd, sorted(file_list)


def create_backup(file_path: str) -> Path | None:
    """파일 변경 전에 backup 폴더에 원본 복사본을 생성합니다."""
    if not file_path or not os.path.exists(file_path):
        return None

    source = Path(file_path)
    backup_dir = Path("backup")
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{source.stem}_{timestamp}{source.suffix}"
    shutil.copy2(source, backup_path)
    return backup_path


@before_agent(can_jump_to=["end"])
def request_safety_middleware(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """에이전트 시작 전에 빈 입력, 프롬프트 인젝션, 민감정보 입력을 차단합니다."""
    user_input = _last_user_text(state).strip()
    if not user_input:
        return {
            "messages": [AIMessage(content="요청 내용을 입력해주세요.")],
            "jump_to": "end",
        }

    lowered = user_input.lower()
    if any(pattern in lowered for pattern in INJECTION_PATTERNS):
        return {
            "messages": [AIMessage(content="죄송합니다. 보안정책에 의해 답변드릴 수 없습니다.")],
            "jump_to": "end",
        }

    detected = [
        pii_type
        for pii_type, pattern in PII_PATTERNS.items()
        if pattern.search(user_input)
    ]
    if detected:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "입력에 개인정보가 포함되어 있습니다. "
                        f"제거 후 다시 요청해주세요. 감지 항목: {', '.join(detected)}"
                    )
                )
            ],
            "jump_to": "end",
        }

    return None


@before_agent
async def workspace_index_middleware(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """에이전트 시작 시 Workspace 문서 파일 목록을 시스템 메시지로 추가합니다."""
    cwd, file_list = await asyncio.to_thread(scan_workspace)
    visible_files = file_list[:MAX_INDEX_FILES]
    omitted_count = max(len(file_list) - len(visible_files), 0)
    index_text = "\n".join(visible_files) if visible_files else "- 문서 파일 없음"
    if len(index_text) > MAX_INDEX_CHARS:
        index_text = index_text[:MAX_INDEX_CHARS] + "\n- ...생략"
    if omitted_count:
        index_text += f"\n- ...외 {omitted_count}개 생략"

    return {
        "messages": [
            SystemMessage(
                content=(
                    "[Workspace Index]\n"
                    "Workspace 문서 인덱스\n"
                    f"문서 파일 수: {len(file_list)}\n"
                    f"표시 파일 수: {len(visible_files)}\n"
                    f"{index_text}\n\n"
                    "문서 요청은 이 목록을 먼저 참고하세요."
                )
            )
        ]
    }


@after_model
def response_pii_masking_middleware(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """모델 응답에 개인정보 형식이 포함되면 출력 직전에 마스킹합니다."""
    messages = state.get("messages") or []
    if not messages:
        return None

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not isinstance(last_message.content, str):
        return None

    masked_content = _mask_pii(last_message.content)
    if masked_content != last_message.content:
        last_message.content = masked_content

    return None


@wrap_tool_call
async def auto_backup_middleware(request, handler):
    """파일 변경 도구 실행 전 백업을 만들고 민감 경로 변경은 차단합니다."""
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})
    file_path = tool_args.get("file_path")

    if file_path and _is_sensitive_path(file_path):
        return ToolMessage(
            content=f"오류: 보안상 민감 경로는 도구로 변경할 수 없습니다: {file_path}",
            tool_call_id=request.tool_call["id"],
        )

    if tool_name == "delete_file":
        return ToolMessage(
            content="오류: delete_file은 미들웨어 정책에 의해 차단되었습니다.",
            tool_call_id=request.tool_call["id"],
        )

    if tool_name in {"write_file", "edit_file"} and file_path:
        try:
            backup_path = await asyncio.to_thread(create_backup, file_path)
            if backup_path:
                print(f"[Auto Backup] 백업 생성: {backup_path}")
        except Exception as error:
            print(f"[Auto Backup] 백업 실패: {error}")

    return await handler(request)
