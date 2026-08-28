"""Open LLM comparison agent middleware."""

import asyncio
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    after_model,
    before_agent,
    wrap_tool_call,
)
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.runtime import Runtime


# ============================================
# 안전 Middleware 설정
# ============================================

DOCUMENT_EXTENSIONS = {".md", ".csv", ".txt"}
MAX_INDEX_FILES = 40
MAX_INDEX_CHARS = 2500

SENSITIVE_PATH_PARTS = {
    ".env",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
}

PII_PATTERNS = {
    "주민등록번호": re.compile(r"\d{6}-\d{7}"),
    "전화번호": re.compile(r"01[0-9]-\d{3,4}-\d{4}"),
    "이메일": re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    ),
    "신용카드": re.compile(
        r"\d{4}-\d{4}-\d{4}-\d{4}"
    ),
}

INJECTION_PATTERNS = (
    "system prompt",
    "시스템 프롬프트",
    "이전 지시",
    "ignore previous",
    "developer mode",
    "관리자 모드",
)


# ============================================
# 안전 Middleware 내부 함수
# ============================================

def _last_user_text(state: AgentState) -> str:
    """상태에서 가장 최근 사용자 메시지를 반환합니다."""
    messages = state.get("messages") or []

    user_messages = [
        message
        for message in messages
        if getattr(message, "type", None) == "human"
    ]

    if not user_messages:
        return ""

    return str(user_messages[-1].content)


def _is_sensitive_path(file_path: str) -> bool:
    """파일 경로가 민감 경로인지 검사합니다."""
    parts = {
        part.lower()
        for part in Path(file_path).parts
    }

    return any(
        part in parts
        for part in SENSITIVE_PATH_PARTS
    )


def _mask_pii(content: str) -> str:
    """텍스트에 포함된 개인정보 형식을 마스킹합니다."""
    masked = content

    masked = PII_PATTERNS["주민등록번호"].sub(
        "******-*******",
        masked,
    )
    masked = PII_PATTERNS["전화번호"].sub(
        "***-****-****",
        masked,
    )
    masked = PII_PATTERNS["이메일"].sub(
        "***@***.***",
        masked,
    )
    masked = PII_PATTERNS["신용카드"].sub(
        "****-****-****-****",
        masked,
    )

    return masked


def scan_workspace() -> tuple[str, list[str]]:
    """Workspace의 문서 파일 목록을 동기 방식으로 스캔합니다."""
    cwd = os.getcwd()
    file_list = []

    for root, dirs, files in os.walk(cwd):
        dirs[:] = [
            directory
            for directory in dirs
            if not directory.startswith(".")
            and directory
            not in {
                "node_modules",
                "venv",
                ".venv",
                ".cache",
                "backup",
                "__pycache__",
            }
        ]

        level = root.replace(cwd, "").count(os.sep)

        if level > 3:
            dirs[:] = []
            continue

        for file_name in files:
            if file_name.startswith("."):
                continue

            extension = os.path.splitext(
                file_name
            )[1].lower()

            if extension not in DOCUMENT_EXTENSIONS:
                continue

            file_path = os.path.join(
                root,
                file_name,
            )
            rel_path = os.path.relpath(
                file_path,
                cwd,
            )
            file_list.append(f"- {rel_path}")

    return cwd, sorted(file_list)


def create_backup(file_path: str) -> Path | None:
    """파일 변경 전에 backup 폴더에 원본 복사본을 생성합니다."""
    if not file_path:
        return None

    source = Path(file_path)

    if not source.exists() or not source.is_file():
        return None

    backup_dir = Path("backup")
    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    backup_path = (
        backup_dir
        / f"{source.stem}_{timestamp}{source.suffix}"
    )

    shutil.copy2(
        source,
        backup_path,
    )

    return backup_path


# ============================================
# 요청 안전성 Middleware
# ============================================

@before_agent(can_jump_to=["end"])
def request_safety_middleware(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """에이전트 실행 전에 위험하거나 민감한 입력을 차단합니다."""
    user_input = _last_user_text(state).strip()

    if not user_input:
        return {
            "messages": [
                AIMessage(
                    content="요청 내용을 입력해주세요."
                )
            ],
            "jump_to": "end",
        }

    lowered = user_input.lower()

    if any(
        pattern in lowered
        for pattern in INJECTION_PATTERNS
    ):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "죄송합니다. 보안정책에 의해 "
                        "답변드릴 수 없습니다."
                    )
                )
            ],
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
                        "개인정보를 제거한 후 다시 요청해주세요. "
                        f"감지 항목: {', '.join(detected)}"
                    )
                )
            ],
            "jump_to": "end",
        }

    return None


# ============================================
# Workspace 인덱스 Middleware
# ============================================

@before_agent
async def workspace_index_middleware(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """에이전트 실행 시 Workspace 문서 목록을 추가합니다."""
    _, file_list = await asyncio.to_thread(
        scan_workspace
    )

    visible_files = file_list[:MAX_INDEX_FILES]
    omitted_count = max(
        len(file_list) - len(visible_files),
        0,
    )

    index_text = (
        "\n".join(visible_files)
        if visible_files
        else "- 문서 파일 없음"
    )

    if len(index_text) > MAX_INDEX_CHARS:
        index_text = (
            index_text[:MAX_INDEX_CHARS]
            + "\n- ...생략"
        )

    if omitted_count:
        index_text += (
            f"\n- ...외 {omitted_count}개 생략"
        )

    return {
        "messages": [
            SystemMessage(
                content=(
                    "[Workspace Index]\n"
                    "Workspace 문서 인덱스\n"
                    f"문서 파일 수: {len(file_list)}\n"
                    f"표시 파일 수: {len(visible_files)}\n"
                    f"{index_text}\n\n"
                    "문서 관련 요청은 이 목록을 "
                    "먼저 참고하세요."
                )
            )
        ]
    }


# ============================================
# 응답 개인정보 마스킹 Middleware
# ============================================

@after_model
def response_pii_masking_middleware(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """모델 응답에 포함된 개인정보 형식을 마스킹합니다."""
    messages = state.get("messages") or []

    if not messages:
        return None

    last_message = messages[-1]

    if not isinstance(last_message, AIMessage):
        return None

    if not isinstance(last_message.content, str):
        return None

    masked_content = _mask_pii(
        last_message.content
    )

    if masked_content != last_message.content:
        last_message.content = masked_content

    return None


# ============================================
# 파일 자동 백업 Middleware
# ============================================

@wrap_tool_call
async def auto_backup_middleware(
    request,
    handler,
):
    """파일 변경 전에 백업하고 민감 경로 변경을 차단합니다."""
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get(
        "args",
        {},
    )
    file_path = tool_args.get("file_path")

    if file_path and _is_sensitive_path(file_path):
        return ToolMessage(
            content=(
                "오류: 보안상 민감 경로는 "
                f"도구로 변경할 수 없습니다: {file_path}"
            ),
            tool_call_id=request.tool_call["id"],
        )

    if tool_name == "delete_file":
        return ToolMessage(
            content=(
                "오류: delete_file은 미들웨어 "
                "정책에 의해 차단되었습니다."
            ),
            tool_call_id=request.tool_call["id"],
        )

    if (
        tool_name in {"write_file", "edit_file"}
        and file_path
    ):
        try:
            backup_path = await asyncio.to_thread(
                create_backup,
                file_path,
            )

            if backup_path:
                print(
                    "[Auto Backup] "
                    f"백업 생성: {backup_path}"
                )

        except Exception as error:
            print(
                "[Auto Backup] "
                f"백업 실패: {error}"
            )

    return await handler(request)


# ============================================
# Skill 메타데이터 파싱
# ============================================

def parse_skill_metadata() -> list[dict[str, str]]:
    """skills 폴더의 SKILL.md에서 이름과 설명을 읽습니다."""
    skills = []

    skills_dir = (
        Path(__file__).resolve().parent
        / "skills"
    )

    if not skills_dir.exists():
        return skills

    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue

        skill_file = skill_dir / "SKILL.md"

        if not skill_file.exists():
            continue

        try:
            content = skill_file.read_text(
                encoding="utf-8"
            )

            frontmatter_match = re.match(
                r"^---\s*\n(.*?)\n---",
                content,
                flags=re.DOTALL,
            )

            if not frontmatter_match:
                skills.append({
                    "name": skill_dir.name,
                    "description": (
                        f"{skill_dir.name} Skill"
                    ),
                })
                continue

            frontmatter = frontmatter_match.group(1)

            name_match = re.search(
                r"^name:\s*(.+)$",
                frontmatter,
                flags=re.MULTILINE,
            )
            description_match = re.search(
                r"^description:\s*(.+)$",
                frontmatter,
                flags=re.MULTILINE,
            )

            name = (
                name_match.group(1).strip()
                if name_match
                else skill_dir.name
            )
            description = (
                description_match.group(1).strip()
                if description_match
                else "Skill description not provided."
            )

            skills.append({
                "name": name,
                "description": description,
            })

        except Exception as error:
            print(
                f"[SkillMiddleware] "
                f"{skill_dir.name} 파싱 실패: {error}"
            )

    return skills


# ============================================
# SkillMiddleware
# ============================================

class SkillMiddleware(AgentMiddleware):
    """사용 가능한 Skill 목록을 모델 시스템 프롬프트에 추가합니다."""

    def __init__(self):
        """현재 등록된 Skill 목록을 읽어 프롬프트를 생성합니다."""
        skills = parse_skill_metadata()

        if skills:
            self.skills_prompt = "\n".join(
                (
                    f"- **{skill['name']}**: "
                    f"{skill['description']}"
                )
                for skill in skills
            )
        else:
            self.skills_prompt = (
                "현재 등록된 Skill이 없습니다."
            )

    def _skill_addendum(self) -> str:
        """시스템 프롬프트에 추가할 Skill 지침을 반환합니다."""
        return (
            "\n\n"
            "## 사용 가능한 Skills (Available Skills)\n\n"
            f"{self.skills_prompt}\n\n"
            "### Skill 사용 규칙\n"
            "- 사용자 요청과 관련된 Skill이 있으면 "
            "반드시 `load_skill` 도구로 상세 지침을 로드하세요.\n"
            "- Skill 이름은 위 목록에 표시된 이름을 "
            "정확히 사용하세요.\n"
            "- Skill을 로드한 뒤에는 해당 Skill의 "
            "프로세스를 순서대로 따르세요.\n"
            "- Open LLM 모델 비교, 평가, 순위 또는 "
            "Markdown 보고서 요청에는 "
            "`advanced-evaluation` Skill을 우선 로드하세요.\n"
            "- 관련 없는 Skill은 로드하지 마세요.\n"
        )

    def _add_skill_prompt(
        self,
        request: ModelRequest,
    ) -> ModelRequest:
        """ModelRequest의 시스템 메시지에 Skill 지침을 추가합니다."""
        current_blocks = list(
            request.system_message.content_blocks
        )

        current_blocks.append({
            "type": "text",
            "text": self._skill_addendum(),
        })

        new_system_message = SystemMessage(
            content=current_blocks
        )

        return request.override(
            system_message=new_system_message
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[
            [ModelRequest],
            ModelResponse,
        ],
    ) -> ModelResponse:
        """동기 모델 호출에 Skill 목록을 주입합니다."""
        modified_request = self._add_skill_prompt(
            request
        )

        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[
            [ModelRequest],
            Awaitable[ModelResponse],
        ],
    ) -> ModelResponse:
        """비동기 LangGraph 호출에 Skill 목록을 주입합니다."""
        modified_request = self._add_skill_prompt(
            request
        )

        return await handler(modified_request)
