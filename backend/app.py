"""
知屿 ZhiYu · 多模态教学智能体 —— 后端服务
==========================================
作为前端网页的"大脑"，安全中转调用 DeepSeek 大模型（OpenAI 兼容接口）。

启动方式（两种任选其一）：
    1) uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    2) python app.py

环境变量（API Key 绝不写死在代码里，全部从环境读取）：
    DEEPSEEK_API_KEY   必填，DeepSeek 开放平台的 API Key
    DEEPSEEK_BASE_URL  可选，默认 https://api.deepseek.com/v1
    DEEPSEEK_MODEL     可选，默认 deepseek-chat
    ALLOWED_ORIGINS    可选，逗号分隔的跨域白名单，默认 *（全部放行，生产请收紧）
"""

import os
import json
import base64
import re
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# --------------------------- 配置 ---------------------------
# 从环境变量读取密钥，绝不在代码中写死
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("zhiyu")

# httpx 客户端（复用连接，提升性能）
_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    """获取全局复用的异步 HTTP 客户端。"""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=60.0)
    return _client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：退出时关闭 HTTP 客户端。"""
    yield
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


app = FastAPI(
    title="知屿 ZhiYu · 多模态教学智能体后端",
    version="1.0.0",
    lifespan=lifespan,
)

# --------------------------- CORS ---------------------------
# 允许前端跨域访问，方便本地联调与前后端分离部署
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # 生产环境建议配置为具体前端域名
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------- 数据模型 ---------------------------
class ChatMessage(BaseModel):
    role: str = Field(..., description="角色：system / user / assistant")
    content: str = Field(..., description="消息文本内容")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="完整对话消息列表")


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="图片的 base64 编码（可选）")
    pdf_text: Optional[str] = Field(None, description="PDF 已提取出的纯文本（可选）")


# --------------------------- 工具函数 ---------------------------
def check_api_key() -> None:
    """校验 API Key 是否已配置，未配置时返回友好提示（而非 500 裸错）。"""
    if not DEEPSEEK_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="服务端未配置 DEEPSEEK_API_KEY，请联系管理员在环境变量中设置。",
        )


def extract_json(text: str) -> Dict[str, Any]:
    """稳健地从模型返回文本中提取 JSON 对象（兼容 ```json 代码块包裹）。"""
    text = text.strip()
    # 1) 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2) 提取 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # 3) 提取第一个平衡的 { ... } 块
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("模型返回内容不是合法的 JSON")


async def call_deepseek(messages: List[Dict[str, str]], temperature: float = 0.5) -> str:
    """调用 DeepSeek 的 OpenAI 兼容 chat/completions 接口，返回回答文本。

    对网络错误、超时、DeepSeek 报错均抛出带友好信息的 HTTPException，
    绝不向用户暴露 500 裸错。
    """
    check_api_key()
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }

    try:
        resp = await get_client().post(url, json=payload, headers=headers)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="调用 DeepSeek 超时，请稍后重试。")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="无法连接 DeepSeek 服务，请检查网络或稍后再试。")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"请求 DeepSeek 失败：{e}")

    if resp.status_code != 200:
        # 透传 DeepSeek 的错误信息，但包装为友好的 502
        try:
            err_msg = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            err_msg = resp.text
        logger.error("DeepSeek 返回 %s: %s", resp.status_code, err_msg)
        raise HTTPException(status_code=502, detail=f"DeepSeek 调用失败：{err_msg}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=502, detail="DeepSeek 返回格式异常，缺少 choices。")


def ocr_extract(image_base64: str) -> Optional[str]:
    """OCR 文字提取（占位实现）。

    真实项目可在此接入：
        - 云服务：腾讯云 OCR / 百度 OCR / 阿里云 OCR
        - 本地：PaddleOCR
    当前仅校验 base64 合法性，并返回 None 表示「OCR 待接入」。
    """
    try:
        base64.b64decode(image_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="image_base64 不是合法的 base64 编码。")
    # TODO: 接入真实 OCR 后，把识别出的文字 return 出来
    return None


def _ensure_list(v: Any) -> List[Any]:
    """确保字段是列表，防御模型输出异常。"""
    return v if isinstance(v, list) else []


# --------------------------- 提示词 ---------------------------
SYSTEM_PROMPT = """你是一位专业的教育知识结构化专家。你的任务是从给定的教学材料中，抽取结构化的知识点图谱。

输出必须是严格的 JSON 对象（不要输出 markdown 代码块、不要任何解释文字），格式如下：
{
  "subject": "材料主题（例如：高等数学 · 微积分）",
  "nodes": [
    {
      "id": "英文小写唯一标识（例如 derivative）",
      "name": "知识点名称",
      "difficulty": "easy | medium | hard",
      "importance": 0.0 到 1.0 之间的小数（越重要数值越大）,
      "summary": "一句话概述",
      "points": ["核心要点1", "核心要点2"],
      "example": "一个简短例证"
    }
  ],
  "edges": [
    {"source": "前置知识点 id", "target": "后置知识点 id", "relation": "先修"}
  ],
  "learning_path": ["按由易到难递进顺序排列的节点 id 列表"]
}

要求：
1. 知识点数量控制在 5 ~ 15 个。
2. edges 表达知识点之间的「先修 / 依赖」关系。
3. learning_path 必须是一条从入门到进阶的递进路径。
4. 只输出 JSON，不要有任何多余内容。"""


def build_knowledge_prompt(text: str) -> str:
    """构造知识点结构化抽取的用户提示词。"""
    # 控制长度，避免超出模型上下文
    clipped = text if len(text) <= 12000 else text[:12000]
    return f"请从以下教学材料中抽取知识点图谱：\n\n{clipped}\n\n请严格按系统要求输出 JSON。"


# --------------------------- 静态目录（前后端同源部署） ---------------------------
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)  # 确保目录存在（即使为空也能启动）


# --------------------------- 接口 ---------------------------
@app.get("/health")
def health():
    """健康检查，方便部署后确认服务在线。"""
    return {
        "service": "知屿 ZhiYu · 多模态教学智能体后端",
        "status": "ok",
        "model": DEEPSEEK_MODEL,
        "api_key_configured": bool(DEEPSEEK_API_KEY),
    }


@app.get("/")
def index():
    """根路径直接返回前端首页，评审点击公网 URL 即可使用。"""
    return FileResponse(STATIC_DIR / "index.html")


# 挂载静态资源（前端 HTML/CSS/JS 等），供 /static/... 访问
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """接收前端消息列表，转发到 DeepSeek，返回 { reply: 回答 }。"""
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages 不能为空。")
    # 手动构造 dict，兼容 pydantic v1/v2
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    reply = await call_deepseek(messages, temperature=0.5)
    return {"reply": reply}


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """图片 / PDF → 提取文字 → 结构化抽取知识点，返回知识点图谱 JSON。"""
    # 1) 提取文字
    text: Optional[str] = None
    if req.pdf_text and req.pdf_text.strip():
        text = req.pdf_text.strip()  # PDF 已带文字，直接使用
    elif req.image_base64:
        text = ocr_extract(req.image_base64)  # 图片走 OCR
        if text is None:
            # OCR 尚未接入，返回占位信息（不报错，让前端可提示用户）
            return {
                "ocr_status": "pending",
                "message": "OCR 待接入",
                "subject": "",
                "nodes": [],
                "edges": [],
                "learning_path": [],
            }
    else:
        raise HTTPException(status_code=400, detail="请提供 image_base64 或 pdf_text 之一。")

    # 2) 调用 DeepSeek 做知识点结构化抽取
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_knowledge_prompt(text)},
    ]
    raw = await call_deepseek(messages, temperature=0.2)

    # 3) 解析 JSON；解析失败时返回原始文本 + 提示（不抛 500）
    try:
        data = extract_json(raw)
    except ValueError:
        logger.warning("知识点抽取结果无法解析为 JSON")
        return {
            "ocr_status": "ok",
            "parse_error": True,
            "message": "知识点抽取完成，但模型返回无法解析为 JSON。",
            "raw": raw,
            "subject": "",
            "nodes": [],
            "edges": [],
            "learning_path": [],
        }

    # 4) 规范化返回，字段与前端 knowledge_points 对齐
    return {
        "ocr_status": "ok",
        "subject": data.get("subject", ""),
        "nodes": _ensure_list(data.get("nodes")),
        "edges": _ensure_list(data.get("edges")),
        "learning_path": _ensure_list(data.get("learning_path")),
    }


# --------------------------- 兜底异常处理 ---------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    """捕获未预期异常，返回友好 JSON（HTTPException 仍由框架默认处理）。"""
    logger.exception("未预期异常：%s", exc)
    return JSONResponse(status_code=500, content={"detail": "服务器内部错误，请稍后重试。"})


# 便于 `python app.py` 直接启动
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
