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

# --------------------------- 加载 .env ---------------------------
# 本地开发时从 .env 读取密钥；平台环境变量优先级更高（不覆盖已存在的值）。
# .env 已被 .gitignore 忽略，绝不提交到代码仓库。
def _load_dotenv() -> None:
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

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
    pdf_base64: Optional[str] = Field(None, description="PDF 文件的 base64 编码（可选）")
    pdf_text: Optional[str] = Field(None, description="已提取出的纯文本（可选，txt/md 或已解析的 PDF）")


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


def extract_pdf_text(pdf_base64: str) -> str:
    """用 PyMuPDF 从 PDF base64 提取文字。

    若 PDF 无文本层（如扫描版 / 图片型 PDF），自动降级为「逐页渲染成图片 -> OCR」
    （离线 RapidOCR 优先，腾讯云 OCR 兜底，复用 ocr_extract 的双链路）。
    """
    # 先校验 base64 合法性
    try:
        pdf_bytes = base64.b64decode(pdf_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="pdf_base64 不是合法的 base64 编码。")
    # 再检查 PDF 解析库
    try:
        import pymupdf as fitz  # PyMuPDF（新版推荐）
    except ImportError:
        try:
            import fitz  # 旧版兼容
        except ImportError:
            raise HTTPException(status_code=503, detail="未安装 PDF 解析库 PyMuPDF，请 pip install pymupdf。")
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"PDF 解析失败：{e}")

    # 有文本层：直接返回
    if text.strip():
        doc.close()
        return text

    # ---- 无文本层：扫描版 / 图片型 PDF → 提取各页内嵌图（原分辨率）后走 OCR ----
    logger.warning("PDF 无文本层（%d 页），提取内嵌图后逐页 OCR …", doc.page_count)
    ocr_pages = []
    try:
        for i, page in enumerate(doc):
            page_text = _ocr_pdf_page(doc, page)
            if page_text:
                ocr_pages.append(f"【第 {i + 1} 页】\n{page_text}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("PDF 逐页 OCR 中途失败：%s", e)
    finally:
        doc.close()

    if not ocr_pages:
        # 文本层与 OCR 均未拿到内容（可能 OCR 未接入，或扫描质量过低）
        raise HTTPException(
            status_code=422,
            detail="PDF 未提取到文字且自动 OCR 未识别到内容（扫描质量过低或 OCR 未配置腾讯云凭证）。",
        )
    return "\n\n".join(ocr_pages)


def _ocr_pdf_page(doc, page) -> str:
    """对单个 PDF 页做 OCR：依次识别页内“内容级”嵌入图（原分辨率），直到拿到文字。

    - 用 page.get_images(full=True) 拿到本页引用的所有图 (xref, width, height)；
    - 按 xref 去重（同一张背景/水印图在相邻页会重复出现，避免重复识别）；
    - 按像素面积从大到小尝试识别，跳过明显是背景/小位图的图；
    - 单页最多扫描 4 张大图，成本与耗时上限可控。
    """
    try:
        imgs = page.get_images(full=True)
    except Exception:
        imgs = []
    if not imgs:
        return ""

    # 收集 (xref, w, h)，按 xref 去重（同 xref 只取一次，尺寸以引用处最大为准）
    seen: dict = {}
    for it in imgs:
        xref = it[0]
        w, h = it[2], it[3]
        prev = seen.get(xref)
        if prev is None or w * h > prev[0] * prev[1]:
            seen[xref] = (w, h)
    if not seen:
        return ""

    # 页面面积（pt），作为内容图阈值参考（与像素面积做相对比较）
    pr = page.rect
    page_area = pr.width * pr.height

    # 面积降序（内容大图优先被识别，水印/小图靠后）
    ordered = sorted(seen.items(), key=lambda kv: kv[1][0] * kv[1][1], reverse=True)

    collected = []
    for xref, (w, h) in ordered[:4]:
        # 小于页面面积 80%（像素级）的图视为背景/装饰，跳过
        if w * h < page_area * 0.8:
            continue
        try:
            raw = doc.extract_image(xref)
            img_bytes = raw.get("image")
        except Exception:
            continue
        if not img_bytes:
            continue
        page_text = ocr_extract(base64.b64encode(img_bytes).decode())
        if page_text:
            collected.append(page_text)
            continue  # 已拿到本页文字，不再识别更多图（控制成本）
    return ("\n".join(collected)).strip()


def ocr_extract(image_base64: str) -> Optional[str]:
    """OCR 文字提取 —— 本地离线 OCR 优先，腾讯云 OCR 兜底。

    优先级：
      1. RapidOCR（onnxruntime，本地离线、免费、无需账号）—— 中文识别效果好；
      2. 腾讯云「通用印刷体识别」—— 当本地离线 OCR 识别不出文字 / 未安装 / 出错时兜底。

    兜底条件（任一命中即降级到腾讯云）：
      - RapidOCR 返回空文本（识别不出字）；
      - RapidOCR 依赖缺失（ImportError）；
      - RapidOCR 运行异常（非 4xx/5xx 编码类错误）。

    若两条链路都不可用/无凭证，则返回 None（前端提示「OCR 待接入」，不报错）。
    """
    # 校验 base64 合法性（两条链路都不需要重复校验）
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="image_base64 不是合法的 base64 编码。")

    # ---- 第一优先：本地离线 OCR（RapidOCR）----
    try:
        import numpy as np
        import cv2
        from rapidocr_onnxruntime import RapidOCR

        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            # 无效图片：本地与云端都无法解码，直接拒绝，不浪费云端调用
            raise HTTPException(
                status_code=422, detail="无法解码图片，请上传有效的 JPG/PNG 图片。"
            )

        engine = RapidOCR()
        result, _ = engine(img)

        if not result:
            # 离线识别不出字（可能是特殊字体 / 复杂背景）→ 交给腾讯云兜底
            logger.info("RapidOCR 未识别到文字，降级到腾讯云 OCR 兜底。")
            return _tencent_fallback(image_bytes, image_base64)

        # result 结构：[ [box, text, score], ... ]，取每个元素的 text（第 2 项）
        text = "\n".join(item[1] for item in result).strip()
        if not text:
            # 识别出空串 → 仍交给腾讯云兜底
            logger.info("RapidOCR 识别结果为空，降级到腾讯云 OCR 兜底。")
            return _tencent_fallback(image_bytes, image_base64)
        return text

    except ImportError:
        # 未安装 RapidOCR 相关依赖 → 腾讯云兜底
        logger.warning("未安装 RapidOCR，尝试腾讯云 OCR 兜底…")
        return _tencent_fallback(image_bytes, image_base64)
    except HTTPException:
        raise
    except Exception as e:
        # 其它非预期错误（模型加载失败等）→ 腾讯云兜底，而非直接失败
        logger.warning("RapidOCR 运行异常（%s），尝试腾讯云 OCR 兜底…", e)
        return _tencent_fallback(image_bytes, image_base64)


def _tencent_fallback(image_bytes: bytes, image_base64: str) -> Optional[str]:
    """腾讯云 OCR 兜底：未知时返回 None（前端提示「OCR 待接入」）。"""
    if not image_bytes:
        raise HTTPException(status_code=422, detail="图片内容为空。")
    try:
        # 延迟导入：避免 SDK 未安装时顶部 import 直接崩溃
        from ocr_tencent import extract_text_from_image_base64
        return extract_text_from_image_base64(image_base64)
    except (ImportError, ModuleNotFoundError):
        logger.warning("未安装 tencentcloud-sdk-python-ocr，OCR 兜底不可用。")
        return None
    except RuntimeError as e:
        # 未配置腾讯云凭证 → 明确提示
        logger.warning("腾讯云 OCR 兜底不可用：%s", e)
        return None
    except Exception as e:
        # 图内没有可识别文字（TencentCloudSDKException FailedOperation.ImageNoText）→
        # 视为“空结果”而非错误，交给上层决定（PDF 逐页时跳过该页）。
        msg = str(e)
        if "ImageNoText" in msg or "未检测到文本" in msg:
            logger.info("腾讯云 OCR 判定图内无文本，按空结果处理。")
            return ""
        logger.error("腾讯云 OCR 兜底识别失败：%s", e)
        raise HTTPException(status_code=502, detail=f"OCR 识别失败：{e}")


def _ensure_list(v: Any) -> List[Any]:
    """确保字段是列表，防御模型输出异常。"""
    return v if isinstance(v, list) else []


# --------------------------- 提示词 ---------------------------
SYSTEM_PROMPT = """你是一位专业的教育知识结构化专家。你的任务是从给定的教学材料中，抽取结构化的知识点信息。

输出必须是严格的 JSON 对象（不要输出 markdown 代码块、不要任何解释文字），格式如下：
{
  "material_type": "材料类型（如：课件 / 讲义 / 教材 / 论文 / 图片，无法判断时用「未分类」）",
  "summary": "对整份材料的一句话概述",
  "knowledge_points": [
    {
      "name": "知识点名称",
      "level": "基础 | 进阶 | 高级",
      "key_idea": "该知识点的核心思想（一句话）",
      "prerequisite": "前置知识点名称，没有则为空字符串",
      "example": "一个简短例证，没有则为空字符串"
    }
  ],
  "relations": [
    {"from": "前置知识点名称", "to": "后置知识点名称", "type": "先修"}
  ]
}

要求：
1. knowledge_points 控制在 5 ~ 15 个，按由易到难顺序排列。
2. level 只能取「基础」「进阶」「高级」三者之一。
3. relations 表达知识点之间的先修/依赖关系，from 与 to 必须是 knowledge_points 里出现过的 name。
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
        text = req.pdf_text.strip()  # 已带文字，直接使用
    elif req.pdf_base64:
        text = extract_pdf_text(req.pdf_base64)  # PDF 走 pymupdf 提取文字
    elif req.image_base64:
        text = ocr_extract(req.image_base64)  # 图片走 OCR
        if text is None:
            # OCR 尚未接入，返回占位信息（不报错，让前端可提示用户）
            return {
                "ocr_status": "pending",
                "message": "OCR 待接入",
                "material_type": "",
                "summary": "",
                "knowledge_points": [],
                "relations": [],
            }
    else:
        raise HTTPException(status_code=400, detail="请提供 image_base64、pdf_base64 或 pdf_text 之一。")

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
            "material_type": "",
            "summary": "",
            "knowledge_points": [],
            "relations": [],
        }

    # 4) 规范化返回，字段按约定输出
    return {
        "ocr_status": "ok",
        "material_type": data.get("material_type", ""),
        "summary": data.get("summary", ""),
        "knowledge_points": _ensure_list(data.get("knowledge_points")),
        "relations": _ensure_list(data.get("relations")),
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
