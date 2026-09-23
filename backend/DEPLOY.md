# 部署说明 · 一键在线可访问（Render.com 免费 Web Service）

把「知屿 ZhiYu」多模态教学智能体（FastAPI 后端 + 前端静态页面）部署到 **Render.com**，
得到一个公网 HTTPS 链接，评审点击即可使用，**无需本地安装任何东西**。

## 为什么选 Render.com

- 免费档 Web Service，自动 HTTPS，直接跑 Python 进程；
- 环境变量在 Web 后台安全配置，**API Key 不进代码、不进仓库**；
- FastAPI 用 `StaticFiles` 把前端页面一并服务，**前后端同源部署**到同一个链接。

---

## 一、目录结构（需提交到 GitHub）

```
backend/
├── app.py              # FastAPI 后端（已含静态文件服务）
├── requirements.txt    # 依赖清单
├── static/
│   └── index.html      # 前端页面（已复制好）
├── README.md
└── DEPLOY.md           # 本文档
```

> 关键：`static/index.html` 就是「多模态教学智能体」前端页，`app.py` 已用 StaticFiles 把它挂载为首页。

## 二、部署步骤

### 1. 把 `backend/` 目录推到 GitHub（公开或私有仓库均可）

### 2. 在 Render 新建 Web Service

1. 登录 [render.com](https://render.com) → 右上角 **New +** → **Web Service**；
2. 选择你的 GitHub 仓库；
3. 关键配置：

| 配置项 | 值 |
|--------|-----|
| Name | `zhiyu`（任意） |
| Root Directory | `backend`（**重要，指向后端目录**） |
| Runtime | `Python 3` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |

### 3. 配置环境变量 DEEPSEEK_API_KEY

在 **Environment → Environment Variables** 添加：

| Key | Value |
|-----|-------|
| `DEEPSEEK_API_KEY` | `sk-你的DeepSeek密钥` |

可选变量：

| Key | 说明 |
|-----|------|
| `ALLOWED_ORIGINS` | 跨域白名单，同源部署可留空（默认 `*`） |
| `DEEPSEEK_MODEL` | 默认 `deepseek-chat` |

### 4. 部署

点 **Create Web Service**，等待 Build → Deploy 完成（首次约 1~3 分钟）。

---

## 三、requirements.txt 需要哪些包

只需 3 个：

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
httpx>=0.27.0
```

> 免费实例单进程用 `uvicorn` 足够；若将来要多进程/高并发，再加 `gunicorn`，并把 Start Command 换成
> `gunicorn -k uvicorn.workers.UvicornWorker app:app`。

## 四、静态文件同源部署（前端一并服务）

前端页面放在 `backend/static/index.html`，后端用 `StaticFiles` 挂载，根路径直接返回前端首页。
这段代码**已写在 `app.py` 里**，如下：

```python
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

@app.get("/")
def index():
    # 根路径直接返回前端首页，评审点击公网 URL 即可打开
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

# 挂载静态资源，供 /static/... 访问
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
```

前端 `API_BASE` 也已自动适配：**本地开发**走 `http://localhost:8000`，**线上部署**走同源相对路径（无需任何配置）。

---

## 五、部署完成后的公网 URL

部署成功后，Render 会分配一个 URL，例如：

```
https://zhiyu.onrender.com
```

- **评审直接点击该链接** → 打开前端首页，即可上传课件解析、进知识星空图、交互答疑；
- 所有 AI 调用经同源后端转发到 DeepSeek，**前端看不到任何 API Key**；
- 健康检查：`https://zhiyu.onrender.com/health` 返回 `{"status":"ok"}`。

---

## 六、注意事项

1. **免费实例会休眠**：15 分钟无访问会自动 sleep，下次访问需冷启动（约 30~60 秒），评审第一次打开请耐心等待，属于正常现象。
2. **API Key 安全**：`DEEPSEEK_API_KEY` 只配置在 Render 环境变量里，绝不要提交进 GitHub。
3. **CORS 收紧（可选）**：同源部署后其实无需跨域；若前后端分离部署，把 `ALLOWED_ORIGINS` 设为前端域名。
4. **验证在线**：部署完成后先打开 `/health` 确认 `api_key_configured: true`，再试用前端。
