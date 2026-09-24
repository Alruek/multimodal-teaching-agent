# 知屿 ZhiYu · 多模态教学智能体 —— 后端服务

作为前端网页的"大脑"，安全中转调用 DeepSeek 大模型（OpenAI 兼容接口）。前端不直接持有 API Key，所有大模型调用都经过本服务。

## 目录

```
backend/
├── app.py            # 后端主服务（FastAPI）
├── requirements.txt  # Python 依赖
└── README.md         # 本说明
```

## 一、安装依赖

```bash
cd backend
pip install -r requirements.txt
```

## 二、配置环境变量（关键，Key 绝不写进代码）

**方式 A：`.env` 文件（推荐，本地开发最方便）**

```bash
# 复制模板，填入你的密钥
cp .env.example .env
# 编辑 .env，把 DEEPSEEK_API_KEY 改成你的真实密钥
```

> `.env` 已被 `.gitignore` 忽略，不会提交到代码仓库。

**方式 B：系统环境变量**

Windows（PowerShell）：

```powershell
$env:DEEPSEEK_API_KEY="sk-你的DeepSeek密钥"
```

macOS / Linux：

```bash
export DEEPSEEK_API_KEY="sk-你的DeepSeek密钥"
```

> 优先级：系统/平台环境变量 > `.env` 文件（`.env` 不会覆盖已存在的环境变量）。
> 可选变量：
> - `DEEPSEEK_BASE_URL`：默认 `https://api.deepseek.com/v1`
> - `DEEPSEEK_MODEL`：默认 `deepseek-chat`
> - `ALLOWED_ORIGINS`：跨域白名单，逗号分隔，默认 `*`（生产建议改成前端具体域名）

## 三、启动

```bash
# 方式一（推荐）
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 方式二
python app.py
```

启动后访问 http://localhost:8000/ 应看到：

```json
{"service": "知屿 ZhiYu · 多模态教学智能体后端", "status": "ok", "model": "deepseek-chat", "api_key_configured": true}
```

## 四、接口说明

### 1. `POST /api/chat` —— 对话

```json
{
  "messages": [
    {"role": "system", "content": "你是一位耐心的 AI 助教"},
    {"role": "user", "content": "导数和微分的区别是什么？"}
  ]
}
```

返回：

```json
{ "reply": "DeepSeek 的回答文本" }
```

### 2. `POST /api/analyze` —— 知识点结构化抽取

请求体三选一（对应不同文件类型）：

```json
{ "pdf_text": "已提取的纯文本（txt/md 文件）" }
```

或

```json
{ "pdf_base64": "PDF 文件的 base64（后端用 PyMuPDF 提取文字）" }
```

或

```json
{ "image_base64": "图片的 base64（后端走 OCR 识别）" }
```

返回知识点结构化结果（字段按约定输出）：

```json
{
  "ocr_status": "ok",
  "material_type": "讲义",
  "summary": "本文介绍微积分基础概念",
  "knowledge_points": [
    { "name": "导数", "level": "进阶", "key_idea": "瞬时变化率", "prerequisite": "函数与极限", "example": "y=x² 在 x=3 斜率 6" }
  ],
  "relations": [
    { "from": "函数与极限", "to": "导数", "type": "先修" }
  ]
}
```

> 图片走 OCR 时，若尚未接入真实 OCR，会返回 `{"ocr_status": "pending", "message": "OCR 待接入", ...}`（不会报错）。

## 五、部署注意

1. **API Key 安全**：务必通过部署平台的环境变量 / Secret 注入 `DEEPSEEK_API_KEY`，不要提交到代码仓库。
2. **CORS 收紧**：上线前把 `ALLOWED_ORIGINS` 改成前端实际域名（如 `https://your-site.com`），避免任意站点跨域调用。
3. **异常友好**：DeepSeek 超时 / 网络错误 / 调用失败均返回带中文提示的 502/504，不会抛 500 裸错。

## 六、前端对接

前端已放在 `static/index.html`（Vue3 版 / 纯 HTML 版均可），通过**相对路径**调同源后端：

```js
// 相对路径，同源部署（前端由后端服务，无需写域名）
const res = await fetch('/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ messages: [...] })
});
const { reply } = await res.json();
```

## 七、验证 API 接入（真实链路）

填好 Key 并启动后，跑一遍验证脚本，确认 DeepSeek 链路真实可用：

```bash
python test_api.py
# 依次验证 GET /health、POST /api/chat、POST /api/analyze
```

## 八、部署与 OCR

- **一键部署**：`render.yaml` 是 Render Blueprint 配置，推到 GitHub 后在 Render 用「Blueprint」即可自动创建服务；详细步骤见 `DEPLOY.md`。
- **OCR（付费示例）**：`ocr_tencent.py` 已接入腾讯云「通用印刷体识别」。配置 `TENCENT_SECRET_ID` / `TENCENT_SECRET_KEY` 后，图片上传走真实 OCR；未配置则返回「OCR 待接入」占位。启用需在 `requirements.txt` 取消注释 `tencentcloud-sdk-python-ocr`。
