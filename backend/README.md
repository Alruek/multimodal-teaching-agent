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

**Windows（PowerShell）：**

```powershell
$env:DEEPSEEK_API_KEY="sk-你的DeepSeek密钥"
```

**macOS / Linux：**

```bash
export DEEPSEEK_API_KEY="sk-你的DeepSeek密钥"
```

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

请求体二选一：

```json
{ "pdf_text": "课件已提取的纯文本..." }
```

或

```json
{ "image_base64": "图片的 base64 字符串" }
```

返回知识点图谱（与前端 `knowledge_points` 对齐）：

```json
{
  "ocr_status": "ok",
  "subject": "高等数学 · 微积分",
  "nodes": [ {"id": "derivative", "name": "导数", "difficulty": "medium", "importance": 0.8, "summary": "...", "points": ["..."], "example": "..."} ],
  "edges": [ {"source": "limits", "target": "derivative", "relation": "先修"} ],
  "learning_path": ["limits", "derivative", "integral"]
}
```

> 图片走 OCR 时，若尚未接入真实 OCR，会返回 `{"ocr_status": "pending", "message": "OCR 待接入", ...}`（不会报错）。

## 五、部署注意

1. **API Key 安全**：务必通过部署平台的环境变量 / Secret 注入 `DEEPSEEK_API_KEY`，不要提交到代码仓库。
2. **CORS 收紧**：上线前把 `ALLOWED_ORIGINS` 改成前端实际域名（如 `https://your-site.com`），避免任意站点跨域调用。
3. **异常友好**：DeepSeek 超时 / 网络错误 / 调用失败均返回带中文提示的 502/504，不会抛 500 裸错。

## 六、前端对接

把前端 HTML 里模拟的 `aiReply` / 上传解析替换为对本服务的 `fetch` 调用即可：

```js
const res = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ messages: [...] })
});
const { reply } = await res.json();
```
