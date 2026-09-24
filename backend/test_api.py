# -*- coding: utf-8 -*-
"""API 接入验证脚本：验证后端三个接口真实可用。

用法：
    1. 先启动后端（确保已配置 DEEPSEEK_API_KEY）：
       uvicorn app:app --host 0.0.0.0 --port 8000
    2. 运行本脚本：
       python test_api.py

若要验证线上部署，把下面的 BASE 改成你的公网 URL（如 https://zhiyu.onrender.com）。
"""

import httpx

BASE = "http://localhost:8000"  # ← 本地；线上改成 https://你的域名.onrender.com


def test_health():
    r = httpx.get(f"{BASE}/health", timeout=30)
    print(f"\n1) GET /health  ->  {r.status_code}")
    d = r.json()
    print("   ", d)
    assert r.status_code == 200, d
    assert d.get("api_key_configured"), "❌ DEEPSEEK_API_KEY 未配置！请先设置环境变量或 .env。"
    print("   ✓ API Key 已配置")


def test_chat():
    payload = {"messages": [{"role": "user", "content": "用一句话解释什么是导数"}]}
    r = httpx.post(f"{BASE}/api/chat", json=payload, timeout=120)
    print(f"\n2) POST /api/chat  ->  {r.status_code}")
    d = r.json()
    assert r.status_code == 200, d
    reply = d.get("reply", "")
    assert reply, "❌ 未返回 reply"
    print("   ✓ 回答:", reply[:100] + ("..." if len(reply) > 100 else ""))


def test_analyze():
    text = "第一章 函数与极限：极限的定义、无穷小与等价替换。第二章 导数：导数的几何意义、基本求导公式与链式法则。"
    r = httpx.post(f"{BASE}/api/analyze", json={"pdf_text": text}, timeout=120)
    print(f"\n3) POST /api/analyze  ->  {r.status_code}")
    d = r.json()
    assert r.status_code == 200, d
    kps = d.get("knowledge_points", [])
    print("   ✓ material_type:", d.get("material_type"))
    print("   ✓ 知识点数:", len(kps))
    for kp in kps[:5]:
        print("      -", kp.get("name"), "/", kp.get("level"))


if __name__ == "__main__":
    test_health()
    test_chat()
    test_analyze()
    print("\n== ✅ 全部通过：后端 API 接入正常，DeepSeek 链路真实可用 ==")
