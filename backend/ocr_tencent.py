# -*- coding: utf-8 -*-
"""腾讯云 OCR 接入示例（付费）—— 通用印刷体识别。

用于把上传的图片 base64 识别成文字，再交给 DeepSeek 做知识点结构化抽取。

前置条件：
    1. 开通腾讯云 OCR（https://console.cloud.tencent.com/ocr），获得 SecretId / SecretKey；
    2. 安装依赖：pip install tencentcloud-sdk-python-ocr
    3. 配置环境变量（或 .env）：
       TENCENT_SECRET_ID  /  TENCENT_SECRET_KEY

费用说明：
    通用印刷体识别按次计费（每月有免费额度，超出后约 ¥0.15~0.3 / 千次量级，
    具体以腾讯云官方价格页为准）。
"""

import os

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ocr.v20181119 import ocr_client, models


_client = None


def _get_client():
    """懒加载并缓存腾讯云 OCR 客户端（跨请求复用，避免反复重建连接池）。"""
    global _client
    if _client is not None:
        return _client
    secret_id = os.environ.get("TENCENT_SECRET_ID", "")
    secret_key = os.environ.get("TENCENT_SECRET_KEY", "")
    if not secret_id or not secret_key:
        raise RuntimeError("未配置腾讯云 OCR 凭证（TENCENT_SECRET_ID / TENCENT_SECRET_KEY）。")

    cred = credential.Credential(secret_id, secret_key)

    # 国内可选地域：ap-guangzhou（广州）、ap-shanghai、ap-beijing 等
    http_profile = HttpProfile()
    http_profile.endpoint = "ocr.tencentcloudapi.com"
    http_profile.reqTimeout = 30          # 单次请求超时（秒），避免无响应拖死整条链路
    http_profile.keepAlive = 1            # 复用长连接，提升多图识别速度

    client_profile = ClientProfile()
    client_profile.httpProfile = http_profile

    _client = ocr_client.OcrClient(cred, "ap-guangzhou", client_profile)
    return _client


def extract_text_from_image_base64(image_base64: str) -> str:
    """调用腾讯云「通用印刷体识别」，返回拼接后的纯文本。

    参数：
        image_base64: 图片的 base64（不含 data:image/...;base64, 前缀）
    返回：
        识别出的文字，多行用换行符拼接。
    """
    client = _get_client()

    req = models.GeneralBasicOCRRequest()
    req.ImageBase64 = image_base64  # 直接传 base64（图片 ≤ 7MB，分辨率建议 ≤ 6000px）

    resp = client.GeneralBasicOCR(req)

    lines = [item.DetectedText for item in resp.TextDetections]
    return "\n".join(lines)
