#!/usr/bin/env python3
import asyncio
import os
from aiohttp import web
import aiohttp_cors
import subprocess

# HLS 세그먼트가 생성될 디렉터리
OUTPUT_DIR = "/home/user/ai_cam/hls"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) libcamera-vid → stdout H.264 → ffmpeg HLS 변환
rpicam = [
    "libcamera-vid",
    "--timeout", "0",
    "--post-process-file", "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json",
    "--inline",
    "--width", "1280", "--height", "720", "--framerate", "30",
    "-o", "-"           # H.264를 stdout으로
]
ffmpeg = [
    "ffmpeg", "-hide_banner", "-loglevel", "error",
    "-i", "pipe:0",     # libcamera-vid stdout
    "-codec", "copy",   # 재인코딩 없이 바로 패키징
    "-f", "hls",
    "-hls_time", "2",        # 세그먼트 길이 (초)
    "-hls_list_size", "3",   # 플레이리스트에 최대 3개 세그먼트 유지
    "-hls_flags", "delete_segments",
    os.path.join(OUTPUT_DIR, "stream.m3u8")
]

# 백그라운드로 subprocess 실행
process = subprocess.Popen(rpicam, stdout=subprocess.PIPE)
hls = subprocess.Popen(ffmpeg, stdin=process.stdout)

# 2) aiohttp로 HLS 디렉터리 서빙
async def index(request):
    return web.Response(text="""
<html><body style="margin:0;text-align:center;">
  <h1>AI 카메라 HLS 스트리밍</h1>
  <video controls autoplay muted style="width:100%;height:auto;"
         src="/hls/stream.m3u8" type="application/vnd.apple.mpegurl">
    브라우저가 HLS를 지원하지 않습니다.
  </video>
</body></html>
""", content_type='text/html')

app = web.Application()
cors = aiohttp_cors.setup(app)
# 정적 파일 라우트: /hls → OUTPUT_DIR
resource = app.router.add_static('/hls', OUTPUT_DIR)
aiohttp_cors.add(resource, {
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
    )
})
app.router.add_get('/', index)

if __name__ == "__main__":
    web.run_app(app, host='0.0.0.0', port=8000)
