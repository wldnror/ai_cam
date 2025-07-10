#!/usr/bin/env python3
import os, subprocess
from aiohttp import web
import aiohttp_cors

# HLS 세그먼트 디렉터리
OUTPUT_DIR = "/home/user/ai_cam/hls"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) libcamera-vid → stdout H.264
rpicam = [
  "libcamera-vid",
  "--nopreview",                                  # ◀ 여기에 추가
  "--timeout", "0",
  "--post-process-file", "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json",
  "--inline",
  "--width", "1280", "--height", "720", "--framerate", "30",
  "-o", "-"                                        # H.264를 STDOUT으로
]
# 2) ffmpeg → HLS 패키징
ffmpeg = [
  "ffmpeg", "-hide_banner", "-loglevel", "error",
  "-i", "pipe:0",              # libcamera-vid stdout
  "-codec", "copy",            # 재인코딩 없이
  "-f", "hls",
  "-hls_time", "2",            # 2초 세그먼트
  "-hls_list_size", "3",
  "-hls_flags", "delete_segments",
  os.path.join(OUTPUT_DIR, "stream.m3u8")
]

# 백그라운드로 파이프라인 시작
proc_cam = subprocess.Popen(rpicam, stdout=subprocess.PIPE)
proc_hls = subprocess.Popen(ffmpeg, stdin=proc_cam.stdout)

# 3) aiohttp 웹서버
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
# CORS 허용 (필요시)
cors = aiohttp_cors.setup(app)
res = app.router.add_static('/hls', OUTPUT_DIR)
aiohttp_cors.add(res, {"*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")})
app.router.add_get('/', index)

if __name__ == "__main__":
    web.run_app(app, host='0.0.0.0', port=8000)
