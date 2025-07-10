#!/usr/bin/env python3
import os
import subprocess
from aiohttp import web

# HLS 세그먼트가 생성될 디렉터리
OUTPUT_DIR = "/home/user/ai_cam/hls"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) libcamera-vid → stdout H.264 → ffmpeg HLS 패키징
rpicam = [
    "libcamera-vid",
    "--nopreview",  # DRM 뷰파인더 윈도우를 띄우지 않음
    "--timeout", "0",
    "--post-process-file", "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json",
    "--inline",
    "--width", "1280", "--height", "720", "--framerate", "30",
    "-o", "-"       # H.264 스트림을 stdout으로
]
ffmpeg = [
    "ffmpeg", "-hide_banner", "-loglevel", "error",
    "-i", "pipe:0",  # libcamera-vid stdout
    "-codec", "copy",
    "-f", "hls",
    "-hls_time", "2",
    "-hls_list_size", "3",
    "-hls_flags", "delete_segments",
    os.path.join(OUTPUT_DIR, "stream.m3u8")
]

proc_cam = subprocess.Popen(rpicam, stdout=subprocess.PIPE)
proc_hls = subprocess.Popen(ffmpeg, stdin=proc_cam.stdout)

# 2) aiohttp로 HLS 디렉터리와 index 페이지 서빙
async def index(request):
    return web.Response(text="""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>AI 카메라 HLS 스트리밍</title>
  <style>body{margin:0;text-align:center;}video{width:100%;height:auto;}</style>
</head>
<body>
  <h1>On-sensor 객체 감지 스트림</h1>
  <video controls autoplay muted type="application/vnd.apple.mpegurl"
         src="/hls/stream.m3u8">
    브라우저가 HLS를 지원하지 않습니다.
  </video>
</body>
</html>
""", content_type='text/html')

app = web.Application()
# /hls 경로로 OUTPUT_DIR 내부 파일들(static) 서빙
app.router.add_static('/hls', OUTPUT_DIR, name='hls')
app.router.add_get('/', index)

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=8000)
