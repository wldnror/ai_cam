#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")  # Python 경고 억제

import os
import sys
import queue
from picamera2 import Picamera2
from flask import Flask, Response

# 0) 화면 절전/DPMS 비활성화 (X 환경일 때만)
try:
    if os.environ.get('DISPLAY'):
        os.system('setterm -blank 0 -powerdown 0 -powersave off')
        os.system('xset s off; xset s noblank; xset -dpms')
        print("⏱️ 화면 절전/스크린세이버 비활성화 완료")
except Exception:
    pass

# 1) CSI 카메라 초기화 (MJPEG 포맷 사용)
try:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "MJPEG"},
        lores={"size": (640, 360)},
        buffer_count=2
    )
    picam2.configure(config)
    picam2.start()
    # 워밍업 캡처
    for _ in range(3):
        picam2.capture_buffer("main")
    print(">>> Using CSI camera module (MJPEG)")
except Exception as e:
    print(f"[ERROR] CSI 카메라 초기화 실패: {e}")
    sys.exit(1)

# 프레임 메시 저장용 큐
frame_queue = queue.Queue(maxsize=1)

# 버퍼 읽기 쓰기 핸들러
class FrameWriter:
    def write(self, buf):
        # buf may be memoryview or bytes
        try:
            data = buf.tobytes() if hasattr(buf, 'tobytes') else bytes(buf)
            if not frame_queue.empty(): frame_queue.get_nowait()
            frame_queue.put(data)
        except Exception:
            pass

# 녹화 시작 (MJPEGStream)
    encoder = None  # 사용 안 함
    # 기본 main 스트림으로 MJPEG 스트림 기록
    picam2.start_recording(FrameWriter())
