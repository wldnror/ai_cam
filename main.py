#!/usr/bin/env python3
import os
import time
import threading
import queue
import subprocess

import cv2
import torch
import psutil
from flask import Flask, Response, render_template, jsonify, request

# ──────────────────────────────────────────────────────────────────────────────
# 0) 실행 중 화면 꺼짐·절전 모드 방지 (콘솔 전용이라 사실 필요 없음)
try:
    os.system('setterm -blank 0 -powerdown 0 -powersave off')
    os.environ.setdefault('DISPLAY', ':0')
    os.system('xset s off')
    os.system('xset s noblank')
    os.system('xset -dpms')
    print("⏱️ 화면 절전/블랭킹 기능 비활성화 완료")
except Exception as e:
    print("⚠️ 전원/스크린세이버 비활성화 중 오류:", e)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# 1) PyTorch 스레드 수 제한 (불필요한 컨텍스트 스위치 방지)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# 모델 로드 & 추론 모드 고정
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.eval()
# ──────────────────────────────────────────────────────────────────────────────

# 2) 카메라 인터페이스 통일 (변경 없음)…
class CSICamera:
    # …생략…

class USBCamera:
    # …생략…

try:
    camera = CSICamera()
    print(">>> Using CSI camera module")
except Exception:
    camera = USBCamera()
    print(">>> Using USB webcam")

# 3) 백그라운드 프레임 처리 스레드 + 큐
frame_queue = queue.Queue(maxsize=1)

def capture_and_process():
    # 변경: FPS를 15 → 10으로 낮춤
    fps = 10
    interval = 1.0 / fps
    last = time.time()

    # 변경: 해상도를 640→320 으로 낮춰서 연산량 4배 절감
    target_size = (320, 320)

    # 변경: 프레임 스킵 (N프레임 당 1회만 추론)
    skip_interval = 2
    frame_count = 0

    while True:
        now = time.time()
        sleep = interval - (now - last)
        if sleep > 0:
            time.sleep(sleep)
        last = time.time()

        ret, frame = camera.read()
        if not ret:
            continue

        frame_count += 1
        # 실제 추론 수행할 때만 모델 호출
        if frame_count % skip_interval == 0:
            # no_grad 컨텍스트로 불필요한 그래디언트 계산 차단
            with torch.no_grad():
                small = cv2.resize(frame, target_size)
                results = model(small)
        # skip_interval 프레임에는 이전 results 재사용

        # 박스 그리기 (기존 비율 계산 유지)
        h_ratio = frame.shape[0] / target_size[1]
        w_ratio = frame.shape[1] / target_size[0]
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, (
                box[0] * w_ratio,
                box[1] * h_ratio,
                box[2] * w_ratio,
                box[3] * h_ratio
            ))
            label = results.names[int(cls)]
            if label in ('person', 'car'):
                color = (0,0,255) if label=='person' else (255,0,0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # JPEG 인코딩 (필요 시 하드웨어 인코더 교체 고려)
        _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = buf.tobytes()

        # 큐에 최신 프레임만 유지
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(data)

threading.Thread(target=capture_and_process, daemon=True).start()

# 4) Flask 앱 & 스트리밍 + 통계 엔드포인트 (변경 없음)…
app = Flask(__name__)
# …이하 생략…
if __name__ == '__main__':
    # nice 옵션으로 Python 프로세스 우선순위 올려 실행
    os.execvp('nice', ['nice', '-n', '-5', 'python3'] + ['-u', __file__])
