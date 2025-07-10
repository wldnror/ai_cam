#!/usr/bin/env python3
from flask import Flask, Response, render_template, jsonify, request
from picamera2 import Picamera2
from picamera2.devices import IMX500
import cv2

# — 설정 — 
MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
THRESHOLD = 0.30  # confidence threshold

# — IMX500 로드 & 인트린직스 — 
imx500 = IMX500(MODEL_PATH)
intrinsics = imx500.network_intrinsics
intrinsics.update_with_defaults()

# — Picamera2 초기화 — 
picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    controls={"FrameRate": intrinsics.inference_rate},
    buffer_count=12
)
imx500.show_network_fw_progress_bar()
picam2.configure(config)
picam2.start()

# — Flask 앱 — 
app = Flask(__name__)

# — 더미 /stats 엔드포인트 (404 로그 제거용) — 
@app.route('/stats')
def stats():
    return jsonify({})

# — 객체 탐지 파싱 함수 — 
last_results = []
def parse_detections(metadata):
    global last_results
    outputs = imx500.get_outputs(metadata, add_batch=True)
    if outputs is None:
        return last_results
    boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
    # 정규화 해제
    if intrinsics.bbox_normalization:
        boxes = boxes * imx500.get_input_size()[::-1]
    # 좌표 순서 교정
    if intrinsics.bbox_order == "xy":
        boxes = boxes[:, [1,0,3,2]]
    detections = []
    for box, score, cls in zip(boxes, scores, classes):
        if score < THRESHOLD:
            continue
        x0, y0, x1, y1 = box.astype(int)
        detections.append((int(cls), float(score), (x0, y0, x1, y1)))
    last_results = detections
    return detections

# — 스트리밍용 프레임 생성기 — 
def gen_frames():
    while True:
        metadata = picam2.capture_metadata()
        detections = parse_detections(metadata)
        frame = picam2.capture_array("main")

        # 터미널 로그 출력
        labels = [intrinsics.labels[cls] for cls, _, _ in detections]
        print("Detections:", labels)

        # 프레임에 박스·라벨 그리기
        for cls, score, (x0, y0, x1, y1) in detections:
            label = f"{intrinsics.labels[cls]}:{score:.2f}"
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0,255,0), 2)
            cv2.putText(frame, label, (x0, y0-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
