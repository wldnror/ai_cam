# ai_stream_korean.py
from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import onnxruntime as ort

# 1) 한글 라벨 매핑 (필요한 클래스만 추가하세요)
KOR_LABELS = {
    0: "background",
    1: "사람",
    2: "자전거",
    3: "자동차",
    4: "오토바이",
    5: "비행기",
    6: "버스",
    7: "기차",
    8: "트럭",
    9: "보트",
    # …COCO 레이블 전체를 매핑하세요
}

# 2) ONNX 세션 준비 (imx500_mobilenet_ssd 모델)
MODEL_PATH = "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def preprocess(frame):
    # Mobilenet SSD: 300×300 입력, RGB, [0,1] 정규화
    img = cv2.resize(frame, (300, 300))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))[None, ...]

def postprocess(frame, outputs, score_threshold=0.5):
    h, w, _ = frame.shape
    # ONNX 출력 형식: [boxes, scores, class_ids, num_detections]
    boxes, scores, class_ids = outputs[0], outputs[1], outputs[2]
    for box, score, cls in zip(boxes[0], scores[0], class_ids[0]):
        if score < score_threshold:
            continue
        # 상대좌표→절대좌표
        y1, x1, y2, x2 = box
        x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        label = KOR_LABELS.get(int(cls), str(int(cls)))
        text = f"{label} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

app = Flask(__name__)

def gen_frames():
    # Picamera2 설정: post_callback 대신 직접 ONNX 추론
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640,480)}
    )
    picam2.configure(config)
    picam2.start()
    while True:
        frame = picam2.capture_array()             # RGB 배열
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 3) 추론
        inp = preprocess(frame)
        outputs = session.run(None, {input_name: inp})
        # 4) 후처리 & 한글 라벨 그리기
        frame = postprocess(frame, outputs, score_threshold=0.4)
        # 5) JPEG 인코딩
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html><body>
     <h1>AI 카메라 실시간 스트리밍</h1>
     <img src="/video_feed" style="width:100%;max-width:640px"/>
    </body></html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
