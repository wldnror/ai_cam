#!/usr/bin/env python3
from flask import Flask, Response, render_template
from picamera2 import Picamera2
from picamera2.devices import IMX500
import cv2

# — 1) IMX500 모델 로드 —
MODEL_PATH = "/usr/share/imx500-models/" \
    "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
imx500 = IMX500(MODEL_PATH)

# — 2) Picamera2 구성 —
picam2 = Picamera2(imx500.camera_num)
# preview 해상도, 프레임 속도 맞추기
config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    controls={"FrameRate": imx500.network_intrinsics.inference_rate}
)
# on-camera 추론만 수행 (draw는 우리가 직접)
config["post_process_file"] = "/usr/share/rpi-camera-assets/" \
    "imx500_mobilenet_ssd.json"
picam2.configure(config)
imx500.show_network_fw_progress_bar()  # (선택) 펌웨어 로드 대기
picam2.start()

app = Flask(__name__)

def gen_frames():
    while True:
        # capture_request() 로 프레임 + 메타데이터 동시 획득
        with picam2.capture_request() as req:
            frame = req.make_image("main")      # JPEG 전용 BGR 배열
            md = req.get_metadata()             # 메타데이터 가져오기

        # 메타데이터에서 객체 탐지 결과 꺼내기
        # JSON 파이프라인 이름에 따라 경로가 달라질 수 있음
        dets = md.get("Inference", {}).get("objects", [])
        for obj in dets:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["label_text"]
            score = obj["score"]
            # 박스·라벨 그리기
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{label} {score:.2f}",
                        (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # JPEG 인코딩 및 스트리밍
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
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    # 시스템 파이썬으로 실행하세요(가상환경 deactivate 상태)
    app.run(host='0.0.0.0', port=5000, threaded=True)
