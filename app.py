#!/usr/bin/env python3
from flask import Flask, Response, render_template
from picamera2 import Picamera2
from picamera2.devices import IMX500
import cv2

# — 1) IMX500 모델 로드 —
MODEL_PATH = "/usr/share/imx500-models/" \
    "imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
imx500 = IMX500(MODEL_PATH)

# — 2) Picamera2 객체 생성 —
picam2 = Picamera2(imx500.camera_num)

# — 3) Preview + on-camera 추론 설정 —
config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    controls={"FrameRate": imx500.network_intrinsics.inference_rate}
)
config["post_process_file"] = "/usr/share/rpi-camera-assets/" \
    "imx500_mobilenet_ssd.json"

# — 4) 설정 적용 & 카메라 시작 —
picam2.configure(config)
imx500.show_network_fw_progress_bar()  # (선택) 펌웨어 업로드 진행 표시
picam2.start()

# — 5) Flask 앱 정의 —
app = Flask(__name__)

def gen_frames():
    """메타데이터 읽어서 박스·레이블 직접 그린 뒤 JPEG 스트리밍"""
    while True:
        # captured_request() 는 with 구문 지원
        with picam2.captured_request() as req:
            # 'main' 스트림을 NumPy 배열로
            frame = req.make_array("main")
            md = req.get_metadata()

        # 메타데이터에서 객체 리스트 꺼내기
        dets = md.get("Inference", {}).get("objects", [])
        for obj in dets:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["label_text"]
            score = obj["score"]
            # 박스와 레이블 그리기
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{label} {score:.2f}",
                        (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # JPEG 인코딩
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
    # venv 비활성화 후 시스템 python3 로 실행하세요!
    app.run(host='0.0.0.0', port=5000, threaded=True)
