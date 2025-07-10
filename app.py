from flask import Flask, Response, render_template
from picamera2 import Picamera2
from picamera2.devices import IMX500
import cv2

# IMX500 모델 로드
imx500 = IMX500("/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk")
picam2 = Picamera2(imx500.camera_num)

# 프리뷰 + 추론 설정
config = picam2.create_preview_configuration(main={"size": (640, 480)},
                                             controls={"FrameRate": imx500.network_intrinsics.inference_rate})
config["post_process_file"] = "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"
picam2.configure(config)
picam2.start()

app = Flask(__name__)

def gen_frames():
    while True:
        # 프레임 + 메타데이터 동시 획득
        request = picam2.capture_request()
        frame = request.make_image("main")  # 오버레이 없이 원본
        metadata = request.get_metadata()
        request.release()

        # 메타데이터에서 박스/레이블 정보 파싱
        dets = metadata.get("Inference", {}).get("objects", [])
        for obj in dets:
            # obj 구조는 JSON 파이프라인에 따라 달라집니다.
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["label_text"]
            score = obj["score"]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (int(x1), int(y1)-5),
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
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
