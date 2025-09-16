#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import logging, os, time, threading, queue, subprocess, json
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, render_template_string, jsonify, request

# -------- 로그 정리
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# -------- 한글 폰트 (없으면 나눔 설치 시도)
FONT_PATH = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if not os.path.exists(FONT_PATH):
    try:
        subprocess.run(['sudo','apt-get','update'], check=True)
        subprocess.run(['sudo','apt-get','install','-y','fonts-nanum'], check=True)
    except Exception as e:
        print(f"[FONT] install failed: {e}")
try:
    font = ImageFont.truetype(FONT_PATH, 20)
except Exception:
    font = ImageFont.load_default()

# -------- Picamera2 + IMX500
from picamera2 import Picamera2
try:
    # Picamera2의 IMX500 헬퍼: 모델 로드/메타데이터 파싱 제공
    from picamera2.imx500 import IMX500
    HAS_IMX_HELPER = True
except Exception:
    HAS_IMX_HELPER = False

# -------- 모드/모델 경로 (필요시 원하는 rpk로 교체)
MODEL_OBJECT = "/usr/share/imx500-models/imx500_network_efficientdet_lite0_pp.rpk"
MODEL_POSE   = "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"

# COCO 포즈 스켈레톤 연결(17 keypoints 기준, HigherHRNet COCO)
# (nose, L/R eye, L/R ear, L/R shoulder, L/R elbow, L/R wrist, L/R hip, L/R knee, L/R ankle)
COCO_PAIRS = [
    (5,7),(7,9),      # L-shoulder->L-elbow->L-wrist
    (6,8),(8,10),     # R-shoulder->R-elbow->R-wrist
    (11,13),(13,15),  # L-hip->L-knee->L-ankle
    (12,14),(14,16),  # R-hip->R-knee->R-ankle
    (5,6),(11,12),    # shoulders, hips
    (5,11),(6,12),    # torso diagonals
    (0,5),(0,6),      # nose->shoulders
    (1,0),(2,0),(1,3),(2,4)  # eyes/ears to nose (가시성 낮으면 생략 가능)
]

# -------- 전역 상태
frame_queue = queue.Queue(maxsize=3)
stats_lock = threading.Lock()
current_fps = 0.0
current_mode = "object"  # "object" | "pose"
running = True

# -------- 카메라 래퍼
class IMX500Camera:
    def __init__(self):
        self.picam2 = Picamera2()
        self.cfg = self.picam2.create_video_configuration(
            main={"size": (960, 540)}, buffer_count=6
        )
        self.picam2.configure(self.cfg)
        self.picam2.start()
        time.sleep(0.2)

        # 현재 로드된 모델 경로
        self.loaded_model = None

    def load_model(self, model_path: str):
        """IMX500 rpk 모델 교체. 안전하게 정지/재시작."""
        if not HAS_IMX_HELPER:
            raise RuntimeError("picamera2.imx500 헬퍼 모듈을 찾을 수 없습니다.")

        # 이미 동일 모델이면 스킵
        if self.loaded_model == model_path:
            return

        # 카메라가 돌아가는 동안에도 로드되는 경우가 있으나
        # 안정성을 위해 잠시 stop/start 권장
        try:
            self.picam2.stop()
        except Exception:
            pass

        IMX500.load_model(self.picam2, model_path)
        self.loaded_model = model_path

        self.picam2.configure(self.cfg)
        self.picam2.start()
        time.sleep(0.1)

    def read(self):
        """프레임 + 메타데이터 획득"""
        req = self.picam2.capture_request()
        try:
            rgb = req.make_array("main")
            md  = req.get_metadata()
        finally:
            req.release()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return True, bgr, md

# USB 백업(없으면 안 써도 됨)
class USBCamera:
    def __init__(self):
        self.cap = None
        for i in range(0, 4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                self.cap = cap
                break
        if not self.cap:
            raise RuntimeError("USB 카메라를 찾을 수 없습니다.")

    def load_model(self, _):  # USB는 모델 없음
        pass

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame, None

# 카메라 선택 (IMX500 우선)
try:
    camera = IMX500Camera()
    print(">>> Using IMX500 AI Camera")
except Exception as e:
    print(f"[WARN] IMX500 init failed: {e}")
    camera = USBCamera()
    print(">>> Using USB webcam (no IMX500)")

# -------- 메타데이터 파서들
def parse_object_boxes(md, conf_thres=0.4):
    boxes = []
    if md is None:
        return boxes
    try:
        if HAS_IMX_HELPER:
            dets = IMX500.get_object_detection(md)  # [{'x1','y1','x2','y2','label','score'}]
            labels = IMX500.get_labels(md) or []
            for d in dets:
                sc = float(d.get("score", 0))
                if sc < conf_thres:
                    continue
                lab_idx = d.get("label", None)
                name = labels[lab_idx] if (labels and isinstance(lab_idx,int) and lab_idx < len(labels)) else str(lab_idx)
                boxes.append((int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]), name, sc))
            return boxes
    except Exception:
        pass

    # Fallback: 직접 키 찾기(펌웨어/모델에 따라 다를 수 있음)
    t = md.get("Tensor") or md.get("tensors") or {}
    bxs, lbs, scs = t.get("boxes"), t.get("labels"), t.get("scores")
    if isinstance(bxs, (list, tuple)) and isinstance(lbs, (list, tuple)) and isinstance(scs, (list, tuple)):
        for (x1,y1,x2,y2), lab, sc in zip(bxs, lbs, scs):
            if sc >= conf_thres:
                boxes.append((int(x1), int(y1), int(x2), int(y2), str(lab), float(sc)))
    return boxes

def parse_pose(md, conf_thres=0.3):
    """여러 사람의 포즈를 반환: [ {'keypoints': [(x,y,score),...]} , ... ]"""
    poses = []
    if md is None:
        return poses
    try:
        if HAS_IMX_HELPER and hasattr(IMX500, "get_pose_estimation"):
            # 일부 배포판에 포함되어 있음
            p = IMX500.get_pose_estimation(md)  # [{'keypoints': [(x,y,score),...]}]
            # 스코어 필터링
            for person in p:
                kps = []
                for (x, y, s) in person.get("keypoints", []):
                    kps.append((int(x), int(y), float(s)))
                poses.append({"keypoints": kps})
            return poses
    except Exception:
        pass

    # Fallback: 메타데이터 직접 파싱(키 이름은 모델에 따라 상이할 수 있음)
    t = md.get("Tensor") or md.get("tensors") or {}
    kps_all = t.get("keypoints") or t.get("poses") or []
    # 예상 형식: [ [x,y,score]*17 ] 또는 dict 리스트
    for kp in kps_all:
        person = []
        if isinstance(kp, dict) and "keypoints" in kp:
            pts = kp["keypoints"]
            for i in range(0, len(pts), 3):
                x, y, s = pts[i:i+3]
                person.append((int(x), int(y), float(s)))
        elif isinstance(kp, (list, tuple)) and len(kp) >= 51:
            for i in range(0, 51, 3):
                x, y, s = kp[i:i+3]
                person.append((int(x), int(y), float(s)))
        if person:
            poses.append({"keypoints": person})
    return poses

# -------- 그리기 유틸
def draw_boxes(frame, boxes):
    for x1,y1,x2,y2,label,conf in boxes:
        color = (0, 128, 255) if (label and "person" in str(label).lower()) else (0, 0, 255)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        txt = f"{label} {conf*100:.1f}%"
        img_pil = Image.fromarray(frame[:, :, ::-1])
        draw = ImageDraw.Draw(img_pil)
        w, h = draw.textsize(txt, font=font)
        draw.rectangle([x1, y1-h-4, x1+w+6, y1], fill=(0,0,0))
        draw.text((x1+3, y1-h-2), txt, font=font, fill=(255,255,255))
        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_poses(frame, poses):
    for person in poses:
        kps = person.get("keypoints", [])
        # 점
        for (x,y,s) in kps:
            if s is None or s < 0.3: 
                continue
            cv2.circle(frame, (x,y), 3, (0,255,0), -1)
        # 선(스켈레톤)
        for a,b in COCO_PAIRS:
            if a < len(kps) and b < len(kps):
                xa,ya,sa = kps[a]
                xb,yb,sb = kps[b]
                if sa is not None and sb is not None and sa >= 0.3 and sb >= 0.3:
                    cv2.line(frame, (xa,ya), (xb,yb), (255,0,0), 2)

# -------- 캡처/추론(=메타데이터 파싱) 스레드
def run_loop():
    global current_fps
    # 초기 모델(객체 탐지)
    if isinstance(camera, IMX500Camera):
        try:
            camera.load_model(MODEL_OBJECT)
        except Exception as e:
            print(f"[IMX500] model load failed: {e}")

    target_fps = 15
    interval = 1.0 / target_fps

    while running:
        t0 = time.time()
        ret, frame, md = camera.read()
        if not ret:
            continue

        mode = current_mode

        if mode == "object":
            boxes = parse_object_boxes(md, conf_thres=0.45)
            draw_boxes(frame, boxes)
        else:
            poses = parse_pose(md, conf_thres=0.3)
            draw_poses(frame, poses)

        # 인코딩 -> 큐
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if ok:
            if not frame_queue.empty():
                try: frame_queue.get_nowait()
                except Exception: pass
            frame_queue.put(buf.tobytes())

        # FPS
        dt = time.time() - t0
        with stats_lock:
            current_fps = (1.0/dt) if dt > 0 else 0.0

        # 타깃 FPS 맞추기
        sleep = interval - dt
        if sleep > 0: 
            time.sleep(sleep)

threading.Thread(target=run_loop, daemon=True).start()

# -------- Flask
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>IMX500 웹 데모</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Apple SD Gothic Neo,Malgun Gothic,sans-serif;padding:16px;background:#0b0b0b;color:#eaeaea}
.wrap{max-width:1080px;margin:0 auto}
.row{display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
.btn{padding:10px 14px;border-radius:10px;border:0;background:#222;color:#fff;cursor:pointer}
.btn.active{background:#4f46e5}
.badge{padding:6px 10px;border-radius:8px;background:#222}
video, img{width:100%;max-width:960px;border-radius:12px;border:1px solid #222}
small{opacity:0.7}
</style>
</head>
<body>
<div class="wrap">
  <h2>Raspberry Pi AI Camera (IMX500) – Object / Pose 전환</h2>
  <div class="row">
    <button id="btn-object" class="btn">객체 탐지</button>
    <button id="btn-pose" class="btn">자세(포즈) 탐지</button>
    <span class="badge">모드: <strong id="mode">-</strong></span>
    <span class="badge">FPS: <strong id="fps">0</strong></span>
    <span class="badge">CPU/메모리: <strong id="sys">-</strong></span>
  </div>
  <img id="vid" src="/video_feed" alt="video"/>
  <p><small>버튼으로 모드를 전환하면 카메라의 IMX500 모델이 교체됩니다.</small></p>
</div>
<script>
const modeEl = document.getElementById('mode');
const fpsEl  = document.getElementById('fps');
const sysEl  = document.getElementById('sys');
const bObj   = document.getElementById('btn-object');
const bPose  = document.getElementById('btn-pose');

async function setMode(m){
  const r = await fetch('/mode', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({mode:m})});
  const j = await r.json();
  modeEl.textContent = j.mode;
  bObj.classList.toggle('active', j.mode==='object');
  bPose.classList.toggle('active', j.mode==='pose');
}

async function poll(){
  try{
    const r = await fetch('/stats');
    const j = await r.json();
    modeEl.textContent = j.mode;
    fpsEl.textContent  = j.fps;
    sysEl.textContent  = `${j.cpu_percent}% / ${j.memory_percent}%`;
    bObj.classList.toggle('active', j.mode==='object');
    bPose.classList.toggle('active', j.mode==='pose');
  }catch(e){}
  setTimeout(poll, 800);
}

bObj.onclick = ()=>setMode('object');
bPose.onclick = ()=>setMode('pose');
poll();
</script>
</body>
</html>
"""

def generate():
    while True:
        frame = frame_queue.get()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/video_feed")
def video_feed():
    resp = Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers.update({'Cache-Control':'no-cache, no-store, must-revalidate','Pragma':'no-cache','Expires':'0'})
    return resp

@app.route("/mode", methods=["POST"])
def set_mode():
    global current_mode
    data = request.get_json(force=True) or {}
    m = data.get("mode")
    if m not in ("object","pose"):
        return jsonify(ok=False, error="invalid mode"), 400

    # 모델 교체
    try:
        if isinstance(camera, IMX500Camera):
            model_path = MODEL_OBJECT if m == "object" else MODEL_POSE
            camera.load_model(model_path)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

    current_mode = m
    return jsonify(ok=True, mode=current_mode)

@app.route("/stats")
def stats():
    import psutil
    with stats_lock:
        fps = round(current_fps, 1)
    cpu = psutil.cpu_percent(interval=0.0)
    mem = psutil.virtual_memory().percent
    return jsonify(mode=current_mode, fps=fps, cpu_percent=cpu, memory_percent=mem)

if __name__ == "__main__":
    # 0.0.0.0 로 모두에게 개방
    app.run(host="0.0.0.0", port=5000, threaded=True)
