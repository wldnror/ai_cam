#!/usr/bin/env python3
import time
from picamera2 import Picamera2
from picamera2.devices import IMX500

MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
PP_FILE   = "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json"

# 1) IMX500 + Picamera2 세팅
imx500 = IMX500(MODEL_PATH)
picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    controls={"FrameRate": imx500.network_intrinsics.inference_rate}
)
config["post_process_file"] = PP_FILE
picam2.configure(config)
imx500.show_network_fw_progress_bar()
picam2.start()

# 2) 메타데이터 덤프
for i in range(5):
    md = picam2.capture_metadata()  # 이미지 없이 메타데이터만 가져옴
    print(f"\n--- frame {i} metadata keys:", md.keys())
    for k, v in md.items():
        print(f"{k!r}:", v)
    time.sleep(1)
