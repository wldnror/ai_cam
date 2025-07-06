# Raspberry Pi AI Camera App

Raspberry Pi AI Camera App은 Sony IMX500 기반 AI 카메라 모듈을 우선 사용하고, CSI 카메라 또는 USB 웹캠으로 폴백하여 객체 탐지 스트리밍을 제공하는 Flask 기반 애플리케이션입니다.

## 주요 기능
- **AI 카메라 지원**  
  Sony IMX500 “Intelligent Vision Sensor” 모듈의 온-센서 NPU를 활용한 초저지연 객체 탐지  
- **폴백 모드**  
  1) IMX500 AI 카메라  
  2) Raspberry Pi CSI 카메라  
  3) USB 웹캠 (OpenCV)  
- **실시간 비디오 스트리밍**  
  Flask + multipart MJPEG 방식으로 웹에서 바로 확인  
- **시스템 상태 모니터링**  
  CPU 사용량, 메모리 사용률, 온도, Wi-Fi 신호 강도 API 제공  

## 요구사항
- Raspberry Pi OS (Bullseye 이상 권장)  
- Python 3.7 이상  

### 패키지
```bash
sudo apt update
sudo apt install -y \
  imx500-all python3-picamera2 libcamera-apps python3-libcamera \
  python3-opencv python3-psutil python3-flask python3-pip
sudo pip3 install --upgrade pip setuptools
sudo pip3 install torch torchvision ultralytics
sudo reboot
