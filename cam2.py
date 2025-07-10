~/ai_cam $ export DISPLAY=:0
python3 cam2.py
setterm: terminal xterm-256color does not support --blank
setterm: cannot (un)set powersave mode: Inappropriate ioctl for device
xset:  unable to open display ":0"
xset:  unable to open display ":0"
xset:  unable to open display ":0"
Traceback (most recent call last):
  File "/home/user/ai_cam/cam2.py", line 42, in <module>
    camera = ScreenCamera()
             ^^^^^^^^^^^^^^
  File "/home/user/ai_cam/cam2.py", line 27, in __init__
    self.sct = mss.mss()
               ^^^^^^^^^
  File "/home/user/ai_cam/camenv/lib/python3.11/site-packages/mss/factory.py", line 32, in mss
    return linux.MSS(**kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/user/ai_cam/camenv/lib/python3.11/site-packages/mss/linux.py", line 319, in __init__
    raise ScreenShotError(msg)
mss.exception.ScreenShotError: Unable to open display: b':0'.
(camenv) user@raspberrypi:~/ai_cam $ 
