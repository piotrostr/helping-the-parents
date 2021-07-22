import cv2
import time
from numpy import array
from mss import mss


def check_fps_mss(m):
    print('Measuring capture FPS...')
    fc = 0
    t0 = time.time()
    while fc < 100:
        array(m.grab(m.monitors[0]))
        fc += 1
    t1 = time.time()
    fps = int(fc / (t1 - t0))
    print(f'Capture FPS: {fps}')
    return fps


def show_frame(frame):
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        return False


def main():
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    m = mss()
    fps = check_fps_mss(m)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception('Webcam could not be initialised.')
    else:
        print('Capture open.')
    if not len(m.monitors):
        raise Exception('No display could be detected.')
    else:
        print('Display detected.')

    screen = m.monitors[0]
    h, w, _ = array(m.grab(screen)).shape
    scene_vid = cv2.VideoWriter('scene.mp4', fourcc, fps, (w, h))

    _, face = cap.read()
    h, w, _ = face.shape
    face_vid = cv2.VideoWriter('face.mp4', fourcc, fps, (w, h))

    print('Capturing...')
    while True:
        try:
            ss = m.grab(screen)
            ss = array(ss)
            ss = cv2.cvtColor(ss, cv2.COLOR_RGBA2RGB)
            _, face = cap.read()
            face_vid.write(face) 
            scene_vid.write(ss)
        except KeyboardInterrupt:
            break

    cap.release()
    face_vid.release()
    scene_vid.release()
    print('Saved the outputs.')


if __name__ == '__main__':
    main()

