import cv2
import time
from numpy import array
from mss import mss


def check_fps(face_vid, scene_vid, cap, m):
    print('Measuring capture FPS...')
    n = 0
    t0 = time.time()
    screen = m.monitors[0]
    while n < 150:
        ss = m.grab(screen)
        ss = array(ss)
        ss = cv2.cvtColor(ss, cv2.COLOR_RGBA2RGB)
        _, face = cap.read()
        face_vid.write(face) 
        scene_vid.write(ss)
        n += 1
    t1 = time.time()
    fps = n / (t1 - t0)
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
    sh, sw, _ = array(m.grab(screen)).shape
    _, face = cap.read()
    fh, fw, _ = face.shape
    fps = 25

    scene_vid = cv2.VideoWriter('videos/scene.mp4', fourcc, fps, (sw, sh))
    face_vid = cv2.VideoWriter('videos/face.mp4', fourcc, fps, (fw, fh))

    # override using the actual fps
    fps = check_fps(face_vid, scene_vid, cap, m)
    scene_vid = cv2.VideoWriter('videos/scene.mp4', fourcc, fps, (sw, sh))
    face_vid = cv2.VideoWriter('videos/face.mp4', fourcc, fps, (fw, fh))

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

