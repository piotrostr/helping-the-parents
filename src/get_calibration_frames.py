import os
import cv2
import time


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    i = 0
    if 'data' not in os.listdir():
        os.mkdir('data')
    files = sorted(os.listdir('data'))
    if len(files):
        latest_idx, ext = files[-1].split('.')
    else:
        latest_idx = 1
    latest_idx = int(latest_idx)
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        i += 1
        if not ret:
            break
        time_now = time.time()
        fps = i / (time_now - start_time)
        cv2.putText(frame, f'fps: {fps:.2f}', (7, 70), font, 1, (255, 0, 0))
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            gray = cv2.cvtColor(frame, 0)
            got_corners, corners = cv2.findChessboardCorners(gray, (7,6), None)
            if got_corners:
                fname = f'data/{latest_idx}.jpg'
                cv2.imwrite(fname, frame)
                latest_idx += 1
                print(f'written {fname}')
            else:
                print('could not detect corners in that frame')
            i = 0
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

