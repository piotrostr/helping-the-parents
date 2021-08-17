import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse 
import face_alignment


def crop_eyes_of_video(path, output_path, fa):
    cap = cv2.VideoCapture(path)
    framerate = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, int(framerate), (256, 128))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        [landmarks] = fa.get_landmarks(frame)
        landmarks = landmarks.astype(int)
        left_x, left_y = landmarks[37]
        right_x, right_y = landmarks[43]
        try:
            print('.')
            eye_right = frame[right_y - 64: right_y + 64,
                              right_x - 64: right_x + 64]
            eye_left = frame[left_y - 64: left_y + 64, 
                             left_x - 64: left_x + 64]
            eyes_bit = np.concatenate((eye_right, eye_left), axis=1)
            out.write(eyes_bit)
        except IndexError:
            h, w, _ = frame.shape
            raise Exception(f'input dims too low ({w}x{h})')
    cap.release()
    out.release()
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='video path')
    parser.add_argument('output_path', type=str, help='output path')
    args = parser.parse_args()

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                      flip_input=False,
                                      device='cpu',
                                      face_detector='blazeface')
    crop_eyes_of_video(args.path, args.output_path, fa)

