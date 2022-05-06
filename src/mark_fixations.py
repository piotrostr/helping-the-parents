from typing import List
import cv2
import torch


def mark_fixations(model_output, scene_path, out_path):
    with open(model_output, 'rb') as f:
        model_output = torch.load(f, map_location='cpu')
    cap = cv2.VideoCapture(scene_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_vid = cv2.VideoWriter(out_path, fourcc, float(fps), (1920, 1080))
    frames = []
    for i in range(len(model_output['PoG_px_final'])):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.resize(frame, (1920, 1080))
        frames.append(frame)
    for frame, (x, y) in zip(frames, model_output['PoG_px_final']):
        coords = (int(x), int(y))
        frame = cv2.circle(frame, coords, 15, (255, 255, 255), thickness=2)
        out_vid.write(frame)
    out_vid.release()
    cap.release()


def mark_fixations_points(fixations: List[List[int]], scene_path, out_path):
    cap = cv2.VideoCapture(scene_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_vid = cv2.VideoWriter(out_path, fourcc, fps, (1920, 1080))
    frames = []
    for _ in range(len(fixations)):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1920, 1080))
        if not ret:
            break
        frames.append(frame)
    for frame, (x, y) in zip(frames, fixations):
        coords = (int(x), int(y))
        frame = cv2.circle(frame, coords, 15, (255, 255, 255), thickness=2)
        out_vid.write(frame)
    out_vid.release()
    cap.release()
    print(f'Written {out_path}.')


if __name__ == '__main__':
    from .fixations import fixations
    mark_fixations_points(fixations, 'data/scene.mp4', 'data/scene_out.mp4')
