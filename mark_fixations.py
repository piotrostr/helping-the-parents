import cv2
import torch


def mark_fixations(model_output, scene_path, out_path):
    with open(model_output, 'rb') as f:
        model_output = torch.load(f, map_location='cpu')
    cap = cv2.VideoCapture(scene_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    out_vid = cv2.VideoWriter(out_path, fourcc, int(fps), (w, h))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    assert len(frames) == len(model_output['PoG_px_final'])
    for frame, (x, y) in zip(frames, model_output['PoG_px_final']):
        frame = cv2.circle(frame, (x, y), 15, (255, 255, 255), thickness=2)
        out_vid.write(frame)
    out_vid.release()
    cap.release()
    

if __name__ == '__main__':
    mark_fixations('./out.pkl', './data/scene.mp4', './data/scene_out.mp4')

