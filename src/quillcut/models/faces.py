"""Face boxes with MediaPipe on CPU. None means 'no detector installed', which is not 'no faces'."""
from __future__ import annotations


def detect_faces(frames) -> list[list[tuple[float, float, float, float]]] | None:
    """RGB uint8 frames -> per-frame boxes (x0, y0, x1, y1), normalised 0-1."""
    try:
        import mediapipe as mp
        detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
    except Exception:
        return None
    out = []
    try:
        for frame in frames:
            boxes = []
            for d in detector.process(frame).detections or []:
                b = d.location_data.relative_bounding_box
                boxes.append((b.xmin, b.ymin, b.xmin + b.width, b.ymin + b.height))
            out.append(boxes)
    finally:
        detector.close()
    return out
