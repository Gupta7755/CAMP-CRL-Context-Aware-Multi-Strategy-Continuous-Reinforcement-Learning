"""Vehicle detectors behind one interface: `detect(frame) -> list[Detection]`.

YoloxDetector  YOLOX (Megvii, Apache-2.0) exported to ONNX, run with onnxruntime on CPU.
               Chosen over Ultralytics YOLO (AGPL-3.0) so the project stays permissively licensed.
               Models: models/yolox_nano.onnx or models/yolox_tiny.onnx from the YOLOX 0.1.1rc0 release.
SyntheticDetector  reads the ground-truth boxes the synthetic camera puts in `frame.meta` and applies
               configurable misses, score noise and false positives. Used for the repeatable demo and
               tests; it is not a vision model and is labelled as such everywhere it is reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from camp_crl.vision.types import Detection, Frame

VEHICLE_CLASSES = (2, 3, 5, 7)  # COCO: car, motorcycle, bus, truck


class Detector(Protocol):
    name: str

    def detect(self, frame: Frame) -> list[Detection]: ...


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy non-maximum suppression; returns kept indices, highest score first."""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


class YoloxDetector:
    name = "yolox"

    def __init__(self, model_path: str | Path, score_thr: float = 0.35, nms_thr: float = 0.45,
                 classes: tuple[int, ...] = VEHICLE_CLASSES, threads: int = 2) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"YOLOX model not found: {path.resolve()} (see README, Phase 3 setup)")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_hw = (int(inp.shape[2]), int(inp.shape[3]))
        self.score_thr, self.nms_thr, self.classes = score_thr, nms_thr, classes
        self.name = f"yolox:{path.stem}"
        self._grids, self._strides = self._make_grids()

    def _make_grids(self) -> tuple[np.ndarray, np.ndarray]:
        grids, strides = [], []
        h, w = self.input_hw
        for s in (8, 16, 32):
            gy, gx = np.meshgrid(np.arange(h // s), np.arange(w // s), indexing="ij")
            grids.append(np.stack((gx, gy), 2).reshape(-1, 2))
            strides.append(np.full((grids[-1].shape[0], 1), s))
        return np.concatenate(grids).astype(np.float32), np.concatenate(strides).astype(np.float32)

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        import cv2

        h, w = self.input_hw
        r = min(h / img.shape[0], w / img.shape[1])
        resized = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR)
        padded = np.full((h, w, 3), 114, dtype=np.uint8)  # YOLOX letterbox, top-left aligned
        padded[: resized.shape[0], : resized.shape[1]] = resized
        return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32), r

    def detect(self, frame: Frame) -> list[Detection]:
        blob, r = self._preprocess(frame.image)
        out = self.session.run(None, {self.input_name: blob})[0][0]
        xy = (out[:, :2] + self._grids) * self._strides
        wh = np.exp(out[:, 2:4]) * self._strides
        cls_scores = out[:, 5:] * out[:, 4:5]
        cls = cls_scores.argmax(1)
        score = cls_scores[np.arange(len(cls)), cls]
        mask = (score >= self.score_thr) & np.isin(cls, self.classes)
        if not mask.any():
            return []
        boxes = np.concatenate([xy[mask] - wh[mask] / 2, xy[mask] + wh[mask] / 2], 1) / r
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame.image.shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame.image.shape[0])
        score, cls = score[mask], cls[mask]
        keep = nms(boxes, score, self.nms_thr)  # class-agnostic: one vehicle should be one box
        return [Detection(tuple(float(v) for v in boxes[i]), float(score[i]), int(cls[i])) for i in keep]


class SyntheticDetector:
    """Ground truth from the synthetic camera with seeded misses, score noise and false positives."""

    name = "synthetic"

    def __init__(self, miss_prob: float = 0.05, false_pos_per_frame: float = 0.02, seed: int = 0) -> None:
        self.miss_prob, self.fp_rate = miss_prob, false_pos_per_frame
        self.rng = np.random.default_rng(seed + 104729)

    def detect(self, frame: Frame) -> list[Detection]:
        if "gt_boxes" not in frame.meta:
            raise ValueError("SyntheticDetector needs frames from SyntheticCamera (no gt_boxes in frame.meta)")
        out = []
        for box in frame.meta["gt_boxes"]:
            if self.rng.random() < self.miss_prob:
                continue
            out.append(Detection(tuple(map(float, box)), float(np.clip(self.rng.normal(0.8, 0.08), 0.3, 1.0))))
        h, w = frame.image.shape[:2]
        for _ in range(self.rng.poisson(self.fp_rate)):
            x, y = self.rng.uniform(0, w - 40), self.rng.uniform(0, h - 40)
            out.append(Detection((x, y, x + 40, y + 40), float(self.rng.uniform(0.3, 0.5))))
        return out
