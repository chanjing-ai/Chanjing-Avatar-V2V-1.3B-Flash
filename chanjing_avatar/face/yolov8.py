from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from ultralytics import YOLO


ArrayLikePath = Union[str, Path]


class Yolov8sFaceDetector:
    """独立版 yolov8s-face 人脸检测器。

    输入:
        image_bgr: OpenCV 读取的图片, 即 numpy.ndarray, shape 为 (H, W, 3), BGR 顺序。

    输出:
        boxes_xyxy: numpy.ndarray, shape 为 (N, 4), dtype float32。
                    每行为 [x1, y1, x2, y2], 坐标单位为原图像素。
        scores: numpy.ndarray, shape 为 (N,), dtype float32。
                每个检测框对应的人脸置信度。
    """

    def __init__(
        self,
        weight_path: Optional[ArrayLikePath] = None,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: Optional[Union[str, int]] = None,
        max_det: int = 300,
        half: bool = False,
    ) -> None:
        """初始化检测器。

        Args:
            weight_path: 模型权重路径。默认使用当前文件同级目录下 weights/face_yolov8s.pt。
            conf_threshold: 置信度阈值。
            iou_threshold: NMS IoU 阈值。
            imgsz: 推理输入尺寸，默认 640。
            device: 推理设备，例如 "cuda:0"、"cpu"。默认有 CUDA 用 cuda:0，否则 cpu。
            max_det: 单张图片最多输出检测框数量。
            half: 是否使用 FP16 推理。通常仅建议 CUDA 上开启。
        """
        if weight_path is None:
            weight_path = Path(__file__).resolve().parent / "weights" / "face_yolov8s.pt"
        self.weight_path = Path(weight_path).expanduser().resolve()
        if not self.weight_path.is_file():
            raise FileNotFoundError(f"未找到 yolov8s-face 权重文件: {self.weight_path}")

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.device = device
        self.max_det = int(max_det)
        self.half = bool(half)

        self.model = YOLO(str(self.weight_path))

    @torch.no_grad()
    def detect(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测单张 OpenCV BGR 图片中的人脸。

        Args:
            image_bgr: OpenCV 读取的图片，np.ndarray，shape=(H, W, 3)，BGR 格式。

        Returns:
            boxes_xyxy, scores:
                boxes_xyxy 为 (N, 4) float32 数组，格式 [x1, y1, x2, y2]。
                scores 为 (N,) float32 数组。
                无检测结果时返回 shape=(0, 4) 和 shape=(0,) 的空数组。
        """
        self._validate_image(image_bgr)

        results = self.model.predict(
            source=image_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_det,
            half=self.half,
            verbose=False,
        )

        if not results:
            return self._empty_result()

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return self._empty_result()

        boxes_xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32, copy=False)
        scores = boxes.conf.detach().cpu().numpy().astype(np.float32, copy=False)
        return boxes_xyxy, scores

    @staticmethod
    def _empty_result() -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    @staticmethod
    def _validate_image(image_bgr: np.ndarray) -> None:
        if not isinstance(image_bgr, np.ndarray):
            raise TypeError(f"image_bgr 必须是 numpy.ndarray，当前类型: {type(image_bgr)!r}")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(
                "image_bgr 必须是 OpenCV BGR 三通道图片，shape 应为 (H, W, 3)，"
                f"当前 shape: {image_bgr.shape}"
            )
        if image_bgr.size == 0:
            raise ValueError("image_bgr 不能为空图像")


def detect_faces(
    image_bgr: np.ndarray,
    weight_path: Optional[ArrayLikePath] = None,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    device: Optional[Union[str, int]] = None,
    max_det: int = 300,
    half: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """一次性检测函数。

    说明:
        如果需要处理多张图片，建议复用 Yolov8sFaceDetector 实例，避免重复加载模型。
    """
    detector = Yolov8sFaceDetector(
        weight_path=weight_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        device=device,
        max_det=max_det,
        half=half,
    )
    return detector.detect(image_bgr)