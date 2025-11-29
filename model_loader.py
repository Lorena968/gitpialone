# model_loader.py
from ultralytics import YOLO
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ModelWrapper:
    """
    Wrapper simples para o modelo Ultralytics YOLO.
    Recebe frames (BGR numpy arrays) e retorna boxes (xyxy), scores e classes como numpy arrays.
    """
    def __init__(self, model_path: str, device: str = "cpu", conf: float = 0.25, iou: float = 0.45):
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.iou = iou

        try:
            self.model = YOLO(model_path)
            # ultralytics aceita .to(device) mas em algumas versões pode falhar silenciosamente
            try:
                self.model.to(self.device)
            except Exception:
                logger.debug("Não foi possível setar device com model.to(device); continuando (pode estar OK).")
        except Exception as e:
            logger.exception("Falha ao carregar o modelo: %s", e)
            raise

    def infer(self, frame):
        """
        frame: BGR numpy array (H, W, C)
        retorna: boxes (N,4) xyxy em pixels, scores (N,), classes (N,)
        """
        if frame is None:
            return np.empty((0,4)), np.empty((0,)), np.empty((0,), dtype=int)

        try:
            # Ultralytics aceita receber numpy arrays diretamente: self.model(frame)
            results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
            if len(results) == 0:
                return np.empty((0,4)), np.empty((0,)), np.empty((0,), dtype=int)

            r = results[0]

            # r.boxes pode ser vazio
            if not hasattr(r, "boxes") or len(r.boxes) == 0:
                return np.empty((0,4)), np.empty((0,)), np.empty((0,), dtype=int)

            # Extrai dados com segurança
            try:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy().astype(int)
            except Exception:
                # fallback: r.boxes.xyxy pode já ser numpy
                boxes = np.array(r.boxes.xyxy).astype(float)
                scores = np.array(r.boxes.conf).astype(float)
                classes = np.array(r.boxes.cls).astype(int)

            return boxes, scores, classes
        except Exception as e:
            logger.exception("Erro na inferência: %s", e)
            return np.empty((0,4)), np.empty((0,)), np.empty((0,), dtype=int)
