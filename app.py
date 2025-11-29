import cv2
import time
import yaml
import json
import numpy as np
from shapely.geometry import Point, Polygon
from model_loader import ModelWrapper
from mqtt_client import MQTTClient
from gpio_alert import GPIOAlert
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sipa-ind")

# carrega config
cfg_path = Path("config.yaml")
if not cfg_path.exists():
    logger.error("Arquivo config.yaml não encontrado.")
    raise SystemExit(1)

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# valores defaults defensivos
camera_cfg = cfg.get("camera", {})
cam_source = camera_cfg.get("source", 0)  # 0 -> webcam
frame_w = camera_cfg.get("width", 1280)
frame_h = camera_cfg.get("height", 720)

model_cfg = cfg.get("model", {})
model_path = model_cfg.get("path", "models/best.pt")
device = model_cfg.get("device", "cpu")
conf_thres = model_cfg.get("conf_thres", 0.4)
iou_thres = model_cfg.get("iou_thres", 0.45)

classes_cfg = cfg.get("classes", {"person": 0, "helmet": 1, "harness": 2})
person_class = classes_cfg.get("person", 0)
helmet_class = classes_cfg.get("helmet", 1)
harness_class = classes_cfg.get("harness", 2)

perimeter_cfg = cfg.get("perimeter", {})
poly_relative = perimeter_cfg.get("polygon", [[0.2,0.6],[0.8,0.6],[0.8,1.0],[0.2,1.0]])
perimeter_polygon = Polygon(poly_relative)

mqtt_cfg = cfg.get("mqtt", {})
mqtt_client = None
try:
    mqtt_client = MQTTClient(
        host=mqtt_cfg.get("host","localhost"),
        port=mqtt_cfg.get("port",1883),
        topic=mqtt_cfg.get("topic","sipa-ind/events"),
        client_id=mqtt_cfg.get("client_id","sipa-edge-01"),
        username=mqtt_cfg.get("username"),
        password=mqtt_cfg.get("password"),
        tls=mqtt_cfg.get("tls", False)
    )
except Exception as e:
    logger.warning("MQTT não inicializado: %s", e)

gpio_cfg = cfg.get("gpio", {})
gpio = GPIOAlert(
    strobe_pin=gpio_cfg.get("strobe_pin", 18),
    buzzer_pin=gpio_cfg.get("buzzer_pin", 23),
    enabled=gpio_cfg.get("enabled", False)  # default false para segurança
)

thresholds = cfg.get("thresholds", {})
min_conf = thresholds.get("min_confidence_for_detection", 0.5)
latency_limit = thresholds.get("alert_latency_limit_s", 1.0)

EVENT_LOG = cfg.get("logging", {}).get("events_db_file", "events.log")

# carrega modelo
try:
    model = ModelWrapper(model_path, device=device, conf=conf_thres, iou=iou_thres)
except Exception as e:
    logger.exception("Não foi possível carregar o modelo: %s", e)
    raise SystemExit(1)

# abre camera
cap = cv2.VideoCapture(cam_source)
if not cap.isOpened():
    logger.error("Falha ao abrir a câmera/source: %s", cam_source)
    raise SystemExit(1)

# tenta definir tamanho, mas VideoCapture pode ignorar
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

def center_pixel_of_bbox(box):
    # box: [x1,y1,x2,y2] (pixels)
    x1,y1,x2,y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return float(cx), float(cy)

def is_center_inside_bbox(center, bbox):
    cx, cy = center
    x1,y1,x2,y2 = bbox
    return (cx >= x1) and (cx <= x2) and (cy >= y1) and (cy <= y2)

def relative_point_from_pixel(cx, cy, w=frame_w, h=frame_h):
    return (cx / w, cy / h)

try:
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame não recebido; tentando novamente.")
            time.sleep(0.05)
            continue

        # garante tamanho esperado
        frame = cv2.resize(frame, (frame_w, frame_h))

        boxes, scores, classes = model.infer(frame)

        # filtra por confiança mínima
        keep_idx = [i for i, s in enumerate(scores) if s >= min_conf]
        if len(keep_idx) == 0:
            latency = time.time() - t0
            if latency > latency_limit:
                logger.warning("Latência de ciclo alta (sem detecções): %.3fs", latency)
            continue

        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        classes = classes[keep_idx]

        # índices por classe
        people_idx = [i for i,c in enumerate(classes) if c == person_class]
        helmet_idx = [i for i,c in enumerate(classes) if c == helmet_class]
        harness_idx = [i for i,c in enumerate(classes) if c == harness_class]

        for pi in people_idx:
            pbox = boxes[pi]  # xyxy pixels
            pcx, pcy = center_pixel_of_bbox(pbox)
            # criar ponto relativo para perímetro
            rel_point = relative_point_from_pixel(pcx, pcy, frame_w, frame_h)
            point = Point(rel_point)
            in_perimeter = perimeter_polygon.contains(point)

            # verifica EPI: procura capacete/harness cujo centro esteja dentro da bbox da pessoa
            has_helmet = False
            for hi in helmet_idx:
                hbox = boxes[hi]
                hcx, hcy = center_pixel_of_bbox(hbox)
                if is_center_inside_bbox((hcx, hcy), pbox):
                    has_helmet = True
                    break

            has_harness = False
            for hi in harness_idx:
                hbox = boxes[hi]
                hcx, hcy = center_pixel_of_bbox(hbox)
                if is_center_inside_bbox((hcx, hcy), pbox):
                    has_harness = True
                    break

            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            event = None

            if in_perimeter and not (has_helmet and has_harness):
                event = {
                    "type": "CRITICAL_VIOLATION",
                    "timestamp": timestamp,
                    "camera": str(cam_source),
                    "person_bbox": [float(v) for v in pbox],
                    "has_helmet": bool(has_helmet),
                    "has_harness": bool(has_harness),
                    "in_perimeter": True
                }
                logger.warning("Alerta crítico: %s", event)
                gpio.alert_on(duration_s=1.0)
            elif in_perimeter:
                event = {
                    "type": "PERIMETER_INTRUSION",
                    "timestamp": timestamp,
                    "camera": str(cam_source),
                    "person_bbox": [float(v) for v in pbox],
                    "has_helmet": bool(has_helmet),
                    "has_harness": bool(has_harness),
                    "in_perimeter": True
                }
                logger.info("Intrusão: %s", event)
                gpio.alert_on(duration_s=0.5)
            elif not (has_helmet and has_harness):
                event = {
                    "type": "EPI_MISSING",
                    "timestamp": timestamp,
                    "camera": str(cam_source),
                    "person_bbox": [float(v) for v in pbox],
                    "has_helmet": bool(has_helmet),
                    "has_harness": bool(has_harness),
                    "in_perimeter": False
                }
                logger.info("EPI faltando (registro): %s", event)

            if event:
                try:
                    with open(EVENT_LOG, "a") as fh:
                        fh.write(json.dumps(event) + "\n")
                except Exception:
                    logger.exception("Falha ao gravar evento localmente.")
                if mqtt_client:
                    mqtt_client.publish_event(event)

        latency = time.time() - t0
        if latency > latency_limit:
            logger.warning("Latência alta no ciclo: %.3f s", latency)

except KeyboardInterrupt:
    logger.info("Encerrando por KeyboardInterrupt")
except Exception as e:
    logger.exception("Erro não tratado no loop principal: %s", e)
finally:
    try:
        cap.release()
    except Exception:
        pass
    try:
        gpio.cleanup()
    except Exception:
        pass
    try:
        if mqtt_client:
            mqtt_client.stop()
    except Exception:
        pass

    logger.info("Processo finalizado.")
