# mqtt_client.py
import json
import logging
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)

class MQTTClient:
    def __init__(self, host, port, topic, client_id="sipa-edge", username=None, password=None, tls=False, keepalive=60):
        self.topic = topic
        self.client = mqtt.Client(client_id=client_id, clean_session=True)

        if username:
            self.client.username_pw_set(username, password)

        if tls:
            try:
                self.client.tls_set()
            except Exception as e:
                logger.warning("Falha ao configurar TLS: %s", e)

        # callbacks simples para debug
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                logger.info("MQTT conectado ao broker %s:%s", host, port)
            else:
                logger.warning("MQTT falhou ao conectar, rc=%s", rc)

        def on_disconnect(client, userdata, rc):
            logger.info("MQTT desconectado rc=%s", rc)

        self.client.on_connect = on_connect
        self.client.on_disconnect = on_disconnect

        try:
            self.client.connect(host, port, keepalive=keepalive)
            self.client.loop_start()
        except Exception as e:
            logger.exception("Erro ao conectar ao broker MQTT: %s", e)
            raise

    def publish_event(self, event: dict, qos=1, retain=False):
        try:
            payload = json.dumps(event)
            self.client.publish(self.topic, payload, qos=qos, retain=retain)
        except Exception as e:
            logger.exception("Falha ao publicar evento MQTT: %s", e)

    def stop(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
