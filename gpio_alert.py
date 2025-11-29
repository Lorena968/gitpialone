# gpio_alert.py
import time
import logging

logger = logging.getLogger(__name__)

# Tenta importar bibliotecas típicas; se não houver, faz fallback para simulação
GPIO = None
PLATFORM = "none"
try:
    import Jetson.GPIO as GPIO
    PLATFORM = "jetson"
except Exception:
    try:
        import RPi.GPIO as GPIO
        PLATFORM = "rpi"
    except Exception:
        GPIO = None
        PLATFORM = "none"

class GPIOAlert:
    def __init__(self, strobe_pin=18, buzzer_pin=23, enabled=True):
        self.enabled = enabled and (GPIO is not None)
        self.strobe_pin = strobe_pin
        self.buzzer_pin = buzzer_pin

        if self.enabled:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.strobe_pin, GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(self.buzzer_pin, GPIO.OUT, initial=GPIO.LOW)
            except Exception as e:
                logger.warning("Erro ao inicializar GPIO (%s). Desativando GPIO: %s", PLATFORM, e)
                self.enabled = False

    def alert_on(self, duration_s=1.0):
        if not self.enabled:
            logger.info("[GPIO_SIM] alert_on por %.2fs (strobe=%s buzzer=%s)", duration_s, self.strobe_pin, self.buzzer_pin)
            return
        try:
            GPIO.output(self.strobe_pin, GPIO.HIGH)
            GPIO.output(self.buzzer_pin, GPIO.HIGH)
            time.sleep(duration_s)
            GPIO.output(self.strobe_pin, GPIO.LOW)
            GPIO.output(self.buzzer_pin, GPIO.LOW)
        except Exception as e:
            logger.exception("Erro ao executar alerta GPIO: %s", e)

    def cleanup(self):
        if self.enabled:
            try:
                GPIO.cleanup()
            except Exception as e:
                logger.warning("Erro ao limpar GPIO: %s", e)
