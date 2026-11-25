"""Replay two MIT-BIH blocks through the ECG firmware over UART."""

import argparse
import math
from pathlib import Path
import time

FIXTURES = Path(__file__).resolve().parent / "tests" / "fixtures"
BLOCK_BYTES = 930 * 4  # float32 little-endian
RECORDINGS = (
    ("116", "mit-bih-116-251.f32le", 2),
    ("119", "mit-bih-119-193.f32le", 1),
)
BAUD_RATE = 115200
READY_MARKER = b"ECG listo por UART"
BOOT_TIMEOUT = 6.0
RESPONSE_TIMEOUT = 4.0


class ReplayError(RuntimeError):
    """The fixture or firmware output did not match the demo protocol."""


def load_block(filename: str) -> bytes:
    payload = (FIXTURES / filename).read_bytes()
    if len(payload) != BLOCK_BYTES:
        raise ReplayError(f"{filename} debe tener {BLOCK_BYTES} bytes. Tiene {len(payload)}.")
    return payload


def restart_firmware(port):
    port.reset_input_buffer()
    port.rts = True
    time.sleep(0.12)
    port.rts = False
    boot = bytearray()
    deadline = time.monotonic() + BOOT_TIMEOUT
    while time.monotonic() < deadline:
        boot.extend(port.read(port.in_waiting or 1))
        marker = boot.find(READY_MARKER)
        if marker != -1 and boot.find(b"\n", marker + len(READY_MARKER)) != -1:
            port.reset_input_buffer()
            return
    raise ReplayError("La placa no indicó que el firmware ECG estaba listo.")


def send_block(port, payload: bytes, expected_responses: int) -> list[float]:
    if port.write(payload) != len(payload):
        raise ReplayError("La escritura por serie quedó incompleta.")
    port.flush()

    pending = bytearray()
    values = []
    # The stream ends each value with a comma and has no end-of-block marker.
    deadline = time.monotonic() + RESPONSE_TIMEOUT
    while time.monotonic() < deadline:
        pending.extend(port.read(port.in_waiting or 1))
        *received, pending = pending.split(b",")
        for field in received:
            try:
                value = float(field)
            except ValueError as error:
                raise ReplayError("La placa devolvió un dato inválido por serie.") from error
            if not math.isfinite(value) or value < 0:
                raise ReplayError("La placa devolvió un error de reconstrucción inválido.")
            values.append(value)
        if len(values) > expected_responses:
            raise ReplayError(f"Se esperaban {expected_responses} respuestas y se recibieron más.")

    if pending.strip():
        raise ReplayError("La placa dejó una respuesta incompleta.")
    if len(values) != expected_responses:
        raise ReplayError(
            f"Se esperaban {expected_responses} respuestas y se recibieron {len(values)}."
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Envía dos tramos de ECG por serie y muestra el error de reconstrucción (MAE)."
        ),
        epilog="La placa se reinicia por RTS antes de cada tramo.",
    )
    parser.add_argument("port", help="Puerto serie de la ESP32-S3, por ejemplo /dev/ttyACM0")
    args = parser.parse_args()

    try:
        import serial
    except ModuleNotFoundError:
        parser.exit(
            1,
            "Falta pySerial. Instalalo con python -m pip install -r requirements-demo.txt.\n",
        )

    try:
        blocks = [(record, load_block(filename), count) for record, filename, count in RECORDINGS]
        port = serial.Serial(baudrate=BAUD_RATE, timeout=0.1, write_timeout=2, exclusive=True)
        port.dtr = False
        port.rts = False
        port.port = args.port
        with port:
            for record, payload, count in blocks:
                try:
                    restart_firmware(port)
                    values = send_block(port, payload, count)
                except (OSError, ReplayError) as error:
                    raise ReplayError(f"MIT-BIH {record}. {error}") from error
                formatted = ", ".join(f"{value:.6f}" for value in values)
                print(f"MIT-BIH {record} · MAE {formatted}", flush=True)
    except (OSError, ReplayError) as error:
        parser.exit(1, f"Falló la prueba por serie. {error}\n")


if __name__ == "__main__":
    main()
