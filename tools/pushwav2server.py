#!/usr/bin/env python3
"""
Stream microphone or virtual-audio input to the EchoAvatar inference server.

This keeps EchoAvatar's original protocol:
  4-byte big-endian pickle payload length, followed by a pickled numpy block.

Run this script on the machine that captures application or microphone audio.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from echoavatar_config import get_int, get_str, load_config, section


def should_stop_on_hotkey() -> bool:
    try:
        import keyboard

        return keyboard.is_pressed("ctrl+space")
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream local audio input to EchoAvatar's audio TCP server."
    )
    parser.add_argument("--config", default=None, help="Path to EchoAvatar TOML config.")
    parser.add_argument(
        "--server-ip",
        default=None,
        help="EchoAvatar inference server IP or hostname.",
    )
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice input device index. Use tools/get_device.py to list devices.",
    )
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--rate", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument(
        "--no-hotkey",
        action="store_true",
        help="Disable Ctrl+Space stop hotkey and stop only with Ctrl+C.",
    )
    args = parser.parse_args()

    cfg = section(load_config(args.config), "audio_sender")
    args.server_ip = args.server_ip or get_str(cfg, "server_host", "127.0.0.1")
    args.server_port = args.server_port or get_int(cfg, "server_port", 12345)
    args.device = args.device if args.device is not None else get_int(cfg, "device", 2)
    args.channels = args.channels or get_int(cfg, "channels", 1)
    args.rate = args.rate or get_int(cfg, "rate", 24000)
    args.chunk = args.chunk or get_int(cfg, "chunk", 1000)
    return args


def main() -> int:
    args = parse_args()

    try:
        import sounddevice as sd
    except ModuleNotFoundError as exc:
        if exc.name != "sounddevice":
            raise
        raise SystemExit(
            "Missing dependency: sounddevice. Install it on the audio-capture "
            "machine with `py -m pip install -r requirements-client.txt`.\n"
            f"Python executable: {sys.executable}\n"
            f"Python path: {sys.path}"
        ) from exc

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((args.server_ip, args.server_port))

    stream = sd.InputStream(
        device=args.device,
        channels=args.channels,
        samplerate=args.rate,
        blocksize=args.chunk,
    )

    try:
        print(
            f"Start transmitting audio to {args.server_ip}:{args.server_port} "
            f"from device={args.device}, rate={args.rate}, chunk={args.chunk}"
        )
        with stream:
            while True:
                data, _overflowed = stream.read(args.chunk)
                data_bytes = pickle.dumps(data)
                client_socket.sendall(len(data_bytes).to_bytes(4, byteorder="big"))
                client_socket.sendall(data_bytes)

                if not args.no_hotkey and should_stop_on_hotkey():
                    print("Detected Ctrl+Space, stopping recording.")
                    client_socket.sendall((0).to_bytes(4, byteorder="big"))
                    break

    except KeyboardInterrupt:
        print("Interrupted, stopping recording.")
        try:
            client_socket.sendall((0).to_bytes(4, byteorder="big"))
        except OSError:
            pass
    finally:
        client_socket.close()
        print("Connection closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
