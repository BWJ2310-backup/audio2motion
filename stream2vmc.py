"""Body-only EchoAvatar motion stream to VMC."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import time
from pathlib import Path

from echoavatar_config import (
    ROOT_DIR,
    get_bool,
    get_float,
    get_float_tuple,
    get_int,
    get_str,
    load_config,
    section,
)
from tools.echoavatar_stream_common import (
    StopStream,
    VmcSender,
    frame_indices,
    read_echo_chunk,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream EchoAvatar body motion to VMC.")
    parser.add_argument("--config", default=None, help="Path to EchoAvatar TOML config.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(ROOT_DIR)
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    cfg = section(load_config(args.config), "stream2vmc")
    listen_host = get_str(cfg, "listen_host", "0.0.0.0")
    listen_port = get_int(cfg, "listen_port", 12346)
    source_fps = get_int(cfg, "source_fps", 60)
    body_fps = get_int(cfg, "body_fps", 60)
    stream_fps = get_int(cfg, "stream_fps", 60)
    frame_stride = max(1, round(source_fps / body_fps)) if body_fps > 0 else 1
    frame_interval = 0.0 if get_bool(cfg, "no_pacing", False) else 1.0 / float(stream_fps)

    sender = VmcSender(
        get_str(cfg, "target_host", "127.0.0.1"),
        get_int(cfg, "target_port", 39539),
        dry_run=get_bool(cfg, "dry_run", False),
        root_scale=get_float(cfg, "root_scale", 1.0),
        root_offset=get_float_tuple(cfg, "root_offset", (0.0, 0.0, 0.0)),
        root_rotation_mode=get_str(cfg, "root_rotation_mode", "identity"),
        debug=get_bool(cfg, "debug", False),
    )
    max_packet_bytes = get_int(cfg, "max_packet_bytes", 100 * 1024 * 1024)
    debug = get_bool(cfg, "debug", False)
    running = True

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal running
        running = False
        raise StopStream(f"received signal {signum}")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    start_time = time.monotonic()
    chunks = 0

    try:
        sender.send_available()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((listen_host, listen_port))
            server.listen(1)
            server.settimeout(0.5)
            print(f"[vmc] listening for EchoAvatar body stream on {listen_host}:{listen_port}")
            print(f"[vmc] output body VMC -> {sender.target[0]}:{sender.target[1]}")

            while running:
                try:
                    client, address = server.accept()
                except socket.timeout:
                    continue
                print(f"[vmc] EchoAvatar connected from {address[0]}:{address[1]}")
                with client:
                    client.settimeout(0.5)
                    while running:
                        chunk = read_echo_chunk(client, max_packet_bytes)
                        if chunk is None:
                            print("[vmc] EchoAvatar disconnected")
                            break

                        chunks += 1
                        frame_count = min(len(chunk.pose), len(chunk.trans))
                        if frame_count <= 0:
                            continue

                        for index in frame_indices(frame_count, frame_stride):
                            timestamp = time.monotonic() - start_time
                            sender.send_frame(chunk.pose[index], chunk.trans[index], timestamp)
                            if frame_interval > 0.0:
                                time.sleep(frame_interval)

                        if debug or chunks % 30 == 0:
                            print(
                                f"[vmc] chunks={chunks} frames={frame_count} "
                                f"body_sent={sender.sent_frames}"
                            )

    except StopStream as exc:
        print(f"[vmc] stopping: {exc}")
    finally:
        sender.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
