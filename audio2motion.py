#!/usr/bin/env python3
"""Config-backed launcher for the EchoAvatar audio-to-motion inference service."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys

from echoavatar_config import ROOT_DIR, get_bool, get_int, get_str, load_config, resolve_repo_path, section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start EchoAvatar audio2motion from TOML config.")
    parser.add_argument("--config", default=None, help="Path to EchoAvatar TOML config.")
    parser.add_argument("--motion-receiver", default=None, help="Override [[motion_receivers]] host.")
    return parser.parse_args()


def enabled_motion_receivers(config: dict) -> list[dict[str, object]]:
    receivers = []
    for item in config.get("motion_receivers", []):
        if not isinstance(item, dict):
            continue
        if not get_bool(item, "enabled", True):
            continue
        receivers.append(
            {
                "name": get_str(item, "name", f"receiver_{len(receivers) + 1}"),
                "host": get_str(item, "host", "127.0.0.1"),
                "port": get_int(item, "port", 12346),
            }
        )
    return receivers


def main() -> int:
    args = parse_args()
    os.chdir(ROOT_DIR)
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    config = load_config(args.config)
    cfg = section(config, "audio2motion")
    receivers = enabled_motion_receivers(config)
    if args.motion_receiver:
        receivers = [{**receiver, "host": args.motion_receiver} for receiver in receivers]
    if not receivers:
        raise SystemExit("No enabled [[motion_receivers]] entries found in config.")

    os.environ["ECHOAVATAR_HOST"] = get_str(cfg, "listen_host", "0.0.0.0")
    os.environ["ECHOAVATAR_AUDIO_PORT"] = str(get_int(cfg, "audio_port", 12345))
    os.environ["ECHOAVATAR_MOTION_RECEIVERS"] = json.dumps(receivers)
    os.environ["PROFILE_TIMING"] = "1" if get_bool(cfg, "profile_timing", False) else "0"
    os.environ["PROFILE_SYNC"] = "1" if get_bool(cfg, "profile_sync", False) else "0"

    cuda_visible_devices = cfg.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    first = receivers[0]
    os.environ["ECHOAVATAR_MOTION_SERVER_HOST"] = str(first["host"])
    os.environ["ECHOAVATAR_MOTION_SERVER_PORT"] = str(first["port"])

    script = resolve_repo_path(get_str(cfg, "script", "scripts/streaming_audio2motion_30fps_bp_attn4_encodec2_multirvq_nbc512_withface_ik.py"))
    model_name = get_str(cfg, "model_name", "./ckpts/body_g_d")

    print(
        "[audio2motion] motion receivers: "
        + ", ".join(f"{item['name']}={item['host']}:{item['port']}" for item in receivers)
    )
    print(f"[audio2motion] audio listen port: {os.environ['ECHOAVATAR_AUDIO_PORT']}")
    print(f"[audio2motion] model: {model_name}")

    sys.argv = [str(script), "--model-name", model_name]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
