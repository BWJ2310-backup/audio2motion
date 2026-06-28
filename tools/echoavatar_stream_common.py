#!/usr/bin/env python3
"""Shared packet and sender utilities for EchoAvatar stream services."""

from __future__ import annotations

import datetime
import json
import math
import socket
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ECHOAVATAR_BONES = """
Root_M Hip_R HipPart1_R Knee_R KneePart1_R Ankle_R Toes_R ToesEnd_R Heel_R
HeelEnd_R Spine1_M Spine1Part1_M Chest_M Scapula_R Shoulder_R ShoulderPart1_R
Elbow_R ElbowPart1_R Wrist_R MiddleFinger1_R MiddleFinger2_R MiddleFinger3_R
MiddleFinger4_R ThumbFinger1_R ThumbFinger2_R ThumbFinger3_R ThumbFinger4_R
IndexFinger1_R IndexFinger2_R IndexFinger3_R IndexFinger4_R Cup_R PinkyFinger1_R
PinkyFinger2_R PinkyFinger3_R PinkyFinger4_R RingFinger1_R RingFinger2_R
RingFinger3_R RingFinger4_R Neck_M NeckPart1_M Head_M Head_angleFix L_eye_jnt
L_eScale_jnt Eye_L L_pupil_jnt R_eye_jnt R_eScale_jnt Eye_R R_pupil_jnt
Scapula_L Shoulder_L ShoulderPart1_L Elbow_L ElbowPart1_L Wrist_L
MiddleFinger1_L MiddleFinger2_L MiddleFinger3_L MiddleFinger4_L ThumbFinger1_L
ThumbFinger2_L ThumbFinger3_L ThumbFinger4_L IndexFinger1_L IndexFinger2_L
IndexFinger3_L IndexFinger4_L Cup_L PinkyFinger1_L PinkyFinger2_L PinkyFinger3_L
PinkyFinger4_L RingFinger1_L RingFinger2_L RingFinger3_L RingFinger4_L Hip_L
HipPart1_L Knee_L KneePart1_L Ankle_L Toes_L ToesEnd_L Heel_L HeelEnd_L
""".split()


BONE_INDEX = {name: index for index, name in enumerate(ECHOAVATAR_BONES)}


_VMC_REFERENCE_BASIS_QUATS: dict[str, tuple[float, float, float, float]] | None = None


# VMC body output uses VRM humanoid bone names. Face data is intentionally not
# sent through VMC; ARKit face coefficients belong to LiveLink.
VMC_TO_ECHO_BONE = {
    "Hips": ("Root_M",),
    "Spine": ("Spine1_M",),
    "Chest": ("Spine1Part1_M",),
    "UpperChest": ("Chest_M",),
    "ChestUpper": ("Chest_M",),
    "Neck": ("Neck_M", "NeckPart1_M"),
    "Head": ("Head_M",),
    "RightShoulder": ("Scapula_R",),
    "RightUpperArm": ("Shoulder_R", "ShoulderPart1_R"),
    "RightLowerArm": ("Elbow_R", "ElbowPart1_R"),
    "RightHand": ("Wrist_R",),
    "LeftShoulder": ("Scapula_L",),
    "LeftUpperArm": ("Shoulder_L", "ShoulderPart1_L"),
    "LeftLowerArm": ("Elbow_L", "ElbowPart1_L"),
    "LeftHand": ("Wrist_L",),
    "RightUpperLeg": ("Hip_R", "HipPart1_R"),
    "RightLowerLeg": ("Knee_R", "KneePart1_R"),
    "RightFoot": ("Ankle_R",),
    "RightToes": ("Toes_R",),
    "LeftUpperLeg": ("Hip_L", "HipPart1_L"),
    "LeftLowerLeg": ("Knee_L", "KneePart1_L"),
    "LeftFoot": ("Ankle_L",),
    "LeftToes": ("Toes_L",),
    "RightThumbProximal": ("ThumbFinger1_R",),
    "RightThumbIntermediate": ("ThumbFinger2_R",),
    "RightThumbDistal": ("ThumbFinger3_R",),
    "RightIndexProximal": ("IndexFinger1_R",),
    "RightIndexIntermediate": ("IndexFinger2_R",),
    "RightIndexDistal": ("IndexFinger3_R",),
    "RightMiddleProximal": ("MiddleFinger1_R",),
    "RightMiddleIntermediate": ("MiddleFinger2_R",),
    "RightMiddleDistal": ("MiddleFinger3_R",),
    "RightRingProximal": ("RingFinger1_R",),
    "RightRingIntermediate": ("RingFinger2_R",),
    "RightRingDistal": ("RingFinger3_R",),
    "RightLittleProximal": ("PinkyFinger1_R",),
    "RightLittleIntermediate": ("PinkyFinger2_R",),
    "RightLittleDistal": ("PinkyFinger3_R",),
    "LeftThumbProximal": ("ThumbFinger1_L",),
    "LeftThumbIntermediate": ("ThumbFinger2_L",),
    "LeftThumbDistal": ("ThumbFinger3_L",),
    "LeftIndexProximal": ("IndexFinger1_L",),
    "LeftIndexIntermediate": ("IndexFinger2_L",),
    "LeftIndexDistal": ("IndexFinger3_L",),
    "LeftMiddleProximal": ("MiddleFinger1_L",),
    "LeftMiddleIntermediate": ("MiddleFinger2_L",),
    "LeftMiddleDistal": ("MiddleFinger3_L",),
    "LeftRingProximal": ("RingFinger1_L",),
    "LeftRingIntermediate": ("RingFinger2_L",),
    "LeftRingDistal": ("RingFinger3_L",),
    "LeftLittleProximal": ("PinkyFinger1_L",),
    "LeftLittleIntermediate": ("PinkyFinger2_L",),
    "LeftLittleDistal": ("PinkyFinger3_L",),
}


class StopStream(Exception):
    pass


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, float(value)))


def normalize_xyzw(quat: Sequence[float]) -> tuple[float, float, float, float]:
    if len(quat) < 4:
        return 0.0, 0.0, 0.0, 1.0
    x, y, z, w = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-8:
        return 0.0, 0.0, 0.0, 1.0
    return x / norm, y / norm, z / norm, w / norm


def multiply_xyzw(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return normalize_xyzw(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        )
    )


def invert_unit_xyzw(
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, z, w = normalize_xyzw(quat)
    return -x, -y, -z, w


def get_vmc_reference_basis_quats() -> dict[str, tuple[float, float, float, float]]:
    global _VMC_REFERENCE_BASIS_QUATS
    if _VMC_REFERENCE_BASIS_QUATS is not None:
        return _VMC_REFERENCE_BASIS_QUATS

    root_dir = Path(__file__).resolve().parents[1]
    basis_path = root_dir / "stats" / "blender2inzoi_vmc_rest_basis.json"
    try:
        data = json.loads(basis_path.read_text())
        basis = data.get("basis", {})
        _VMC_REFERENCE_BASIS_QUATS = {
            str(name): normalize_xyzw(entry["quat_xyzw"])
            for name, entry in basis.items()
            if "quat_xyzw" in entry
        }
    except Exception:
        _VMC_REFERENCE_BASIS_QUATS = {}
    return _VMC_REFERENCE_BASIS_QUATS


def build_osc_message(address: str, *args: object) -> bytes:
    def pad4(data: bytes) -> bytes:
        return data + (b"\x00" * ((4 - len(data) % 4) % 4))

    address_bytes = pad4(address.encode("utf-8") + b"\x00")
    type_tags = ","
    arg_bytes = bytearray()

    for arg in args:
        if isinstance(arg, str):
            type_tags += "s"
            arg_bytes.extend(pad4(arg.encode("utf-8") + b"\x00"))
        elif isinstance(arg, bool):
            type_tags += "i"
            arg_bytes.extend(struct.pack(">i", int(arg)))
        elif isinstance(arg, int):
            type_tags += "i"
            arg_bytes.extend(struct.pack(">i", int(arg)))
        else:
            type_tags += "f"
            arg_bytes.extend(struct.pack(">f", float(arg)))

    return address_bytes + pad4(type_tags.encode("utf-8") + b"\x00") + bytes(arg_bytes)


class VmcSender:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        dry_run: bool,
        root_offset: tuple[float, float, float],
        root_position_mode: str,
        root_rotation_mode: str,
        hips_rotation_mode: str,
        coordinate_mode: str,
        debug: bool,
    ) -> None:
        self.dry_run = dry_run
        self.root_offset = root_offset
        self.root_position_mode = root_position_mode
        self.root_rotation_mode = root_rotation_mode
        self.hips_rotation_mode = hips_rotation_mode
        self.coordinate_mode = coordinate_mode
        self.debug = debug
        self.target = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent_frames = 0
        self.root_anchor: tuple[float, float, float] | None = None

    def close(self) -> None:
        self.sock.close()

    def send(self, address: str, *args: object) -> None:
        packet = build_osc_message(address, *args)
        if not self.dry_run:
            self.sock.sendto(packet, self.target)

    def send_available(self) -> None:
        self.send("/VMC/Ext/VRM", "loaded", 1)
        self.send("/VMC/Ext/Rcv", "enable", 1)

    def send_frame(
        self,
        pose_frame: Sequence[Sequence[float]],
        trans_frame: Sequence[float],
        timestamp: float,
    ) -> None:
        root_x = float(trans_frame[0]) if len(trans_frame) > 0 else 0.0
        root_y = float(trans_frame[1]) if len(trans_frame) > 1 else 0.0
        root_z = float(trans_frame[2]) if len(trans_frame) > 2 else 0.0
        root_pos = self._convert_position(
            (
                root_x,
                root_y,
                root_z,
            )
        )
        root_pos = self._apply_root_position_mode(root_pos)
        root_pos = (
            root_pos[0] + self.root_offset[0],
            root_pos[1] + self.root_offset[1],
            root_pos[2] + self.root_offset[2],
        )

        root_quat = (
            self._bone_quat(pose_frame, "Root_M")
            if self.root_rotation_mode == "root_bone"
            else (0.0, 0.0, 0.0, 1.0)
        )
        self.send("/VMC/Ext/Root/Pos", "root", *root_pos, *root_quat)

        for vmc_name, echo_names in VMC_TO_ECHO_BONE.items():
            if vmc_name == "Hips" and self.hips_rotation_mode == "identity":
                quat = (0.0, 0.0, 0.0, 1.0)
            elif vmc_name == "Hips" and self.hips_rotation_mode == "skip":
                continue
            else:
                quat = self._bone_quat(pose_frame, echo_names, vmc_name)
            self.send("/VMC/Ext/Bone/Pos", vmc_name, 0.0, 0.0, 0.0, *quat)

        self.send("/VMC/Ext/OK", 1)
        self.send("/VMC/Ext/T", float(timestamp))
        self.sent_frames += 1

        if self.debug and self.sent_frames % 60 == 0:
            print(
                f"[vmc] frame={self.sent_frames} "
                f"root=({root_pos[0]:.3f},{root_pos[1]:.3f},{root_pos[2]:.3f})"
            )

    def _bone_quat(
        self,
        pose_frame: Sequence[Sequence[float]],
        echo_names: str | Sequence[str],
        vmc_name: str | None = None,
    ) -> tuple[float, float, float, float]:
        names = (echo_names,) if isinstance(echo_names, str) else tuple(echo_names)
        quat = (0.0, 0.0, 0.0, 1.0)
        found = False
        for echo_name in names:
            index = BONE_INDEX.get(echo_name)
            if index is None or index >= len(pose_frame):
                continue
            quat = multiply_xyzw(quat, normalize_xyzw(pose_frame[index]))
            found = True
        if not found:
            return 0.0, 0.0, 0.0, 1.0
        return self._convert_quat(quat, vmc_name)

    def _convert_position(
        self,
        pos: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        x, y, z = pos
        if self.coordinate_mode in ("echoavatar_to_vmc", "echoavatar_raw"):
            return -x, y, z
        if self.coordinate_mode == "blender_to_vmc":
            return -x, z, -y
        if self.coordinate_mode == "unity_to_unreal":
            return z, x, y
        return x, y, z

    def _apply_root_position_mode(
        self,
        pos: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if self.root_position_mode == "absolute":
            return pos

        if self.root_anchor is None:
            self.root_anchor = pos

        ax, ay, az = self.root_anchor
        x, y, z = pos
        if self.root_position_mode == "relative":
            return x - ax, y - ay, z - az
        if self.root_position_mode == "relative_horizontal":
            if self.coordinate_mode in ("echoavatar_to_vmc", "echoavatar_raw"):
                return x - ax, 0.0, z - az
            if self.coordinate_mode == "unity_to_unreal":
                return x - ax, y - ay, z
            return x - ax, y, z - az
        return pos

    def _convert_quat(
        self,
        quat: tuple[float, float, float, float],
        vmc_name: str | None = None,
    ) -> tuple[float, float, float, float]:
        x, y, z, w = quat
        if self.coordinate_mode == "echoavatar_raw":
            return quat
        if self.coordinate_mode == "echoavatar_to_vmc":
            basis = get_vmc_reference_basis_quats().get(vmc_name or "")
            if basis is not None:
                quat = multiply_xyzw(multiply_xyzw(basis, quat), invert_unit_xyzw(basis))
                x, y, z, w = quat
            return normalize_xyzw((x, -z, y, w))
        if self.coordinate_mode == "blender_to_vmc":
            return normalize_xyzw((x, -z, y, w))
        if self.coordinate_mode == "unity_to_unreal":
            return normalize_xyzw((z, x, y, w))
        return quat


class LiveLinkFacePacket:
    def __init__(self, name: str, fps: int) -> None:
        self.name = name
        self.uuid = "$" + str(uuid.uuid1())
        self.fps = fps
        self.version = 6
        self.sub_frame = 1056060032
        self.denominator = max(1, int(fps / 60))
        self.values = [0.0] * 61

    def encode(self) -> bytes:
        now = datetime.datetime.now()
        total_seconds = (
            now.hour * 3600
            + now.minute * 60
            + now.second
            + now.microsecond / 1_000_000
        )
        frames = int(total_seconds * self.fps)
        return (
            struct.pack("<I", self.version)
            + self.uuid.encode("utf-8")
            + struct.pack("!i", len(self.name))
            + self.name.encode("utf-8")
            + struct.pack("!II", frames, self.sub_frame)
            + struct.pack("!II", self.fps, self.denominator)
            + struct.pack("!B61f", 61, *self.values)
        )


class LiveLinkFaceSender:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        dry_run: bool,
        fps: int,
        subject: str,
        debug: bool,
    ) -> None:
        self.dry_run = dry_run
        self.debug = debug
        self.packet = LiveLinkFacePacket(subject, fps)
        self.target = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent_frames = 0

    def close(self) -> None:
        self.sock.close()

    def send_discovery(self) -> None:
        self._send_packet()

    def send_blendshapes(
        self,
        values52: Sequence[float],
    ) -> None:
        values = [0.0] * 61
        for index, value in enumerate(values52[:52]):
            values[index] = float(value)
        self.packet.values = values
        self._send_packet()
        self.sent_frames += 1

        if self.debug and self.sent_frames % 60 == 0:
            active = sum(1 for value in values if abs(value) > 0.01)
            print(f"[livelink] frame={self.sent_frames} active_curves={active}")

    def _send_packet(self) -> None:
        packet = self.packet.encode()
        if not self.dry_run:
            try:
                self.sock.sendto(packet, self.target)
            except OSError as exc:
                if self.debug:
                    print(f"[livelink] send failed: {exc}")


@dataclass
class EchoChunk:
    pose: list
    trans: list
    blendshape: list
    audio: list

    @property
    def frame_count(self) -> int:
        counts = [len(item) for item in (self.pose, self.trans, self.blendshape) if item]
        return min(counts) if counts else 0


def recv_exact(sock: socket.socket, size: int) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        try:
            chunk = sock.recv(size - len(data))
        except socket.timeout:
            continue
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def read_echo_chunk(sock: socket.socket, max_bytes: int) -> EchoChunk | None:
    header = recv_exact(sock, 4)
    if not header:
        return None

    size = int.from_bytes(header, byteorder="big")
    if size <= 0:
        return None
    if size > max_bytes:
        raise ValueError(f"EchoAvatar packet is too large: {size} bytes")

    payload = recv_exact(sock, size)
    if payload is None:
        return None

    obj = json.loads(payload.decode("utf-8"))
    return EchoChunk(
        pose=obj.get("pose") or [],
        trans=obj.get("trans") or [],
        blendshape=obj.get("blendshape") or [],
        audio=obj.get("audio") or [],
    )


def frame_indices(frame_count: int, stride: int) -> Iterable[int]:
    for index in range(0, frame_count, max(1, stride)):
        yield index
