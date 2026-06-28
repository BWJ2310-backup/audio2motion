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


# VMC body output uses VRM humanoid bone names. Face data is intentionally not
# sent through VMC; ARKit face coefficients belong to LiveLink.
VMC_TO_ECHO_BONE = {
    "Hips": "Root_M",
    "Spine": "Spine1_M",
    "Chest": "Chest_M",
    "UpperChest": "Chest_M",
    "ChestUpper": "Chest_M",
    "Neck": "Neck_M",
    "Head": "Head_M",
    "RightShoulder": "Scapula_R",
    "RightUpperArm": "Shoulder_R",
    "RightLowerArm": "Elbow_R",
    "RightHand": "Wrist_R",
    "LeftShoulder": "Scapula_L",
    "LeftUpperArm": "Shoulder_L",
    "LeftLowerArm": "Elbow_L",
    "LeftHand": "Wrist_L",
    "RightUpperLeg": "Hip_R",
    "RightLowerLeg": "Knee_R",
    "RightFoot": "Ankle_R",
    "RightToes": "Toes_R",
    "LeftUpperLeg": "Hip_L",
    "LeftLowerLeg": "Knee_L",
    "LeftFoot": "Ankle_L",
    "LeftToes": "Toes_L",
    "RightThumbProximal": "ThumbFinger1_R",
    "RightThumbIntermediate": "ThumbFinger2_R",
    "RightThumbDistal": "ThumbFinger3_R",
    "RightIndexProximal": "IndexFinger1_R",
    "RightIndexIntermediate": "IndexFinger2_R",
    "RightIndexDistal": "IndexFinger3_R",
    "RightMiddleProximal": "MiddleFinger1_R",
    "RightMiddleIntermediate": "MiddleFinger2_R",
    "RightMiddleDistal": "MiddleFinger3_R",
    "RightRingProximal": "RingFinger1_R",
    "RightRingIntermediate": "RingFinger2_R",
    "RightRingDistal": "RingFinger3_R",
    "RightLittleProximal": "PinkyFinger1_R",
    "RightLittleIntermediate": "PinkyFinger2_R",
    "RightLittleDistal": "PinkyFinger3_R",
    "LeftThumbProximal": "ThumbFinger1_L",
    "LeftThumbIntermediate": "ThumbFinger2_L",
    "LeftThumbDistal": "ThumbFinger3_L",
    "LeftIndexProximal": "IndexFinger1_L",
    "LeftIndexIntermediate": "IndexFinger2_L",
    "LeftIndexDistal": "IndexFinger3_L",
    "LeftMiddleProximal": "MiddleFinger1_L",
    "LeftMiddleIntermediate": "MiddleFinger2_L",
    "LeftMiddleDistal": "MiddleFinger3_L",
    "LeftRingProximal": "RingFinger1_L",
    "LeftRingIntermediate": "RingFinger2_L",
    "LeftRingDistal": "RingFinger3_L",
    "LeftLittleProximal": "PinkyFinger1_L",
    "LeftLittleIntermediate": "PinkyFinger2_L",
    "LeftLittleDistal": "PinkyFinger3_L",
}


LIVELINK_HEAD_YAW = 52
LIVELINK_HEAD_PITCH = 53
LIVELINK_HEAD_ROLL = 54
LIVELINK_LEFT_EYE_YAW = 55
LIVELINK_LEFT_EYE_PITCH = 56
LIVELINK_LEFT_EYE_ROLL = 57
LIVELINK_RIGHT_EYE_YAW = 58
LIVELINK_RIGHT_EYE_PITCH = 59
LIVELINK_RIGHT_EYE_ROLL = 60

ARKIT_EYE_LOOK_DOWN_LEFT = 1
ARKIT_EYE_LOOK_IN_LEFT = 2
ARKIT_EYE_LOOK_OUT_LEFT = 3
ARKIT_EYE_LOOK_UP_LEFT = 4
ARKIT_EYE_LOOK_DOWN_RIGHT = 8
ARKIT_EYE_LOOK_IN_RIGHT = 9
ARKIT_EYE_LOOK_OUT_RIGHT = 10
ARKIT_EYE_LOOK_UP_RIGHT = 11


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


def quat_xyzw_to_euler_xyz(quat: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = normalize_xyzw(quat)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch_y = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch_y = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(siny_cosp, cosy_cosp)
    return roll_x, pitch_y, yaw_z


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
        root_scale: float,
        root_offset: tuple[float, float, float],
        root_rotation_mode: str,
        debug: bool,
    ) -> None:
        self.dry_run = dry_run
        self.root_scale = root_scale
        self.root_offset = root_offset
        self.root_rotation_mode = root_rotation_mode
        self.debug = debug
        self.target = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent_frames = 0

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
        root_pos = (
            root_x * self.root_scale + self.root_offset[0],
            root_y * self.root_scale + self.root_offset[1],
            root_z * self.root_scale + self.root_offset[2],
        )

        root_quat = (
            self._bone_quat(pose_frame, "Root_M")
            if self.root_rotation_mode == "root_bone"
            else (0.0, 0.0, 0.0, 1.0)
        )
        self.send("/VMC/Ext/Root/Pos", "root", *root_pos, *root_quat)

        for vmc_name, echo_name in VMC_TO_ECHO_BONE.items():
            quat = self._bone_quat(pose_frame, echo_name)
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
        echo_name: str,
    ) -> tuple[float, float, float, float]:
        index = BONE_INDEX.get(echo_name)
        if index is None or index >= len(pose_frame):
            return 0.0, 0.0, 0.0, 1.0
        return normalize_xyzw(pose_frame[index])


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
        eye_rotation_mode: str,
        eye_yaw_scale: float,
        eye_pitch_scale: float,
        debug: bool,
    ) -> None:
        self.dry_run = dry_run
        self.debug = debug
        self.eye_rotation_mode = eye_rotation_mode
        self.eye_yaw_scale = eye_yaw_scale
        self.eye_pitch_scale = eye_pitch_scale
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
        pose_frame: Sequence[Sequence[float]] | None,
    ) -> None:
        values = [0.0] * 61
        for index, value in enumerate(values52[:52]):
            values[index] = clamp(value, 0.0, 1.0)
        self._apply_eye_rotations(values, pose_frame)
        self.packet.values = values
        self._send_packet()
        self.sent_frames += 1

        if self.debug and self.sent_frames % 60 == 0:
            active = sum(1 for value in values if abs(value) > 0.01)
            print(
                f"[livelink] frame={self.sent_frames} active_curves={active} "
                f"left_eye=({values[LIVELINK_LEFT_EYE_YAW]:.3f},"
                f"{values[LIVELINK_LEFT_EYE_PITCH]:.3f}) "
                f"right_eye=({values[LIVELINK_RIGHT_EYE_YAW]:.3f},"
                f"{values[LIVELINK_RIGHT_EYE_PITCH]:.3f})"
            )

    def _send_packet(self) -> None:
        packet = self.packet.encode()
        if not self.dry_run:
            try:
                self.sock.sendto(packet, self.target)
            except OSError as exc:
                if self.debug:
                    print(f"[livelink] send failed: {exc}")

    def _apply_eye_rotations(
        self,
        values61: list[float],
        pose_frame: Sequence[Sequence[float]] | None,
    ) -> None:
        if self.eye_rotation_mode == "off":
            return

        if self.eye_rotation_mode == "pose" and pose_frame:
            left = self._eye_rotation_from_pose(pose_frame, "Eye_L")
            right = self._eye_rotation_from_pose(pose_frame, "Eye_R")
            if left and right:
                values61[LIVELINK_LEFT_EYE_YAW] = left[0]
                values61[LIVELINK_LEFT_EYE_PITCH] = left[1]
                values61[LIVELINK_LEFT_EYE_ROLL] = left[2]
                values61[LIVELINK_RIGHT_EYE_YAW] = right[0]
                values61[LIVELINK_RIGHT_EYE_PITCH] = right[1]
                values61[LIVELINK_RIGHT_EYE_ROLL] = right[2]
                return

        values61[LIVELINK_LEFT_EYE_YAW] = clamp(
            (values61[ARKIT_EYE_LOOK_OUT_LEFT] - values61[ARKIT_EYE_LOOK_IN_LEFT])
            * self.eye_yaw_scale,
            -1.0,
            1.0,
        )
        values61[LIVELINK_LEFT_EYE_PITCH] = clamp(
            (values61[ARKIT_EYE_LOOK_DOWN_LEFT] - values61[ARKIT_EYE_LOOK_UP_LEFT])
            * self.eye_pitch_scale,
            -1.0,
            1.0,
        )
        values61[LIVELINK_RIGHT_EYE_YAW] = clamp(
            (values61[ARKIT_EYE_LOOK_IN_RIGHT] - values61[ARKIT_EYE_LOOK_OUT_RIGHT])
            * self.eye_yaw_scale,
            -1.0,
            1.0,
        )
        values61[LIVELINK_RIGHT_EYE_PITCH] = clamp(
            (values61[ARKIT_EYE_LOOK_DOWN_RIGHT] - values61[ARKIT_EYE_LOOK_UP_RIGHT])
            * self.eye_pitch_scale,
            -1.0,
            1.0,
        )

    def _eye_rotation_from_pose(
        self,
        pose_frame: Sequence[Sequence[float]],
        echo_name: str,
    ) -> tuple[float, float, float] | None:
        index = BONE_INDEX.get(echo_name)
        if index is None or index >= len(pose_frame):
            return None
        pitch_x, _unused_y, yaw_z = quat_xyzw_to_euler_xyz(pose_frame[index])
        return (
            clamp(yaw_z * self.eye_yaw_scale, -1.0, 1.0),
            clamp(pitch_x * self.eye_pitch_scale, -1.0, 1.0),
            0.0,
        )


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
        chunk = sock.recv(size - len(data))
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
