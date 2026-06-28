# EchoAvatar: Real-time Generative Avatar Animation from Audio Streams

<center>
  <a href="https://robinwitch.github.io/EchoAvatar-Page/">Project Page</a> •
  <a href="https://arxiv.org/abs/2605.28272">Arxiv Paper</a> •
  <a href="https://youtu.be/GydR3H6YwBQ">Demo Video</a> •
  <a href="#citation">Citation</a>
</center>

## Release Plans

- [x] Real-time deployment code.
- [ ] Code for evaluating BEATv2 benchmark results.
- [ ] Training code.


## Environment Setup

We recommend using Conda with Python 3.13:

```bash
conda create -n echoavatar python=3.13
conda activate echoavatar
pip install -r requirements.txt
```

If you run the real-time deployment across two machines, install this environment on the Ubuntu inference server. On the local Windows machine, only the following packages are required:

```bash
pip install sounddevice keyboard huggingface_hub
```

## Checkpoints

Download the checkpoints into the repository root before running inference:

```bash
hf download robinwitch/EchoAvatar --local-dir . --include "ckpts/**"
git clone https://huggingface.co/robinwitch/hf_transformer_mhubert_base_vp_en_es_fr_it3 ./ckpts/hf_transformer_mhubert_base_vp_en_es_fr_it3
```

## Real-time deployment

EchoAvatar is a generic real-time audio-to-avatar-motion service. It receives
audio, generates body motion and ARKit-compatible face coefficients, and streams
the generated data to configurable output services.

```text
audio source
  -> tools/pushwav2server.py
  -> audio2motion.py
  -> stream2vmc.py       -> body VMC receiver
  -> stream2livelink.py  -> facial LiveLink receiver
```

Recommended setup:

- Audio capture machine: captures browser, system, music, or microphone output.
- Inference server: runs EchoAvatar with NVIDIA GPU. For simultaneous face and body generation, dual RTX 3090 or better is recommended. If you only generate face motion or only body motion, one GPU is enough.
- Output receivers: body motion is sent through VMC only; facial ARKit
  coefficients are sent through LiveLink only.

### 1. Configure Runtime

Edit [config/echoavatar.toml](config/echoavatar.toml). This file controls the
inference model, audio ports, body VMC target, facial LiveLink target, and the
two internal TCP ports used by the separated stream services.

Default local wiring:

```text
audio2motion.py
  -> body_vmc      127.0.0.1:12346
  -> face_livelink 127.0.0.1:12348

stream2vmc.py
  listens on 12346
  sends body VMC UDP to 127.0.0.1:39539

stream2livelink.py
  listens on 12348
  sends facial LiveLink UDP to 127.0.0.1:11111
```

For multi-machine deployment, change the `host` fields under
`[[motion_receivers]]` to the machines running `stream2vmc.py` and
`stream2livelink.py`, and change each streamer's `target_host` to the final
receiver machine.

Remote audio2motion hosting:

- Run `audio2motion.py` on the GPU machine that has the EchoAvatar environment,
  checkpoints, CUDA, and vLLM installed.
- Run `stream2vmc.py` and `stream2livelink.py` on the machine closest to the
  InZOI receiver, usually the local Windows or middleware machine.
- In `[audio_sender]`, set `server_host` to the GPU machine IP or DNS name.
- In `[[motion_receivers]]`, set `host` to the machine running each streamer as
  reachable from the GPU machine.
- In `[stream2vmc]`, set `target_host` and `target_port` for the final VMC UDP
  receiver.
- In `[stream2livelink]`, set `target_host` and `target_port` for the final
  LiveLink UDP receiver.

If the machines are not on the same LAN, use a VPN/tunnel such as Tailscale,
WireGuard, or ZeroTier. Do not rely on public open ports unless you also add
firewall allowlists and authentication at the network layer.

### 2. Start Body VMC Streamer

Start the body-only VMC service before `audio2motion.py`:

```bash
python stream2vmc.py
```

This service only reads `pose` and `trans` from the EchoAvatar packet and only
sends VMC body bone messages. It does not send ARKit face data.

### 3. Start Facial LiveLink Streamer

Start the face-only LiveLink service before `audio2motion.py`:

```bash
python stream2livelink.py
```

This service only sends facial ARKit coefficients over LiveLink. Eye rotation
output is controlled by `stream2livelink.eye_rotation_mode`,
`eye_yaw_scale`, and `eye_pitch_scale` in the TOML config.

### 4. Start Audio2Motion

Run the audio-to-motion inference service on the GPU server:

```bash
python audio2motion.py
```

`audio2motion.py` reads `config/echoavatar.toml`, starts the original
EchoAvatar inference script, and fans out the neutral generated packet to each
enabled `[[motion_receivers]]` entry.

### 5. Audio Capture

Install a virtual audio device such as [VB-CABLE](https://vb-audio.com/Cable/) if
you need to capture browser or application output. Route the application output
to the virtual cable, then list local audio devices:

```bash
python tools/get_device.py
```

Start the audio sender:

```bash
python tools/pushwav2server.py
```

The audio sender reads `[audio_sender]` from `config/echoavatar.toml`. You can
still override values from the CLI, for example:

```bash
python tools/pushwav2server.py --server-ip <gpu-server-ip> --device <audio-device-index>
```

### 6. Audio Input Protocol

EchoAvatar has one input contract: audio. Route audio to the capture device used
by `tools/pushwav2server.py`, or implement the same TCP protocol directly.

Audio sender protocol:

```text
TCP -> [audio2motion].audio_port
4-byte big-endian payload length
pickle payload containing a numpy-compatible audio block
```


## Citation

If you find our code or paper helps, please consider citing:

```bibtex
@inproceedings{chen2026echo,
  author = {Bohong Chen and Yumeng Li and Yinglin Xu and Youyi Zheng and Yanlin Weng and Kun Zhou},
  title = {EchoAvatar: Real-time Generative Avatar Animation from Audio Streams},
  year = {2026},
  isbn = {9798400725548},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3799902.3811066},
  doi = {10.1145/3799902.3811066},
  booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  series = {SIGGRAPH Conference Papers '26}
}
```


## Acknowledgments
Thanks to [EMAGE](https://github.com/PantoMatrix/PantoMatrix), [ZeroEGGS](https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS), [MotoricaDanceDataset](https://github.com/simonalexanderson/MotoricaDanceDataset), [motorica-retarget
](https://github.com/orangeduck/motorica-retarget),[zeroeggs-retarget
](https://github.com/orangeduck/zeroeggs-retarget) , [torchtune](https://github.com/meta-pytorch/torchtune), [ichigo](https://github.com/menloresearch/ichigo),  [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MoMask](https://github.com/EricGuo5513/momask-codes), [MECo](https://github.com/RobinWitch/MECo), [verl](https://github.com/verl-project/verl), [vLLM](https://github.com/vllm-project/vllm), [encodec](https://github.com/facebookresearch/encodec), our code is partially borrowing from them. Please check these useful repos.
