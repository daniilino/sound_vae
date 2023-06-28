import multiprocessing as mp
import logging
from copy import copy
import time
from typing import Tuple

import cv2
import pyaudio
import torch
import torch.nn.functional as F
import torchaudio.transforms as TAT

logger = logging.getLogger(__name__)

def get_sound_device(audio, query):
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get("deviceCount")
    available_devices = []

    for i in range(0, numdevices):
        sound_device = audio.get_device_info_by_host_api_device_index(0, i)
        num_channels = max(
            sound_device.get("maxInputChannels"),
            sound_device.get("maxOutputChannels"),
        )
        if num_channels > 0:
            name = sound_device.get("name")
            # print(name)
            available_devices.append("              " + name)
            if query != "" and query in name:
                return sound_device, num_channels

    available_devices = "\n".join(available_devices)
    raise Exception(
        f"Device not found, here are available devices: \n{available_devices}"
    )


def start_streaming(queue, query, chunk, sample_rate, is_streaming):
    p = pyaudio.PyAudio()
    sound_device, _ = get_sound_device(p, query)
    device_index = sound_device.get("index")
    sample_rate.value = int(sound_device.get("defaultSampleRate"))

    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        input_device_index=device_index,
        rate=sample_rate.value,
        input=True,
        frames_per_buffer=chunk,
    )

    while True:
        if is_streaming.value:
            data = stream.read(chunk, exception_on_overflow=False)
            queue.put(copy(data))


class RealTimeAudioStream:
    def __init__(
        self,
        query: str = "",
        chunk=1024,
        cv2_window_size: Tuple[int, int] = (256, 512),
        z_dim=256,
    ):
        """This object takes stream from input\output device
        and then calculates:
            RMS (Root Mean Square),
            ZCR (Zero Crossing Rate),
            FFT (Fast Fourier Transform) - MelSpectrogram"""

        self.query = query
        self.queue = mp.Queue()
        self.cv2_window_size = cv2_window_size  # (H, W)
        self.chunk = chunk
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = mp.Value("i", 0)
        self.is_streaming = mp.Value("i", 0)

        self.start_audio_process()
        time.sleep(0.5)  # wait for the process to start

        self._new_wav = torch.zeros(
            (1, self.chunk),
            dtype=float,
            device=self.device,
            requires_grad=False,
        )  # this is the last piece of information we obtained

        self.current_rms = torch.zeros(
            (1, 1),
            dtype=float,
            device=self.device,
            requires_grad=False,
        )
        self.current_zcr = self.current_rms.clone()
        self.current_fft = self.current_rms.clone()

        self.processor_fft = (
            TAT.MelSpectrogram(
                sample_rate=self.sample_rate.value,
                n_mels=z_dim,
                n_fft=self.chunk,
                win_length=self.chunk,
                hop_length=self.chunk,
            )
            .to(self.device)
            .double()
        )  # here double as well

        self.is_streaming.value = 1  # start streaming

    def start_audio_process(self):
        process = mp.Process(
            target=start_streaming,
            args=(
                self.queue,
                self.query,
                self.chunk,
                self.sample_rate,
                self.is_streaming
            ),
        )

        process.start()

    def _rms(self):
        self.current_rms = self._new_wav.pow(2).mean(1, keepdim=True).sqrt()

    def _zcr(self):
        self.current_zcr = (
            torch.diff(self._new_wav > 0, dim=1).type(torch.int).abs() > 0
        ).sum(dim=1, keepdim=True) / self.window_size

    def _fft(self):
        self.current_fft = self.processor_fft(self._new_wav).amax(
            2, keepdim=True
        )  # [C, z_dims, 3] - > [C, z_dims, 1]

    def _vis(self):
        show_R = self.buffer_rms  # [C, time]
        show_G = self.buffer_zcr  # [C, time]
        show_B = self.buffer_fft.mean(
            0, keepdim=True
        )  # [C, z_dim, time] - > [1, z_dim, time]

        C, TIME = show_R.shape

        H, W = self.cv2_window_size
        show_R = torch.clamp(H - ((show_R + 1) * H // 2), 0, H - 1).type(
            torch.LongTensor
        )[
            0, :
        ]  # [TIME]
        show_G = torch.clamp(H - ((show_G + 1) * H // 2), 0, H - 1).type(
            torch.LongTensor
        )[
            0, :
        ]  # [TIME]
        image = torch.zeros(
            (3, H, TIME), dtype=float, requires_grad=False, device=self.device
        )  # [3, H, W1]

        image[2, show_R, torch.arange(0, TIME)] = 1
        image[1, show_G, torch.arange(0, TIME)] = 1

        image = F.interpolate(image.unsqueeze(0), size=(H, W)).squeeze()
        image[0, :, :] = F.interpolate(show_B.unsqueeze(0), size=(H, W)).squeeze()

        return image.permute(1, 2, 0).cpu().numpy()

    def step_process(self, rms=False, zcr=False, fft=True):
        data = self.queue.get()
        while not self.queue.empty():
            data = self.queue.get()
        stream = copy(data)

        self._new_wav = (
            torch.frombuffer(stream, dtype=torch.float32)
            .reshape(1, -1)
            .to(self.device)
        )  # [C, overlap]

        if rms:
            self._rms()
        if zcr:
            self._zcr()
        if fft:
            self._fft()

        return self.current_rms.cuda(), self.current_zcr.cuda(), self.current_fft.cuda()
