import torch
import cv2
import torchaudio.transforms as TAT
import torch.nn.functional as F

import numpy as np

import pyaudio

from typing import Tuple

import keyboard

from copy import copy

class RealTimeAudioStream:
    def __init__(self, sound_device: str = "", window_size = 1024, overlap = 512, buffer_seconds = 5, cv2_window_size: Tuple[int, int] = (256, 512), z_dim = 256):
        
        """ This object takes stream from input\output device 
        and then calculates:
            RMS (Root Mean Square), 
            ZCR (Zero Crossing Rate), 
            FFT (Fast Fourier Transform) - MelSpectrogram"""

        self.done = None
        self.cv2_window_size = cv2_window_size # (H, W)

        ##### SETTING AUDIO STREAMER DEVICE #####
        self.audio = pyaudio.PyAudio()
        self.sound_device, self._num_channels = self._get_sound_device(sound_device)

        self.window_size = window_size # samples per processsing step
        self.overlap = overlap
        self.sample_rate = int(self.sound_device.get("defaultSampleRate"))
        self._buffer_size = self.sample_rate * buffer_seconds # samples memory size
        print(f"RealTimeAudioStream initialized with {self.sample_rate} sample rate")

        self._streamer = self.audio.open(format=pyaudio.paInt16,
                    channels=self._num_channels,
                    input_device_index=self.sound_device.get('index'),
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.overlap)
        
        ##### SETTING PYTORCH PROCESSORS AND BUFFERS ####

        self._d =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._new_wav = torch.zeros((self._num_channels, self.overlap), dtype=float, device=self._d, requires_grad=False) # this is the last piece of information we obtained 
        self._current_window = torch.zeros((self._num_channels, self.window_size), dtype=float, device=self._d, requires_grad=False) # pre-last + last piece of wav (because sliding window overlaps!)
        self._buffer_wav = torch.zeros((self._num_channels, self._buffer_size), dtype=float, device=self._d, requires_grad=False)

        self.current_rms = torch.zeros((self._num_channels, 1), dtype=float, device=self._d, requires_grad=False)
        self.current_zcr = self.current_rms.clone()
        self.current_fft = self.current_rms.clone()


        self.buffer_rms = torch.zeros((self._num_channels, self._buffer_size // overlap), dtype=float, device=self._d, requires_grad=False)
        self.buffer_zcr = self.buffer_rms.clone()
        self.buffer_fft = torch.zeros((self._num_channels, z_dim, self._buffer_size // overlap), dtype=torch.double, device=self._d, requires_grad=False) # note: dtype is double, otherwise yields an Exception

        self.processor_fft = TAT.MelSpectrogram(
                            sample_rate=self.sample_rate,
                            n_mels=z_dim,
                            n_fft=self.window_size,
                            win_length=self.window_size,
                            hop_length=self.overlap,
                            ).to(self._d).double() # here double as well

    def _get_sound_device(self, query):

        info = self.audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        available_devices = []

        for i in range(0, numdevices):
            sound_device = self.audio.get_device_info_by_host_api_device_index(0, i)
            num_channels = max(sound_device.get('maxInputChannels'), sound_device.get('maxOutputChannels'))
            if num_channels > 0:
                name = sound_device.get('name')
                available_devices.append("              " + name)
                if query!= "" and query in name:
                    return sound_device, num_channels
        
        available_devices = "\n".join(available_devices)
        raise Exception(f"Device not found, here are available devices: \n{available_devices}")

    def _rms(self):
        self.current_rms = self._current_window.pow(2).mean(1, keepdim=True).sqrt() 
        self.buffer_rms = torch.cat((self.buffer_rms, self.current_rms), dim=1)[:, 1:]

    def _zcr(self):
        self.current_zcr = (torch.diff(self._current_window > 0, dim=1).type(torch.int).abs() > 0).sum(dim=1, keepdim=True) / self.window_size
        self.buffer_zcr = torch.cat((self.buffer_zcr, self.current_zcr), dim=1)[:, 1:]

    def _fft(self):
        self.current_fft = self.processor_fft(self._current_window ).amax(2, keepdim=True) # [C, z_dims, 3] - > [C, z_dims, 1]
        self.buffer_fft = torch.cat((self.buffer_fft, self.current_fft), dim=2)[:, :, 1:]

    def _vis(self):

        show_R = self.buffer_rms # [C, time]
        show_G = self.buffer_zcr # [C, time]
        show_B = self.buffer_fft.mean(0, keepdim=True) # [C, z_dim, time] - > [1, z_dim, time]

        C, TIME = show_R.shape

        H, W = self.cv2_window_size
        show_R = torch.clamp(  H - ((show_R + 1) * H // 2)    , 0, H-1).type(torch.LongTensor)[0, :] # [TIME]
        show_G = torch.clamp(  H - ((show_G + 1) * H // 2)    , 0, H-1).type(torch.LongTensor)[0, :] # [TIME]
        image = torch.zeros((3, H, TIME), dtype=float, requires_grad=False, device=self._d) # [3, H, W1]

        image[2, show_R, torch.arange(0, TIME)] = 1
        image[1, show_G, torch.arange(0, TIME)] = 1

        image = F.interpolate(image.unsqueeze(0), size=(H,W)).squeeze()
        image[0, :, :] = F.interpolate(show_B.unsqueeze(0), size=(H,W)).squeeze()

        return image.permute(1,2,0).cpu().numpy()

    def step(self, rms=False, zcr=False, fft=True):
        stream = copy(self._streamer.read(self.overlap, exception_on_overflow=False)) # bytes; we're copying, because torch doesn't like read-only streams and gives a warning
        self._new_wav = torch.frombuffer(stream, dtype=torch.int16).reshape(self._num_channels, -1).to(self._d) # [C, overlap]
        self._new_wav = self._new_wav/(2**15) # normalizes data into [0, 1] range

        self._buffer_wav = torch.cat((self._buffer_wav, self._new_wav), dim=1)[:, self.overlap:] # xx[.::.:.:.:.::::..]+[..:]
        self._current_window = self._buffer_wav[:, -self.window_size:] # [C, window_size]
        if rms: self._rms()
        if zcr: self._zcr()
        if fft: self._fft()

        keyboard.on_press_key("ESC", lambda _: self._done())

        return self.current_rms, self.current_zcr, self.current_fft
    
    def _done(self): 
        """ because keyboard.on_press_key doesn't work otherwise, so i made this function"""
        self.done = True

    def stream(self, rms=False, zcr=False, fft=True):
        self.done = False

        while not self.done:
            self.step(rms, zcr, fft)
            cv2.imshow('stream', self._vis())

            k = cv2.waitKey(33)
            if k==27:    # Esc key to stop
                self.done = True
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()