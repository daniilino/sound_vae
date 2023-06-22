import torch
import cv2
import torchaudio.transforms as TAT
import torch.nn.functional as F

import soundcard as sc
import soundfile as sf

import keyboard

class RealTimeAudioStream:
    def __init__(self, sample_rate = 44100, window_size = 1024, overlap = 512, buffer_seconds = 5, cv2_window_size = (256, 512), z_dim = 256):
        
        self.cv2_window_size = cv2_window_size # (H, W)

        self.done = None
        self.current_rms = None
        self.current_zcr = None
        self.current_fft = None

        self._d =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sample_rate = (sample_rate // window_size) * window_size # samples per seconds, a.k.a [Hz]
        print(f"RealTimeAudioStream initialized with {self.sample_rate} sample rate")
        self.window_size = window_size # samples per processsing step
        self.overlap = overlap # overlap

        self._mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
        self._num_channels = self._mic.channels

        self._buffer_size = self.sample_rate * buffer_seconds # samples memory size
        self._buffer_wav = torch.zeros((self._buffer_size, self._num_channels), dtype=float, device=self._d, requires_grad=False)

        self.buffer_rms = torch.zeros((self._buffer_size // overlap, self._num_channels), dtype=float, device=self._d, requires_grad=False)
        self.buffer_zcr = torch.zeros((self._buffer_size // overlap, self._num_channels), dtype=float, device=self._d, requires_grad=False)
        self.buffer_fft = torch.zeros((self._num_channels, z_dim, self._buffer_size // overlap), device=self._d, requires_grad=False).double()

        self.processor_fft = TAT.MelSpectrogram(
                            sample_rate=self.sample_rate,
                            n_mels=z_dim,
                            n_fft=self.window_size,
                            win_length=self.window_size,
                            hop_length=self.overlap,
                            normalized=True,
                            ).to(self._d).double()

    def _rms(self):
        current = self._buffer_wav[-self.window_size:, :]
        self.current_rms = current.pow(2).mean(0, keepdim=True).sqrt()
        self.buffer_rms = torch.cat((self.buffer_rms, self.current_rms), dim=0)[1:,:]

    def _zcr(self):
        current = self._buffer_wav[-self.window_size:, :]
        self.current_zcr = (torch.diff(current > 0, dim=0).type(torch.int).abs() > 0).sum(dim=0, keepdim=True) / self.window_size
        self.buffer_zcr = torch.cat((self.buffer_zcr, self.current_zcr), dim=0)[1:,:]

    def _fft(self):
        current = self._buffer_wav[-self.window_size:, :].permute(1,0) # [C, window_size]
        self.current_fft = self.processor_fft(current).amax(2, keepdim=True) # [C, z_dims, 3] - > [C, z_dims, 1]
        self.buffer_fft = torch.cat((self.buffer_fft, self.current_fft), dim=2)[:, :, 1:]

    def _vis(self):

        show_R = self.buffer_rms
        show_G = self.buffer_zcr
        show_B = self.buffer_fft[0, None, :, :] # [C, z_dim, time]

        W1, C = show_R.shape

        H, W = self.cv2_window_size
        sound_R = torch.clamp(  H - ((show_R + 1) * H // 2)    , 0, H-1).type(torch.LongTensor)[:,0] # [W1]
        sound_G = torch.clamp(  H - ((show_G + 1) * H // 2)    , 0, H-1).type(torch.LongTensor)[:,0] # [W1]
        image = torch.zeros((3, H, W1), dtype=float) # [3, H, W1]

        image[2, sound_R, torch.arange(0, W1)] = 1
        image[1, sound_G, torch.arange(0, W1)] = 1

        image = F.interpolate(image.unsqueeze(0), size=(H,W)).squeeze()
        image[0, :, :] = F.interpolate(show_B.unsqueeze(0), size=(H,W)).squeeze()

        return image.permute(1,2,0).cpu().numpy()

    def step(self, mic):
        self._current = torch.from_numpy(mic.record(numframes=self.overlap)).to(self._d) # [window_size, num_channels] ~ [1024, 2]

        self._buffer_wav = torch.cat((self._buffer_wav, self._current), dim=0)[self.overlap:,:]
        self._rms()
        self._zcr()
        self._fft()

        keyboard.on_press_key("ESC", lambda _: self._done())

        return self.current_rms, self.current_zcr, self.current_fft
    
    def get_recorder(self):
        return self._mic.recorder(samplerate=self.sample_rate)
    
    def _done(self):
        self.done = True

    def stream(self):
        self.done = False

        with self.get_recorder() as mic:
            while not self.done:
                self.step(mic)
                cv2.imshow('stream', self._vis())

                k = cv2.waitKey(33)
                if k==27:    # Esc key to stop
                    self.done = True
                    cv2.destroyAllWindows()
                    break

            cv2.destroyAllWindows()