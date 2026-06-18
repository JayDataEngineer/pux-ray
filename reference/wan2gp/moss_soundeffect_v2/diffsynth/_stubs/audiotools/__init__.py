"""Stub for descript-audiotools — avoids protobuf>=6 conflict.

Only AudioSignal is referenced (in DAC compress/decompress), which isn't
used during pipeline inference. BaseModel is a parent class of DAC but
never calls its methods.
"""


class AudioSignal:
    def __init__(self, *a, **kw):
        pass

    @staticmethod
    def load_from_file_with_ffmpeg(path):
        return AudioSignal()

    def to_mono(self):
        return self

    def resample(self, sr):
        pass

    def loudness(self):
        return 0.0

    def normalize(self, db):
        pass

    def ensure_max_of_audio(self):
        pass

    def zero_pad(self, left, right):
        pass

    def clone(self):
        return self

    @property
    def sample_rate(self):
        return 44100

    @property
    def signal_duration(self):
        return 0.0

    @property
    def signal_length(self):
        return 0

    @property
    def device(self):
        import torch
        return torch.device("cpu")

    @property
    def audio_data(self):
        import torch
        return torch.zeros(1)
