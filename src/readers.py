"""Hardware-neutral EEG input contract for future home BCI integrations."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass
class EEGRecording:
    samples: np.ndarray
    sampling_rate: float
    channel: str
    unit: str = 'V'
    start_seconds: float = 0.0
    preprocessed: bool = False

    def validate(self):
        if self.samples.ndim != 1 or not len(self.samples) or not np.isfinite(self.samples).all():
            raise ValueError('A recording requires a nonempty finite single-channel vector')
        if not np.isfinite(self.sampling_rate) or self.sampling_rate <= 0:
            raise ValueError('Sampling rate must be positive and finite')
        if self.unit not in ('V', 'uV'):
            raise ValueError('Explicit V or uV units are required')
        if not self.channel or not np.isfinite(self.start_seconds):
            raise ValueError('Channel and finite start time are required')
        return self

    def volts(self):
        self.validate()
        values = self.samples.astype(np.float64) * (1e-6 if self.unit == 'uV' else 1.0)
        if (np.abs(values) > np.finfo(np.float32).max).any():
            raise ValueError('Converted samples exceed finite float32 range')
        return values.astype(np.float32)


class EEGReader(Protocol):
    def read(self, source: str) -> EEGRecording:
        """Return samples and explicit rate/channel/unit/preprocessing metadata."""
        ...


class TextEEGReader:
    def __init__(self, sampling_rate=100, channel='EEG Fpz-Cz', unit='V', preprocessed=False):
        self.sampling_rate = sampling_rate
        self.channel = channel
        self.unit = unit
        self.preprocessed = preprocessed

    def read(self, source):
        from data_preprocessing import load_eeg_txt
        return EEGRecording(load_eeg_txt(source), self.sampling_rate, self.channel,
                            self.unit, preprocessed=self.preprocessed).validate()
