# Project: Generative Adversarial Network

In this project, you can either work with [Images](#task-image) or [Audio](#task-audio)

## Task, Image

Implement [Pix2Pix](https://arxiv.org/abs/1611.07004). Use the datasets from the paper.

## Task, Audio
Implement [HiFiGAN](https://arxiv.org/pdf/2010.05646.pdf) vocoder.

Use dataset [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) you already know.

To avoid a mismatch of training and test features, please use the following code to generate MelSpecs
<details>
<summary>Click me</summary>

```python
from dataclasses import dataclass

import torch
from torch import nn

import torchaudio

import librosa  


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            center=False,
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        audio = torch.nn.functional.pad(audio.unsqueeze(1),
             (int((self.config.n_fft-self.config.hop_length)/2),
              int((self.config.n_fft-self.config.hop_length)/2)),
             mode='reflect')

        audio = audio.squeeze(1)

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel
```
</details>

--------------
## General requirements
In general, the format of the current project follows that of the ASR project assignment. You must organize the repository as in the first homework assignment, which is to break up the code into modules and follow the code style.

Requirements:
* The code should be situated in a public github (or gitlab) repository
* All the necessary packages should be mentioned in `./requirements.txt` or in an installation guide section of README.md
* Use W&B for logging losses and synthesized images/audio. 
* All necessary resources (such as model checkpoints) should be downloadable with a script
  Mention the script (or lines of code) in the `README.md`
* Attach a report that includes:
  * Description and result of each experiment
  * How to reproduce your model?
  * Attach training logs showing the rate of convergence
  * What worked and what didn't work?
  * What were the major challenges?

--------------
## Evaluation, Images

You can use either the metrics from the paper or more state-of-the-art ones, like: [FID](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html), [SSIM](https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html), [PSNR](https://lightning.ai/docs/torchmetrics/stable/image/peak_signal_noise_ratio.html)

## Evaluation, Audio

To evaluate the MOS (Mean Opinion Score), add a synthesis of the following sentences to the report:
* `A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest`
* `Massachusetts Institute of Technology may be best known for its math, science and engineering education`
* `Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space`

The corresponding `MelSpectograms` will be announced in Discord.