# Project 1: Generative Adversarial Network

In this project, you can either work with [Images](#task-image) or [Audio](#task-audio).

We recommend using our [project template](https://github.com/Blinorot/pytorch_project_template) and modify it to support GAN training (having two optimizers instead of one, etc.). Though, a structured notebook with markdown comments is fine for beginners.

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

---

You can use either the metrics from the paper or more state-of-the-art ones, like: [FID](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html), [SSIM](https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html), [PSNR](https://lightning.ai/docs/torchmetrics/stable/image/peak_signal_noise_ratio.html). For audio metrics, please refer to the paper itself.

# Project 2: Transformer for Machine Translation

Implement simple machine translation task with [vanilla Transformer](https://arxiv.org/abs/1706.03762). You can use `nn.Transformer` or rewrite the network yourself without `PyTorch` implementations of Transformer and its layers. For the datasets, you can choose those from the paper, for example, [WMT 14](https://huggingface.co/datasets/wmt/wmt14). You can use metrics from the paper, like [BLEU](https://lightning.ai/docs/torchmetrics/stable/text/bleu_score.html)

Also, follow [General Recommendations](#general-recommendations)

---

# Project 3: Anti-spoofing (Deepfake Detection)

## LightCNN

Implement [LightCNN (LCCN)](https://arxiv.org/abs/1511.02683) following the Speech Technology Center [paper](https://arxiv.org/abs/1904.05576).

**Hints**:

1. Take training recipe and data preparation scheme from [this paper](https://arxiv.org/abs/2103.11326). Also, read the comparative study and think whether you should use A-Softmax or Cross-Entropy loss function.

2. Use STFT (FFT in the paper) as front-end. (Though others may work too.)

3. Put dropout layer as it is done in [this paper](https://ieeexplore.ieee.org/document/9428313).

## RawNet2

Implement [RawNet2](https://arxiv.org/abs/2011.01108). There are three types of sinc filters:

- S1: fixed Mel-scaled
- S2: fixed inverse Mel-scaled
- S3: fixed linear-scaled

You are free to choose any of them.

**Hints**:

- You have to take the absolute value of sinc-layer output.
- Use 3-layer GRU, do BN and LeakyReLU before.
- Change hyperparameters to the ASVspoof2021 style:
  - Sinc filter length: 1024
  - `Conv(3,1,128) -> Conv(3,1,20)`, `Conv(3,1,512) -> Conv(3,1,128)`.
- Use weighted cross-entropy: $1.0$ for spoof class, $9.0$ for bona-fide.
- Set $0.0001$ weight decay in optimizer.
- Take Sinc-layer from the [seminar](https://github.com/LauzHack/deep-learning-bootcamp/blob/summer25/day08/Seminar_AntiSpoofing.ipynb). Note the difference between SincNet And RawNet2. Set `min_low_hz, min_band_hz` to zero.

---

# General recommendations

You should organize the repository in a project-style (see `day05` and [Project Template](https://github.com/Blinorot/pytorch_project_template)) or use a structured (with `markdown` comments) notebook.

Good to do:

- All the necessary packages should be mentioned in `./requirements.txt` or in an installation guide section of README.md
- Use W&B for logging losses and your input/output text/images/audio.
- All necessary resources (such as model checkpoints) should be downloadable with a script
  Mention the script (or lines of code) in the `README.md`
