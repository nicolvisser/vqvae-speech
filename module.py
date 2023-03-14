import pytorch_lightning as pl
import torch
from model import Encoder, Decoder
from torchaudio import transforms
from torch.nn.functional import cross_entropy


class VQVAE(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        self.encoder = Encoder(**cfg.model.encoder)
        self.decoder = Decoder(**cfg.model.decoder)

        self.t_mel_spec = transforms.MelSpectrogram(
            sample_rate=cfg.preprocessing.sample_rate,
            n_fft=cfg.preprocessing.n_fft,
            win_length=cfg.preprocessing.win_length,
            hop_length=cfg.preprocessing.hop_length,
            f_min=cfg.preprocessing.f_min,
            n_mels=cfg.preprocessing.n_mels,
            power=1.0
        )

        self.t_amp_to_db = transforms.AmplitudeToDB(
            stype="magnitude",
            top_db=cfg.preprocessing.top_db)

        self.t_mu_law = transforms.MuLawEncoding(
            quantization_channels=cfg.preprocessing.quantization_channels)

    def training_step(self, batch, batch_idx):
        wav, speakers = batch
        print(f"wav shape: {wav.shape}")

        # resample here
        log_mel = self.t_amp_to_db(self.t_mel_spec(wav)) / self.cfg.preprocessing.top_db + 1
        print(f"log_mel shape: {log_mel.shape}")

        wav_mu_law = self.t_mu_law(wav)
        print(f"wav_mu_law shape: {wav_mu_law.shape}")

        # might have to revisit benji's code in dataset to see if log_mel and wav corresponds to that

        z, vq_loss, perplexity = self.encoder(log_mel)
        print(f"z shape: {z.shape}")

        output = self.decoder(wav_mu_law[:, :-1], z, speakers)
        print(f"output shape: {output.shape}")

        recon_loss = cross_entropy(output.transpose(1, 2), wav_mu_law[:, 1:])

        loss = recon_loss + vq_loss

        return {"loss": loss, "reconstruction loss": recon_loss, "vq loss": vq_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.training.optimizer.lr
        )
        return optimizer
