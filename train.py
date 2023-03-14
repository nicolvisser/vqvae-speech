import hydra
import torch.nn.functional
import wandb
from torch.utils.data import DataLoader
import torchaudio
import pytorch_lightning as pl
from module import VQVAE
from torch.utils.data import Dataset
from sklearn import preprocessing
from torch.nn.functional import cross_entropy


class SpeechCommandsDataset(Dataset):
    def __init__(self, subset):
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root="datasets/speechcommands/",
            url="speech_commands_v0.01",
            download=True,
            subset=subset)

        speaker_ids = []
        print('encoding speaker labels...')
        for wav, sr, label, speaker_id, utterance_number in self.dataset:
            speaker_ids.append(speaker_id)

        self.le = preprocessing.LabelEncoder()
        self.le.fit(speaker_ids)

        self.num_speakers = torch.IntTensor([len(self.le.classes_)])

    def __getitem__(self, item):
        wav, sr, label, speaker_id, utterance_number = self.dataset.__getitem__(item)
        if wav.shape[1] != 16000:
            wav = torch.nn.functional.pad(wav, (0, 16000 - wav.shape[1]), mode='constant')
        wav = wav.squeeze()
        speaker_id = torch.IntTensor(self.le.transform([speaker_id]))
        return wav, speaker_id

    def __len__(self):
        return self.dataset.__len__()


@hydra.main(version_base=None, config_path="config", config_name="train")
def train_model(cfg):
    train_dataset = SpeechCommandsDataset(subset='training')
    val_dataset = SpeechCommandsDataset(subset='validation')

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=3,  # cfg.training.batch_size,
        shuffle=True,
        drop_last=True)
    #
    # val_dataloader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=cfg.training.batch_size,
    #     shuffle=False,
    #     drop_last=False)
    #
    # #wandb.login()
    #
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     monitor='validation loss',
    #     mode='min'
    # )

    model = VQVAE(cfg)

    for i, batch in enumerate(train_dataloader):
        wavs, speakers = batch
        print(f"wav shape: {wavs.shape}")

        # resample here
        log_mel = model.t_amp_to_db(model.t_mel_spec(wavs)) / model.cfg.preprocessing.top_db + 1
        print(f"log_mel shape: {log_mel.shape}")

        wav_mu_law = model.t_mu_law(wavs)
        print(f"wav_mu_law shape: {wav_mu_law.shape}")

        # might have to revisit benji's code in dataset to see if log_mel and wav corresponds to that

        z, vq_loss, perplexity = model.encoder(log_mel)
        print(f"z shape: {z.shape}")

        output = model.decoder(wav_mu_law, z, speakers)
        print(f"output shape: {output.shape}")

        recon_loss = cross_entropy(output.transpose(1, 2), wav_mu_law[:, 1:])

        loss = recon_loss + vq_loss

        print( {"loss": loss, "reconstruction loss": recon_loss, "vq loss": vq_loss})

        if i == 1:
            break

    return

    pl.loggers.WandbLogger(
        project=cfg.wandb.project,
        log_model='all',
        save_dir='checkpoints'
    )

    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator='gpu',
        precision=16,
        max_epochs=cfg.training.num_epochs,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )


if __name__ == "__main__":
    train_model()
