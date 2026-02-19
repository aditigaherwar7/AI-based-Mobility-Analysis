import torch
from torch.utils.data import DataLoader

from setting import Setting
from dataloader import PoiDataloader
from dataset import Split
from trainer import FlashbackTrainer
from evaluation import Evaluation
from network import create_h0_strategy  # <-- IMPORTANT (same as train.py)

CKPT_PATH = "checkpoints/gowalla_flashback.pt"


def main():
    setting = Setting()
    setting.parse()

    # Load data
    poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)
    poi_loader.read(setting.dataset_file)

    dataset_test = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # Build trainer/model
    trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s)
    trainer.prepare(
        loc_count=poi_loader.locations(),
        user_count=poi_loader.user_count(),
        hidden_size=setting.hidden_dim,
        gru_factory=setting.rnn_factory,
        device=setting.device
    )

    # Load checkpoint
    trainer.model.load_state_dict(torch.load(CKPT_PATH, map_location=setting.device))
    trainer.model.eval()

    # Use official h0 strategy from repo (has on_init, on_reset, on_reset_test)
    h0_strategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)

    # Official evaluation (same pipeline as train.py)
    evaluation = Evaluation(
        dataset_test,
        dataloader_test,
        poi_loader.user_count(),
        h0_strategy,
        trainer,
        setting
    )

    evaluation.evaluate()


if __name__ == "__main__":
    main()
