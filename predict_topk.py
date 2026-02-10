import os
import torch
from torch.utils.data import DataLoader

from setting import Setting
from dataloader import PoiDataloader
from dataset import Split
from trainer import FlashbackTrainer


# ===== CONFIG (keep simple for demo) =====
CKPT_PATH = "checkpoints/gowalla_flashback.pt"
TOP_K = 5


def main():
    setting = Setting()
    setting.parse()

    # Load data
    poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)
    poi_loader.read(setting.dataset_file)

    dataset_test = poi_loader.create_dataset(
        setting.sequence_length,
        setting.batch_size,
        Split.TEST
    )
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
    state = torch.load(CKPT_PATH, map_location=setting.device)
    trainer.model.load_state_dict(state)
    trainer.model.eval()

    # Take first batch
    batch = next(iter(dataloader_test))

    x = batch[0].squeeze().to(setting.device)
    t = batch[1].squeeze().to(setting.device)
    s = batch[2].squeeze().to(setting.device)
    y = batch[3].squeeze().to(setting.device)
    y_t = batch[4].squeeze().to(setting.device)
    y_s = batch[5].squeeze().to(setting.device)
    active_users = batch[7].to(setting.device)

    h = None

    # Predict
    with torch.no_grad():
        out, h = trainer.evaluate(x, t, s, y_t, y_s, h, active_users)

    out = out.reshape(-1, out.shape[-1])
    last_scores = out[-1]

    scores, idx = torch.topk(last_scores, k=TOP_K)

    print(f"\nTop-{TOP_K} predicted next locations:")
    for i, (loc, sc) in enumerate(zip(idx.tolist(), scores.tolist()), 1):
        print(f"{i}. LocationID={loc} | score={sc:.4f}")

    if y.numel() > 0:
        true_loc = y.view(-1)[-1].item()
        print(f"\nTrue next location: {true_loc}")
    if y.numel() > 0:
        true_loc = y.view(-1)[-1].item()
        print(f"\nTrue next location: {true_loc}")

    # Save results to file (INSIDE main)
    os.makedirs("results", exist_ok=True)
    with open("results/topk_prediction_sample.txt", "w", encoding="utf-8") as f:
        f.write(f"Top-{TOP_K} predicted next locations:\n")
        for i, (loc, sc) in enumerate(zip(idx.tolist(), scores.tolist()), 1):
            f.write(f"{i}. LocationID={loc} | score={sc:.4f}\n")

        if y.numel() > 0:
            true_loc = y.view(-1)[-1].item()
            f.write(f"\nTrue next location: {true_loc}\n")


if __name__ == "__main__":
    main()

