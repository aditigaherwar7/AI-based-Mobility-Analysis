import torch
from torch.utils.data import DataLoader

from setting import Setting
from dataloader import PoiDataloader
from dataset import Split
from trainer import FlashbackTrainer


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
    ckpt = "checkpoints/gowalla_flashback.pt"
    trainer.model.load_state_dict(torch.load(ckpt, map_location=setting.device))
    trainer.model.eval()

    # Take first batch
    batch = next(iter(dataloader_test))

    # Unpack (8 items)
    x, t, s, y, y_t, y_s, reset_h, active_users = batch

    # Remove batch dimension=1
    x = x.squeeze(0).to(setting.device)        # [20, 200]
    t = t.squeeze(0).to(setting.device)        # [20, 200]
    s = s.squeeze(0).to(setting.device)        # [20, 200, 2]
    y = y.squeeze(0).to(setting.device)        # [20, 200]
    y_t = y_t.squeeze(0).to(setting.device)    # [20, 200]
    y_s = y_s.squeeze(0).to(setting.device)    # [20, 200, 2]
    active_users = active_users.squeeze(0).to(setting.device)  # [200]

    # Hidden state (start fresh)
    h = None

    # Predict logits for each timestep/user
    out, h = trainer.evaluate(x, t, s, y_t, y_s, h, active_users)

    # out likely: [20, 200, loc_count]
    # We'll pick ONE user from this batch and predict next location for the last timestep.
    user_idx = 0  # change 0..199 to see different users
    last_scores = out[-1, user_idx, :]  # [loc_count]

    topk = 5
    scores, idx = torch.topk(last_scores, k=topk)

    print("\nTop-5 predicted next locations for user_idx=0 (Location IDs):")
    for rank, (loc_id, sc) in enumerate(zip(idx.tolist(), scores.tolist()), 1):
        print(f"{rank}. LocationID={loc_id} | score={sc:.4f}")

    # True label for comparison (last timestep)
    true_loc = int(y[-1, user_idx].item())
    print(f"\nTrue next location label (LocationID): {true_loc}")


if __name__ == "__main__":
    main()
