import torch
from torch.utils.data import DataLoader

from setting import Setting
from dataloader import PoiDataloader
from dataset import Split
from trainer import FlashbackTrainer

CKPT_PATH = "checkpoints/gowalla_flashback.pt"
TOP_K = 5
MAX_BATCHES = 50
USER_IDX = 0
DEBUG_FIRST = True


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

    total = 0
    correct_topk = 0

    global DEBUG_FIRST

    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader_test):
            if b_idx >= MAX_BATCHES:
                break

            x = batch[0].squeeze().to(setting.device)
            t = batch[1].squeeze().to(setting.device)
            s = batch[2].squeeze().to(setting.device)
            y = batch[3].squeeze().to(setting.device)
            y_t = batch[4].squeeze().to(setting.device)
            y_s = batch[5].squeeze().to(setting.device)
            active_users = batch[7].to(setting.device)

            h = None

            # Predict scores for all timesteps/users
            out, h = trainer.evaluate(x, t, s, y_t, y_s, h, active_users)

            # out should be [seq_len, batch_users, loc_count]
            # Pick last timestep and one user for evaluation
            last_scores = out[-1, USER_IDX, :]  # (loc_count,)
            _, idx = torch.topk(last_scores, k=TOP_K)

            # True label for same timestep/user
            true_loc = int(y[-1, USER_IDX].item())

            if DEBUG_FIRST:
                print("\n[DEBUG] Shapes:")
                print("out:", tuple(out.shape))
                print("x:", tuple(x.shape))
                print("y:", tuple(y.shape))
                print("TopK idx sample:", idx[:5].tolist())
                print("True loc (y[-1,USER_IDX]):", true_loc)
                DEBUG_FIRST = False

            if true_loc in idx.tolist():
                correct_topk += 1
            total += 1

    acc = correct_topk / total if total > 0 else 0
    print(f"\nTop-{TOP_K} Accuracy over {total} samples: {acc:.4f}")


if __name__ == "__main__":
    main()
