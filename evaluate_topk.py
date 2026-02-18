import torch
from torch.utils.data import DataLoader

from setting import Setting
from dataloader import PoiDataloader
from dataset import Split
from trainer import FlashbackTrainer

CKPT_PATH = "checkpoints/gowalla_flashback.pt"
TOP_K = 5
MAX_BATCHES = 50
DEBUG_FIRST = True


def pick_scores_last(out: torch.Tensor, seq_len: int, batch_users: int) -> torch.Tensor:
    """
    Returns scores for the last timestep with shape (batch_users, loc_count),
    regardless of whether out is (seq_len, batch_users, loc) or (batch_users, seq_len, loc),
    or flattened.
    """
    if out.dim() == 3:
        a, b, loc = out.shape

        # Case 1: (seq_len, batch_users, loc)
        if a == seq_len and b == batch_users:
            return out[-1]  # (batch_users, loc)

        # Case 2: (batch_users, seq_len, loc)
        if a == batch_users and b == seq_len:
            return out[:, -1, :]  # (batch_users, loc)

        raise RuntimeError(f"Unexpected 3D out shape {tuple(out.shape)} for seq_len={seq_len}, batch_users={batch_users}")

    if out.dim() == 2:
        n, loc = out.shape

        # Flattened case: (seq_len*batch_users, loc)
        if n == seq_len * batch_users:
            out3 = out.reshape(seq_len, batch_users, loc)
            return out3[-1]  # (batch_users, loc)

        # Already last-step: (batch_users, loc)
        if n == batch_users:
            return out  # (batch_users, loc)

        raise RuntimeError(f"Unexpected 2D out shape {tuple(out.shape)} for seq_len={seq_len}, batch_users={batch_users}")

    raise RuntimeError(f"Unexpected out dims {out.dim()} with shape {tuple(out.shape)}")


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
    state = torch.load(CKPT_PATH, map_location=setting.device)
    trainer.model.load_state_dict(state)
    trainer.model.eval()

    total_users = 0
    hit_users = 0

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

            seq_len, batch_users = y.shape  # expected (20, 200)

            out, _ = trainer.evaluate(x, t, s, y_t, y_s, None, active_users)

            # get (batch_users, loc_count) scores for last timestep
            scores_last = pick_scores_last(out, seq_len=seq_len, batch_users=batch_users)

            # labels for last timestep: (batch_users,)
            labels_last = [-1].reshape(-1).long()

            # top-k: (batch_users, TOP_K)
            topk_idx = torch.topk(scores_last, k=TOP_K, dim=1).indices

            # hit@k: (batch_users,)
            hits = (topk_idx == labels_last.unsqueeze(1)).any(dim=1)
            hit_users += int(hits.sum().item())
            total_users += int(hits.numel())

            if DEBUG_FIRST:
                print("\n[DEBUG]")
                print("y shape:", tuple(y.shape))
                print("out shape:", tuple(out.shape))
                print("scores_last shape:", tuple(scores_last.shape))
                print("labels_last shape:", tuple(labels_last.shape))
                print("sample topk:", topk_idx[0].tolist(), " true:", int(labels_last[0].item()))
                DEBUG_FIRST = False

    acc = hit_users / total_users if total_users > 0 else 0.0
    print(f"\nHit@{TOP_K} over {total_users} user-samples (across {MAX_BATCHES} batches): {acc:.4f}")


if __name__ == "__main__":
    main()
