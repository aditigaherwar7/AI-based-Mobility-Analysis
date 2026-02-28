import argparse
import json
import os
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from dataloader import PoiDataloader
from dataset import Split
from setting import Setting
from trainer import FlashbackTrainer


def pick_scores_last(out: torch.Tensor, seq_len: int, batch_users: int) -> torch.Tensor:
    """Return scores for the last timestep with shape (batch_users, loc_count)."""
    if out.dim() == 3:
        a, b, _ = out.shape
        if a == seq_len and b == batch_users:
            return out[-1]
        if a == batch_users and b == seq_len:
            return out[:, -1, :]
        raise RuntimeError(f"Unexpected 3D output shape: {tuple(out.shape)}")

    if out.dim() == 2:
        n, loc_count = out.shape
        if n == seq_len * batch_users:
            return out.reshape(seq_len, batch_users, loc_count)[-1]
        if n == batch_users:
            return out
        raise RuntimeError(f"Unexpected 2D output shape: {tuple(out.shape)}")

    raise RuntimeError(f"Unexpected output dimensions: {out.dim()} shape={tuple(out.shape)}")


def build_user_entry(
    user_id: str,
    sequence_length: int,
    visited_coords: List[tuple],
    pred_coord: tuple,
    confidence: float,
    true_coord: tuple,
) -> Dict:
    return {
        "user_id": user_id,
        "sequence_length": int(sequence_length),
        "visited": [{"lat": float(lat), "lng": float(lng)} for lat, lng in visited_coords],
        "predicted": {
            "lat": float(pred_coord[0]),
            "lng": float(pred_coord[1]),
            "confidence": float(confidence),
        },
        "true_next": {
            "lat": float(true_coord[0]),
            "lng": float(true_coord[1]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Flashback predictions as JSON for dashboard")
    parser.add_argument("--num-users", type=int, default=10, help="Number of users to export")
    parser.add_argument("--sequence-length", type=int, default=10, help="Visited history length per exported user")
    parser.add_argument("--output", type=str, default="results/predictions.json", help="Output JSON path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/gowalla_flashback.pt", help="Model checkpoint path")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    setting = Setting()
    setting.parse()

    poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)
    poi_loader.read(setting.dataset_file)

    dataset_test = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s)
    trainer.prepare(
        loc_count=poi_loader.locations(),
        user_count=poi_loader.user_count(),
        hidden_size=setting.hidden_dim,
        gru_factory=setting.rnn_factory,
        device=setting.device,
    )

    state = torch.load(args.checkpoint, map_location=setting.device)
    trainer.model.load_state_dict(state)
    trainer.model.eval()

    # Map internal user ids back to original dataset ids when available.
    reverse_user_id = {mapped: original for original, mapped in poi_loader.user2id.items()}

    exported: List[Dict] = []
    seen_users = set()

    with torch.no_grad():
        for batch in dataloader_test:
            if len(exported) >= args.num_users:
                break

            x = batch[0].squeeze(0).to(setting.device)
            t = batch[1].squeeze(0).to(setting.device)
            s = batch[2].squeeze(0).to(setting.device)
            y = batch[3].squeeze(0).to(setting.device)
            y_t = batch[4].squeeze(0).to(setting.device)
            y_s = batch[5].squeeze(0).to(setting.device)
            active_users = batch[7].squeeze(0).to(setting.device)

            seq_len, batch_users = y.shape
            out, _ = trainer.evaluate(x, t, s, y_t, y_s, None, active_users)
            scores_last = pick_scores_last(out, seq_len=seq_len, batch_users=batch_users)
            probs_last = torch.softmax(scores_last, dim=1)

            pred_ids = torch.argmax(scores_last, dim=1)
            true_ids = y[-1].long()
            confidences = probs_last.gather(1, pred_ids.unsqueeze(1)).squeeze(1)

            for col in range(batch_users):
                if len(exported) >= args.num_users:
                    break

                dataset_user_index = int(active_users[col].item())
                remapped_user_id = int(dataset_test.users[dataset_user_index])
                original_user_id = reverse_user_id.get(remapped_user_id, remapped_user_id)
                user_id_str = str(original_user_id)

                if user_id_str in seen_users:
                    continue

                visited_loc_ids = x[:, col].detach().cpu().tolist()
                visited_loc_ids = visited_loc_ids[-args.sequence_length:]
                visited_coords = [poi_loader.get_coord(int(loc_id)) for loc_id in visited_loc_ids]
                visited_coords = [coord for coord in visited_coords if coord is not None]

                pred_coord = poi_loader.get_coord(int(pred_ids[col].item()))
                true_coord = poi_loader.get_coord(int(true_ids[col].item()))

                if pred_coord is None or true_coord is None or len(visited_coords) == 0:
                    continue

                entry = build_user_entry(
                    user_id=user_id_str,
                    sequence_length=min(args.sequence_length, len(visited_coords)),
                    visited_coords=visited_coords,
                    pred_coord=pred_coord,
                    confidence=float(confidences[col].item()),
                    true_coord=true_coord,
                )
                exported.append(entry)
                seen_users.add(user_id_str)

    if len(exported) == 0:
        raise RuntimeError("No users could be exported. Check dataset/checkpoint compatibility.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {"users": exported}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Predictions exported to {args.output}")
    print(f"Exported users: {len(exported)}")


if __name__ == "__main__":
    main()
