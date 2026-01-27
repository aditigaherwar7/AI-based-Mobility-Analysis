import folium
import torch
from torch.utils.data import DataLoader

from setting import Setting
from dataloader import PoiDataloader
from dataset import Split
from trainer import FlashbackTrainer


def main():
    setting = Setting()
    setting.parse()

    loader = PoiDataloader(setting.max_users, setting.min_checkins)
    loader.read(setting.dataset_file)

    dataset_test = loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s)
    trainer.prepare(
        loc_count=loader.locations(),
        user_count=loader.user_count(),
        hidden_size=setting.hidden_dim,
        gru_factory=setting.rnn_factory,
        device=setting.device
    )

    trainer.model.load_state_dict(torch.load("checkpoints/gowalla_flashback.pt", map_location=setting.device))
    trainer.model.eval()

    # One batch
    batch = next(iter(dataloader_test))
    x, t, s, y, y_t, y_s, _, active_users = batch

    x = x.squeeze(0)
    t = t.squeeze(0)
    s = s.squeeze(0)
    y = y.squeeze(0)
    y_t = y_t.squeeze(0)
    y_s = y_s.squeeze(0)
    active_users = active_users.squeeze(0)

    # Predict
    out, _ = trainer.evaluate(x, t, s, y_t, y_s, None, active_users)

    user_idx = 0
    pred_loc = int(torch.argmax(out[-1, user_idx]).item())
    true_loc = int(y[-1, user_idx].item())

    # Build visited path from last 10 steps (use x which stores location ids)
    visited_ids = x[-10:, user_idx].tolist()
    visited_coords = [loader.get_coord(int(i)) for i in visited_ids]
    visited_coords = [c for c in visited_coords if c is not None]

    pred_coord = loader.get_coord(pred_loc)
    true_coord = loader.get_coord(true_loc)

    # Center map
    center = pred_coord or (visited_coords[-1] if visited_coords else (0, 0))

    m = folium.Map(location=center, zoom_start=13)

    # Plot visited points
    for c in visited_coords:
        folium.CircleMarker(location=c, radius=4, popup="Visited", fill=True).add_to(m)

    # Plot predicted point
    if pred_coord is not None:
        folium.Marker(location=pred_coord, popup="Predicted Next", icon=folium.Icon(color="red")).add_to(m)

    # Plot true point
    if true_coord is not None:
        folium.Marker(location=true_coord, popup="True Next", icon=folium.Icon(color="green")).add_to(m)

    # Save
    out_file = "day13_map.html"
    m.save(out_file)

    print("\nSaved:", out_file)
    print("Predicted LocationID:", pred_loc, "->", pred_coord)
    print("True LocationID:", true_loc, "->", true_coord)


if __name__ == "__main__":
    main()
