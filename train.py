import torch
from torch.utils.data import DataLoader
import numpy as np
import os

from setting import Setting
from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from network import create_h0_strategy
from evaluation import Evaluation

'''
Main train script to invoke from commandline.
'''

### parse settings ###
setting = Setting()
setting.parse()
print(setting)

### load dataset ###
poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)
poi_loader.read(setting.dataset_file)
dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset_test = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
assert setting.batch_size < poi_loader.user_count(), 'batch size must be lower than the amount of available users'

### create flashback trainer ###
trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s)
h0_strategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)
trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.hidden_dim, setting.rnn_factory, setting.device)
evaluation_test = Evaluation(dataset_test, dataloader_test, poi_loader.user_count(), h0_strategy, trainer, setting)
print('{} {}'.format(trainer, setting.rnn_factory))

###  training loop ###
optimizer = torch.optim.Adam(trainer.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.2)
train_user_chunk = setting.train_user_chunk
if train_user_chunk is None or train_user_chunk <= 0:
    train_user_chunk = 8 if setting.device.type == 'cuda' else setting.batch_size
train_user_chunk = min(train_user_chunk, setting.batch_size)
print(f'Using train user chunk size: {train_user_chunk}')

for e in range(setting.epochs):
    h = h0_strategy.on_init(setting.batch_size, setting.device)    
    dataset.shuffle_users() # shuffle users before each epoch!
    
    losses = []
    
    for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in enumerate(dataloader):
        # reset hidden states for newly added users
        for j, reset in enumerate(reset_h):
            if reset:
                if setting.is_lstm:
                    hc = h0_strategy.on_reset(active_users[0][j])
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])
        
        x = x.squeeze().to(setting.device)
        t = t.squeeze().to(setting.device)
        s = s.squeeze().to(setting.device)
        y = y.squeeze().to(setting.device)
        y_t = y_t.squeeze().to(setting.device)
        y_s = y_s.squeeze().to(setting.device)                
        active_users = active_users.squeeze(0).to(setting.device)
        # IMPORTANT: detach hidden state from previous batch graph
        if h is not None:
            if isinstance(h, tuple):  # LSTM
                h = (h[0].detach(), h[1].detach())
            else:  # RNN / GRU
                h = h.detach()

        optimizer.zero_grad()
        batch_users = x.size(1)
        batch_loss = 0.0
        if setting.is_lstm:
            next_h0_chunks = []
            next_h1_chunks = []
        else:
            next_h_chunks = []

        for start in range(0, batch_users, train_user_chunk):
            end = min(start + train_user_chunk, batch_users)
            chunk_weight = (end - start) / batch_users

            x_chunk = x[:, start:end]
            t_chunk = t[:, start:end]
            s_chunk = s[:, start:end]
            y_chunk = y[:, start:end]
            y_t_chunk = y_t[:, start:end]
            y_s_chunk = y_s[:, start:end]
            active_users_chunk = active_users[start:end]

            if setting.is_lstm:
                h_chunk = (
                    h[0][:, start:end, :],
                    h[1][:, start:end, :]
                )
            else:
                h_chunk = h[:, start:end, :]

            loss_chunk, next_h_chunk = trainer.loss(
                x_chunk, t_chunk, s_chunk, y_chunk, y_t_chunk, y_s_chunk, h_chunk, active_users_chunk
            )
            (loss_chunk * chunk_weight).backward()
            batch_loss += loss_chunk.item() * chunk_weight

            if setting.is_lstm:
                next_h0_chunks.append(next_h_chunk[0])
                next_h1_chunks.append(next_h_chunk[1])
            else:
                next_h_chunks.append(next_h_chunk)

        if setting.is_lstm:
            h = (torch.cat(next_h0_chunks, dim=1), torch.cat(next_h1_chunks, dim=1))
        else:
            h = torch.cat(next_h_chunks, dim=1)

        losses.append(batch_loss)
        optimizer.step()
    
    # schedule learning rate once per epoch:
    scheduler.step()
    
    # statistics:
    if (e+1) % 1 == 0:
        epoch_loss = np.mean(losses)
        print(f'Epoch: {e+1}/{setting.epochs}')
        print(f'Used learning rate: {scheduler.get_last_lr()[0]}')
        print(f'Avg Loss: {epoch_loss}')
    if (e+1) % setting.validate_epoch == 0:        
        print(f'~~~ Test Set Evaluation (Epoch: {e+1}) ~~~')
        if setting.device.type == 'cuda':
            torch.cuda.empty_cache()
        evaluation_test.evaluate()

os.makedirs("checkpoints", exist_ok=True)
torch.save(trainer.model.state_dict(), "checkpoints/gowalla_flashback.pt")
print("Saved checkpoint: checkpoints/gowalla_flashback.pt")
