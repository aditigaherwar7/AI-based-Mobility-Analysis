
import torch
import numpy as np

class Evaluation:
    
    '''
    Handles evaluation on a given POI dataset and loader.
    
    The two metrics are MAP and recall@n. Our model predicts sequencse of
    next locations determined by the sequence_length at one pass. During evaluation we
    treat each entry of the sequence as single prediction. One such prediction
    is the ranked list of all available locations and we can compute the two metrics.
    
    As a single prediction is of the size of all available locations,
    evaluation takes its time to compute. The code here is optimized.
    
    Using the --report_user argument one can access the statistics per user.
    '''
    
    def __init__(self, dataset, dataloader, user_count, h0_strategy, trainer, setting):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.h0_strategy = h0_strategy
        self.trainer = trainer
        self.setting = setting

    def evaluate(self):
        original_device = next(self.trainer.model.parameters()).device
        eval_device = torch.device('cpu') if self.setting.device.type == 'cuda' else self.setting.device
        if eval_device != original_device:
            self.trainer.model.to(eval_device)
        if self.setting.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.dataset.reset()
        h = self.h0_strategy.on_init(self.setting.batch_size, eval_device)
        eval_user_chunk = self.setting.eval_user_chunk
        if eval_user_chunk is None or eval_user_chunk <= 0:
            eval_user_chunk = 8 if eval_device.type == 'cuda' else self.setting.batch_size
        eval_user_chunk = min(eval_user_chunk, self.setting.batch_size)
        
        try:
            with torch.no_grad():        
                iter_cnt = 0
                recall1 = 0
                recall5 = 0
                recall10 = 0
                average_precision = 0.
            
            u_iter_cnt = np.zeros(self.user_count)
            u_recall1 = np.zeros(self.user_count)
            u_recall5 = np.zeros(self.user_count)
            u_recall10 = np.zeros(self.user_count)
            u_average_precision = np.zeros(self.user_count)        
            reset_count = torch.zeros(self.user_count)
            
            for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in enumerate(self.dataloader):
                active_users = active_users.squeeze()
                active_users_cpu = active_users.clone()
                for j, reset in enumerate(reset_h):
                    if reset:
                        if self.setting.is_lstm:
                            hc = self.h0_strategy.on_reset_test(active_users[j], eval_device)
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            h[0, j] = self.h0_strategy.on_reset_test(active_users[j], eval_device)
                        reset_count[active_users[j]] += 1
                
                # squeeze for reasons of "loader-batch-size-is-1"
                x = x.squeeze().to(eval_device)
                t = t.squeeze().to(eval_device)
                s = s.squeeze().to(eval_device)            
                y = y.squeeze()
                y_t = y_t.squeeze().to(eval_device)
                y_s = y_s.squeeze().to(eval_device)
                
                active_users = active_users.to(eval_device)            
            
                # evaluate:
                out_chunks = []
                if self.setting.is_lstm:
                    next_h0_chunks = []
                    next_h1_chunks = []
                else:
                    next_h_chunks = []

                batch_users = x.size(1)
                for start in range(0, batch_users, eval_user_chunk):
                    end = min(start + eval_user_chunk, batch_users)
                    x_chunk = x[:, start:end]
                    t_chunk = t[:, start:end]
                    s_chunk = s[:, start:end]
                    y_t_chunk = y_t[:, start:end]
                    y_s_chunk = y_s[:, start:end]
                    active_users_chunk = active_users[start:end]

                    if self.setting.is_lstm:
                        h_chunk = (h[0][:, start:end, :], h[1][:, start:end, :])
                    else:
                        h_chunk = h[:, start:end, :]

                    with torch.no_grad():
                        out_chunk, next_h_chunk = self.trainer.evaluate(
                            x_chunk, t_chunk, s_chunk, y_t_chunk, y_s_chunk, h_chunk, active_users_chunk
                        )

                    out_chunks.append(out_chunk.cpu())
                    if self.setting.is_lstm:
                        next_h0_chunks.append(next_h_chunk[0])
                        next_h1_chunks.append(next_h_chunk[1])
                    else:
                        next_h_chunks.append(next_h_chunk)

                out_chunks = [o.cpu() for o in out_chunks]
                out = torch.cat(out_chunks, dim=0)
                if self.setting.is_lstm:
                    h = (torch.cat(next_h0_chunks, dim=1), torch.cat(next_h1_chunks, dim=1))
                else:
                    h = torch.cat(next_h_chunks, dim=1)
                
                for j in range(self.setting.batch_size):  
                    # o contains a per user list of votes for all locations for each sequence entry
                    o = out[j]                        
                                        
                    # partition elements
                    o_n = o.cpu().detach().numpy()
                    ind = np.argpartition(o_n, -10, axis=1)[:, -10:] # top 10 elements
                                       
                    y_j = y[:, j]
                    user_id = int(active_users_cpu[j].item())
                    
                    for k in range(len(y_j)):                    
                        if (reset_count[user_id] > 1):
                            continue # skip already evaluated users.
                                                                                                            
                        # resort indices for k:
                        ind_k = ind[k]
                        r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)] # sort top 10 elements descending
                                                    
                        r = torch.tensor(r)
                        t = y_j[k]
                        
                        # compute MAP:
                        r_kj = o_n[k, :]
                        t_val = r_kj[t]
                        upper = np.where(r_kj > t_val)[0]
                        precision = 1. / (1+len(upper))
                        
                        # store
                        u_iter_cnt[user_id] += 1
                        u_recall1[user_id] += t in r[:1]
                        u_recall5[user_id] += t in r[:5]
                        u_recall10[user_id] += t in r[:10]
                        u_average_precision[user_id] += precision
            
                formatter = "{0:.8f}"
                for j in range(self.user_count):
                    iter_cnt += u_iter_cnt[j]
                    recall1 += u_recall1[j]
                    recall5 += u_recall5[j]
                    recall10 += u_recall10[j]
                    average_precision += u_average_precision[j]
    
                    if (self.setting.report_user > 0 and (j+1) % self.setting.report_user == 0):
                        print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1', formatter.format(u_recall1[j]/u_iter_cnt[j]), 'MAP', formatter.format(u_average_precision[j]/u_iter_cnt[j]), sep='\t')
            
                print('recall@1:', formatter.format(recall1/iter_cnt))
                print('recall@5:', formatter.format(recall5/iter_cnt))
                print('recall@10:', formatter.format(recall10/iter_cnt))
                print('MAP', formatter.format(average_precision/iter_cnt))
                print('predictions:', iter_cnt)
        finally:
            if eval_device != original_device:
                self.trainer.model.to(original_device)
