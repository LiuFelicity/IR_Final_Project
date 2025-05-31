import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch, length, dataloader, model, optimizer, batch_size, writer):    
    model.train()
    print('Now, training start ...')
    sum_loss = 0.0
    sum_model_loss = 0.0
    sum_reg_loss = 0.0
    sum_contrastive_loss = 0.0
    sum_weight_loss = 0.0
    step = 0.0
    pbar = tqdm(total=length)
    num_pbar = 0
    sum_mat = 0.0

    for user_tensor, item_tensor in dataloader:
        optimizer.zero_grad()
        loss, model_loss, reg_loss = model.loss(user_tensor.to(device), item_tensor.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        sum_mat += model.mat.detach().cpu().item()
        sum_loss += loss.cpu().item()
        sum_model_loss += model_loss.cpu().item()
        sum_reg_loss += reg_loss.cpu().item()
        pbar.update(batch_size)
        num_pbar += batch_size
        step += 1.0

    pbar.close()
    print('----------------- loss value:{}  model_loss value:{} contrastive_loss value:{} reg_loss value:{} --------------'
        .format(sum_loss/step, sum_model_loss/step, sum_contrastive_loss/step, sum_reg_loss/step))

    return loss, sum_mat/step
