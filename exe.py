import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=10,
    foldername='',
):
    optimizer = Adam(model.parameters(), lr=config['train']['lr'], weight_decay=1e-6)
    if foldername != '':
        output_path = foldername + '/model.pth'
    m = []
    for i in range(int(config['train']['epochs'] / 10)):
        m.append(i * 10)
        
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=m[1:], gamma=0.8)
    # earlystopping count
    ct = 0
    _, target_var = pickle.load(open('preprocess/data/var.pkl', 'rb'))
    size_y = 10 * len(target_var)
    best_valid_loss = np.inf
    for epoch_no in range(config['train']['epochs']):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch, config['diffusion']['size'], size_y)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        'avg_epoch_loss': avg_loss / batch_no,
                        'epoch': epoch_no,
                        'lr': optimizer.param_groups[0]['lr']
                    },
                    refresh=False,
                )
                
            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0 and epoch_no > int((config['train']['epochs']) / 2 - 5):
            model.eval()
            CRPS_valid, _ = evaluate(0, model, valid_loader, nsample=5, foldername=foldername)
            print('best: {}_current: {}'.format(best_valid_loss, CRPS_valid))
            if best_valid_loss > CRPS_valid:
                ct = 0
                best_valid_loss = CRPS_valid
                torch.save(model.state_dict(), output_path)
                print('model updated')
            else:
                ct += 1
                print('ct: {}'.format(ct))
            # earlystopping
            if ct > 2:
                model.load_state_dict(torch.load(output_path))
                print('stop')
                break

def calc_metrics(is_test, all_generation, all_samples_y):
    MSE = None
    target = all_samples_y[:, 2]
    if is_test == 1:
        quantiles = np.arange(0.05, 1.0, 0.05)
        # calculate MSE
        gt = all_samples_y[:, 2]
        mask = all_samples_y[:, 3]
        prediction = all_generation.median(dim=2)
        MSE = ((prediction.values - gt) * mask) ** 2
        MSE = MSE.sum() / mask.sum()
    else:
        quantiles = np.arange(0.25, 1.0, 0.25)
    denom = torch.sum(torch.abs(target))
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(all_generation)):
            q_pred.append(torch.quantile(all_generation[j], quantiles[i], dim = -1))
        q_pred = torch.cat(q_pred, 0).reshape(-1)
        target = target.reshape(-1)
        q_loss = 2 * torch.sum(torch.abs((q_pred - target) * all_samples_y[:, 3].reshape(-1) * ((target <= q_pred) * 1.0 - quantiles[i])))
        CRPS += q_loss / denom
    
    return CRPS.item() / len(quantiles), MSE

def evaluate(is_test, model, data_loader, nsample=100, foldername=""):
    with torch.no_grad():
        model.eval()
        all_samples_x = []
        all_samples_y = []
        all_generation = []
        with tqdm(data_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, batch in enumerate(it, start=1):
                # ground truth values will be replaced with pure noise before generation
                output = model.evaluate(batch, nsample)
                generation, samples_y, samples_x = output
                all_generation.append(generation)
                all_samples_x.append(samples_x)
                all_samples_y.append(samples_y)
            
            all_generation = torch.cat(all_generation)
            all_samples_x = torch.cat(all_samples_x)
            all_samples_y = torch.cat(all_samples_y)
            CRPS, MSE = calc_metrics(is_test, all_generation, all_samples_y)
            if is_test == 1:
                pickle.dump([all_generation, all_samples_y, all_samples_x], open(foldername + "/generated_outputs" + str(nsample) + ".pkl", "wb"))
                pickle.dump([CRPS, MSE], open(foldername + "/result_nsample" + str(nsample) + ".pkl", "wb"))
            return CRPS, MSE
