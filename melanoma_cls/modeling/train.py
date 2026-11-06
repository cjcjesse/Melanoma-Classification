import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchmetrics.classification import BinaryAveragePrecision
from torchvision.models import resnet50, densenet121, swin_t
from sklearn.model_selection import StratifiedGroupKFold
from melanoma_cls.dataset import ISICDataset

def staged_ft_loop(df, train_transform, val_transform, root_dir, epochs, batch_size, seed, device, models_dir, model_type, n_splits=5, lr=1e-3, weight_decay=1e-3, pos_repeat=5):
    if model_type not in ['densenet', 'resnet', 'swin']:
        raise ValueError("Input model type is not one of ['densenet', 'resnet', 'swin']")
    
    fold_state_dict = {fold: None for fold in range(1, n_splits + 1)}
    sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    start = time.time()
    for fold_idx, (train_idx, val_idx) in tqdm(enumerate(sgkf.split(df['isic_id'], df['target'], df['patient_id']), start=1), desc='[FOLD]', leave=False): # df cols: ['isic_id', 'patient_id' 'target']
        best = {
            'eval_metric': float('-inf'),
            'stage': None,
            'epoch': None,
            'probs': None,
            'preds': None,
            'true': None,
            'path': None,
        }
        train_epoch_losses_s1 = []
        val_epoch_losses_s1 = []
        train_epoch_losses_s2 = []
        val_epoch_losses_s2 = []

        model, criterion, optimizer, scheduler, ap_score, train_loader, val_loader = fold_init(df, root_dir, fold_idx, train_idx, val_idx, train_transform, val_transform, batch_size, lr, weight_decay, pos_repeat, seed, device, model_type)
        print(F'Fold {fold_idx} Start\n')
        for epoch in range(epochs):
            train_loss, train_total = epoch_train(model, train_loader, criterion, optimizer, device, model_type)
            ap_score, val_loss, val_total, probs_list, true_list = epoch_eval(model, ap_score, val_loader, criterion, device)
            train_epoch_losses_s1, val_epoch_losses_s1 = epoch_checkpoint(model, ap_score, probs_list, true_list, train_loss, train_total, train_epoch_losses_s1, val_loss, val_total, val_epoch_losses_s1, scheduler, fold_idx, epoch, epochs, models_dir, 1, model_type, best)

        model, optimizer, scheduler = s1_model_loading(model, best, lr, weight_decay, model_type)
        
        for epoch in range(epochs):
            train_loss, train_total = epoch_train(model, train_loader, criterion, optimizer, device, model_type, True)
            ap_score, val_loss, val_total, probs_list, true_list = epoch_eval(model, ap_score, val_loader, criterion, device)
            train_epoch_losses_s2, val_epoch_losses_s2 = epoch_checkpoint(model, ap_score, probs_list, true_list, train_loss, train_total, train_epoch_losses_s2, val_loss, val_total, val_epoch_losses_s2, scheduler, fold_idx, epoch, epochs, models_dir, 2, model_type, best)

        model = s2_model_loading(model, best, model_type)

        fold_state_dict[fold_idx] = {
            'state_dict_path':best['path'],
            'fold_best_model':model,
            'fold_best_probs':best['probs'], # (n_examples, )
            'fold_best_preds':best['preds'], # (n_examples, )
            'fold_best_true':best['true'], # (n_examples, )
            'fold_best_stage':best['stage'],
            'fold_best_epoch':best['epoch'],
            'fold_best_ap':best['eval_metric'],
            'fold_train_losses_s1':train_epoch_losses_s1,
            'fold_val_losses_s1':val_epoch_losses_s1,
            'fold_train_losses_s2':train_epoch_losses_s2,
            'fold_val_losses_s2':val_epoch_losses_s2
        }

    end = time.time() - start
    print(f'Time Taken (min): {int(end // 60)}\n')

    return fold_state_dict

def fold_init(df, root_dir, fold_idx, train_idx, val_idx, train_transform, val_transform, batch_size, lr, weight_decay, pos_repeat, seed, device, model_type):
    if model_type == 'densenet':
        model = densenet121(weights='IMAGENET1K_V1')
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, 2, bias=True)
        model = model.to(device)
    if model_type == 'resnet':
        model = resnet50(weights='IMAGENET1K_V1')
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2, bias=True)
        model = model.to(device)
    if model_type == 'swin':
        model = swin_t(weights='IMAGENET1K_V1')
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, 2, bias=True)
        model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda param:param.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    ap_score = BinaryAveragePrecision().to(device)
    
    train_dataset = ISICDataset(df.iloc[train_idx, :], root_dir, train_transform)
    y_train = df.iloc[train_idx]['target'].to_numpy()
    w = (1.0 / np.bincount(y_train))[y_train]
    n_pos = int(np.sum(y_train))
    num_samples = 2 * n_pos * pos_repeat
    generator = torch.Generator().manual_seed(seed + fold_idx)
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=num_samples, generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True, pin_memory=True, num_workers=2, persistent_workers=True)

    val_dataset = ISICDataset(df.iloc[val_idx, :], root_dir, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2, persistent_workers=True)

    return model, criterion, optimizer, scheduler, ap_score, train_loader, val_loader

def s1_model_loading(model, best, lr, weight_decay, model_type):
    model.load_state_dict(torch.load(best['path'], weights_only=True)) # initialize with best S1 weights
    if model_type == 'densenet':
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.norm5.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW([
            {'params':model.features.denseblock4.parameters(), 'lr':(lr / 10)},
            {'params':model.features.norm5.parameters(), 'lr':(lr / 10)},
            {'params':model.classifier.parameters(), 'lr':(lr * 3 / 10)}
            ], lr=lr, weight_decay=weight_decay)
    if model_type == 'resnet':
        for param in model.layer4.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW([
            {'params':model.layer4.parameters(), 'lr':(lr / 10)},
            {'params':model.fc.parameters(), 'lr':(lr * 3 / 10)}
            ], lr=lr, weight_decay=weight_decay)
    if model_type == 'swin':
        for param in model.features[-1][-1].parameters():
            param.requires_grad = True
        for param in model.norm.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW([
            {'params':model.features[-1][-1].parameters(), 'lr':(lr / 10)},
            {'params':model.norm.parameters(), 'lr':(lr / 100)},
            {'params':model.head.parameters(), 'lr':(lr * 3 / 10)}
            ], lr=lr, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    return model, optimizer, scheduler

def s2_model_loading(model, best, model_type):
    if best['stage'] == 2:
        checkpoint = torch.load(best['path'], weights_only=True)
        if model_type == 'densenet':
            model.features.denseblock4.load_state_dict(checkpoint['features.denseblock4'])
            model.features.norm5.load_state_dict(checkpoint['features.norm5'])
            model.classifier.load_state_dict(checkpoint['classifier'])
        if model_type == 'resnet':
            model.layer4.load_state_dict(checkpoint['layer4'])
            model.fc.load_state_dict(checkpoint['fc'])
        if model_type == 'swin':
            model.features[-1][-1].load_state_dict(checkpoint['last_block'])
            model.norm.load_state_dict(checkpoint['norm'])
            model.head.load_state_dict(checkpoint['head'])
    else:
        model.load_state_dict(torch.load(best['path'], weights_only=True))

    return model


def epoch_train(model, train_loader, criterion, optimizer, device, model_type, stage_two=False):
    model.eval()
    if model_type == 'densenet':
        model.classifier.train()
        if stage_two:
            model.features.denseblock4.train()
            model.features.norm5.train()
    if model_type == 'resnet':
        model.fc.train()
        if stage_two:
            model.layer4.train()
    if model_type == 'swin':
        model.head.train()
        if stage_two:
            model.features[-1][-1].train()
            model.norm.train()
            
    train_loss = 0.0
    train_total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels) # (batch_size, n_classes), (batch_size, )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item() * labels.size(0)
        train_total += labels.size(0)

    return train_loss, train_total

def epoch_eval(model, ap_score, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_total = 0
    ap_score.reset()
    with torch.no_grad():
        probs_list = []
        true_list = []
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels) # (batch_size, n_classes), (batch_size, )
            probs = torch.softmax(outputs, dim=-1)[:, -1] # (batch_size, )

            val_loss += loss.item() * labels.size(0)
            val_total += labels.size(0)
            ap_score.update(probs, labels)
            probs_list.append(probs.detach().cpu().numpy())
            true_list.append(labels.detach().cpu().numpy())

    return ap_score, val_loss, val_total, probs_list, true_list

def epoch_checkpoint(model, ap_score, probs_list, true_list, train_loss, train_total, train_epoch_losses, val_loss, val_total, val_epoch_losses, scheduler, fold_idx, epoch, epochs, models_dir, stage, model_type, best):
    epoch_eval_metric = ap_score.compute().item()
    train_epoch_loss = train_loss / train_total
    val_epoch_loss = val_loss / val_total
    train_epoch_losses.append(train_epoch_loss)
    val_epoch_losses.append(val_epoch_loss)

    if (epoch == 0) or ((epoch + 1) % 4 == 0) or (epoch == (epochs - 1)):
        print(f"[STAGE {stage}] Epoch {epoch + 1}:\n\tTrain Loss: {train_epoch_loss:.3f}\n\tVal Loss: {val_epoch_loss:.3f}\n\tAUPRC: {epoch_eval_metric:.3f}\n")
    
    scheduler.step(epoch_eval_metric)

    if epoch_eval_metric > best['eval_metric']:
        state_dict_path = models_dir / f'{model_type}_s{stage}_{fold_idx}.pt'
        best_probs = np.concatenate(probs_list)
        best_preds = (best_probs >= 0.5).astype(int)
        best_true = np.concatenate(true_list)
        best.update(eval_metric=epoch_eval_metric, stage=stage, epoch=epoch, probs=best_probs, preds=best_preds, true=best_true, path=state_dict_path)
        
        if stage == 1:
            torch.save(model.state_dict(), state_dict_path)
        else:
            if model_type == 'densenet':
                torch.save({'features.denseblock4': model.features.denseblock4.state_dict(), 'features.norm5': model.features.norm5.state_dict(), 'classifier': model.classifier.state_dict()}, state_dict_path)
            if model_type == 'resnet':
                torch.save({'layer4': model.layer4.state_dict(), 'fc': model.fc.state_dict()}, state_dict_path)
            if model_type == 'swin':
                torch.save({'last_block':model.features[-1][-1].state_dict(), 'norm':model.norm.state_dict(), 'head':model.head.state_dict()}, state_dict_path)
    
    return train_epoch_losses, val_epoch_losses