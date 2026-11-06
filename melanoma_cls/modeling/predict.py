import numpy as np
import torch

from torch.utils.data import DataLoader
from melanoma_cls.dataset import PH2Dataset

def eval_model(df, fold_state_dict, test_transform, root_dir, device, batch_size):
    fold_eval_dict = {fold: None for fold in range(1, len(fold_state_dict) + 1)}
    
    for fold_idx, fold_dict in fold_state_dict.items():
        logits_list = []
        probs_list = []

        fold_model = fold_dict['fold_best_model'].to(device)
        fold_model.eval()
        test_dataset = PH2Dataset(df, root_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2, persistent_workers=True)

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = fold_model(inputs) # (batch_size, n_classes)
                probs = torch.softmax(outputs, dim=-1)[:, -1] # (batch_size, )

                logits_list.append(outputs.detach().cpu().numpy()) # (batch_size, n_classes)
                probs_list.append(probs.detach().cpu().numpy()) # (batch_size, )

        fold_eval_dict[fold_idx] = {
            'fold_logits':np.concatenate(logits_list, axis=0), # (examples, n_classes)
            'fold_probs':np.concatenate(probs_list) # (examples, )
        }

    return fold_eval_dict, test_dataset.targets.to_numpy()