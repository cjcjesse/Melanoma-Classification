import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

def plot_one_image(meta_df, train_root_dir):
    with Image.open(train_root_dir / f"{meta_df['isic_id'][0]}.jpg") as sample:
        sample_arr = np.asarray(sample)
        plt.imshow(sample_arr)
        plt.title(f"Sample ISIC Dermoscopy Image (ID: {meta_df['isic_id'][0]})")
        plt.yticks([])
        plt.xticks([])
        plt.show()
        plt.close()

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axes = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(10, 6))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axes[0, i].imshow(np.asarray(img))
        axes[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def viz_batch_transform(viz_loader):
    inputs, _ = next(iter(viz_loader))
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    mean = mean.reshape(1, 3, 1, 1).to(inputs.device)
    std = std.reshape(1, 3, 1, 1).to(inputs.device)

    denorm_inputs = (inputs * std) + mean
    denorm_inputs = torch.clamp(denorm_inputs, 0.0, 1.0)

    grid = vutils.make_grid(denorm_inputs, nrow=4, padding=5)
    show(grid)

def plot_fold_lc(fold_state_dict, model_type):
    for fold_idx, fold_dict in fold_state_dict.items():
        train_epoch_losses_s1 = fold_dict['fold_train_losses_s1']    
        X = np.arange(1, len(train_epoch_losses_s1) + 1)
        #fig, axes = plt.subplots(2, 1, figsize=(10, 12))
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        for idx, ax in enumerate(axes.flat, start=1):
            ax.plot(X, fold_dict[f'fold_val_losses_s{idx}'], label='Val')
            ax.plot(X, fold_dict[f'fold_train_losses_s{idx}'], label='Train')
            ax.set_title(f'[FOLD {fold_idx}] {model_type.capitalize()} Stage {idx} Learning Curves')
            ax.set_ylabel('Loss')
            ax.set_xlabel('Epochs')
            ax.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

def generate_val_report(val_report_dict):
    pooled_probs = np.concatenate([fold_dict['fold_best_probs'] for fold_dict in val_report_dict.values()], axis=0) # (n_examples, )
    #pooled_preds = np.concatenate([fold_dict['fold_best_preds'] for fold_dict in val_report_dict.values()]) # (n_examples, )
    pooled_true = np.concatenate([fold_dict['fold_best_true'] for fold_dict in val_report_dict.values()]) # (n_examples, )

    fold_auroc_list = [] # list of (n_examples, ) 1d-arrays
    fold_auprc_list = [] # list of (n_examples, ) 1d-arrays

    print(f"{'=' * 30}\nValidation Performance Summary\n{'=' * 30}\n")
    for fold_idx, fold_dict in val_report_dict.items():
        fold_probs = fold_dict['fold_best_probs']
        fold_true = fold_dict['fold_best_true']
        fold_auroc = roc_auc_score(fold_true, fold_probs)
        fold_auprc = average_precision_score(fold_true, fold_probs)
        fold_auroc_list.append(fold_auroc)
        fold_auprc_list.append(fold_auprc)
        print(f"[FOLD {fold_idx}]:\n\tBest AUROC: {fold_auroc:.3f}\n\tBest AUPRC: {fold_auprc:.3f}\n")
        #print(f"[FOLD {fold_idx}]:\n\tBest AUROC: {roc_auc_score(fold_true, fold_probs):.3f}\n\tBest AUPRC: {average_precision_score(fold_true, fold_probs):.3f}\n")
    val_avg_auroc = np.mean(np.stack(fold_auroc_list , axis=0), axis=0) # (folds, n_examples) -> (n_examples, )
    val_avg_auprc = np.mean(np.stack(fold_auprc_list , axis=0), axis=0) # (folds, n_examples) -> (n_examples, )
    std_avg_auroc = np.std(np.stack(fold_auroc_list , axis=0), axis=0) # (folds, n_examples) -> (n_examples, )
    std_avg_auprc = np.std(np.stack(fold_auprc_list , axis=0), axis=0) # (folds, n_examples) -> (n_examples, )

    print(f"[FOLD AVG]:\n\tAUROC: {val_avg_auroc:.3f} \u00B1 {std_avg_auroc:.3f}\n\tAUPRC: {val_avg_auprc:.3f} \u00B1 {std_avg_auprc:.3f}\n")
    print(f"[POOLED]\n\tBest AUROC: {roc_auc_score(pooled_true, pooled_probs):.3f}\n\tBest AUPRC: {average_precision_score(pooled_true, pooled_probs):.3f}")

def generate_test_report(test_report_dict, test_true):
    avg_probs = np.mean(np.stack([fold_dict['fold_probs'] for fold_dict in test_report_dict.values()], axis=0), axis=0) # (folds, examples) -> (examples, )
    for fold_idx, fold_dict in test_report_dict.items():
        eval_fold_probs = fold_dict['fold_probs']
        print(f"[FOLD {fold_idx}]:\n\tTest AUROC: {roc_auc_score(test_true, eval_fold_probs):.3f}\n\tTest AUPRC: {average_precision_score(test_true, eval_fold_probs):.3f}\n")

    print(f"[ENSEMBLE AVERAGE]\n\tTest AUROC: {roc_auc_score(test_true, avg_probs):.3f}\n\tTest AUPRC: {average_precision_score(test_true, avg_probs):.3f}")

def load_for_cam(im_path, test_transform, device):
    img = Image.open(im_path).convert('RGB')
    input = test_transform(img).unsqueeze(0).to(device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    rgb_img = (input * std + mean).clamp(0, 1)[0]
    rgb_img = rgb_img.permute(1, 2, 0).cpu().numpy().astype(np.float32)

    return rgb_img, input

def reshape_transform(tensor):
    return tensor.permute(0, 3, 1, 2) # turn swin outputs into (B, C, H, W)

def generate_test_gradcam(ph2_df, ph2_img_path, fold_state_dict, test_transform, seed, device, model_type, n=2): # expect to pass ph2_img_path = ph2_cache / 'PH2Dataset/PH2 Dataset images'
    if model_type not in ['densenet', 'resnet', 'swin']:
        raise ValueError("Input model type is not one of ['densenet', 'resnet', 'swin']")
    
    chosen = ph2_df.groupby('labels')[['Name', 'labels']].sample(n=n, random_state=seed).sort_values('labels', ascending=True) # samples n numbers of samples per label groups, returns df with name and labels sorted by labels in ascending order (False/neg samples first)
    p_list = [ph2_img_path / f'{name}/{name}_Dermoscopic_Image/{name}.bmp' for name in chosen['Name']] # list of path objects pointing to sampled test examples
    labels = [int(label) for label in chosen['labels']]

    for fold_idx, fold_dict in fold_state_dict.items():
        fold_model = fold_dict['fold_best_model'].to(device)
        fold_model.eval()

        if model_type == 'densenet':
            target_layers = [fold_model.features.norm5]
        if model_type == 'resnet':
            target_layers = [fold_model.layer4[-1].conv2]
        if model_type == 'swin':
            target_layers = [fold_model.norm]

        plt.figure(figsize=(10, 6))
        rt = reshape_transform if model_type =='swin' else None
        with GradCAM(model=fold_model, target_layers=target_layers, reshape_transform=rt) as cam:
            for i, im_path in enumerate(p_list, start=1):
                rgb_img, input = load_for_cam(im_path, test_transform, device)

                with torch.no_grad():
                    output = fold_model(input)
                    _, pred = torch.max(output, 1)
                    f_pred = int(pred.item())
                    pos_prob = float(torch.softmax(output, dim=1)[0, 1].item())

                targets = [ClassifierOutputTarget(f_pred)]
                grayscale_cam = cam(input, targets=targets)[0]
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                plt.subplot(2, n, i)
                plt.imshow(visualization)
                plt.axis('off')
                plt.title(f'{im_path.stem} - Label={labels[i-1]}, Pred={f_pred}, P(pos)={pos_prob:.3f}', fontsize=9)

        plt.suptitle(f'FOLD {fold_idx}')
        plt.tight_layout()
        plt.show()
        plt.close()