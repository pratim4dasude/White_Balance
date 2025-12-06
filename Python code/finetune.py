import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import tifffile
from PIL import Image
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler

CONFIG = {
    'TRAIN_CSV': r'C:\Users\KIIT\PycharmProjects\aftrsht\dataset\Train\sliders.csv',
    'TRAIN_IMG_DIR': r'C:\Users\KIIT\PycharmProjects\aftrsht\dataset\Train\images',

    'PREVIOUS_BEST_MODEL_PATH': 'model_new_woo.pth',

    'MODEL_NAME': 'efficientnet_v2_s',

    'BATCH_SIZE': 6,
    'LR': 1e-5,
    'EPOCHS': 15,
    'IMG_SIZE': 384,

    'NUM_WORKERS': 0,

    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'BEST_MODEL_PATH': 'best_efficientnet_v2_s_384_finetune_color_jitter_rem.pth',
    'PRED_CSV_PATH': 'val_predictions_efficientnet_v2_s_384_finetune_color_jitter_rem.csv',

    'WEIGHT_DECAY': 5e-4,
    'DROPOUT_RATE': 0.4,
}

TEMP_SCALE = 10000.0
TINT_SCALE = 200.0
TEMP_MIN, TEMP_MAX = 1500.0, 10000.0
TINT_MIN, TINT_MAX = -50.0, 50.0


class WhiteBalanceDataset(Dataset):
    def __init__(self, df, img_dir, processor, id_col='id_global'):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.processor = processor
        self.id_col = id_col

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, img_id: str) -> str:
        exts = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']
        for ext in exts:
            cand = os.path.join(self.img_dir, img_id + ext)
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(
            f"No image file found for id '{img_id}' in {self.img_dir} with extensions: {exts}"
        )

    def _read_image(self, img_path: str) -> Image.Image:
        ext = os.path.splitext(img_path)[1].lower()
        if ext in ['.tif', '.tiff']:
            img = tifffile.imread(img_path)
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                elif img.ndim == 3 and img.shape[0] in [1, 3]:
                    img = np.transpose(img, (1, 2, 0))
                img = Image.fromarray(img.astype(np.uint8))
        else:
            img = Image.open(img_path).convert('RGB')
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row[self.id_col])
        img_path = self._resolve_image_path(img_id)
        img = self._read_image(img_path)

        # transforms pipeline handles the resizing to 384x384
        pixel_values = self.processor(img)

        temp = float(row["Temperature"])
        tint = float(row["Tint"])
        temp_scaled = temp / TEMP_SCALE
        tint_scaled = tint / TINT_SCALE
        targets = torch.tensor([temp_scaled, tint_scaled], dtype=torch.float32)

        return pixel_values, targets, img_id


class EfficientNetV2Regressor(nn.Module):
    def __init__(self, model_name, num_targets, dropout_rate):
        super().__init__()

        self.effnet = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT
        )

        in_features = self.effnet.classifier[-1].in_features

        self.effnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_targets)
        )

    def forward(self, x):
        return self.effnet(x)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, cfg, scaler=None):
    model.train()
    running_loss = 0.0

    all_gt_temp = []
    all_gt_tint = []
    all_pred_temp = []
    all_pred_tint = []

    for pixel_values, targets_scaled, _ in tqdm(
            loader, desc=f"Fine-Tune Epoch {epoch}/{cfg['EPOCHS']}", leave=True
    ):
        pixel_values = pixel_values.to(device)
        targets_scaled = targets_scaled.to(device)

        optimizer.zero_grad()

        if scaler is not None and device == 'cuda':
            with torch.amp.autocast('cuda'):
                preds_scaled = model(pixel_values)
                loss = criterion(preds_scaled, targets_scaled)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds_scaled = model(pixel_values)
            loss = criterion(preds_scaled, targets_scaled)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * targets_scaled.size(0)

        preds_scaled_np = preds_scaled.detach().cpu().numpy()
        targets_scaled_np = targets_scaled.detach().cpu().numpy()

        gt_temp = targets_scaled_np[:, 0] * TEMP_SCALE
        gt_tint = targets_scaled_np[:, 1] * TINT_SCALE
        pred_temp = preds_scaled_np[:, 0] * TEMP_SCALE
        pred_tint = preds_scaled_np[:, 1] * TINT_SCALE

        all_gt_temp.append(gt_temp)
        all_gt_tint.append(gt_tint)
        all_pred_temp.append(pred_temp)
        all_pred_tint.append(pred_tint)

    all_gt_temp = np.concatenate(all_gt_temp)
    all_gt_tint = np.concatenate(all_gt_tint)
    all_pred_temp = np.concatenate(all_pred_temp)
    all_pred_tint = np.concatenate(all_pred_tint)

    temp_mae = mean_absolute_error(all_gt_temp, all_pred_temp)
    tint_mae = mean_absolute_error(all_gt_tint, all_pred_tint)
    avg_mae = (temp_mae + tint_mae) / 2.0

    epoch_loss = running_loss / len(loader)

    return epoch_loss, avg_mae, temp_mae, tint_mae


def validate_one_epoch(model, loader, criterion, device, epoch, cfg):
    model.eval()
    running_loss = 0.0

    all_gt_temp = []
    all_gt_tint = []
    all_pred_temp = []
    all_pred_tint = []

    with torch.no_grad():
        for pixel_values, targets_scaled, _ in tqdm(
                loader, desc=f"Valid Epoch {epoch}/{cfg['EPOCHS']}", leave=True
        ):
            pixel_values = pixel_values.to(device)
            targets_scaled = targets_scaled.to(device)

            if device == 'cuda':
                with torch.amp.autocast('cuda'):
                    preds_scaled = model(pixel_values)
                    loss = criterion(preds_scaled, targets_scaled)
            else:
                preds_scaled = model(pixel_values)
                loss = criterion(preds_scaled, targets_scaled)

            running_loss += loss.item() * targets_scaled.size(0)

            preds_scaled_np = preds_scaled.detach().cpu().numpy()
            targets_scaled_np = targets_scaled.detach().cpu().numpy()

            gt_temp = targets_scaled_np[:, 0] * TEMP_SCALE
            gt_tint = targets_scaled_np[:, 1] * TINT_SCALE
            pred_temp = preds_scaled_np[:, 0] * TEMP_SCALE
            pred_tint = preds_scaled_np[:, 1] * TINT_SCALE

            all_gt_temp.append(gt_temp)
            all_gt_tint.append(gt_tint)
            all_pred_temp.append(pred_temp)
            all_pred_tint.append(pred_tint)

    epoch_loss = running_loss / len(loader)

    all_gt_temp = np.concatenate(all_gt_temp)
    all_gt_tint = np.concatenate(all_gt_tint)
    all_pred_temp = np.concatenate(all_pred_temp)
    all_pred_tint = np.concatenate(all_pred_tint)

    temp_mae = mean_absolute_error(all_gt_temp, all_pred_temp)
    tint_mae = mean_absolute_error(all_gt_tint, all_pred_tint)
    avg_mae = (temp_mae + tint_mae) / 2.0

    return epoch_loss, avg_mae, temp_mae, tint_mae


def save_predictions_csv(model, dataset, device, cfg):
    loader = DataLoader(
        dataset,
        batch_size=cfg['BATCH_SIZE'],
        shuffle=False,
        num_workers=cfg['NUM_WORKERS'],
        pin_memory=(device == 'cuda')
    )

    model.eval()
    all_ids = []
    all_temp = []
    all_tint = []
    all_pred_temp_raw = []
    all_pred_tint_raw = []

    with torch.no_grad():
        for pixel_values, targets_scaled, img_ids in tqdm(
                loader, desc="Generating CSV", leave=True
        ):
            pixel_values = pixel_values.to(device)

            if device == 'cuda':
                with torch.amp.autocast('cuda'):
                    preds_scaled = model(pixel_values)
            else:
                preds_scaled = model(pixel_values)

            preds_scaled_np = preds_scaled.detach().cpu().numpy()
            targets_scaled_np = targets_scaled.detach().cpu().numpy()

            gt_temp = targets_scaled_np[:, 0] * TEMP_SCALE
            gt_tint = targets_scaled_np[:, 1] * TINT_SCALE
            pred_temp = preds_scaled_np[:, 0] * TEMP_SCALE
            pred_tint = preds_scaled_np[:, 1] * TINT_SCALE

            for i in range(len(img_ids)):
                all_ids.append(img_ids[i])
                all_temp.append(gt_temp[i])
                all_tint.append(gt_tint[i])
                all_pred_temp_raw.append(pred_temp[i])
                all_pred_tint_raw.append(pred_tint[i])

    df_out = pd.DataFrame({
        'id_global': all_ids,
        'Temperature': all_temp,
        'Tint': all_tint,
        'pred_Temperature': all_pred_temp_raw,
        'pred_Tint': all_pred_tint_raw,
    })

    df_out["pred_Temperature"] = (
        df_out["pred_Temperature"]
        .clip(lower=TEMP_MIN, upper=TEMP_MAX)
        .round()
        .astype(int)
    )
    df_out["pred_Tint"] = (
        df_out["pred_Tint"]
        .clip(lower=TINT_MIN, upper=TINT_MAX)
        .round()
        .astype(int)
    )

    df_out["Temperature"] = df_out["Temperature"].round().astype(int)
    df_out["Tint"] = df_out["Tint"].round().astype(int)

    # Save
    df_out.to_csv(cfg['PRED_CSV_PATH'], index=False)
    print(f"\nSaved predictions CSV to: {cfg['PRED_CSV_PATH']}")
    print(df_out.head())

    # Also print a clean MAE summary using the rounded predictions
    mae_temp = np.mean(np.abs(df_out["Temperature"] - df_out["pred_Temperature"]))
    mae_tint = np.mean(np.abs(df_out["Tint"] - df_out["pred_Tint"]))
    mae_avg = (mae_temp + mae_tint) / 2.0

    print("\n--- Prediction Summary (rounded ints) ---")
    print(f"Temp MAE: {mae_temp:.2f}")
    print(f"Tint MAE: {mae_tint:.2f}")
    print(f"Avg MAE:  {mae_avg:.2f}")


def main():
    cfg = CONFIG
    device = cfg['DEVICE']
    print(f"Using device: {device}")
    print(f"Starting {cfg['IMG_SIZE']}x{cfg['IMG_SIZE']} Fine-Tuning with LR: {cfg['LR']:.2e}")

    # 6.1 Load CSV and split
    df = pd.read_csv(cfg['TRAIN_CSV'])
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # (using new IMG_SIZE for upscaling)
    IMG_SIZE = cfg['IMG_SIZE']
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    base_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    aug_transforms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    train_dataset = WhiteBalanceDataset(train_df, cfg['TRAIN_IMG_DIR'], processor=aug_transforms, id_col='id_global')
    val_dataset = WhiteBalanceDataset(val_df, cfg['TRAIN_IMG_DIR'], processor=base_transforms, id_col='id_global')

    train_loader = DataLoader(
        train_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=True,
        num_workers=cfg['NUM_WORKERS'], pin_memory=(device == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False,
        num_workers=cfg['NUM_WORKERS'], pin_memory=(device == 'cuda')
    )

    model = EfficientNetV2Regressor(
        cfg['MODEL_NAME'], num_targets=2, dropout_rate=cfg['DROPOUT_RATE']
    ).to(device)

    # --- CRITICAL FINE-TUNING STEP: LOAD PREVIOUS WEIGHTS ---
    if os.path.exists(cfg['PREVIOUS_BEST_MODEL_PATH']):
        try:
            model.load_state_dict(torch.load(cfg['PREVIOUS_BEST_MODEL_PATH'], map_location=device))
            print(f"✅ Successfully loaded best weights from: {cfg['PREVIOUS_BEST_MODEL_PATH']} for fine-tuning.")
        except RuntimeError as e:
            print(f"❌ Error loading state dict. Starting from ImageNet pre-trained weights. Error: {e}")
    else:
        print(
            f"⚠️ Warning: Previous model checkpoint not found at {cfg['PREVIOUS_BEST_MODEL_PATH']}. Starting from ImageNet pre-trained weights.")
    # --------------------------------------------------------

    criterion = nn.MSELoss(reduction='mean')

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['LR'],  # Starting at low LR for fine-tuning
        weight_decay=cfg['WEIGHT_DECAY']
    )

    # Scheduler will reduce the already low LR further if improvement plateaus
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=True
    )

    scaler = GradScaler() if device == 'cuda' else None

    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    best_val_mae = float('inf')

    for epoch in range(1, cfg['EPOCHS'] + 1):
        train_loss, train_mae, train_temp_mae, train_tint_mae = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg, scaler=scaler
        )

        val_loss, val_mae, val_temp_mae, val_tint_mae = validate_one_epoch(
            model, val_loader, criterion, device, epoch, cfg
        )

        scheduler.step(val_mae)

        print(
            f"Epoch {epoch}/{cfg['EPOCHS']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Valid Avg_MAE: {val_mae:.2f} "
            f"(Temp: {val_temp_mae:.2f}, Tint: {val_tint_mae:.2f}) | "
            f"Current LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), cfg['BEST_MODEL_PATH'])
            print(f"[Checkpoint] New best Avg_MAE saved: {best_val_mae:.2f}")

    print("\nTraining finished.")
    print(f"Best validation Avg_MAE achieved: {best_val_mae:.2f}")

    if os.path.exists(cfg['BEST_MODEL_PATH']):
        print(f"\nLoading best model from: {cfg['BEST_MODEL_PATH']}")
        model.load_state_dict(torch.load(cfg['BEST_MODEL_PATH'], map_location=device))
    else:
        print("\n[Warning] Best model file not found, using last-epoch weights.")

    save_predictions_csv(model, val_dataset, device, cfg)


if __name__ == "__main__":
    main()