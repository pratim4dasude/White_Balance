import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import tifffile
from PIL import Image
from torchvision import transforms, models
from tqdm.auto import tqdm

CONFIG = {

    'NEW_DATA_CSV': r'C:\Users\KIIT\PycharmProjects\aftrsht\dataset\Validation\sliders_input.csv',
    'NEW_IMG_DIR': r'C:\Users\KIIT\PycharmProjects\aftrsht\dataset\Validation\images',

    'MODEL_NAME': 'efficientnet_v2_s',
    'BEST_MODEL_PATH': 'best_efficientnet_v2_s_384_finetune_111.pth',
    'OUTPUT_PRED_CSV_PATH': 'PRED_TEST_VALL__regressor_model_s_384_finetune_111_passthrough_v2.csv',


    'BATCH_SIZE': 8,
    'NUM_WORKERS': 0,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',

    'IMG_SIZE': 256,
    'DROPOUT_RATE': 0.4,
}

TEMP_SCALE = 10000.0
TINT_SCALE = 200.0

TEMP_MIN, TEMP_MAX = 1500.0, 10000.0
TINT_MIN, TINT_MAX = -50.0, 50.0


class DatasetPrepWorker(Dataset):

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
        return None

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
        if img_path is None:
            raise FileNotFoundError(f"Image not found for ID: {img_id}")

        img = self._read_image(img_path)
        pixel_values = self.processor(img)

        try:
            # Existing logic for ground truth: uses 'currTemp' and 'currTint'
            temp = float(row["currTemp"])
            tint = float(row["currTint"])
        except KeyError:
            temp, tint = np.nan, np.nan

        temp_scaled = temp / TEMP_SCALE
        tint_scaled = tint / TINT_SCALE
        targets = torch.tensor([temp_scaled, tint_scaled], dtype=torch.float32)

        return pixel_values, targets, img_id


class EfficientNetV2Regressor(nn.Module):

    def __init__(self, model_name: str, num_targets: int, dropout_rate: float):
        super().__init__()

        # Base model
        self.effnet = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT
        )

        in_features = self.effnet.classifier[-1].in_features

        self.effnet.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_targets),
        )

    def forward(self, x):
        return self.effnet(x)


def make_predictions(model: nn.Module,
                     dataset: Dataset,
                     device: str,
                     cfg: dict,
                     df_full_input: pd.DataFrame = None) -> None:

    loader = DataLoader(
        dataset,
        batch_size=cfg['BATCH_SIZE'],
        shuffle=False,
        num_workers=cfg['NUM_WORKERS'],
        pin_memory=(device == 'cuda'),
    )

    model.eval()

    all_ids = []
    all_actual_temp = []
    all_actual_tint = []
    all_pred_temp = []
    all_pred_tint = []

    print(f"\nStarting prediction on {len(dataset)} samples...")

    with torch.inference_mode():
        for pixel_values, targets_scaled, img_ids in tqdm(loader, desc="Predicting", leave=True):
            pixel_values = pixel_values.to(device)

            if device == 'cuda':
                with torch.amp.autocast('cuda'):
                    preds_scaled = model(pixel_values)
            else:
                preds_scaled = model(pixel_values)

            preds_scaled_np = preds_scaled.detach().cpu().numpy()
            targets_scaled_np = targets_scaled.detach().cpu().numpy()

            pred_temp = preds_scaled_np[:, 0] * TEMP_SCALE
            pred_tint = preds_scaled_np[:, 1] * TINT_SCALE

            actual_temp = targets_scaled_np[:, 0] * TEMP_SCALE
            actual_tint = targets_scaled_np[:, 1] * TINT_SCALE

            for i in range(len(img_ids)):
                all_ids.append(img_ids[i])
                all_actual_temp.append(actual_temp[i])
                all_actual_tint.append(actual_tint[i])
                all_pred_temp.append(pred_temp[i])
                all_pred_tint.append(pred_tint[i])

    df_out = pd.DataFrame({
        'id_global': all_ids,
        'Actual_currTemp': all_actual_temp,
        'Actual_currTint': all_actual_tint,
        'pred_Temperature': all_pred_temp,
        'pred_Tint': all_pred_tint,
    })

    if df_full_input is not None:
        passthrough_cols = []
        if 'Temperature' in df_full_input.columns:
            passthrough_cols.append('Temperature')
        if 'Tint' in df_full_input.columns:
            passthrough_cols.append('Tint')

        if passthrough_cols:

            df_out = df_out.merge(df_full_input[['id_global'] + passthrough_cols], on='id_global', how='left')
    # --------------------------------------------------------------------------------------

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

    df_out["Actual_currTemp"] = df_out["Actual_currTemp"].round()
    df_out["Actual_currTint"] = df_out["Actual_currTint"].round()

    df_out.to_csv(cfg['OUTPUT_PRED_CSV_PATH'], index=False)
    print(f"\n✅ Predictions saved successfully to: {cfg['OUTPUT_PRED_CSV_PATH']}")

    print("\n--- Sample Predictions ---")
    print(df_out.head())

    print("\n--- Prediction Summary ---")
    if not df_out['Actual_currTemp'].isnull().all():
        mae_temp = np.nanmean(np.abs(df_out['Actual_currTemp'] - df_out['pred_Temperature']))
        mae_tint = np.nanmean(np.abs(df_out['Actual_currTint'] - df_out['pred_Tint']))
        mae_avg = (mae_temp + mae_tint) / 2.0

        print(f"Calculated Avg MAE: {mae_avg:.2f}")
        print(f"Calculated Temp MAE: {mae_temp:.2f}")
        print(f"Calculated Tint MAE: {mae_tint:.2f}")
    else:
        print("Ground-truth currTemp/currTint not available; MAE not computed.")


def build_transforms(img_size: int):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])


def main():
    cfg = CONFIG
    device = cfg['DEVICE']
    print(f"Using device: {device}")

    if not os.path.exists(cfg['BEST_MODEL_PATH']):
        print(f"\n❌ ERROR: Best model file not found at {cfg['BEST_MODEL_PATH']}")
        print("Please ensure you have completed training and the file path is correct.")
        return

    df_new_data = pd.read_csv(cfg['NEW_DATA_CSV'])
    print(f"Loaded new data CSV: {cfg['NEW_DATA_CSV']}")
    print(f"Rows: {len(df_new_data)}")

    transforms_infer = build_transforms(cfg['IMG_SIZE'])

    pred_dataset = DatasetPrepWorker(
        df=df_new_data,
        img_dir=cfg['NEW_IMG_DIR'],
        processor=transforms_infer,
        id_col='id_global',
    )

    model = EfficientNetV2Regressor(
        model_name=cfg['MODEL_NAME'],
        num_targets=2,
        dropout_rate=cfg['DROPOUT_RATE'],
    ).to(device)

    model.load_state_dict(torch.load(cfg['BEST_MODEL_PATH'], map_location=device))
    print(f"\n✅ Successfully loaded weights from: {cfg['BEST_MODEL_PATH']}")

    make_predictions(model, pred_dataset, device, cfg, df_full_input=df_new_data)


if __name__ == "__main__":
    main()