"""Step 6: Temporal Fusion Transformer via PyTorch Forecasting."""
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from utils import PROC, OUT, HORIZON, evaluate, save_metrics

pl.seed_everything(42)
cols = ["id", "item_id", "store_id", "state_id", "date", "sales", "sell_price", "price_ratio",
        "snap", "is_event", "dow", "month", "is_test"]
df = pd.read_parquet(PROC / "features.parquet", columns=cols).sort_values(["id", "date"]).reset_index(drop=True)

date0 = df["date"].min()
df["time_idx"] = (df["date"] - date0).dt.days          # integer clock the model needs
for c in ("item_id", "store_id", "state_id", "dow", "month"):
    df[c] = df[c].astype(str)                           # categoricals must be strings
for c in ("sales", "sell_price", "price_ratio", "snap", "is_event"):
    df[c] = df[c].astype("float32")

train_end = int(df.loc[~df["is_test"], "time_idx"].max())  # last training day
ENC = 56                                                    # look-back window (days)

# training windows: everything up to 28 days before the end of training
training = TimeSeriesDataSet(
    df[df["time_idx"] <= train_end - HORIZON],
    time_idx="time_idx", target="sales", group_ids=["id"],
    min_encoder_length=ENC, max_encoder_length=ENC,
    min_prediction_length=HORIZON, max_prediction_length=HORIZON,
    static_categoricals=["item_id", "store_id", "state_id"],
    time_varying_known_categoricals=["dow", "month"],
    time_varying_known_reals=["time_idx", "sell_price", "price_ratio", "snap", "is_event"],
    time_varying_unknown_reals=["sales"],
    target_normalizer=GroupNormalizer(groups=["id"], transformation="softplus"),
    add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
)
# validation = the last 28 training days; testing = the real held-out 28 days
validation = TimeSeriesDataSet.from_dataset(training, df[df["time_idx"] <= train_end], predict=True, stop_randomization=True)
testing = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

train_dl = training.to_dataloader(train=True, batch_size=128, num_workers=0)
val_dl = validation.to_dataloader(train=False, batch_size=256, num_workers=0)
test_dl = testing.to_dataloader(train=False, batch_size=256, num_workers=0)

tft = TemporalFusionTransformer.from_dataset(
    training, learning_rate=0.03, hidden_size=16, attention_head_size=2, dropout=0.1,
    hidden_continuous_size=8, loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]), output_size=3,
    optimizer="adam", reduce_on_plateau_patience=3,
)
print("TFT parameters:", sum(p.numel() for p in tft.parameters()))

ckpt = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=str(OUT / "tft_ckpt"), filename="tft-best")
trainer = pl.Trainer(
    max_epochs=10, accelerator="auto", gradient_clip_val=0.1, limit_train_batches=400,
    callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min"), ckpt],
)
trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
best = TemporalFusionTransformer.load_from_checkpoint(ckpt.best_model_path)

# ---- predict the 28 test days: quantiles (10%, 50%, 90%) ----
res = best.predict(test_dl, mode="quantiles", return_index=True)
out = res.output.cpu().numpy()                 # shape: (series, 28 days, 3 quantiles)
idx = res.index.reset_index(drop=True)         # which series / first forecast day
frames = []
for h in range(HORIZON):
    f = idx[["id"]].copy()
    f["date"] = date0 + pd.to_timedelta(idx["time_idx"] + h, unit="D")
    f["lo"], f["pred"], f["hi"] = out[:, h, 0], out[:, h, 1], out[:, h, 2]
    frames.append(f)
pred = pd.concat(frames, ignore_index=True)
pred[["pred", "lo", "hi"]] = pred[["pred", "lo", "hi"]].clip(lower=0)
pred.to_parquet(OUT / "pred_tft.parquet", index=False)

test = df[df["is_test"]].merge(pred[["id", "date", "pred"]], on=["id", "date"])
save_metrics(evaluate(test, df[~df["is_test"]], "TFT"))