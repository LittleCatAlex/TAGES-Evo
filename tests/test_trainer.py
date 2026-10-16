"""Trainer, Loss 與 Checkpointing 單元測試 (Unit Tests for Trainer & Losses)。"""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytest

from tages.models.losses import MaskedFocalLoss, HierarchicalSequenceLoss
from tages.models.well_seq_net import WellboreSeqNet
from tages.models.trainer import WellboreSeqTrainer
from tages.features.sequence_features import SequenceFeatureExtractor
from tages.models.dataset import WellboreSequenceDataset, collate_well_sequences

def test_masked_focal_loss():
    loss_fn = MaskedFocalLoss(gamma=2.0)
    logits = torch.randn(2, 10, 6, requires_grad=True)
    targets = torch.randint(0, 6, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, 8:] = False
    
    loss = loss_fn(logits, targets, mask=mask)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None

def test_trainer_train_and_save_load(tmp_path):
    # 建立小型合成訓練井資料
    records = []
    for st_id in ["W1", "W2", "W3", "W4"]:
        for d in np.linspace(1.0, 30.0, 30):
            records.append({
                "station_id": st_id,
                "depth_m": d,
                "z_msl": 10.0 - d,
                "vp_ms": 1700.0 + d * 5.0,
                "vs_ms": 220.0 + d * 2.0,
                "vp_vs_ratio": 2.2,
                "dvs_dz": 0.02,
                "poisson_ratio": 0.28,
                "x_twd97": 190000.0,
                "y_twd97": 2570000.0,
                "lithology_id": 2 if d < 10 else (4 if d < 20 else 5),
                "lithology_l1_id": 2 if d < 10 else 3
            })
    df = pd.DataFrame(records)
    
    extractor = SequenceFeatureExtractor()
    extractor.fit(df)
    
    dataset = WellboreSequenceDataset(df, extractor, l2_classes=[2, 3, 4, 5, 6, 7])
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_well_sequences)
    
    model = WellboreSeqNet(
        in_features=14,
        cnn_channels=32,
        lstm_hidden=32,
        lstm_layers=1,
        n_classes_l2=6,
        n_classes_l1=4
    )
    
    trainer = WellboreSeqTrainer(
        model=model,
        extractor=extractor,
        l2_classes=[2, 3, 4, 5, 6, 7],
        lr=0.01
    )
    
    # 執行 2 個 Epoch
    res1 = trainer.train_epoch(loader)
    res2 = trainer.train_epoch(loader)
    assert res1["loss"] > 0.0
    
    eval_res = trainer.evaluate(loader)
    assert 0.0 <= eval_res["accuracy_l2"] <= 1.0
    assert 0.0 <= eval_res["macro_f1_l2"] <= 1.0
    assert eval_res["isolated_noise_count"] >= 0
    
    # 測試 Checkpointing
    ckpt_file = tmp_path / "test_seq_model.pt"
    trainer.save_checkpoint(ckpt_file)
    assert ckpt_file.exists()
    
    loaded_model, loaded_ext, ckpt = WellboreSeqTrainer.load_checkpoint(ckpt_file)
    assert loaded_ext.fitted
    assert len(ckpt["l2_classes"]) == 6
    
    # 測試載入模型推論
    sample_batch = next(iter(loader))
    pred = loaded_model.predict(sample_batch["x"], mask=sample_batch["mask"])
    assert len(pred["l2_paths"]) == len(sample_batch["station_ids"])
