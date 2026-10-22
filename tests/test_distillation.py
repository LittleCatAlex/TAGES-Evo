"""特徵蒸餾與電測預訓練單元測試 (Unit Tests for Petrophysics Distillation Model)"""
import pytest
import torch
import numpy as np

from tages.models.distillation import PetrophysicsTeacher, DistilledStudentNet, DistillationLoss

def test_teacher_forward():
    B, T, C_in = 4, 32, 17
    teacher = PetrophysicsTeacher(in_features=C_in, cnn_channels=32, lstm_hidden=48, n_classes_l2=6, n_classes_l1=4)
    x = torch.randn(B, T, C_in)
    logits_l2, logits_l1, lstm_out = teacher(x)
    assert logits_l2.shape == (B, T, 6)
    assert logits_l1.shape == (B, T, 4)
    assert lstm_out.shape == (B, T, 96)


def test_student_forward_and_predict():
    B, T, C_in = 4, 32, 14
    student = DistilledStudentNet(in_features=C_in, cnn_channels=32, lstm_hidden=48, n_classes_l2=6, n_classes_l1=4, n_petro_targets=2)
    x = torch.randn(B, T, C_in)
    out = student(x)
    assert out["logits_l2"].shape == (B, T, 6)
    assert out["logits_l1"].shape == (B, T, 4)
    assert out["pred_petro"].shape == (B, T, 2)

    # 測試 predict 介面
    mask = torch.ones(B, T, dtype=torch.bool)
    res = student.predict(x, mask=mask, use_crf=True)
    assert len(res["l2_paths"]) == B
    assert len(res["l2_paths"][0]) == T
    assert res["pred_petro"].shape == (B, T, 2)


def test_distillation_loss():
    B, T, C_l2 = 2, 16, 6
    loss_fn = DistillationLoss(temperature=2.0)

    s_logits = torch.randn(B, T, C_l2, requires_grad=True)
    t_logits = torch.randn(B, T, C_l2)
    s_petro = torch.randn(B, T, 2, requires_grad=True)
    t_petro = torch.randn(B, T, 2)
    targets = torch.randint(0, C_l2, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    petro_mask = torch.zeros(B, T, dtype=torch.bool)
    petro_mask[0, :] = True  # 第 1 筆樣本具有電測

    loss_dict = loss_fn(
        student_logits_l2=s_logits,
        teacher_logits_l2=t_logits,
        student_pred_petro=s_petro,
        true_petro=t_petro,
        targets_l2=targets,
        mask=mask,
        petro_mask=petro_mask
    )

    assert "loss" in loss_dict
    assert loss_dict["loss"].item() > 0.0
    assert loss_dict["loss_kd"].item() >= 0.0
    assert loss_dict["loss_petro"].item() >= 0.0

    # 反向傳播檢驗
    loss_dict["loss"].backward()
    assert s_logits.grad is not None
    assert s_petro.grad is not None
