"""WellboreSeqNet 與 LinearChainCRF 單元測試 (Unit Tests for WellboreSeqNet & CRF)。"""
import numpy as np
import torch
import pytest

from tages.models.crf import LinearChainCRF
from tages.models.well_seq_net import WellboreSeqNet, MultiScaleConv1DBlock

def test_linear_chain_crf():
    num_tags = 6
    crf = LinearChainCRF(num_tags=num_tags)
    
    # 測試以實證轉移矩陣熱啟動
    synth_trans = np.eye(num_tags) * 0.6 + 0.4 / num_tags
    crf.init_from_empirical_matrix(synth_trans)
    
    batch_size = 3
    seq_len = 25
    emissions = torch.randn(batch_size, seq_len, num_tags, requires_grad=True)
    tags = torch.randint(0, num_tags, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 20:] = False  # 第 0 筆長度 20
    mask[1, 15:] = False  # 第 1 筆長度 15
    
    # 前向計算 NLL Loss
    loss = crf.forward_loss(emissions, tags, mask=mask)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    
    # 測試反向傳播
    loss.backward()
    assert emissions.grad is not None
    assert crf.transitions.grad is not None
    
    # 測試維特比解碼
    paths = crf.decode(emissions.detach(), mask=mask)
    assert len(paths) == batch_size
    assert len(paths[0]) == 20
    assert len(paths[1]) == 15
    assert len(paths[2]) == 25
    for p in paths:
        assert all(0 <= tag < num_tags for tag in p)

def test_multiscale_conv1d_block():
    block = MultiScaleConv1DBlock(in_channels=14, out_channels=96)
    x = torch.randn(2, 14, 50)
    out = block(x)
    assert out.shape == (2, 96, 50)

def test_wellbore_seq_net_forward_and_predict():
    net = WellboreSeqNet(
        in_features=14,
        cnn_channels=48,
        lstm_hidden=64,
        lstm_layers=1,
        n_classes_l2=6,
        n_classes_l1=4
    )
    
    batch_size = 2
    seq_len = 30
    x = torch.randn(batch_size, seq_len, 14)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 22:] = False
    
    # 測試 forward
    out = net(x)
    assert "emissions_l2" in out and "logits_l1" in out
    assert out["emissions_l2"].shape == (batch_size, seq_len, 6)
    assert out["logits_l1"].shape == (batch_size, seq_len, 4)
    
    # 測試 predict
    res = net.predict(x, mask=mask)
    assert len(res["l2_paths"]) == batch_size
    assert len(res["l2_paths"][0]) == 22
    assert len(res["l2_paths"][1]) == 30
    assert res["l1_preds"].shape == (batch_size, seq_len)
    assert res["l2_probs"].shape == (batch_size, seq_len, 6)
    assert res["l2_conf"].shape == (batch_size, seq_len)
    assert np.all(res["l2_probs"] >= 0.0)
    assert np.allclose(np.sum(res["l2_probs"], axis=-1), 1.0, atol=1e-5)
