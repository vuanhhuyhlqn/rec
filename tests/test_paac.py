"""Tests for Task 9: PAAC L_sa and L_cl losses + integration."""

import torch
from torch.utils.data import DataLoader

from newsrec.data.collate import stack_collate
from newsrec.data.finetune_dataset import FinetuneTripletDataset
from newsrec.data.news_tokens import NewsTokenTable
from newsrec.data.popularity import ItemPopularity
from newsrec.losses.paac_losses import (
    augment_views,
    reweighting_contrastive_loss,
    supervised_alignment_loss,
)
from newsrec.models.rec_model import build_rec_model
from newsrec.training.finetuner import Finetuner


# --------------------------------------------------------------------------- #
# L_sa                                                                        #
# --------------------------------------------------------------------------- #
def test_l_sa_zero_when_pop_equals_unpop():
    B, S, D = 2, 4, 8
    vecs = torch.randn(B, S, D)
    # make all history vectors identical → pop/unpop distance is 0
    vecs[:] = vecs[:, :1, :]
    mask = torch.ones(B, S)
    pop = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    loss = supervised_alignment_loss(vecs, mask, pop)
    assert loss.shape == ()
    assert torch.allclose(loss, torch.zeros(()), atol=1e-6)


def test_l_sa_positive_when_groups_differ():
    B, S, D = 1, 4, 8
    vecs = torch.zeros(B, S, D)
    # popular items (high pop) at e0, unpopular at e1 → non-zero distance
    vecs[0, 0, 0] = 1.0
    vecs[0, 1, 0] = 1.0
    vecs[0, 2, 1] = 1.0
    vecs[0, 3, 1] = 1.0
    mask = torch.ones(B, S)
    pop = torch.tensor([[10.0, 9.0, 1.0, 0.0]])
    loss = supervised_alignment_loss(vecs, mask, pop, ratio=0.5)
    assert loss.item() > 0


def test_l_sa_respects_mask():
    B, S, D = 1, 5, 8
    vecs = torch.randn(B, S, D)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])  # only 2 valid -> 1 pop 1 unpop
    pop = torch.tensor([[5.0, 1.0, 9.0, 9.0, 9.0]])
    loss = supervised_alignment_loss(vecs, mask, pop)
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# Augmentation + L_cl                                                         #
# --------------------------------------------------------------------------- #
def test_augment_views_shapes_and_differ():
    h = torch.randn(6, 16)
    v1, v2 = augment_views(h, dropout_p=0.2, noise_std=0.1)
    assert v1.shape == h.shape and v2.shape == h.shape
    assert not torch.allclose(v1, v2)


def test_l_cl_finite_and_grouping():
    torch.manual_seed(0)
    N, D = 8, 16
    items = torch.randn(N, D)
    pop = torch.arange(N, dtype=torch.float)  # 0..7
    loss = reweighting_contrastive_loss(items, pop, x_percent=50.0, beta=1.0, gamma=0.5)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_l_cl_lower_when_views_agree():
    torch.manual_seed(0)
    N, D = 8, 16
    items = torch.randn(N, D)
    pop = torch.arange(N, dtype=torch.float)
    # No augmentation noise/dropout → the two views are identical → positives
    # dominate → lower loss than the heavily-augmented case.
    low = reweighting_contrastive_loss(items, pop, dropout_p=0.0, noise_std=0.0, tau=0.1)
    high = reweighting_contrastive_loss(items, pop, dropout_p=0.5, noise_std=1.0, tau=0.1)
    assert low.item() < high.item()


def test_l_cl_gamma_weighting():
    torch.manual_seed(1)
    N, D = 8, 16
    items = torch.randn(N, D)
    pop = torch.arange(N, dtype=torch.float)
    only_pop = reweighting_contrastive_loss(items, pop, gamma=1.0, dropout_p=0.0, noise_std=0.0)
    only_unpop = reweighting_contrastive_loss(items, pop, gamma=0.0, dropout_p=0.0, noise_std=0.0)
    mixed = reweighting_contrastive_loss(items, pop, gamma=0.5, dropout_p=0.0, noise_std=0.0)
    # gamma=0.5 result lies between the two extremes (uses same RNG seed path)
    lo, hi = sorted([only_pop.item(), only_unpop.item()])
    assert lo - 1e-4 <= mixed.item() <= hi + 1e-4


def test_l_cl_centering_breaks_anisotropy_degeneracy():
    """Anisotropic items pin L_cl at ~log(N) unless mean-centered."""
    import math

    torch.manual_seed(0)
    N, D = 32, 16
    # Strong shared direction => off-diagonal cosine ~= 1 (anisotropy).
    base = torch.randn(D)
    items = base.unsqueeze(0).repeat(N, 1) + 0.05 * torch.randn(N, D)
    pop = torch.arange(N, dtype=torch.float)
    lnN = math.log(N)

    # No augmentation so the only difference is the centering.
    raw = reweighting_contrastive_loss(
        items, pop, x_percent=80.0, dropout_p=0.0, noise_std=0.0, center=False
    )
    centered = reweighting_contrastive_loss(
        items, pop, x_percent=80.0, dropout_p=0.0, noise_std=0.0, center=True
    )
    # Without centering the loss sits at the no-information fixed point ~log(N);
    # with centering the positive separates and the loss collapses toward 0.
    assert raw.item() > 0.9 * lnN
    assert centered.item() < 0.25 * lnN


def test_l_cl_runs_under_bf16_autocast_on_cpu():
    """The autocast-disable guard must not error and must stay finite."""
    torch.manual_seed(0)
    items = torch.randn(8, 16)
    pop = torch.arange(8, dtype=torch.float)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = reweighting_contrastive_loss(items, pop, x_percent=50.0)
    assert torch.isfinite(loss) and loss.dtype == torch.float32


def test_l_cl_robust_to_norm_collapse():
    """L_cl must stay low even when embeddings collapse in norm (L_sa effect).

    With absolute augmentation noise, shrinking the embedding norm drives L_cl
    toward log(N); scale-invariant noise keeps it bounded.
    """
    import math

    torch.manual_seed(0)
    N, D = 64, 32
    base = torch.randn(D)
    items = base.unsqueeze(0).repeat(N, 1) + 0.15 * torch.randn(N, D)
    pop = torch.arange(N, dtype=torch.float)
    lnN = math.log(N)

    healthy = reweighting_contrastive_loss(items, pop, x_percent=80.0)
    collapsed = reweighting_contrastive_loss(items * 0.02, pop, x_percent=80.0)
    # Both stay well below the no-information fixed point despite the collapse.
    assert healthy.item() < 0.3 * lnN
    assert collapsed.item() < 0.3 * lnN


def test_augment_noise_scales_with_magnitude():
    """A 10x larger embedding should get ~10x larger absolute noise."""
    torch.manual_seed(0)
    small = torch.ones(4, 64)
    big = torch.ones(4, 64) * 10.0
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    sa, _ = augment_views(small, dropout_p=0.0, noise_std=0.1, generator=g1)
    ba, _ = augment_views(big, dropout_p=0.0, noise_std=0.1, generator=g2)
    small_noise = (sa - small).abs().mean()
    big_noise = (ba - big).abs().mean()
    assert torch.allclose(big_noise, small_noise * 10.0, rtol=1e-4)
    """The generator arg must actually seed the augmentation (was a dead param)."""
    h = torch.randn(6, 16)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a1, b1 = augment_views(h, dropout_p=0.2, noise_std=0.1, generator=g1)
    a2, b2 = augment_views(h, dropout_p=0.2, noise_std=0.1, generator=g2)
    assert torch.allclose(a1, a2) and torch.allclose(b1, b2)


# --------------------------------------------------------------------------- #
# Integration                                                                 #
# --------------------------------------------------------------------------- #
class _Impr:
    def __init__(self, history, candidates):
        self.history = history
        self.candidates = candidates

    @property
    def clicked(self):
        return [n for n, l in self.candidates if l == 1]

    @property
    def non_clicked(self):
        return [n for n, l in self.candidates if l == 0]


def _setup():
    max_len = 8
    tokens = {f"N{i}": {"input_ids": [1 + i % 4] * max_len,
                        "attention_mask": [1] * max_len} for i in range(12)}
    table = NewsTokenTable(tokens, max_len)
    imprs = [
        _Impr(["N0", "N1", "N2", "N3"], [("N4", 1), ("N5", 0), ("N6", 0)]),
        _Impr(["N7", "N8"], [("N9", 1), ("N10", 0)]),
    ]
    pop = ItemPopularity({f"N{i}": (12 - i) for i in range(12)})
    cfg = {
        "plm": {"pretrained": False, "use_lora": True, "lora_r": 4,
                "small_config": dict(hidden_size=32, num_hidden_layers=1,
                                     num_attention_heads=4, intermediate_size=64,
                                     max_position_embeddings=16, vocab_size=50)},
        "model_dim": 32,
        "news_encoder": {"num_layers": 1, "num_heads": 4, "dropout": 0.0},
        "user_encoder": {"num_layers": 1, "num_heads": 4, "dropout": 0.0},
        "max_title_len": 8, "max_history_len": 5,
    }
    model = build_rec_model(cfg)
    ds = FinetuneTripletDataset(imprs, table, max_history=5, negatives_per_pos=2,
                                popularity=pop)
    return model, ds


def test_finetuner_paac_loss_breakdown():
    torch.manual_seed(0)
    model, ds = _setup()
    loader = DataLoader(ds, batch_size=4, collate_fn=stack_collate)
    tuner = Finetuner(model, config={"lr": 1e-2, "lambda1": 0.1, "lambda2": 0.1,
                                     "lambda3": 1e-4})
    batch = next(iter(loader))
    losses = tuner.compute_losses(batch)
    for key in ("L_rec", "L_sa", "L_cl", "L_reg", "total"):
        assert key in losses
        assert torch.isfinite(losses[key])
    # total should reflect the weighted sum
    expected = (losses["L_rec"] + 0.1 * losses["L_sa"] + 0.1 * losses["L_cl"]
                + 1e-4 * losses["L_reg"])
    assert torch.allclose(losses["total"], expected, atol=1e-5)


def test_finetuner_paac_train_step_runs():
    torch.manual_seed(0)
    model, ds = _setup()
    loader = DataLoader(ds, batch_size=4, collate_fn=stack_collate)
    tuner = Finetuner(model, config={"lr": 1e-2, "lambda1": 0.1, "lambda2": 0.1})
    batch = next(iter(loader))
    out = tuner.train_step(batch)
    assert out["total"] == out["total"]  # not NaN
    assert "L_sa" in out and "L_cl" in out
