"""Tests for newsrec.eval.metrics and newsrec.eval.evaluator."""

import math

import torch

from newsrec.eval.evaluator import ImpressionEvaluator
from newsrec.eval.metrics import (
    auc_score,
    compute_impression_metrics,
    mrr_score,
    ndcg_score,
)
from newsrec.models.rec_model import build_rec_model


# --------------------------------------------------------------------------- #
# Metric correctness                                                          #
# --------------------------------------------------------------------------- #
def test_perfect_ranking():
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    assert auc_score(scores, labels) == 1.0
    # Official MIND MRR averages reciprocal ranks over ALL positives:
    # (1/1 + 1/2) / 2 = 0.75
    assert math.isclose(mrr_score(scores, labels), 0.75, abs_tol=1e-9)
    assert ndcg_score(scores, labels, 5) == 1.0
    assert ndcg_score(scores, labels, 10) == 1.0


def test_worst_ranking_auc():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [1, 1, 0, 0]
    assert auc_score(scores, labels) == 0.0


def test_auc_known_value():
    # 1 positive, ranked 2nd of 3 → AUC should be 0.5
    scores = [0.9, 0.5, 0.1]
    labels = [0, 1, 0]
    assert math.isclose(auc_score(scores, labels), 0.5, abs_tol=1e-9)


def test_mrr_second_position():
    scores = [0.9, 0.8, 0.1]
    labels = [0, 1, 0]
    # single positive at rank 2 → MRR = 1/2
    assert math.isclose(mrr_score(scores, labels), 0.5, abs_tol=1e-9)


def test_ndcg_known_value():
    scores = [0.9, 0.8, 0.7]
    labels = [0, 1, 0]
    # DCG = 1/log2(3); IDCG = 1/log2(2)=1 → nDCG = 1/log2(3)
    expected = (1 / math.log2(3)) / 1.0
    assert math.isclose(ndcg_score(scores, labels, 10), expected, abs_tol=1e-9)


def test_compute_skips_degenerate_auc():
    # second impression all-positive → AUC undefined, skipped from average
    scores_list = [[0.9, 0.1], [0.5, 0.6]]
    labels_list = [[1, 0], [1, 1]]
    out = compute_impression_metrics(scores_list, labels_list, metrics=["auc"])
    assert out["auc"] == 1.0  # only first impression counts


# --------------------------------------------------------------------------- #
# Evaluator                                                                   #
# --------------------------------------------------------------------------- #
class _Impr:
    def __init__(self, history, candidates):
        self.history = history
        self.candidates = candidates


def _tiny_model():
    cfg = {
        "plm": {"pretrained": False, "use_lora": False,
                "small_config": dict(hidden_size=16, num_hidden_layers=1,
                                     num_attention_heads=4, intermediate_size=32,
                                     max_position_embeddings=32, vocab_size=100)},
        "model_dim": 16,
        "news_encoder": {"num_layers": 1, "num_heads": 4, "dropout": 0.0},
        "user_encoder": {"num_layers": 1, "num_heads": 4, "dropout": 0.0},
    }
    return build_rec_model(cfg).eval()


def test_evaluator_oracle_perfect_metrics():
    """Inject vectors so clicked candidates align with the user vector.

    Uses a lightweight fake model whose user encoder mean-pools the history
    vectors and whose ``score`` is cosine similarity — enough to exercise the
    evaluator's batching + metric machinery deterministically.
    """
    import torch.nn.functional as F

    class FakeModel:
        def eval(self):
            return self

        def user_encoder(self, hist, mask):
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            z = (hist * mask.unsqueeze(-1)).sum(dim=1) / denom
            return hist, z

        def score(self, z_u, cand):
            u = F.normalize(z_u, dim=-1).unsqueeze(1)
            c = F.normalize(cand, dim=-1)
            return (u * c).sum(-1)

    dim = 16
    e0 = torch.zeros(dim); e0[0] = 1.0
    e1 = torch.zeros(dim); e1[1] = 1.0
    news_vectors = {
        "click1": e0.clone(), "click2": e0.clone(),
        "noclick1": e1.clone(), "noclick2": e1.clone(),
        "h1": e0.clone(),
    }

    evaluator = ImpressionEvaluator(FakeModel())
    imprs = [
        _Impr(["h1"], [("click1", 1), ("noclick1", 0), ("noclick2", 0)]),
        _Impr(["h1"], [("noclick1", 0), ("click2", 1)]),
    ]
    out = evaluator.evaluate(imprs, news_vectors, metrics=["auc", "mrr", "ndcg@5"])
    assert math.isclose(out["auc"], 1.0, abs_tol=1e-6)
    assert math.isclose(out["mrr"], 1.0, abs_tol=1e-6)
    assert math.isclose(out["ndcg@5"], 1.0, abs_tol=1e-6)


def test_evaluator_runs_with_real_user_encoder():
    model = _tiny_model()
    evaluator = ImpressionEvaluator(model)
    dim = model.model_dim
    torch.manual_seed(0)
    news_vectors = {f"N{i}": torch.randn(dim) for i in range(10)}
    imprs = [
        _Impr(["N0", "N1"], [("N2", 1), ("N3", 0), ("N4", 0)]),
        _Impr(["N5"], [("N6", 0), ("N7", 1), ("N8", 0)]),
    ]
    out = evaluator.evaluate(imprs, news_vectors)
    for key in ("auc", "mrr", "ndcg@5", "ndcg@10"):
        assert key in out
        assert not math.isnan(out[key])
