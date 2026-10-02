from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from breastpath_copilot.models.moe import (  # noqa: E402
    DEFAULT_MAG_ROUTING,
    ExpertProjector,
    HardMagRoutingMoE,
    MoEClassifier,
    MoEConfig,
    MoEHead,
    MoEOutput,
    SoftTopKRoutingMoE,
    StackingMoE,
    build_moe,
)


EXPERT_DIMS = {"uni": 1024, "virchow2": 2560, "phikon_v2": 1024}
SHARED_DIM = 64
BATCH = 4


def _make_embeddings(batch: int = BATCH, seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(batch, dim, generator=gen)
        for name, dim in EXPERT_DIMS.items()
    }


# --------------------------------------------------------------------------- #
# Projector
# --------------------------------------------------------------------------- #

def test_projector_stacks_to_shared_dim():
    proj = ExpertProjector(EXPERT_DIMS, shared_dim=SHARED_DIM)
    out = proj(_make_embeddings())
    assert out.shape == (BATCH, len(EXPERT_DIMS), SHARED_DIM)


def test_projector_raises_on_missing_expert():
    proj = ExpertProjector(EXPERT_DIMS, shared_dim=SHARED_DIM)
    incomplete = {"uni": torch.randn(BATCH, EXPERT_DIMS["uni"])}
    with pytest.raises(KeyError):
        proj(incomplete)


def test_projector_requires_at_least_one_expert():
    with pytest.raises(ValueError):
        ExpertProjector({}, shared_dim=SHARED_DIM)


# --------------------------------------------------------------------------- #
# Hard magnification routing
# --------------------------------------------------------------------------- #

def test_hard_mag_routing_shapes_and_gate_sums_to_one():
    head = build_moe(EXPERT_DIMS, MoEConfig(routing="hard_mag", shared_dim=SHARED_DIM))
    out = head(_make_embeddings(), magnifications=["40X", "100X", "200X", "400X"])
    assert out.routed_embedding.shape == (BATCH, SHARED_DIM)
    assert out.gate_weights.shape == (BATCH, len(EXPERT_DIMS))
    torch.testing.assert_close(
        out.gate_weights.sum(dim=-1), torch.ones(BATCH), atol=1e-5, rtol=1e-5
    )
    assert out.aux_loss.item() == 0.0


def test_hard_mag_routing_table_uses_domain_priors():
    head = build_moe(EXPERT_DIMS, MoEConfig(routing="hard_mag", shared_dim=SHARED_DIM))
    out = head(_make_embeddings(batch=2), magnifications=["40X", "400X"])
    # 40X favors virchow2 (index 1); 400X favors uni (index 0).
    assert out.gate_weights[0].argmax().item() == 1
    assert out.gate_weights[1].argmax().item() == 0


def test_hard_mag_routing_unknown_mag_falls_back_uniform():
    head = build_moe(EXPERT_DIMS, MoEConfig(routing="hard_mag", shared_dim=SHARED_DIM))
    out = head(_make_embeddings(batch=1), magnifications=["something_unseen"])
    expected = torch.full((len(EXPERT_DIMS),), 1.0 / len(EXPERT_DIMS))
    torch.testing.assert_close(out.gate_weights[0], expected, atol=1e-5, rtol=1e-5)


def test_hard_mag_routing_defaults_to_unknown_when_no_mag_provided():
    head = build_moe(EXPERT_DIMS, MoEConfig(routing="hard_mag", shared_dim=SHARED_DIM))
    out = head(_make_embeddings(batch=2))
    expected = torch.full((len(EXPERT_DIMS),), 1.0 / len(EXPERT_DIMS))
    torch.testing.assert_close(out.gate_weights[0], expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out.gate_weights[1], expected, atol=1e-5, rtol=1e-5)


def test_hard_mag_routing_mag_count_mismatch_raises():
    head = build_moe(EXPERT_DIMS, MoEConfig(routing="hard_mag", shared_dim=SHARED_DIM))
    with pytest.raises(ValueError):
        head(_make_embeddings(batch=2), magnifications=["40X"])


def test_default_mag_routing_covers_breakhis_levels():
    assert {"40X", "100X", "200X", "400X"} <= set(DEFAULT_MAG_ROUTING)
    for mag, weights in DEFAULT_MAG_ROUTING.items():
        assert set(weights) == set(EXPERT_DIMS), f"{mag} mismatch with default experts"


# --------------------------------------------------------------------------- #
# Soft top-k routing (the headline Slicing-and-Dicing-grounded variant)
# --------------------------------------------------------------------------- #

def test_soft_topk_default_picks_exactly_two_experts():
    head = build_moe(
        EXPERT_DIMS, MoEConfig(routing="soft_topk", shared_dim=SHARED_DIM, top_k=2)
    )
    out = head(_make_embeddings())
    nonzero_per_row = (out.gate_weights > 0).sum(dim=-1)
    assert torch.equal(nonzero_per_row, torch.full((BATCH,), 2))
    torch.testing.assert_close(
        out.gate_weights.sum(dim=-1), torch.ones(BATCH), atol=1e-5, rtol=1e-5
    )


def test_soft_topk_top1_picks_one_expert():
    head = build_moe(
        EXPERT_DIMS, MoEConfig(routing="soft_topk", shared_dim=SHARED_DIM, top_k=1)
    )
    out = head(_make_embeddings())
    nonzero_per_row = (out.gate_weights > 0).sum(dim=-1)
    assert torch.equal(nonzero_per_row, torch.full((BATCH,), 1))


def test_soft_topk_with_all_experts_collapses_to_full_ensemble():
    head = build_moe(
        EXPERT_DIMS,
        MoEConfig(routing="soft_topk", shared_dim=SHARED_DIM, top_k=len(EXPERT_DIMS)),
    )
    out = head(_make_embeddings())
    assert (out.gate_weights > 0).all().item()


def test_soft_topk_rejects_invalid_k():
    with pytest.raises(ValueError):
        build_moe(EXPERT_DIMS, MoEConfig(routing="soft_topk", shared_dim=SHARED_DIM, top_k=0))
    with pytest.raises(ValueError):
        build_moe(
            EXPERT_DIMS,
            MoEConfig(routing="soft_topk", shared_dim=SHARED_DIM, top_k=len(EXPERT_DIMS) + 1),
        )


def test_soft_topk_aux_loss_is_positive_and_differentiable():
    head = build_moe(
        EXPERT_DIMS,
        MoEConfig(routing="soft_topk", shared_dim=SHARED_DIM, top_k=2, load_balance_weight=0.05),
    )
    out = head(_make_embeddings())
    assert out.aux_loss.item() > 0
    out.aux_loss.backward()
    # Gate MLP must receive gradient from the aux loss.
    gate_grads = [p.grad for p in head.gate.parameters() if p.grad is not None]
    assert gate_grads and any(g.abs().sum() > 0 for g in gate_grads)


def test_soft_topk_load_balance_weight_zero_disables_aux_loss():
    head = build_moe(
        EXPERT_DIMS,
        MoEConfig(routing="soft_topk", shared_dim=SHARED_DIM, top_k=2, load_balance_weight=0.0),
    )
    out = head(_make_embeddings())
    assert out.aux_loss.item() == 0.0


# --------------------------------------------------------------------------- #
# Stacking baseline
# --------------------------------------------------------------------------- #

def test_stacking_has_no_aux_loss_and_shared_gate_across_batch():
    head = build_moe(EXPERT_DIMS, MoEConfig(routing="stacking", shared_dim=SHARED_DIM))
    out = head(_make_embeddings())
    assert out.aux_loss.item() == 0.0
    torch.testing.assert_close(out.gate_weights[0], out.gate_weights[1])
    torch.testing.assert_close(
        out.gate_weights.sum(dim=-1), torch.ones(BATCH), atol=1e-5, rtol=1e-5
    )


def test_stacking_per_dim_weights_have_correct_shape():
    head = build_moe(
        EXPERT_DIMS, MoEConfig(routing="stacking", shared_dim=SHARED_DIM, per_dim_weights=True)
    )
    assert isinstance(head, StackingMoE)
    assert head.weight_logits.shape == (len(EXPERT_DIMS), SHARED_DIM)


# --------------------------------------------------------------------------- #
# Cross-routing gradient flow
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("routing", ["hard_mag", "soft_topk", "stacking"])
def test_gradients_flow_to_expert_projections(routing: str):
    head = build_moe(EXPERT_DIMS, MoEConfig(routing=routing, shared_dim=SHARED_DIM))
    classifier = nn.Linear(SHARED_DIM, 5)
    embeddings = _make_embeddings()
    out = head(embeddings, magnifications=["40X", "100X", "200X", "400X"])
    logits = classifier(out.routed_embedding)
    target = torch.tensor([0, 1, 2, 3])
    loss = F.cross_entropy(logits, target) + out.aux_loss
    loss.backward()
    for name in EXPERT_DIMS:
        grad = head.projector.projectors[name].weight.grad
        assert grad is not None, f"{routing}: missing grad for {name}"
        assert grad.abs().sum().item() > 0, f"{routing}: zero grad for {name}"


# --------------------------------------------------------------------------- #
# Convenience classifier wrapper
# --------------------------------------------------------------------------- #

def test_moe_classifier_returns_logits_and_moe_output():
    clf = MoEClassifier(
        EXPERT_DIMS,
        num_classes=5,
        config=MoEConfig(routing="soft_topk", shared_dim=SHARED_DIM, top_k=2),
        dropout=0.1,
    )
    logits, out = clf(_make_embeddings(), magnifications=["40X", "100X", "200X", "400X"])
    assert logits.shape == (BATCH, 5)
    assert isinstance(out, MoEOutput)
    assert out.routed_embedding.shape == (BATCH, SHARED_DIM)


# --------------------------------------------------------------------------- #
# Registry sanity
# --------------------------------------------------------------------------- #

def test_build_moe_rejects_unknown_routing():
    with pytest.raises(ValueError):
        build_moe(EXPERT_DIMS, MoEConfig(routing="not_a_routing"))  # type: ignore[arg-type]


def test_all_routings_share_moehead_interface():
    for routing in ("hard_mag", "soft_topk", "stacking"):
        head = build_moe(EXPERT_DIMS, MoEConfig(routing=routing, shared_dim=SHARED_DIM))
        assert isinstance(head, MoEHead)
        assert isinstance(head, (HardMagRoutingMoE, SoftTopKRoutingMoE, StackingMoE))
