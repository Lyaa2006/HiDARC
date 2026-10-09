import unittest
from pathlib import Path
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.model.relation_text_utils import build_task_anchor_bank


class _VisionConfig:
    image_size = 8
    patch_size = 2


class _VisionTower:
    config = _VisionConfig()


class _SpectralHarness(LlavaMetaForCausalLM):
    def __init__(self):
        self.relation_routing_config = {
            "use_spectral_image_routing": True,
            "use_spectral_role_prototype": True,
            "use_text_anchor_routing": True,
            "spectral_cutoff": 0.33,
            "spectral_low_bins": 4,
            "spectral_high_bins": 4,
            "spectral_image_weight": 0.5,
            "text_weight": 0.5,
            "history_weight": 0.15,
            "text_activation_ema_decay": 0.9,
            "routing_score_normalization": "cosine",
            "routing_score_scale": 2.0,
            "routing_temperature": 0.1,
            "routing_min_similarity": -1.0,
            "role_member_top_k": 2,
            "role_assignment_member_temperature": 0.35,
            "role_assignment_member_support_mode": "blended_excess_mass",
            "role_assignment_member_excess_alpha": 0.35,
            "routing_early_uniform_mix": 0.25,
            "routing_early_role_temperature": 0.12,
            "routing_early_task_temperature": 0.18,
            "routing_early_role_strength": 0.45,
            "routing_middle_temperature": 0.10,
            "routing_middle_role_temperature": 0.10,
            "routing_middle_role_strength": 0.30,
            "routing_middle_role_gamma": 1.15,
            "routing_middle_role_uniform_mix": 0.10,
            "routing_middle_task_uniform_mix": 0.10,
            "routing_middle_role_margin_low": 0.10,
            "routing_middle_role_margin_high": 0.30,
            "routing_middle_intra_margin_low": 0.05,
            "routing_middle_intra_margin_high": 0.20,
            "routing_late_role_temperature": 0.06,
            "routing_late_task_temperature": 0.05,
            "routing_late_role_strength": 0.15,
            "routing_role_task_floor": 0.05,
        }
        self._vision_tower = _VisionTower()
        self.max_task_slots = 4
        self.max_role_slots = 4
        self.expert_num = 4
        self.text_anchors = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 768)) for _ in range(self.max_task_slots)]
        )
        self.text_boundary = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
                for _ in range(self.max_task_slots)
            ]
        )
        self.spectral_image_boundary = nn.ParameterList(
            [
                nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
                for _ in range(self.max_task_slots)
            ]
        )
        self.spectral_image_dim = 768 * (
            self.relation_routing_config["spectral_low_bins"]
            + self.relation_routing_config["spectral_high_bins"]
        )
        self.spectral_image_anchors = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.spectral_image_dim)) for _ in range(self.max_task_slots)]
        )
        self.role_spectral_prototypes = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.spectral_image_dim)) for _ in range(self.max_role_slots)]
        )
        self.role_text_prototypes = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 768)) for _ in range(self.max_role_slots)]
        )
        self.expert_usage_prior = nn.Parameter(torch.zeros(self.max_task_slots))
        self.role_task_count = nn.Parameter(torch.zeros(self.max_role_slots))
        self.role_usage_prior = nn.Parameter(torch.zeros(self.max_role_slots))
        self.task_role_membership = nn.Parameter(
            torch.zeros(self.max_task_slots, self.max_role_slots)
        )
        self.active_role_count = nn.Parameter(torch.zeros(1))

    def get_model(self):
        return self

    def get_vision_tower(self):
        return self._vision_tower


class SpectralRoutingTest(unittest.TestCase):
    def test_task_anchor_bank_removes_global_mean_and_separates_tasks(self):
        raw_bank = torch.tensor(
            [
                [5.0, 5.0, 1.0],
                [5.0, 5.0, -1.0],
                [5.0, -5.0, 0.0],
            ],
            dtype=torch.float32,
        )

        refined = build_task_anchor_bank(
            raw_bank,
            remove_global_mean=True,
            contrast_weight=0.0,
            hard_negative_top_k=1,
            preserve_mean_weight=0.0,
        )

        raw_sim = torch.matmul(
            torch.nn.functional.normalize(raw_bank, dim=-1),
            torch.nn.functional.normalize(raw_bank, dim=-1).T,
        )
        refined_sim = torch.matmul(refined, refined.T)
        self.assertTrue(torch.isfinite(refined).all())
        self.assertLess(float(refined_sim[0, 1].item()), float(raw_sim[0, 1].item()))

    def test_task_anchor_bank_contrastive_step_reduces_nearest_negative_overlap(self):
        raw_bank = torch.tensor(
            [
                [1.0, 0.9, 0.0],
                [1.0, 0.7, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        centered = build_task_anchor_bank(
            raw_bank,
            remove_global_mean=True,
            contrast_weight=0.0,
            hard_negative_top_k=1,
            preserve_mean_weight=0.0,
        )
        contrasted = build_task_anchor_bank(
            raw_bank,
            remove_global_mean=True,
            contrast_weight=0.4,
            hard_negative_top_k=1,
            preserve_mean_weight=0.0,
        )

        centered_sim = torch.matmul(centered, centered.T)
        contrasted_sim = torch.matmul(contrasted, contrasted.T)
        self.assertLess(
            float(contrasted_sim[0, 1].item()),
            float(centered_sim[0, 1].item()),
        )

    def test_descriptor_shape_finite_and_batch_independence(self):
        harness = _SpectralHarness()
        patches = torch.randn(2, 16, 768)

        descriptors = harness._extract_image_spectral_descriptor(patches)
        first = harness._extract_image_spectral_descriptor(patches[:1])
        second = harness._extract_image_spectral_descriptor(patches[1:])

        self.assertEqual(tuple(descriptors.shape), (2, harness.spectral_image_dim))
        self.assertEqual(descriptors.dtype, torch.float32)
        self.assertTrue(torch.isfinite(descriptors).all())
        self.assertTrue(torch.allclose(descriptors[:1], first, atol=1e-5))
        self.assertTrue(torch.allclose(descriptors[1:], second, atol=1e-5))

    def test_text_activation_index_shape_finite_and_batch_independence(self):
        harness = _SpectralHarness()
        text_features = torch.randn(2, 768)

        descriptors = harness._extract_text_activation_index(text_features)
        first = harness._extract_text_activation_index(text_features[:1])
        second = harness._extract_text_activation_index(text_features[1:])

        self.assertEqual(tuple(descriptors.shape), (2, 768))
        self.assertEqual(descriptors.dtype, torch.float32)
        self.assertTrue(torch.isfinite(descriptors).all())
        self.assertTrue(torch.allclose(descriptors[:1], first, atol=1e-5))
        self.assertTrue(torch.allclose(descriptors[1:], second, atol=1e-5))

    def test_text_activation_index_ignores_global_dc_offset(self):
        harness = _SpectralHarness()
        text_features = torch.randn(2, 768)

        base = harness._extract_text_activation_index(text_features)
        shifted = harness._extract_text_activation_index(text_features + 17.0)

        self.assertTrue(torch.allclose(base, shifted, atol=1e-5))

    def test_text_activation_index_uses_ema_updates(self):
        harness = _SpectralHarness()
        harness.relation_routing_config["text_activation_ema_decay"] = 0.8
        harness.text_boundary[0].data.zero_()
        first_batch = torch.zeros(2, 768)
        first_batch[:, 0] = 1.0
        second_batch = torch.zeros(2, 768)
        second_batch[:, 1] = 1.0

        first_activation = harness._extract_text_activation_index(first_batch)
        second_activation = harness._extract_text_activation_index(second_batch)
        harness._update_running_text_activation_index(
            harness.text_anchors[0],
            harness.text_boundary[0],
            first_batch,
        )
        first_anchor = harness.text_anchors[0].detach().clone()
        harness._update_running_text_activation_index(
            harness.text_anchors[0],
            harness.text_boundary[0],
            second_batch,
        )
        second_anchor = harness.text_anchors[0].detach().clone().squeeze(0)

        expected = harness._safe_normalize(
            0.8 * first_anchor.squeeze(0)
            + 0.2 * harness._safe_normalize(second_activation.mean(dim=0))
        )
        self.assertTrue(
            torch.allclose(
                first_anchor.squeeze(0),
                harness._safe_normalize(first_activation.mean(dim=0)),
                atol=1e-5,
            )
        )
        self.assertTrue(torch.allclose(second_anchor, expected, atol=1e-5))
        self.assertAlmostEqual(
            float(harness.text_boundary[0].detach().item()),
            4.0,
            places=5,
        )

    def test_score_calibration_preserves_absolute_cosine_scale(self):
        harness = _SpectralHarness()
        scores = torch.tensor([[0.9, 0.8], [0.2, 0.1]], dtype=torch.float32)

        calibrated = harness._normalize_task_score_branch(scores)

        self.assertTrue(torch.allclose(calibrated, scores * 2.0))
        self.assertNotEqual(float(calibrated[0, 0] - calibrated[0, 1]), 0.0)
        self.assertTrue(
            torch.allclose(
                calibrated[0] - calibrated[1],
                torch.full((2,), 1.4),
            )
        )

    def test_radial_masks_cover_frequency_grid_without_overlap(self):
        harness = _SpectralHarness()
        low_bins, high_bins, low, high = harness._build_radial_frequency_masks(
            24, 24, torch.device("cpu")
        )

        self.assertEqual(tuple(low_bins.shape), (4, 24, 24))
        self.assertEqual(tuple(high_bins.shape), (4, 24, 24))
        self.assertTrue(torch.equal(low | high, torch.ones_like(low)))
        self.assertFalse(torch.any(low & high))
        self.assertTrue(torch.equal(low_bins.any(dim=0), low))
        self.assertTrue(torch.equal(high_bins.any(dim=0), high))

    def test_eval_only_counts_contiguous_experts_with_required_anchors(self):
        harness = _SpectralHarness()
        harness.text_boundary[0].data.fill_(5.0)
        harness.text_boundary[1].data.fill_(6.0)
        harness.text_boundary[2].data.fill_(7.0)
        harness.spectral_image_boundary[0].data.fill_(4.0)
        harness.spectral_image_boundary[1].data.fill_(3.0)
        harness.spectral_image_boundary[2].data.fill_(0.0)

        self.assertEqual(harness._get_available_eval_expert_count(), 2)
        self.assertTrue(harness._spectral_image_bank_available(2))
        self.assertFalse(harness._spectral_image_bank_available(3))

    def test_eval_requires_trained_text_anchor_when_text_routing_enabled(self):
        harness = _SpectralHarness()
        harness.spectral_image_boundary[0].data.fill_(4.0)
        harness.spectral_image_boundary[1].data.fill_(3.0)
        harness.text_boundary[0].data.fill_(5.0)
        harness.text_boundary[1].data.fill_(1.0)

        self.assertEqual(harness._get_available_eval_expert_count(), 1)

    def test_mix_with_uniform_preserves_secondary_mass(self):
        harness = _SpectralHarness()
        weights = torch.tensor([0.999, 0.001], dtype=torch.float32)

        mixed = harness._mix_with_uniform(weights, 0.25)

        self.assertGreater(mixed[1].item(), weights[1].item())
        self.assertLess(mixed[0].item(), weights[0].item())
        self.assertAlmostEqual(float(mixed.sum().item()), 1.0, places=6)

    def test_role_build_stage_uses_spectral_image_anchor(self):
        harness = _SpectralHarness()
        harness.spectral_image_boundary[0].data.fill_(4.0)
        harness.spectral_image_boundary[1].data.fill_(4.0)
        harness.text_boundary[0].data.fill_(5.0)
        harness.text_boundary[1].data.fill_(5.0)
        harness.spectral_image_anchors[0].data[0, 0] = 1.0
        harness.spectral_image_anchors[1].data[0, 1] = 1.0
        harness.text_anchors[0].data[0, 0] = 1.0
        harness.text_anchors[1].data[0, 0] = 1.0
        current_spec = torch.zeros(harness.spectral_image_dim)
        current_spec[0] = 1.0
        current_text = torch.zeros(768)
        current_text[0] = 1.0

        scores = harness._score_tasks([0, 1], current_spec, current_text, torch.device("cpu"))

        self.assertGreater(scores[0].item(), scores[1].item())

    def test_role_eval_stage_uses_spectral_image_prototype(self):
        harness = _SpectralHarness()
        harness.active_role_count.data[0] = 2.0
        harness.role_task_count.data[:2] = 1.0
        harness.role_spectral_prototypes[0].data[0, 0] = 1.0
        harness.role_spectral_prototypes[1].data[0, 1] = 1.0
        harness.role_text_prototypes[0].data[0, 0] = 1.0
        harness.role_text_prototypes[1].data[0, 0] = 1.0
        image_anchor = torch.zeros(harness.spectral_image_dim)
        image_anchor[0] = 1.0
        text_anchor = torch.zeros(768)
        text_anchor[0] = 1.0

        role_scores = harness._score_roles(
            active_experts=2,
            expert_logits=torch.zeros(2),
            image_anchor=image_anchor,
            text_anchor=text_anchor,
            device=torch.device("cpu"),
        )

        self.assertGreater(role_scores[0].item(), role_scores[1].item())

    def test_role_pair_calibration_maps_cosine_to_unit_interval(self):
        harness = _SpectralHarness()

        calibrated = harness._calibrate_role_pair_compatibility(
            torch.tensor([-1.0, -0.2, 0.6, 1.0], dtype=torch.float32)
        )

        self.assertTrue(torch.all(calibrated >= 0.0))
        self.assertTrue(torch.all(calibrated <= 1.0))
        self.assertTrue(
            torch.allclose(
                calibrated,
                torch.tensor([0.0, 0.4, 0.8, 1.0], dtype=torch.float32),
            )
        )

    def test_role_assignment_member_support_uses_bounded_task_probability(self):
        harness = _SpectralHarness()
        expert_logits = torch.tensor([2.5, 0.5, -0.5], dtype=torch.float32)

        support_strong = harness._compute_role_assignment_member_support(
            expert_logits,
            member_tasks=[0],
            active_task_count=3,
            mode="member_max",
            top_k=1,
            temperature=0.35,
        )
        support_weak = harness._compute_role_assignment_member_support(
            expert_logits,
            member_tasks=[1, 2],
            active_task_count=3,
            mode="member_max",
            top_k=1,
            temperature=0.35,
        )

        self.assertGreater(support_strong.item(), support_weak.item())
        self.assertGreaterEqual(support_strong.item(), 0.0)
        self.assertLessEqual(support_strong.item(), 1.0)
        self.assertGreaterEqual(support_weak.item(), 0.0)
        self.assertLessEqual(support_weak.item(), 1.0)

    def test_role_assignment_member_support_subtracts_random_role_mass(self):
        harness = _SpectralHarness()
        harness.relation_routing_config["role_assignment_member_support_mode"] = "excess_mass"
        expert_logits = torch.zeros(4, dtype=torch.float32)

        singleton_support = harness._compute_role_assignment_member_support(
            expert_logits,
            member_tasks=[0],
            active_task_count=4,
            mode="member_max",
            top_k=1,
            temperature=0.35,
        )
        pair_support = harness._compute_role_assignment_member_support(
            expert_logits,
            member_tasks=[0, 1],
            active_task_count=4,
            mode="member_max",
            top_k=1,
            temperature=0.35,
        )

        self.assertAlmostEqual(float(singleton_support.item()), 0.0, places=6)
        self.assertAlmostEqual(float(pair_support.item()), 0.0, places=6)

    def test_role_assignment_member_support_blends_base_and_excess_terms(self):
        harness = _SpectralHarness()
        harness.relation_routing_config["role_assignment_member_excess_alpha"] = 0.5
        expert_logits = torch.tensor([2.0, 0.0, -1.0, -1.0], dtype=torch.float32)

        blended = harness._compute_role_assignment_member_support(
            expert_logits,
            member_tasks=[0],
            active_task_count=4,
            mode="member_max",
            top_k=1,
            temperature=0.35,
        )
        harness.relation_routing_config["role_assignment_member_support_mode"] = "excess_mass"
        excess_only = harness._compute_role_assignment_member_support(
            expert_logits,
            member_tasks=[0],
            active_task_count=4,
            mode="member_max",
            top_k=1,
            temperature=0.35,
        )
        harness.relation_routing_config["role_assignment_member_support_mode"] = "member_prob"
        base_only = harness._compute_role_assignment_member_support(
            expert_logits,
            member_tasks=[0],
            active_task_count=4,
            mode="member_max",
            top_k=1,
            temperature=0.35,
        )

        self.assertGreaterEqual(blended.item(), min(base_only.item(), excess_only.item()))
        self.assertLessEqual(blended.item(), max(base_only.item(), excess_only.item()))

    def test_early_role_conditioned_weights_are_smoothed_by_uniform_mix(self):
        harness = _SpectralHarness()
        harness.task_role_membership.data.zero_()
        harness.task_role_membership.data[0, 0] = 1.0
        harness.task_role_membership.data[1, 1] = 1.0
        harness.task_role_membership.data[2, 1] = 1.0
        task_scores = torch.tensor([3.0, 1.0, 0.5], dtype=torch.float32)
        role_weights = torch.tensor([0.999, 0.001], dtype=torch.float32)

        weights = harness._build_role_conditioned_task_weights(
            3,
            task_scores,
            role_weights,
            role_temperature=0.12,
            task_temperature=0.18,
            role_strength=0.20,
            role_uniform_mix=0.12,
        )

        self.assertGreater(weights[1].item(), 0.0)
        self.assertGreater(weights[2].item(), 0.0)
        self.assertLess(weights[0].item(), 1.0)

    def test_sparse_late_route_with_top1_is_effectively_one_hot(self):
        harness = _SpectralHarness()
        logits = torch.tensor([0.1, 2.5, 1.7], dtype=torch.float32)

        weights = harness._build_sparse_relation_weights(logits, top_k=1)

        self.assertAlmostEqual(float(weights.sum().item()), 1.0, places=6)
        self.assertEqual(int(weights.argmax().item()), 1)
        self.assertAlmostEqual(float(weights[1].item()), 1.0, places=6)
        self.assertAlmostEqual(float(weights[0].item()), 0.0, places=6)
        self.assertAlmostEqual(float(weights[2].item()), 0.0, places=6)

    def test_middle_role_conditioned_weights_are_smoothed_by_uniform_mix(self):
        harness = _SpectralHarness()
        harness.task_role_membership.data.zero_()
        harness.task_role_membership.data[0, 0] = 1.0
        harness.task_role_membership.data[1, 1] = 1.0
        harness.task_role_membership.data[2, 1] = 1.0
        task_scores = torch.tensor([3.0, 1.5, 1.0], dtype=torch.float32)
        role_weights = torch.tensor([0.999, 0.001], dtype=torch.float32)

        weights = harness._build_role_conditioned_task_weights(
            3,
            task_scores,
            role_weights,
            role_temperature=0.10,
            task_temperature=0.10,
            role_strength=0.12,
            role_uniform_mix=0.04,
        )

        self.assertGreater(weights[1].item(), 0.0)
        self.assertGreater(weights[2].item(), 0.0)
        self.assertLess(weights[0].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
