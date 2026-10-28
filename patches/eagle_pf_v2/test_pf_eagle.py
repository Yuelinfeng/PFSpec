"""Tests for real causal blocks, phase recovery, shared readout and probability laws."""
import unittest

import torch

from model.gpt.nanochat_gpt import GPT, GPTConfig
from speculative.pf_draft import DraftConfig, PFDraft, TokenSpec
from speculative.pf_engine import accept, decode, policy, target_cache


def tiny_core():
    torch.manual_seed(17)
    cfg = GPTConfig(sequence_len=64, vocab_size=14, n_layer=2, n_head=4, n_kv_head=4,
                    n_embd=32, rope_basis="position", sparse_tokens_per_latent=2,
                    sparse_position_vocab_size=8, sparse_feature_vocab_size=4,
                    sparse_feature_token_offset=8, sparse_position_hard_constraints=True,
                    value_embed_every_n_layers=None)
    core = GPT(cfg)
    core.init_weights()
    # Nontrivial logits make acceptance and readout tests sensitive to indexing.
    torch.nn.init.normal_(core.lm_head.weight, std=0.15)
    return core.eval().requires_grad_(False)


class ToyRanker:
    def initialize(self, prefix):
        self.state = 0

    def propose(self, state, previous, eos):
        # Deliberately choose a distant position, exercising P recovery.
        return (7 if previous is None or previous < 7 else eos), state, state

    def committed(self, pending, token, eos):
        return pending

    def after_target_position(self, prefix):
        self.state = 0


class PFEagleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.core = tiny_core()
        self.tokens = torch.tensor([[12, 0, 8, 2, 9, 5, 11]])
        self.coords = self.core._build_sparse_position_indices(self.tokens)

    def draft(self, mode):
        return PFDraft(self.core, DraftConfig(mode=mode, width=32, layers=1, heads=4)).eval()

    def test_hidden_api_and_target_block_cache(self):
        with torch.no_grad():
            plain = self.core(self.tokens)
            logits, hidden = self.core(self.tokens, return_hidden=True)
            torch.testing.assert_close(plain, logits, atol=0, rtol=0)
            reconstructed = 15*torch.tanh(self.core.lm_head(hidden).float()/15)
            legal = torch.isfinite(logits)
            torch.testing.assert_close(reconstructed[..., :14][legal], logits[legal])
            cache = target_cache(self.core, 64)
            parts = [self.core(self.tokens[:, :3], kv_cache=cache, return_hidden=True)[1],
                     self.core(self.tokens[:, 3:], kv_cache=cache, return_hidden=True)[1]]
            torch.testing.assert_close(torch.cat(parts, 1), hidden, atol=2e-6, rtol=2e-5)

    def test_draft_incremental_and_causality(self):
        draft = self.draft("joint")
        torch.nn.init.normal_(draft.hidden_out.weight, std=0.1)
        with torch.no_grad():
            _, h = self.core(self.tokens, return_hidden=True)
            full = draft(h[:, :-1], self.tokens[:, 1:], self.coords[:, 1:])
            cache = draft.new_cache(64)
            pieces = [draft(h[:, i:i+1], self.tokens[:, i+1:i+2], self.coords[:, i+1:i+2], cache)
                      for i in range(6)]
            torch.testing.assert_close(torch.cat(pieces, 1), full, atol=2e-6, rtol=2e-5)
            changed = h[:, :-1].clone()
            changed[:, -1] += 10
            altered = draft(changed, self.tokens[:, 1:], self.coords[:, 1:])
            torch.testing.assert_close(full[:, :-1], altered[:, :-1], atol=0, rtol=0)
        self.assertEqual(cache.get_pos(), 6)
        self.assertTrue(torch.equal(self.coords[:, 1, :3], self.coords[:, 2, :3]))
        self.assertEqual(self.coords[0, 1, 3].item(), 0)
        self.assertEqual(self.coords[0, 2, 3].item(), 1)

    def test_training_modes_and_no_target_gradients(self):
        with torch.no_grad():
            _, h = self.core(self.tokens, return_hidden=True)
        for mode in ("joint", "feature"):
            draft = self.draft(mode)
            values = draft.losses(h[:, :-2], self.tokens[:, 1:-1], h[:, 1:-1],
                                  self.coords[:, 1:-1], self.tokens[:, 2:],
                                  self.core.sparse_position_rank_map)
            self.assertEqual("ce_P" in values, mode == "joint")
            self.assertTrue(torch.isfinite(values["loss"]))
            values["loss"].backward()
            self.assertGreater(draft.hidden_out.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in self.core.parameters()))
            if mode == "feature":
                with self.assertRaises(ValueError):
                    draft.logits(h[0, 0], "P")

    def test_greedy_matches_serial_for_all_windows_and_bos(self):
        for prompt in ([12], [12, 0, 8], [12, 0, 8, 2, 9]):
            expected, suffix = decode(self.core, None, prompt, 14, mode="target", temperature=0)
            for mode in ("feature", "joint"):
                for gamma in (1, 2, 3, 4, 5):
                    stats, output = decode(self.core, self.draft(mode), prompt, 14, mode=mode,
                                           gamma=gamma, ranker=ToyRanker(), temperature=0, trace=True)
                    self.assertEqual(output, suffix, (prompt, mode, gamma))
                    if mode == "feature" and gamma == 1:
                        self.assertEqual(stats["draft_calls"], 0)

    def test_sampling_branches_budgets_and_profile_replay(self):
        rejects = set()
        for mode in ("joint", "feature"):
            draft = self.draft(mode)
            for gamma in (1, 3, 4, 5):
                for seed in range(4):
                    kwargs = dict(mode=mode, gamma=gamma, seed=seed, ranker=ToyRanker(), trace=True)
                    stats, output = decode(self.core, draft, [12, 0, 8], 11, **kwargs)
                    replay, again = decode(self.core, draft, [12, 0, 8], 11, profile=True, **kwargs)
                    self.assertEqual(output, again)
                    self.assertLessEqual(len(output), 11)
                    self.assertEqual(stats["target_calls"], replay["target_calls"])
                    for field in ("proposed", "verified", "accepted"):
                        self.assertEqual(sum(r[field] for r in stats["feature_by_depth"].values()), stats["feature_"+field])
                    self.assertEqual(sum(stats["reject_by_slot"].values()), stats["reject_position"]+stats["reject_feature"])
                    for r in stats["trace"]:
                        if r["rejection"]:
                            rejects.add(r["rejection"])
                        self.assertIn(r["after"]-r["consumed_after"], (0, 1))
                        if gamma % 2 and r["bonus_kind"]:
                            self.assertEqual(r["bonus_kind"], "feature")
        self.assertEqual(rejects, {"position", "feature"})

    def test_persistent_draft_cache_matches_full_history_replay(self):
        for gamma in (3, 5):
            draft = self.draft("feature")
            for seed in (21, 22, 23):
                args = dict(mode="feature", gamma=gamma, seed=seed, ranker=ToyRanker(), trace=True)
                full_stats, full_output = decode(self.core, draft, [12, 0, 8, 2, 9], 18, **args)
                cache_stats, cache_output = decode(self.core, draft, [12, 0, 8, 2, 9], 18,
                                                   reuse_draft_cache=True, **args)
                self.assertEqual(cache_output, full_output, (gamma, seed))
                for key in ("target_calls", "verify_calls", "recovery_calls",
                            "position_accepted", "feature_accepted", "position_verified",
                            "feature_verified", "bonus_feature", "reject_feature"):
                    self.assertEqual(cache_stats[key], full_stats[key], (gamma, seed, key))
                self.assertGreater(cache_stats["draft_commit_tokens"], 0)

    def test_cached_commit_coordinates_match_full_context(self):
        prefix = [12, 0, 8, 2, 9, 5, 11]
        for from_pos, to_pos in ((2, 4), (3, 5), (4, 6)):
            tokens_per_latent = self.core.sparse_tokens_per_latent
            first_position = 1+(from_pos//tokens_per_latent)*tokens_per_latent
            suffix = [12]+prefix[first_position:to_pos+1]
            full_coords = self.core._build_sparse_position_indices(torch.tensor([prefix]))
            suffix_coords = self.core._build_sparse_position_indices(torch.tensor([suffix]))
            local_start = 1+from_pos+1-first_position
            local_stop = 1+to_pos+1-first_position
            torch.testing.assert_close(full_coords[:, from_pos+1:to_pos+1],
                                       suffix_coords[:, local_start:local_stop])

    def test_actual_acceptance_function_restores_distribution(self):
        p, q = torch.tensor([0.65, 0.25, 0.1]), torch.tensor([0.1, 0.1, 0.8])
        rng = torch.Generator().manual_seed(923)
        counts = torch.zeros(3)
        for _ in range(5000):
            token = int(torch.multinomial(q, 1, generator=rng))
            _, actual = accept(p, q, token, rng)
            counts[actual] += 1
        self.assertLess((counts/5000-p).abs().max().item(), 0.025)
        delta = torch.tensor([0., 0., 1.])
        for _ in range(100):
            _, actual = accept(p, delta, 2, rng)
            self.assertIn(actual, (0, 1, 2))

    def test_feature_distillation_dense_kl_and_no_false_labels(self):
        from speculative.pf_distill import distribution_terms
        teacher = torch.tensor([[1., -2., 3., 0.]])
        scores = teacher.clone().requires_grad_()
        values = distribution_terms(scores, teacher, torch.tensor([-1]))
        torch.testing.assert_close(values["kl"], torch.tensor(0.), atol=1e-7, rtol=0)
        torch.testing.assert_close(values["overlap"], torch.tensor(1.))
        self.assertEqual(float(values["ce"].detach()), 0.)
        changed = torch.tensor([[0., 3., -2., 1.]], requires_grad=True)
        loss = distribution_terms(changed, teacher)["kl"]
        loss.backward()
        self.assertGreater(changed.grad.abs().sum().item(), 0)

    def test_recurrent_training_matches_consumed_and_pending_inference(self):
        from speculative.pf_distill import recurrent_predictions, conditioned_loss
        draft = self.draft("feature")
        torch.nn.init.normal_(draft.hidden_out.weight, std=.05)
        prefix, candidates = self.tokens[:, :3], self.tokens[:, 3:7]
        with torch.no_grad():
            _, hidden = self.core(self.tokens, return_hidden=True)
        row = dict(prefix=prefix, candidates=candidates, history=hidden[:, :3], target=hidden[:, 3:7],
                   coordinates=self.coords, labels=torch.tensor([[-1, -1]]))
        for pending in (False, True):
            row["pending"] = pending
            with torch.no_grad():
                cache = draft.new_cache(64)
                consumed = 3-int(pending)
                draft(hidden[:, :consumed-1], prefix[:, 1:consumed], self.coords[:, 1:consumed], cache)
                state = hidden[:, consumed-1:consumed]
                if pending:
                    state = draft(state, prefix[:, -1:], self.coords[:, 2:3], cache)
                outputs = []
                for j in range(4):
                    state = draft(state, candidates[:, j:j+1], self.coords[:, 3+j:4+j], cache)
                    outputs.append(state)
                actual = recurrent_predictions(draft, row)
                torch.testing.assert_close(actual, torch.cat(outputs, 1), atol=2e-6, rtol=2e-5)
            draft.zero_grad(set_to_none=True)
            values = conditioned_loss(self.core, draft, row)
            torch.testing.assert_close(values["weighted_kl"], values["kl"])
            weighted = conditioned_loss(self.core, draft, row, f2_weight=2.)
            torch.testing.assert_close(weighted["weighted_kl"], (weighted["kl_F1"]+2*weighted["kl_F2"])/3)
            values["loss"].backward()
            self.assertGreater(draft.hidden_out.weight.grad.abs().sum().item(), 0)
            self.assertEqual(float(values["ce"].detach()), 0)
            self.assertTrue(all(p.grad is None for p in self.core.parameters()))

    def test_old_readout_input_and_feature_teacher_alignment(self):
        from speculative.pf_distill import teacher_path, feature_scores
        from model.gpt.nanochat_gpt import norm
        row, _, logits = teacher_path(self.core, self.tokens[0, :3].tolist(), self.tokens[0, 3:5].tolist(), raw_history=True)
        torch.testing.assert_close(norm(row["history_raw"]), row["history"], atol=0, rtol=0)
        teacher = feature_scores(self.core, row["target"][0, 0])
        torch.testing.assert_close(teacher, logits[0, 8:12], atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
