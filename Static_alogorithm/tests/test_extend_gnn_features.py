import importlib
import importlib.util
import sys
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

STATIC_DIR = Path(__file__).resolve().parents[1]
if str(STATIC_DIR) not in sys.path:
    sys.path.insert(0, str(STATIC_DIR))

if TORCH_AVAILABLE:
    import torch  # noqa: E402
    import GA.GA as GA  # noqa: E402
    import operation_policy  # noqa: E402
    from operation_policy import (  # noqa: E402
        action_id,
        action_mask,
        apply_fast_action,
        decode_action_id,
        initial_operation_state,
    )

    EG = importlib.import_module("extend_GNN.extend_GNN")
else:
    torch = None
    GA = None
    EG = None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not available in this Python environment")
class ExtendGNNFeatureTests(unittest.TestCase):

    def setUp(self):
        # Feature-value assertions assume the uncalibrated fast model; pin
        # identity so a fitted calibration artifact does not shift them.
        operation_policy.set_calibration(None)

    def tearDown(self):
        operation_policy.set_calibration(operation_policy.load_calibration())

    def make_jobs(self):
        return [
            GA.Job(idx=0, type_="A", duration=5.0, station="station1", inbound_dock="dock1", arrival_time=0.0),
            GA.Job(idx=1, type_="B", duration=10.0, station="station2", inbound_dock="dock1", arrival_time=0.0),
            GA.Job(idx=2, type_="C", duration=15.0, station="station1", inbound_dock="dock2", arrival_time=0.0),
        ]

    def make_state(self):
        return initial_operation_state(None)

    def extract(self, jobs=None, dock_service_events=None, state=None, picked=None, completed=None, carrier=None, order=None):
        if jobs is None:
            jobs = self.make_jobs()
        if state is None:
            state = self.make_state()
        amr_positions, amr_availabilities, station_availabilities, amr_inventory = state
        return EG.extract_state_extend_gnn(
            jobs,
            picked or set(),
            completed or set(),
            carrier or {},
            amr_positions,
            amr_availabilities,
            amr_inventory,
            station_availabilities,
            order or [],
            dock_service_events,
        )

    def test_feature_tensor_shapes(self):
        jobs = self.make_jobs()
        amr_feat, job_feat, dock_feat, action_feat, job_mask, adj, inbound_idx, outbound_idx = self.extract(jobs)

        self.assertEqual(tuple(amr_feat.shape), (1, len(GA.AMR_KEYS), EG.AMR_IN_DIM))
        self.assertEqual(tuple(job_feat.shape), (1, len(jobs), EG.JOB_IN_DIM))
        self.assertEqual(tuple(dock_feat.shape), (1, 10, EG.DOCK_IN_DIM))
        self.assertEqual(tuple(action_feat.shape), (1, 2, len(GA.AMR_KEYS), len(jobs), EG.ACTION_IN_DIM))
        self.assertEqual(tuple(job_mask.shape), (1, len(jobs)))
        self.assertEqual(tuple(adj.shape), (1, len(jobs), len(jobs)))
        self.assertEqual(tuple(inbound_idx.shape), (1, len(jobs)))
        self.assertEqual(tuple(outbound_idx.shape), (1, len(jobs)))

    def test_job_and_action_feature_values(self):
        jobs = self.make_jobs()
        _, job_feat, _, action_feat, _, _, inbound_idx, outbound_idx = self.extract(jobs)

        self.assertEqual(job_feat[0, 0, 0].item(), 1.0)
        self.assertAlmostEqual(job_feat[0, 0, 1].item(), 5.0 / 25.0)
        self.assertEqual(job_feat[0, 0, 5].item(), 1.0)
        self.assertAlmostEqual(job_feat[0, 0, 8].item(), 1.0 / 5.0)
        self.assertAlmostEqual(job_feat[0, 0, 9].item(), 1.0 / 5.0)
        self.assertEqual(inbound_idx[0, 0].item(), EG.DOCK_INDEX["dock1"])
        self.assertEqual(outbound_idx[0, 0].item(), EG.DOCK_INDEX["station1"])

        pickup = action_feat[0, 0, 0, 0]
        unload = action_feat[0, 1, 0, 0]
        self.assertAlmostEqual(pickup[0].item(), 2.0 / 20.0)
        self.assertAlmostEqual(pickup[1].item(), 9.0 / 20.0)
        self.assertAlmostEqual(unload[0].item(), 7.0 / 20.0)
        self.assertEqual(unload[1].item(), 0.0)

    def test_action_feature_dock_wait_is_operation_specific(self):
        jobs = self.make_jobs()
        state = self.make_state()
        _, _, station_availabilities, _ = state
        station_availabilities["dock1"] = 50.0
        station_availabilities["station1"] = 40.0

        _, _, _, action_feat, _, _, _, _ = self.extract(jobs, state=state)

        self.assertAlmostEqual(action_feat[0, 0, 0, 0, 2].item(), 48.0 / 100.0)
        self.assertAlmostEqual(action_feat[0, 1, 0, 0, 2].item(), 33.0 / 100.0)

    def test_dock_event_features(self):
        dock_service_events = EG._empty_dock_service_events()
        dock_service_events["dock1"].append((0.0, 10.0, 0, GA.PICKUP))
        dock_service_events["dock1"].append((12.0, 20.0, 1, GA.PICKUP))

        _, _, dock_feat, _, _, _, _, _ = self.extract(dock_service_events=dock_service_events)
        dock1 = dock_feat[0, EG.DOCK_INDEX["dock1"]]

        self.assertEqual(dock1[0].item(), 1.0)
        self.assertEqual(dock1[1].item(), 0.0)
        self.assertAlmostEqual(dock1[3].item(), 1.0 / len(GA.AMR_KEYS))
        self.assertGreater(dock1[4].item(), 0.0)
        self.assertAlmostEqual(dock1[5].item(), 10.0 / 100.0)
        self.assertAlmostEqual(dock1[6].item(), 18.0 / 500.0)

    def test_operation_mask_enforces_pickup_before_unload(self):
        jobs = self.make_jobs()
        state = self.make_state()
        amr_positions, amr_availabilities, station_availabilities, amr_inventory = state
        picked = set()
        completed = set()
        carrier = {}

        initial_mask = action_mask(jobs, picked, completed, carrier, amr_inventory)
        pickup_job0_amr1 = action_id(GA.PICKUP, 0, 0, len(jobs))
        unload_job0_amr1 = action_id(GA.UNLOAD, 0, 0, len(jobs))
        self.assertFalse(initial_mask[pickup_job0_amr1])
        self.assertTrue(initial_mask[unload_job0_amr1])

        action = decode_action_id(pickup_job0_amr1, jobs)
        apply_fast_action(
            action,
            jobs,
            picked,
            completed,
            carrier,
            amr_positions,
            amr_availabilities,
            station_availabilities,
            amr_inventory,
        )
        after_pickup_mask = action_mask(jobs, picked, completed, carrier, amr_inventory)
        unload_job0_amr2 = action_id(GA.UNLOAD, 1, 0, len(jobs))

        self.assertTrue(after_pickup_mask[pickup_job0_amr1])
        self.assertFalse(after_pickup_mask[unload_job0_amr1])
        self.assertTrue(after_pickup_mask[unload_job0_amr2])

    def test_adjacency_has_sequence_arcs_only(self):
        # Sharing a dock alone must not create edges (dense cliques collapse
        # job embeddings under aggregation); arcs appear only as jobs are
        # scheduled in sequence on the same AMR or station.
        _, _, _, _, _, adj, _, _ = self.extract()
        self.assertEqual(adj.abs().sum().item(), 0.0)

        # Jobs 0 and 2 share station1; scheduling both creates a station arc.
        order = [GA.Operation(0, GA.PICKUP), GA.Operation(2, GA.PICKUP)]
        carrier = {0: GA.AMR_KEYS[0], 2: GA.AMR_KEYS[1]}
        _, _, _, _, _, adj, _, _ = self.extract(carrier=carrier, order=order)

        self.assertEqual(adj[0, 0, 2].item(), 1.0)
        self.assertEqual(adj[0, 2, 0].item(), 1.0)
        self.assertEqual(adj[0, 0, 1].item(), 0.0)
        self.assertEqual(adj[0, 1, 0].item(), 0.0)

    def test_model_forward_shape(self):
        jobs = self.make_jobs()
        tensors = self.extract(jobs)
        model = EG.ExtendSchedulerGNN(hidden_dim=32, gin_layers=2)
        (
            job_embeddings,
            amr_embeddings,
            dock_embeddings,
            action_feat,
            _job_mask,
            inbound_idx,
            outbound_idx,
        ) = EG._encode_state_tensors(model, tensors, torch.device("cpu"))
        op_mask = torch.tensor(
            [action_mask(jobs, set(), set(), {}, self.make_state()[3])],
            dtype=torch.bool,
        )
        logits = model.forward_operation_actor(
            job_embeddings,
            amr_embeddings,
            dock_embeddings,
            action_feat,
            inbound_idx,
            outbound_idx,
            op_mask,
        )

        self.assertEqual(tuple(logits.shape), (1, 2, len(GA.AMR_KEYS), len(jobs)))
        self.assertTrue(torch.isneginf(logits[0, 1, 0, 0]))


if __name__ == "__main__":
    unittest.main()
