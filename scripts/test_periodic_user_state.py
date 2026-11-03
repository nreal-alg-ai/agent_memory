"""External user-state signals, durable pending claims and explicit offline drain."""

from __future__ import annotations

import logging
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "integrations" / "qwen_audio_agent"))

from memory.memory_database import UserMemoryDB
from task_scheduler import TaskOperationReporter, TaskScheduler
from agent_memory_sidecar import AgentMemoryRuntime


class PeriodicUserStateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.db = UserMemoryDB(Path(directory.name) / "memory.db")
        self.addCleanup(self.db.close)
        self.reporter = TaskOperationReporter()
        self.scheduler = TaskScheduler(
            operation_reporter=self.reporter, claim_state_database=self.db,
            task_scheduling_config={"user_state_claim_batch_limit": 1},
            logger=Mock(),
        )
        self.addCleanup(lambda: self.scheduler.shutdown(timeout=5))
        self.handler = Mock(return_value={"status": "ok"})
        self.scheduler.register_handler("user_state_update_from_user_claims", self.handler)

    def record(self, ids):
        self.scheduler.record_pending_user_state_claims(
            ids, reference_timestamp="2026-10-01T10:00:00+08:00",
        )

    def pending_count(self):
        return self.scheduler._task_execution_db.get_pending_user_state_claims()["pending_count"]

    def flush(self):
        self.assertTrue(self.scheduler.flush(timeout=5))

    def test_start_and_unrelated_work_never_trigger_state_updates(self):
        self.record([1])
        self.scheduler.start()
        self.scheduler.register_handler("unrelated", lambda: {"status": "ok"})
        self.scheduler.submit("unrelated", {})
        self.flush()
        self.handler.assert_not_called()
        self.assertEqual(self.pending_count(), 1)
        self.assertNotIn("_next_claim_state_tick_at", vars(self.scheduler))

    def test_one_claim_runs_on_external_signal_and_preserves_evidence_time(self):
        self.record([1])
        report = self.scheduler.trigger_user_state_update()
        self.assertTrue(report["queued"])
        self.assertEqual(report["operation_type"], "user_state_update_from_user_claims")
        self.assertEqual(report["pending_claim_count"], 1)
        self.flush()
        self.handler.assert_called_once_with(
            new_claim_ids=[1], claim_relations=[],
            reference_timestamp="2026-10-01T10:00:00+08:00",
        )
        self.assertEqual(self.pending_count(), 0)
        before = self.reporter.snapshot()["counts"]
        self.assertEqual(self.scheduler.trigger_user_state_update(), {
            "status": "empty", "queued": False, "reason": "no_changed_claims",
        })
        self.assertEqual(self.reporter.snapshot()["counts"], before)
        self.flush()
        self.handler.assert_called_once()

    def test_claim_produced_after_signal_waits_for_the_next_signal(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def producer():
            entered.set()
            self.assertTrue(release.wait(5))
            self.record([1])
            return {"status": "ok"}

        self.scheduler.register_handler("producer", producer)
        self.scheduler.submit("producer", {})
        self.assertTrue(entered.wait(2))
        self.assertEqual(self.scheduler.trigger_user_state_update()["reason"], "no_changed_claims")
        release.set()
        self.flush()
        self.handler.assert_not_called()
        self.assertEqual(self.pending_count(), 1)
        self.assertTrue(self.scheduler.trigger_user_state_update()["queued"])
        self.flush()
        self.handler.assert_called_once()

    def test_query_failure_and_rejected_submission_allow_retry(self):
        self.record([1])
        with patch.object(self.scheduler._task_execution_db, "get_pending_user_state_claims",
                          side_effect=RuntimeError("database unavailable")):
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                self.scheduler.trigger_user_state_update()
        with patch.object(self.scheduler, "submit", return_value={
            "queued": False, "reason": "worker_queue_full",
        }):
            self.assertEqual(self.scheduler.trigger_user_state_update()["reason"], "worker_queue_full")
        self.assertEqual(self.pending_count(), 1)
        self.assertTrue(self.scheduler.trigger_user_state_update()["queued"])
        self.flush()
        self.assertEqual(self.pending_count(), 0)

    def test_signal_does_not_wait_for_model_and_duplicate_is_coalesced(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def blocked():
            entered.set()
            self.assertTrue(release.wait(5))
            return {"status": "ok"}

        self.scheduler.register_handler("blocked", blocked)
        self.scheduler.submit("blocked", {})
        self.assertTrue(entered.wait(2))
        self.record([1])
        started = time.monotonic()
        report = self.scheduler.trigger_user_state_update()
        self.assertLess(time.monotonic() - started, 1)
        self.assertTrue(report["queued"])
        self.assertEqual(self.scheduler.trigger_user_state_update()["reason"], "claim_state_task_pending")
        release.set()
        self.flush()
        self.handler.assert_called_once()

    def test_failed_state_task_retains_claims_for_next_signal(self):
        self.record([1])
        self.handler.side_effect = [RuntimeError("model unavailable"), {"status": "ok"}]
        self.scheduler.trigger_user_state_update()
        self.flush()
        self.assertEqual(self.pending_count(), 1)
        self.assertTrue(self.scheduler.trigger_user_state_update()["queued"])
        self.flush()
        self.assertEqual(self.pending_count(), 0)

    def test_periodic_signal_processes_one_batch_and_explicit_drain_finishes_rest(self):
        self.record([1, 2, 3])
        self.scheduler.trigger_user_state_update()
        self.flush()
        self.assertEqual(self.handler.call_count, 1)
        self.assertEqual(self.pending_count(), 2)
        self.scheduler.drain_pending_user_state_claims()
        self.flush()
        self.assertEqual(self.handler.call_count, 3)
        self.assertEqual(self.pending_count(), 0)

    def test_claim_changed_during_model_execution_remains_pending(self):
        self.record([1])

        def update(**_payload):
            self.record([1])
            return {"status": "ok"}

        self.handler.side_effect = update
        self.scheduler.trigger_user_state_update()
        self.flush()
        self.assertEqual(self.pending_count(), 1)
        self.handler.side_effect = None
        self.scheduler.trigger_user_state_update()
        self.flush()
        self.assertEqual(self.pending_count(), 0)

    def test_drain_requested_during_state_update_continues_after_current_batch(self):
        self.record([1, 2])
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def update(**_payload):
            entered.set()
            self.assertTrue(release.wait(5))
            return {"status": "ok"}

        self.handler.side_effect = update
        self.scheduler._logger = logging.getLogger(__name__ + ".scheduler")
        with self.assertLogs(self.scheduler._logger, level="INFO") as logs:
            self.scheduler.trigger_user_state_update()
            self.assertTrue(entered.wait(2))
            report = self.scheduler.drain_pending_user_state_claims()
            self.assertFalse(report["queued"])
            self.assertEqual(report["reason"], "claim_state_task_pending")
            release.set()
            self.flush()
        self.assertTrue(any("reason=claim_state_task_pending drain_requested=True" in line
                            for line in logs.output))
        self.assertEqual(self.handler.call_count, 2)
        self.assertEqual(self.pending_count(), 0)

    def test_trigger_logs_explain_empty_unavailable_failure_rejection_and_submission(self):
        self.scheduler._logger = logging.getLogger(__name__ + ".scheduler")
        with self.assertLogs(self.scheduler._logger, level="INFO") as logs:
            self.assertFalse(self.scheduler.trigger_user_state_update()["queued"])
            self.record([1, 2])
            with patch.object(self.scheduler, "_task_execution_db", None):
                self.assertFalse(self.scheduler.trigger_user_state_update()["queued"])
            with patch.object(self.scheduler._task_execution_db, "get_pending_user_state_claims",
                              side_effect=RuntimeError("query failed")):
                with self.assertRaises(RuntimeError):
                    self.scheduler.trigger_user_state_update()
            with patch.object(self.scheduler, "submit", return_value={
                "queued": False, "reason": "worker_queue_full",
            }):
                self.assertFalse(self.scheduler.trigger_user_state_update()["queued"])
            report = self.scheduler.trigger_user_state_update()
            self.flush()
        output = "\n".join(logs.output)
        for expected in ["trigger_user_state_update received", "reason=no_changed_claims",
                         "reason=task_execution_database_unavailable",
                         "trigger_user_state_update failed", "reason=worker_queue_full",
                         "trigger_user_state_update submitting", "pending_claims=2 selected_claims=1",
                         "queued=True task_id=" + report["task_id"]]:
            self.assertIn(expected, output)


class SidecarPeriodicTaskTests(unittest.TestCase):
    def test_validates_task_and_owner_before_resolving_runtime(self):
        runtime = object.__new__(AgentMemoryRuntime)
        runtime._owner_runtime = Mock()
        runtime._logger = Mock()
        with self.assertRaises(ValueError):
            runtime.trigger_periodic_task({"taskId": "unknown", "ownerId": "owner"})
        with self.assertRaises(ValueError):
            runtime.trigger_periodic_task({"taskId": "user_state_from_claims"})
        runtime._owner_runtime.assert_not_called()
        holder = runtime._owner_runtime.return_value
        holder.pwm.trigger_user_state_update.return_value = {"queued": True}
        result = runtime.trigger_periodic_task({
            "taskId": "user_state_from_claims", "ownerId": "authenticated-owner",
        })
        runtime._owner_runtime.assert_called_once_with("authenticated-owner")
        holder.pwm.trigger_user_state_update.assert_called_once_with()
        self.assertEqual(result, {"supported": True, "taskId": "user_state_from_claims", "queued": True})


if __name__ == "__main__":
    unittest.main()
