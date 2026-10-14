#!/usr/bin/env python3
"""Offline tests for joint claim-status and user-state resolution."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from memory.memory_database import SessionDB
from user_state.user_state_resolver import UserStateResolver


class _Scheduler:
    def register_handler(self, *_args: object) -> None:
        pass


class _LLM:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.prompts: list[str] = []

    def call_llm(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.response, ensure_ascii=False)

    @staticmethod
    def parse_json_object_from_llm_text(text: str) -> dict:
        return json.loads(text)


class UserClaimStateResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = SessionDB(Path(self.temp_dir.name) / "memory.db")
        self.owner_id = self.db.add_entity_names(["用户本人"])["用户本人"]
        self.old_id, _ = self.db.upsert_user_claim(
            subject_entity_id=self.owner_id, predicate="prefers",
            claim_text="用户喜欢早上开会", claim_type="preference",
            claim_origin="explicit", primary_domain="work",
            status="active", confidence=0.8,
        )
        self.new_id, _ = self.db.upsert_user_claim(
            subject_entity_id=self.owner_id, predicate="prefers",
            claim_text="用户现在更喜欢下午开会", claim_type="preference",
            claim_origin="explicit", primary_domain="work",
            status="active", confidence=0.9,
        )
        self.db.record_user_claim_relation(
            target_claim_id=self.old_id, trigger_claim_id=self.new_id,
            semantic_relation="supersedes", effective_at="2026-10-13 10:00:00",
        )
        self.relation_id = self.db.get_user_claim_relations_for_claim_ids(
            [self.new_id]
        )[0]["id"]
        self.state_id, _, _ = self.db.upsert_user_state(
            user_entity_id=self.owner_id, state_domain="work",
            state_class="enduring_state", state_key="time_preference",
            state_text="用户喜欢早上开会", status="active", confidence=0.8,
        )
        self.db.upsert_user_state_evidence([{
            "state_id": self.state_id, "evidence_type": "user_claim",
            "evidence_id": self.old_id, "role": "support", "weight": 1.0,
        }])
        self.db.upsert_memory_recall_document(
            object_type="user_claim", object_id=self.old_id,
            identity_text="claim: 用户喜欢早上开会\nstatus: active",
            identity_text_embedding=None, status="active",
        )

    def tearDown(self) -> None:
        self.db.close()
        self.temp_dir.cleanup()

    def _resolve(self, response: dict) -> dict:
        llm = _LLM(response)
        resolver = UserStateResolver(
            db=self.db, task_scheduler=_Scheduler(), llm_client=llm,
        )
        result = resolver.process_update_task(
            changed_claim_ids=[self.new_id],
            reference_timestamp="2026-10-13 10:00:00",
        )
        self.assertIn("用户喜欢早上开会", llm.prompts[0])
        self.assertIn("state_claim_evidence", llm.prompts[0])
        return result

    def test_supersede_claim_and_revise_state_together(self) -> None:
        result = self._resolve({
            "claim_updates": [{
                "claim_id": self.old_id, "new_status": "superseded",
                "relation_event_id": self.relation_id,
                "reason": "用户明确改变了偏好",
            }],
            "operations": [{
                "operation": "revise", "target_state_id": self.state_id,
                "state_class": "enduring_state", "state_key": "time_preference",
                "state_text": "用户目前更喜欢下午开会", "conditions": {},
                "confidence": 0.9,
                "evidence": [
                    {"evidence_type": "user_claim", "evidence_id": self.new_id,
                     "role": "support", "weight": 1.0},
                    {"evidence_type": "user_claim", "evidence_id": self.old_id,
                     "role": "counterexample", "weight": 1.0},
                ],
                "reason": "偏好发生变化",
            }],
        })
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["claim_status_update_count"], 1)
        self.assertEqual(result["revised"], 1)
        self.assertEqual(self.db.get_user_claims_by_ids([self.old_id])[0]["status"], "superseded")
        document_status = self.db._conn.execute(
            "SELECT status FROM memory_recall_documents WHERE object_type = 'user_claim' AND object_id = ?",
            (self.old_id,),
        ).fetchone()["status"]
        self.assertEqual(document_status, "superseded")
        self.assertEqual(
            self.db.get_user_states_by_ids([self.state_id])[0]["state_text"],
            "用户目前更喜欢下午开会",
        )
        evidence = self.db.get_user_state_claim_evidence([self.state_id])
        self.assertEqual(
            {(row["evidence_id"], row["role"]) for row in evidence},
            {(self.new_id, "support"), (self.old_id, "counterexample")},
        )
        same_id, _ = self.db.upsert_user_claim(
            subject_entity_id=self.owner_id, predicate="prefers",
            claim_text="用户喜欢早上开会", claim_type="preference",
            claim_origin="explicit", primary_domain="work",
            status="active", confidence=0.8,
        )
        self.assertEqual(same_id, self.old_id)
        self.assertEqual(self.db.get_user_claims_by_ids([self.old_id])[0]["status"], "superseded")

    def test_invalid_relation_rejects_whole_response(self) -> None:
        result = self._resolve({
            "claim_updates": [{
                "claim_id": self.old_id, "new_status": "superseded",
                "relation_event_id": 999999,
            }],
            "operations": [{
                "operation": "expire", "target_state_id": self.state_id,
                "state_class": "enduring_state", "state_key": "time_preference",
                "state_text": "用户喜欢早上开会", "confidence": 0.5,
                "evidence": [{"evidence_type": "user_claim", "evidence_id": self.new_id,
                              "role": "counterexample", "weight": 1.0}],
            }],
        })
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.db.get_user_claims_by_ids([self.old_id])[0]["status"], "active")
        self.assertEqual(self.db.get_user_states_by_ids([self.state_id])[0]["status"], "active")

    def test_contradiction_cannot_directly_supersede_claim(self) -> None:
        self.db.record_user_claim_relation(
            target_claim_id=self.old_id, trigger_claim_id=self.new_id,
            semantic_relation="contradicts", effective_at="2026-10-13 10:01:00",
        )
        contradiction_id = next(
            row["id"] for row in self.db.get_user_claim_relations_for_claim_ids([self.new_id])
            if row["semantic_relation"] == "contradicts"
        )
        result = self._resolve({
            "claim_updates": [{
                "claim_id": self.old_id, "new_status": "superseded",
                "relation_event_id": contradiction_id,
            }],
            "operations": [],
        })
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.db.get_user_claims_by_ids([self.old_id])[0]["status"], "active")

    def test_state_can_expire_without_invalidating_claim(self) -> None:
        result = self._resolve({
            "claim_updates": [],
            "operations": [{
                "operation": "expire", "target_state_id": self.state_id,
                "state_class": "enduring_state", "state_key": "time_preference",
                "state_text": "用户喜欢早上开会", "confidence": 0.5,
                "evidence": [{"evidence_type": "user_claim", "evidence_id": self.new_id,
                              "role": "counterexample", "weight": 1.0}],
                "reason": "原有聚合不再可靠",
            }],
        })
        self.assertEqual(result["status"], "ok")
        self.assertEqual(self.db.get_user_claims_by_ids([self.old_id])[0]["status"], "active")
        self.assertEqual(self.db.get_user_states_by_ids([self.state_id])[0]["status"], "expired")

    def test_state_write_failure_rolls_back_claim_status(self) -> None:
        resolver = UserStateResolver(
            db=self.db, task_scheduler=_Scheduler(),
            llm_client=_LLM({
                "claim_updates": [{
                    "claim_id": self.old_id, "new_status": "superseded",
                    "relation_event_id": self.relation_id,
                }],
                "operations": [],
            }),
        )
        with patch.object(resolver, "apply_operations", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                resolver.process_update_task(
                    changed_claim_ids=[self.new_id],
                    reference_timestamp="2026-10-13 10:00:00",
                )
        self.assertEqual(self.db.get_user_claims_by_ids([self.old_id])[0]["status"], "active")
        transitions = self.db._conn.execute(
            "SELECT COUNT(*) FROM memory_user_claim_events WHERE event_type = 'status_transition'"
        ).fetchone()[0]
        self.assertEqual(transitions, 0)


if __name__ == "__main__":
    unittest.main()
