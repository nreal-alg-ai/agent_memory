"""Regression coverage for offline-first speaker model resolution and hub failover."""
from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from voice.modelscope_model import (  # noqa: E402
    DOMESTIC_ENDPOINT,
    INTERNATIONAL_ENDPOINT,
    resolve_modelscope_model_path,
)


class ModelScopeModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.model_id = "iic/speech_campplus_sv_zh-cn_16k-common"
        self.snapshot = self.directory / "snapshot"
        self.snapshot.mkdir()
        (self.snapshot / "configuration.json").write_text(json.dumps({
            "file_path_metas": {"config": "config.yaml", "init_param": "weights.bin"},
        }))
        (self.snapshot / "config.yaml").write_text("model: CAMPPlus\nmodel_conf: {}\n")
        (self.snapshot / "weights.bin").write_bytes(b"test weights")
        self.download = Mock()
        module = ModuleType("modelscope.hub.snapshot_download")
        module.snapshot_download = self.download
        self.modules = patch.dict(sys.modules, {"modelscope.hub.snapshot_download": module})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.logger = Mock(spec=logging.Logger)

    def resolve(self, **kwargs):
        return resolve_modelscope_model_path(self.model_id, logger=self.logger, **kwargs)

    def test_complete_cache_never_contacts_a_hub(self):
        self.download.return_value = str(self.snapshot)
        self.assertEqual(self.resolve(), str(self.snapshot.resolve()))
        self.download.assert_called_once_with(
            self.model_id, revision="master", local_files_only=True,
        )

    def test_missing_cache_downloads_from_domestic_hub(self):
        self.download.side_effect = [FileNotFoundError("not cached"), str(self.snapshot)]
        self.assertEqual(self.resolve(revision="v1"), str(self.snapshot.resolve()))
        self.assertEqual(self.download.call_args_list[1].kwargs, {
            "revision": "v1", "endpoint": DOMESTIC_ENDPOINT,
        })

    def test_partial_or_malformed_cache_is_not_used(self):
        partial = self.directory / "partial"
        partial.mkdir()
        for contents in ["{broken", json.dumps({"file_path_metas": {
            "config": "config.yaml", "init_param": "missing.bin",
        }}), "null"]:
            with self.subTest(contents=contents):
                (partial / "configuration.json").write_text(contents)
                self.download.reset_mock(side_effect=True)
                self.download.side_effect = [str(partial), str(self.snapshot)]
                self.assertEqual(self.resolve(), str(self.snapshot.resolve()))
                self.assertEqual(self.download.call_count, 2)

    def test_domestic_file_list_failure_switches_to_international(self):
        self.download.side_effect = [
            FileNotFoundError("not cached"),
            RuntimeError("HTTP 500: model file tree unavailable"),
            str(self.snapshot),
        ]
        self.assertEqual(self.resolve(), str(self.snapshot.resolve()))
        self.assertEqual(
            [call.kwargs.get("endpoint") for call in self.download.call_args_list],
            [None, DOMESTIC_ENDPOINT, INTERNATIONAL_ENDPOINT],
        )
        self.assertTrue(self.logger.warning.call_args.kwargs["exc_info"])

    def test_domestic_timeout_switches_to_international(self):
        self.download.side_effect = [
            FileNotFoundError("not cached"), TimeoutError("file list timeout"), str(self.snapshot),
        ]
        self.assertEqual(self.resolve(), str(self.snapshot.resolve()))
        self.assertEqual(self.download.call_args.kwargs["endpoint"], INTERNATIONAL_ENDPOINT)

    def test_incomplete_domestic_download_switches_to_international(self):
        partial = self.directory / "empty"
        partial.mkdir()
        self.download.side_effect = [FileNotFoundError("not cached"), str(partial), str(self.snapshot)]
        self.assertEqual(self.resolve(), str(self.snapshot.resolve()))
        self.assertEqual(self.download.call_args.kwargs["endpoint"], INTERNATIONAL_ENDPOINT)

    def test_both_hubs_fail_preserves_the_original_errors(self):
        last = TimeoutError("international timeout")
        self.download.side_effect = [
            FileNotFoundError("not cached"), RuntimeError("domestic HTTP 500"), last,
        ]
        with self.assertRaisesRegex(RuntimeError, "domestic HTTP 500.*international timeout") as result:
            self.resolve()
        self.assertIs(result.exception.__cause__, last)

    def test_explicit_local_directory_bypasses_cache_and_network(self):
        self.assertEqual(
            resolve_modelscope_model_path(str(self.snapshot)), str(self.snapshot.resolve()),
        )
        self.download.assert_not_called()

    def test_plain_funasr_config_cache_is_supported(self):
        (self.snapshot / "configuration.json").unlink()
        (self.snapshot / "weights.bin").rename(self.snapshot / "model.pt")
        self.download.return_value = str(self.snapshot)
        self.assertEqual(self.resolve(), str(self.snapshot.resolve()))

    def test_speaker_detector_passes_resolved_path_to_funasr(self):
        from voice.speaker_detector import SpeakerReferenceDetector

        auto_model = Mock(return_value=SimpleNamespace())
        with patch.dict(sys.modules, {"funasr": SimpleNamespace(AutoModel=auto_model)}):
            self.download.return_value = str(self.snapshot)
            SpeakerReferenceDetector({}, sample_rate=16000, device="cpu", logger=self.logger)
        self.assertEqual(auto_model.call_args.kwargs, {
            "model": str(self.snapshot.resolve()), "disable_update": True,
        })

    def test_speaker_alias_uses_the_same_modelscope_cache(self):
        self.download.return_value = str(self.snapshot)
        self.assertEqual(resolve_modelscope_model_path("cam++"), str(self.snapshot.resolve()))
        self.assertEqual(self.download.call_args.args[0], self.model_id)


if __name__ == "__main__":
    unittest.main()
