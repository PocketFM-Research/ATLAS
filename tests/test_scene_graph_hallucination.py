"""
Deterministic tests for the scene-graph hallucination evaluation stack.

Covers:
- hallucination format parsing (graph_verifier.parse_hallucination_string)
- gold/pred matching (hallucination_metrics.match_predictions)
- precision/recall/F1 computation
- manifest loading
- scene-graph output path naming
- Azure GPT-5.4 mini config is passed consistently to graph construction
  and the LLM judge baseline (run_scene_graph_hallucination_eval._resolve_secret +
  llm.__init__.get_llm dispatch)

Run with:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stage_kg.evaluation.graph_by_graph.graph_verifier import (  # noqa: E402
    HALLUCINATION_FORMAT,
    parse_hallucination_string,
)
from stage_kg.evaluation.graph_by_graph.hallucination_metrics import (  # noqa: E402
    DEFAULT_THRESHOLD,
    aggregate_metrics,
    compute_metrics,
    match_predictions,
    precision_recall_f1,
)


class TestHallucinationFormatParsing(unittest.TestCase):
    def test_format_constant_round_trips(self):
        rendered = HALLUCINATION_FORMAT.format(
            x=2, fact="Bob was free", y=5, event="Bob escapes from jail"
        )
        parsed = parse_hallucination_string(rendered)
        self.assertIsNotNone(parsed)
        x, fact, y, event = parsed
        self.assertEqual(x, 2)
        self.assertEqual(y, 5)
        self.assertEqual(fact, "Bob was free")
        self.assertEqual(event, "Bob escapes from jail")

    def test_rejects_malformed(self):
        self.assertIsNone(parse_hallucination_string("random sentence"))
        self.assertIsNone(parse_hallucination_string(""))
        self.assertIsNone(parse_hallucination_string(None))  # type: ignore[arg-type]
        self.assertIsNone(parse_hallucination_string("In scene 2, foo happened"))

    def test_trailing_period_allowed(self):
        rendered = (
            HALLUCINATION_FORMAT.format(
                x=1, fact="Alice owns a cat", y=3, event="Alice has a dog"
            )
            + "."
        )
        self.assertIsNotNone(parse_hallucination_string(rendered))


class TestMatching(unittest.TestCase):
    def _h(self, x, fact, y, event):
        return {
            "id": 0,
            "hallucination": HALLUCINATION_FORMAT.format(x=x, fact=fact, y=y, event=event),
        }

    def test_exact_match(self):
        gold = [self._h(1, "Bob is free at home", 3, "Bob is shown in jail")]
        pred = [self._h(1, "Bob is free at home", 3, "Bob is shown in jail")]
        r = match_predictions(gold, pred)
        self.assertEqual((r.tp, r.fp, r.fn), (1, 0, 0))

    def test_paraphrase_match(self):
        gold = [self._h(1, "Bob is free at home", 3, "Bob is shown in jail")]
        pred = [self._h(1, "Bob free home", 3, "Bob shown in jail")]
        r = match_predictions(gold, pred)
        self.assertEqual(r.tp, 1)

    def test_scene_pair_disagreement_blocks_match(self):
        gold = [self._h(1, "Bob is free at home", 3, "Bob is shown in jail")]
        # Same text, different scene pair — must NOT match.
        pred = [self._h(2, "Bob is free at home", 4, "Bob is shown in jail")]
        r = match_predictions(gold, pred)
        self.assertEqual((r.tp, r.fp, r.fn), (0, 1, 1))

    def test_missing_prediction_is_fn(self):
        gold = [
            self._h(1, "Bob is free", 3, "Bob in jail"),
            self._h(2, "Car is red", 4, "Car is blue"),
        ]
        pred = [self._h(1, "Bob is free", 3, "Bob in jail")]
        r = match_predictions(gold, pred)
        self.assertEqual((r.tp, r.fp, r.fn), (1, 0, 1))

    def test_extra_prediction_is_fp(self):
        gold = [self._h(1, "Bob is free", 3, "Bob in jail")]
        pred = [
            self._h(1, "Bob is free", 3, "Bob in jail"),
            self._h(2, "Car is red", 4, "Car is blue"),
        ]
        r = match_predictions(gold, pred)
        self.assertEqual((r.tp, r.fp, r.fn), (1, 1, 0))

    def test_greedy_does_not_double_count(self):
        gold = [
            self._h(1, "Bob is free", 3, "Bob in jail"),
            self._h(1, "Bob is free", 3, "Bob in jail"),  # duplicate gold
        ]
        pred = [self._h(1, "Bob is free", 3, "Bob in jail")]
        r = match_predictions(gold, pred)
        self.assertEqual(r.tp, 1)
        self.assertEqual(r.fp, 0)
        self.assertEqual(r.fn, 1)


class TestMetrics(unittest.TestCase):
    def test_precision_recall_f1_basic(self):
        p, r, f = precision_recall_f1(tp=2, fp=1, fn=1)
        self.assertAlmostEqual(p, 2 / 3, places=4)
        self.assertAlmostEqual(r, 2 / 3, places=4)
        self.assertAlmostEqual(f, 2 / 3, places=4)

    def test_zero_predictions_zero_gold(self):
        p, r, f = precision_recall_f1(0, 0, 0)
        self.assertEqual((p, r, f), (0.0, 0.0, 0.0))

    def test_compute_metrics_row_shape(self):
        gold = [
            {
                "id": 1,
                "hallucination": HALLUCINATION_FORMAT.format(
                    x=1, fact="a b c", y=3, event="d e f"
                ),
            }
        ]
        pred = [
            {
                "id": 1,
                "hallucination": HALLUCINATION_FORMAT.format(
                    x=1, fact="a b c", y=3, event="d e f"
                ),
            }
        ]
        row = compute_metrics(story_id="s1", method="graph_verifier", gold=gold, predicted=pred)
        for key in ["story_id", "method", "Gold", "Pred.", "TP", "FP", "FN",
                    "precision", "recall", "F1"]:
            self.assertIn(key, row)
        self.assertEqual(row["TP"], 1)
        self.assertEqual(row["precision"], 1.0)
        self.assertEqual(row["recall"], 1.0)
        self.assertEqual(row["F1"], 1.0)

    def test_aggregate_micro_average(self):
        rows = [
            {"story_id": "a", "method": "m", "Gold": 2, "Pred.": 2, "TP": 1, "FP": 1, "FN": 1,
             "precision": 0.5, "recall": 0.5, "F1": 0.5},
            {"story_id": "b", "method": "m", "Gold": 2, "Pred.": 2, "TP": 2, "FP": 0, "FN": 0,
             "precision": 1.0, "recall": 1.0, "F1": 1.0},
        ]
        agg = aggregate_metrics(rows)
        self.assertEqual(len(agg), 1)
        a = agg[0]
        self.assertEqual(a["TP"], 3)
        self.assertEqual(a["FP"], 1)
        self.assertEqual(a["FN"], 1)
        self.assertAlmostEqual(a["precision"], 0.75, places=4)
        self.assertAlmostEqual(a["recall"], 0.75, places=4)


class TestManifestLoading(unittest.TestCase):
    def test_round_trip(self):
        from run_scene_graph_hallucination_eval import load_manifest

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / "case_one").mkdir()
            (base / "case_one" / "annotations.json").write_text("[]")
            manifest_path = base / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "story_id": "case_one",
                            "movie_dir": "case_one",
                            "annotations": "case_one/annotations.json",
                        }
                    ]
                )
            )
            cases = load_manifest(manifest_path)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].story_id, "case_one")
            self.assertEqual(cases[0].movie_dir.name, "case_one")
            self.assertTrue(cases[0].annotations.exists())


class TestSceneGraphPathNaming(unittest.TestCase):
    def test_scene_dir_name(self):
        from stage_kg.ingest.loader import SceneRecord
        from stage_kg.pipeline import _scene_dir_name, _scene_num

        scene = SceneRecord(
            scene_id="7",
            order=6,
            title="7、INT. ROOM - DAY",
            subtitle="",
            content="x",
            scene_category="INT",
            region="",
            main_location="",
            sub_location="",
            summary="",
            chunks=[],
            source_doc_id="doc_6",
        )
        self.assertEqual(_scene_num(scene), 7)
        self.assertEqual(_scene_dir_name(scene), "scene_007")

    def test_fallback_to_order(self):
        from stage_kg.ingest.loader import SceneRecord
        from stage_kg.pipeline import _scene_dir_name, _scene_num

        scene = SceneRecord(
            scene_id="abc",
            order=2,
            title="x",
            subtitle="",
            content="x",
            scene_category="INT",
            region="",
            main_location="",
            sub_location="",
            summary="",
            chunks=[],
            source_doc_id="doc_2",
        )
        self.assertEqual(_scene_num(scene), 3)
        self.assertEqual(_scene_dir_name(scene), "scene_003")


class TestAzureConfigConsistency(unittest.TestCase):
    """
    Verify the runner builds ONE llm via get_llm() and passes the SAME
    instance to both graph construction (run_pipeline_per_scene) and the
    LLM judge (run_llm_judge). The factory must also accept api_version.
    """

    def test_factory_accepts_azure_args(self):
        from stage_kg.llm import get_llm

        with mock.patch("stage_kg.llm.azure_openai_client.AzureOpenAILLM") as mocked:
            get_llm(
                provider="azure_openai",
                model="gpt-5.4-mini",
                api_key="KEY",
                base_url="https://example.openai.azure.com/openai/deployments/gpt-5.4-mini/chat/completions",
                api_version="2024-12-01-preview",
            )
            mocked.assert_called_once()
            kwargs = mocked.call_args.kwargs
            self.assertEqual(kwargs["model"], "gpt-5.4-mini")
            self.assertEqual(kwargs["api_key"], "KEY")
            self.assertIn("openai/deployments", kwargs["base_url"])
            self.assertEqual(kwargs["api_version"], "2024-12-01-preview")

    def test_azure_target_uri_normalization(self):
        from stage_kg.llm.azure_openai_client import (
            _normalize_azure_url,
            _should_send_api_version,
        )

        url, api_version = _normalize_azure_url(
            "https://example.openai.azure.com/openai/deployments/gpt-5.4-mini/"
            "chat/completions?api-version=2024-12-01-preview"
        )
        self.assertEqual(
            url,
            "https://example.openai.azure.com/openai/deployments/gpt-5.4-mini",
        )
        self.assertEqual(api_version, "2024-12-01-preview")
        self.assertTrue(_should_send_api_version(url))

    def test_azure_ai_services_responses_url_normalization(self):
        from stage_kg.llm.azure_openai_client import (
            _normalize_azure_url,
            _should_send_api_version,
        )

        url, api_version = _normalize_azure_url(
            "https://pocketfm-key.services.ai.azure.com/api/projects/PocketFM-ATLAS/"
            "openai/v1/responses"
        )
        self.assertEqual(
            url,
            "https://pocketfm-key.services.ai.azure.com/api/projects/PocketFM-ATLAS/openai/v1",
        )
        self.assertIsNone(api_version)
        self.assertFalse(_should_send_api_version(url))

    def test_azure_ai_services_url_uses_openai_compatible_client(self):
        from stage_kg.llm.azure_openai_client import _is_openai_compatible_url

        self.assertTrue(
            _is_openai_compatible_url("https://pocketfm-key.services.ai.azure.com/api/projects/x")
        )

    def test_runner_shares_same_llm(self):
        """
        Mock the LLM factory + pipeline + judge and verify the same llm
        object is passed to both. We invoke the runner's process_case
        directly with a stub Case.
        """
        import run_scene_graph_hallucination_eval as runner

        sentinel_llm = mock.MagicMock(name="LLM")
        sentinel_llm.model_id = "azure_openai/gpt-5.4-mini"

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            out.mkdir()
            movie_dir = Path(td) / "movie"
            movie_dir.mkdir()
            (movie_dir / "script.json").write_text(
                json.dumps(
                    [
                        {"_id": 0, "title": "1、INT. A - DAY", "subtitle": "", "content": "x"},
                        {"_id": 1, "title": "2、INT. B - DAY", "subtitle": "", "content": "y"},
                    ]
                )
            )
            ann_path = movie_dir / "annotations.json"
            ann_path.write_text(json.dumps([]))

            case = runner.Case(
                story_id="t1", movie_dir=movie_dir, annotations=ann_path
            )

            args = mock.MagicMock()
            args.max_scenes = None
            args.skip_normalization = False
            args.reuse_existing_graphs = False
            args.skip_graph_method = False
            args.skip_llm_judge = False
            args.match_threshold = DEFAULT_THRESHOLD
            args.api_key = None
            args.api_key_file = None
            args.language = "en"

            with mock.patch.object(runner, "run_pipeline_per_scene") as p_pipe, \
                 mock.patch.object(runner, "load_scene_graphs", return_value=[]) as _, \
                 mock.patch.object(runner, "verify_scene_graphs", return_value=[]) as p_verify, \
                 mock.patch.object(runner, "run_llm_judge", return_value=[]) as p_judge, \
                 mock.patch.object(runner, "extract_all_scene_attributes", return_value={}) as p_attr:
                logger = mock.MagicMock()
                runner.process_case(
                    case=case, output_dir=out, llm=sentinel_llm, args=args, logger=logger
                )

            self.assertIs(p_pipe.call_args.kwargs["llm"], sentinel_llm,
                          "Graph construction must receive the same LLM instance.")
            self.assertIs(p_verify.call_args.kwargs["llm"], sentinel_llm,
                          "Graph verifier must receive the same LLM instance.")
            self.assertIs(p_judge.call_args.kwargs["llm"], sentinel_llm,
                          "LLM judge must receive the same LLM instance.")
            self.assertIs(p_attr.call_args.kwargs["llm"], sentinel_llm,
                          "Attribute extractor must receive the same LLM instance.")


if __name__ == "__main__":
    unittest.main()
