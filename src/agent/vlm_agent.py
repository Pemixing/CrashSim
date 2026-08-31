import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent.base_agent import AgentMessage
from agent.memory import SceneContext
from agent.retrieval_agent import RetrievalAgent
from agent.scene_agent import (
    SceneAgent,
    _filter_adversarial_vehicle_ids_posterior,
    _filter_norm_fields_by_ids,
    _read_text,
)

def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).parent.parent.parent
    default_out = root_dir / "out"
    parser = argparse.ArgumentParser(
        description="VLM agent: read NuScenes scene image + text and output structured scene understanding JSON."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(default_out / "viz_nusc_scene_graph" / "val"),
        help="Directory containing sample subfolders; each subfolder has <folder_name>.png and <folder_name>.txt.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=str(default_out / "llm_out" / "val" / "vlm_agent_output.jsonl"),
        help="Path to merged JSONL output.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="https://your-api-endpoint/v1",
        help="VLM API base URL (OpenAI-compatible).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="VLM API key (pass via --api_key).",
    )
    parser.add_argument(
        "--rag_db_dir",
        type=str,
        default=str(default_out / "nhtsa_rag"),
        help="Directory containing the three RAG vector DB JSON files.",
    )
    parser.add_argument(
        "--vector_mode",
        type=str,
        default="diff",
        choices=["trajectory", "control", "diff"],
        help="Vector feature mode used by RetrievalAgent. Allowed: trajectory control diff.",
    )
    parser.add_argument("--max_samples", type=int, default=-1, help="Limit number of samples. -1 means all.")
    parser.add_argument(
        "--read_file",
        action="store_true",
        help="Skip SceneAgent.run_directory; load rows from JSONL (--read_jsonl or --output_jsonl).",
    )
    parser.add_argument(
        "--read_jsonl",
        type=str,
        default=None,
        help="JSONL path when --read_file or --filter (default: same as --output_jsonl).",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help=(
            "Load rows from JSONL (--read_jsonl or --output_jsonl), re-apply "
            "_filter_adversarial_vehicle_ids_posterior without calling VLM (skip call_openai_chat)."
        ),
    )
    return parser.parse_args()


class SceneInterpreter:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        if not self.args.filter and not self.args.api_key:
            raise ValueError("api_key is required (pass --api_key).")

    def _get_resume_start_index(self) -> Tuple[int, Optional[str], List[Dict[str, Any]]]:
        """
        Resume from `input_dir/scene_agent_rows.jsonl` if present.

        The checkpoint written by SceneAgent.run_directory always contains the
        *full* cumulative history (prev rows + new rows), so n_lines directly
        equals the number of samples already handled.  We re-process the last
        entry to guard against a truncated/corrupt final write, so:
            start_index = max(0, n_lines - 1)
            prev_rows   = all lines except the last

        The returned prev_rows are passed back into run_directory as
        `initial_rows` so that any subsequent checkpoint written mid-run also
        includes the full history, preventing start_index from regressing on a
        second crash.
        """
        root = Path(self.args.input_dir)
        ckpt = root / "scene_agent_rows.jsonl"
        if not ckpt.exists():
            return 0, None, []

        try:
            content = ckpt.read_text(encoding="utf-8", errors="ignore")
            lines = [ln for ln in content.splitlines() if ln.strip()]
            n_lines = len(lines)
        except Exception:
            return 0, str(ckpt), []

        if n_lines <= 0:
            return 0, str(ckpt), []

        prev_rows: List[Dict[str, Any]] = []
        for ln in lines[:-1]:
            try:
                prev_rows.append(json.loads(ln))
            except Exception:
                continue

        return max(0, n_lines - 1), str(ckpt), prev_rows

    def _load_rows_from_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"read_jsonl not found: {path}")
        content = path.read_text(encoding="utf-8", errors="ignore")
        rows: List[Dict[str, Any]] = []
        for ln in content.splitlines():
            if not ln.strip():
                continue
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return rows

    def _apply_posterior_filter_to_rows(self, rows: List[Dict[str, Any]]) -> None:
        for idx, row in enumerate(rows):
            if row.get("error"):
                print(f"[vlm_agent] filter skip row {idx}: error={row.get('error')!r}")
                continue
            scene_text = row.get("scene_text")
            text_path = row.get("text_path")
            if not scene_text and text_path:
                try:
                    scene_text = _read_text(Path(str(text_path)))
                except Exception as e:
                    print(f"[vlm_agent] filter skip {row.get('sample_id')}: cannot read text_path: {e!r}")
                    continue
            if not scene_text:
                print(f"[vlm_agent] filter skip {row.get('sample_id')}: no scene_text/text_path")
                continue

            adv_ids = list(row.get("adversarial_vehicle_ids", []) or [])
            txt_parent_dir = str(Path(str(text_path)).parent) if text_path else None
            kept_ids = _filter_adversarial_vehicle_ids_posterior(
                scene_text,
                adv_ids,
                txt_parent_dir=txt_parent_dir,
            )
            row.update(_filter_norm_fields_by_ids(row, kept_ids))
            print(
                f"[vlm_agent] filter [{idx + 1}/{len(rows)}] sample={row.get('sample_id')} "
                f"ids {adv_ids} -> {kept_ids}"
            )

    def run(self) -> None:
        output_jsonl = Path(self.args.output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        rag_dir = Path(self.args.rag_db_dir)
        retrieval_agent = RetrievalAgent(
            mode_db_path=str(rag_dir / "nhtsa_mode_vector_db.json"),
            variant_db_path=str(rag_dir / "nhtsa_variant_vector_db.json"),
            sample_db_path=str(rag_dir / "nhtsa_sample_vector_db.json"),
            vector_mode=self.args.vector_mode,
        )

        if self.args.filter:
            read_path = Path(self.args.read_jsonl or self.args.output_jsonl)
            print(f"[vlm_agent] filter: loading rows from {read_path}")
            rows = self._load_rows_from_jsonl(read_path)
            if self.args.max_samples > 0:
                rows = rows[: self.args.max_samples]
            self._apply_posterior_filter_to_rows(rows)
        elif self.args.read_file:
            read_path = Path(self.args.read_jsonl or self.args.output_jsonl)
            print(f"[vlm_agent] read_file: loading rows from {read_path}")
            rows = self._load_rows_from_jsonl(read_path)
            if self.args.max_samples > 0:
                rows = rows[: self.args.max_samples]
        else:
            scene_agent = SceneAgent(
                api_base=self.args.api_base,
                api_key=self.args.api_key,
                model=self.args.model,
            )

            start_index, ckpt_path, prev_rows = self._get_resume_start_index()
            if start_index > 0:
                print(f"[vlm_agent] resume enabled: start_index={start_index} (from {ckpt_path})")
            else:
                print("[vlm_agent] resume disabled: start_index=0")

            # Bug fix: when resuming, max_samples must be reduced by start_index so
            # that the total number of processed samples does not exceed the user's
            # original limit.
            effective_max_samples = self.args.max_samples
            if effective_max_samples > 0 and start_index > 0:
                effective_max_samples = max(0, effective_max_samples - start_index)

            rows = scene_agent.run_directory(
                input_dir=self.args.input_dir,
                max_samples=effective_max_samples,
                start_index=start_index,
                # Bug fix: pass prev_rows so that every checkpoint written inside
                # run_directory includes the full cumulative history.  Without this,
                # a second crash during a resumed run would overwrite the checkpoint
                # with only the new-batch rows, causing start_index to regress on
                # the next resume attempt.
                initial_rows=prev_rows if prev_rows else None,
            )
            if not rows and not prev_rows:
                raise FileNotFoundError("No valid sample folders with matched png/txt were found.")
            if prev_rows:
                rows = prev_rows + rows

        if not rows:
            if self.args.filter or self.args.read_file:
                rp = Path(self.args.read_jsonl or self.args.output_jsonl)
                raise ValueError(f"No valid JSON lines in {rp}")
            raise FileNotFoundError("No valid sample folders with matched png/txt were found.")

        for idx, row in enumerate(rows):
            # RAG-enhance potential behaviors with NHTSA retrieval.
            adversarial_vehicle_ids = row.get("adversarial_vehicle_ids")
            row["retrieval_agent_messages"] = []
            if adversarial_vehicle_ids:
                tmp_ctx = SceneContext(
                    sample_id=row.get("sample_id", str(idx)),
                    bev_image_path=row.get("image_path"),
                    scene_text_path=row.get("text_path"),
                    scene_text=row.get("scene_text"),
                    scene_understanding=row.get("scene_understanding"),
                    ego_driving_intention=row.get("ego_driving_intention"),
                    adversarial_vehicle_ids=adversarial_vehicle_ids,
                    spatial_relationships_with_ego=row.get("spatial_relationships_with_ego", []),
                    potential_adversarial_behaviors=row.get("potential_adversarial_behaviors", []),
                )
                rag_reply = retrieval_agent.run(
                    AgentMessage(
                        sender="vlm_agent",
                        receiver="RetrievalAgent",
                        msg_type="request",
                        payload={"context": tmp_ctx},
                        trace_id=row.get("sample_id", str(idx)),
                    )
                )
                rag_replies = rag_reply if isinstance(rag_reply, list) else [rag_reply]
                rag_payloads = [r.payload for r in rag_replies]

                # Write all retrieval messages/results to row (one per behavior).
                row["retrieval_agent_messages"] = rag_payloads

            sample_folder = Path(str(row["image_path"])).parent
            per_sample_path = sample_folder / f"{Path(str(row['image_path'])).stem}_vlm.json"
            with per_sample_path.open("w", encoding="utf-8") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)
            print(f"[{idx + 1}/{len(rows)}] saved {per_sample_path}")

        # If resuming, append to avoid clobbering previously merged results.
        # NOTE: We already merged previous rows into `rows` above when resuming,
        # so always write a full merged JSONL to avoid duplication.
        out_mode = "w"
        with output_jsonl.open(out_mode, encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"Done. merged JSONL saved to: {output_jsonl}")

        # Clean up the intermediate checkpoint so that the next invocation
        # starts fresh instead of incorrectly triggering resume logic.
        if not self.args.read_file and not self.args.filter:
            ckpt_file = Path(self.args.input_dir) / "scene_agent_rows.jsonl"
            try:
                if ckpt_file.exists():
                    ckpt_file.unlink()
                    print(f"[vlm_agent] checkpoint removed: {ckpt_file}")
            except Exception as e:
                print(f"[vlm_agent] could not remove checkpoint {ckpt_file}: {e!r}")


def main() -> None:
    args = parse_args()
    SceneInterpreter(args).run()


if __name__ == "__main__":
    main()
