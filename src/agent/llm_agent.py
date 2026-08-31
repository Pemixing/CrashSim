import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from agent.trajectory_agent import TrajectoryAgent


def _str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM agent: read VLM outputs and generate adversarial vehicle 8s trajectory."
    )
    parser.add_argument("--vlm_jsonl", type=str, default="/path/to/llm_out/val/vlm_agent_output.jsonl")
    parser.add_argument(
        "--scene_text_dir",
        type=str,
        default="/path/to/viz_nusc_scene_graph/val",
        help="Directory containing scene text files named <sample_id>.txt",
    )
    parser.add_argument("--output_jsonl", type=str, default="/path/to/llm_out/val/llm_agent_output.jsonl")
    parser.add_argument("--output_dir", type=str, default="./out/llm_agent_json")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument(
        "--api_base",
        type=str,
        default="https://your-api-endpoint/v1",
        help="LLM API base URL (OpenAI-compatible).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="LLM API key (pass via --api_key).",
    )
    parser.add_argument(
        "--use_rag",
        type=_str2bool,
        default=True,
        help="Whether TrajectoryAgent uses RAG anchors. Set False to disable RAG.",
    )
    parser.add_argument("--llm_output_trajectory", type=_str2bool, default=True, help="Whether TrajectoryAgent outputs trajectory.")    
    parser.add_argument("--max_samples", type=int, default=-1, help="Limit number of samples. -1 means all.")
    parser.add_argument(
        "--resume",
        type=_str2bool,
        default=True,
        help="Resume from existing output_jsonl by skipping completed sample_ids.",
    )
    parser.add_argument(
        "--append_jsonl",
        type=_str2bool,
        default=True,
        help="Write output_jsonl incrementally (append per sample). Recommended for long runs.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                preview = line[:240].replace("\n", "\\n")
                print(
                    f"[warn] skip invalid JSONL line {line_no} in {path}: {e}. "
                    f"line_preview={preview!r}"
                )
    return rows


class AnchorGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        if not self.args.api_key:
            raise ValueError("api_key is required (pass --api_key).")

    @staticmethod
    def _load_completed_sample_ids(output_jsonl: Path) -> Set[str]:
        completed: Set[str] = set()
        if not output_jsonl.exists():
            return completed
        try:
            with output_jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sid = obj.get("sample_id")
                    if isinstance(sid, str) and sid.strip():
                        completed.add(sid.strip())
        except Exception as e:
            print(f"[warn] failed to read completed ids from {output_jsonl}: {e}")
        return completed

    @staticmethod
    def _apply_rank_id_mapping_to_messages(
        messages: object, rank_id_to_agent_idx: Dict[str, int]
    ) -> None:
        if not isinstance(messages, list) or not rank_id_to_agent_idx:
            return
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            adv_id = msg.get("adversarial_vehicle_id")
            if adv_id is None:
                continue
            key = str(adv_id)
            mapped = rank_id_to_agent_idx.get(key)
            if mapped is None:
                continue
            msg["adversarial_vehicle_id"] = mapped

    def run(self) -> None:
        vlm_jsonl = Path(self.args.vlm_jsonl)
        output_jsonl = Path(self.args.output_jsonl)
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

        rows = _read_jsonl(vlm_jsonl)
        if self.args.max_samples > 0:
            rows = rows[: self.args.max_samples]
        if not rows:
            raise ValueError(f"No rows found in {vlm_jsonl}")

        trajectory_agent = TrajectoryAgent(
            api_base=self.args.api_base,
            api_key=self.args.api_key,
            model=self.args.model,
            use_rag=self.args.use_rag,
            llm_output_trajectory=self.args.llm_output_trajectory,
        )
        completed_ids: Set[str] = set()
        if self.args.resume:
            completed_ids = self._load_completed_sample_ids(output_jsonl)
            if completed_ids:
                print(f"[info] resume enabled: loaded {len(completed_ids)} completed sample_ids")

        # "断点续跑"的 checkpoint：逐样本落盘 + JSONL 逐行追加。
        out_rows: List[Dict] = []
        jsonl_mode = "a" if self.args.append_jsonl else "w"
        if self.args.append_jsonl and (not self.args.resume):
            # If not resuming but still appending, ensure we start fresh.
            jsonl_mode = "w"

        with output_jsonl.open(jsonl_mode, encoding="utf-8") as merged_f:
            for idx, row in enumerate(rows):
                sample_id = str(row.get("sample_id", "")).strip()
                image_path_in_row = row.get("image_path")
                if isinstance(image_path_in_row, str) and image_path_in_row.strip():
                    sample_id = Path(image_path_in_row).stem
                if not sample_id:
                    sample_id = f"row_{idx}"

                if self.args.resume and sample_id in completed_ids:
                    continue

                item = trajectory_agent.run_rows(rows=[row], max_samples=1)[0]
                out_rows.append(item)

                image_path_in_row = row.get("image_path")
                text_path_in_row = row.get("text_path")

                sample_folder: Optional[Path] = None
                if isinstance(image_path_in_row, str) and image_path_in_row.strip():
                    sample_folder = Path(image_path_in_row).parent
                elif isinstance(text_path_in_row, str) and text_path_in_row.strip():
                    sample_folder = Path(text_path_in_row).parent
                else:
                    candidate = Path(self.args.scene_text_dir) / sample_id
                    if candidate.is_dir():
                        sample_folder = candidate

                if sample_folder is None:
                    sample_folder = output_dir
                sample_folder.mkdir(parents=True, exist_ok=True)

                rank_mapping_path: Optional[Path] = None
                if isinstance(image_path_in_row, str) and image_path_in_row.strip():
                    candidate = Path(image_path_in_row).parent / "rank_id_mapping.json"
                    if candidate.is_file():
                        rank_mapping_path = candidate
                if rank_mapping_path is None:
                    candidate = sample_folder / "rank_id_mapping.json"
                    if candidate.is_file():
                        rank_mapping_path = candidate

                if rank_mapping_path is not None:
                    try:
                        with rank_mapping_path.open("r", encoding="utf-8") as f:
                            mapping_obj = json.load(f)
                        rank_id_to_agent_token = mapping_obj.get("rank_id_to_agent_token", {})
                        if isinstance(rank_id_to_agent_token, dict):
                            self._apply_rank_id_mapping_to_messages(
                                item.get("trajectory_agent_messages"), rank_id_to_agent_token
                            )
                            self._apply_rank_id_mapping_to_messages(
                                row.get("trajectory_agent_messages"), rank_id_to_agent_token
                            )
                    except Exception as e:
                        print(f"[warn] failed to load {rank_mapping_path}: {e}")

                json_stem = sample_id
                if isinstance(image_path_in_row, str) and image_path_in_row.strip():
                    json_stem = Path(image_path_in_row).stem
                per_sample_path = sample_folder / f"{json_stem}_llm.json"
                with per_sample_path.open("w", encoding="utf-8") as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)

                merged_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                merged_f.flush()
                print(f"[{idx + 1}/{len(out_rows)}] saved {per_sample_path}")
        print(f"Done. merged JSONL saved to: {output_jsonl}")


def main() -> None:
    args = parse_args()
    AnchorGenerator(args).run()


if __name__ == "__main__":
    main()
