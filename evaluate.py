"""
# Usage: python evaluate.py --train_output_dir ./CLIP_ViT-L-14-DataComp.XL-s13B-b90K --output_dir ./output --batch_size 16
"""
import argparse
import json
import os
import pickle
import time
import warnings
from pathlib import Path
import yaml

from eval_utils.main import evaluate_model

warnings.filterwarnings("ignore", message="Length of IterableDataset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_output_dir",
        required=True,
        help="Path to output directory from training.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Path to output directory to use for evaluation. If nothing is passed, use the training output dir.",
    )
    parser.add_argument(
        "--data_dir",
        help="(Optional) Path to directory containing downloaded evaluation datasets.",
        default=None,
    )
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")

    # Debug-only flags. Using any of these might invalidate your submission.
    parser_debug = parser.add_argument_group("debug-only")
    parser_debug.add_argument(
        "--use_model",
        type=str,
        help='If set, manually specify a model architecture and checkpoint path ("model path")',
        default=None,
    )

    args = parser.parse_args()

    args.train_output_dir = Path(args.train_output_dir)
    if args.output_dir is None:
        args.output_dir = args.train_output_dir
    args.output_dir = Path(args.output_dir)

    if args.use_model is not None:
        args.train_output_dir = args.output_dir
        # Generate barebones info.pkl
        model_arch, model_checkpoint = args.use_model.split(maxsplit=1)
        Path.mkdir(args.output_dir, parents=True, exist_ok=True)
        with open(args.train_output_dir / "info.pkl", "wb") as f:
            pickle.dump(
                {"scale_config": {"model": model_arch}, "checkpoint": model_checkpoint},
                f,
            )

    # Read training information
    train_info_filename = args.train_output_dir / "info.pkl"
    # train_info = pickle.load(open(train_info_filename, "rb"))
    # we hack here.
    # train_info = {"scale_config": {"model": "ViT-L-14"}, "checkpoint": "/home/develop/datacomp/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin"}
    # this line should be the same
    train_info = {"scale_config": {"model": "ViT-L-14"}, "checkpoint": "datacomp_xl_s13b_b90k"}
    # for debug
    # train_info = {"scale_config": {"model": "ViT-L-14"}, "checkpoint": "laion2b_s32b_b82k"}
    # train_info = {"scale_config": {"model": "ViT-L-14"}, "checkpoint": "openai"}

    results_filename = args.output_dir / "eval_results.jsonl"

    # Get list of datasets
    with open(os.path.join(os.path.dirname(__file__), "tasklist.yml")) as f:
        tasks = yaml.safe_load(f)

    # Check for cached results
    results = {}
    cached_train_info_filename = args.output_dir / "info.pkl"
    if args.output_dir.exists() and cached_train_info_filename.exists():
        # If the output directory already exists, the training information should match.
        cached_train_info = pickle.load(open(cached_train_info_filename, "rb"))
        error_message = (
            "Error: output directory exists, but the training configs do not match. "
            "If you are re-using an output directory for evals, please be sure that "
            "the training output directory is consistent."
        )
        assert cached_train_info == train_info, error_message

        # Read existing results
        if results_filename.exists():
            with open(results_filename, "r") as f:
                lines = [json.loads(s) for s in f.readlines()]
                for line in lines:
                    if line["key"] not in tasks:
                        continue
                    results[line["dataset"]] = line
            print(f"Found {len(results)} eval result(s) in {results_filename}.")
    else:
        Path.mkdir(args.output_dir, parents=True, exist_ok=True)
        pickle.dump(train_info, open(cached_train_info_filename, "wb"))

    train_checkpoint = Path(train_info["checkpoint"])
    try:
        exists = Path(train_info["checkpoint"]).exists()
    except:
        exists = False
    # if not exists and args.use_model is None:
    #     print(
    #         "Warning, did not find or could not read checkpoint at",
    #         train_info["checkpoint"],
    #     )
    #     default_checkpoint_name = (
    #         args.train_output_dir / "checkpoints" / "epoch_latest.pt"
    #     )
    #     print("Defaulting to", default_checkpoint_name)
    #     train_info["checkpoint"] = default_checkpoint_name

    print("Evaluating")

    starttime = int(time.time())

    for task_key in tasks:
        task_name = tasks[task_key].get("name", task_key)
        if task_name in results:
            print(
                f"Skipping {task_name} since results are already in {results_filename}"
            )
        else:
            print(f"Evaluating on {task_name}")
            metrics = evaluate_model(
                task_key,
                train_info,
                args.data_dir,
                tasks[task_key].get("size"),
                batch_size=args.batch_size,
            )
            metrics["main_metric"] = metrics.get(
                tasks[task_key].get("main_metric", "acc1")
            )
            results[task_name] = {
                "key": task_key,
                "dataset": task_name,
                "metrics": metrics,
            }
            with open(results_filename, "a+") as f:
                f.write(json.dumps(results[task_name]) + "\n")

        if results[task_name]["metrics"]["main_metric"] is not None:
            print(f"Score: {results[task_name]['metrics']['main_metric']:.4f}")
        else:
            print(f"Score: No summary metric")

    elapsed = int(time.time()) - starttime
    print(
        f"Evaluation time: {elapsed // 3600} hour(s) {elapsed % 3600 // 60} minute(s) {elapsed % 60} second(s)"
    )
    print()
    print("=== Final results ===")
    for line in results.values():
        print(f"{line['dataset']}: {line['metrics']['main_metric']}")