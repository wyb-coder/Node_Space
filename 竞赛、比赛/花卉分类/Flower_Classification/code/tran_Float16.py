#!/usr/bin/env python3
"""Convert exported checkpoints to float16 storage in-place."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


def cast_state_dict(state_dict: Optional[Dict[str, Any]], dtype: torch.dtype) -> Optional[Dict[str, Any]]:
	if state_dict is None:
		return None
	converted: Dict[str, Any] = {}
	for key, value in state_dict.items():
		if torch.is_tensor(value):
			converted[key] = value.to(dtype)
		else:
			converted[key] = value
	return converted


def iter_checkpoint_paths(patterns: Iterable[str]) -> Iterable[Path]:
	seen = set()
	for pattern in patterns:
		for path in sorted(Path().glob(pattern)):
			if path.is_file() and path not in seen:
				seen.add(path)
				yield path


def process_file(path: Path, dtype: torch.dtype, backup: bool) -> None:
	checkpoint = torch.load(path, map_location="cpu")
	checkpoint["model_state"] = cast_state_dict(checkpoint.get("model_state"), dtype)
	if "ema_state" in checkpoint:
		checkpoint["ema_state"] = cast_state_dict(checkpoint.get("ema_state"), dtype)
	checkpoint["storage_dtype"] = str(dtype).replace("torch.", "")
	temp_path = path.with_suffix(path.suffix + ".tmp")
	torch.save(checkpoint, temp_path)
	if backup:
		backup_path = path.with_suffix(path.suffix + ".bak")
		if not backup_path.exists():
			shutil.copy2(path, backup_path)
	shutil.move(str(temp_path), str(path))
	print(f"[OK] Converted {path} -> {dtype} storage")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Convert checkpoints to float16 storage")
	parser.add_argument(
		"--pattern",
		action="append",
		default=["model/best_model_Fold*.pth"],
		help="Glob pattern(s) for checkpoint files",
	)
	parser.add_argument(
		"--dtype",
		default="float16",
		choices=["float16", "bfloat16"],
		help="Target storage dtype",
	)
	parser.add_argument(
		"--no-backup",
		action="store_true",
		help="Overwrite without creating .bak backup",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	target_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
	paths = list(iter_checkpoint_paths(args.pattern))
	if not paths:
		raise FileNotFoundError(f"No checkpoint files matched patterns: {args.pattern}")
	for path in paths:
		process_file(path, target_dtype, backup=not args.no_backup)


if __name__ == "__main__":
	main()
