#!/usr/bin/env python3
"""Unified training script for the flower classification challenge.

This script implements the end-to-end pipeline described in `ref/2.md`.
It handles configuration parsing, data preprocessing, class-imbalance
mitigation, progressive resizing, mixup/cutmix, EMA, early stopping, and
multi-fold training with automatic checkpointing.

The code intentionally lives in a single file for now; future refactors
can split the helpers into `model.py`, `utils.py`, etc.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.model_selection import StratifiedKFold
from timm import create_model
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets.folder import default_loader


# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
	"seed": 42,
	"experiment_name": "flower_convnextv2_base",
	"output_dir": "results/model",
	"results_dir": "results/results",
	"device": "cuda",
	"gpu": None,
	"best_model_path": "model/temp_best_model.pth",
	"data": {
		"root": "data/train",
		"csv": "data/train_labels.csv",
		"image_col": "filename",
		"label_col": "category_id",
		"num_classes": "auto",
		"n_splits": 5,
		"fold": -1,
		"val_split_seed": 42,
	},
	"sampler": {
		"type": "weighted_en",  # {none, weighted_en, class_balanced}
		"beta": 0.999,
		"oversample_cap": 3.0,
		"balanced_max_dup": 3.0,
	},
	"model": {
		"name": "swin_small",
		"backbone": "swin_small_patch4_window7_224",
		"pretrained_file": "swin-small-patch4-window7-224.bin",
		"drop_path": 0.18,
		"checkpoint": None,
		"pretrained_dir": "model/pretrained_model",
		"checkpoint_key": None,
		"head": "linear",
		"margin": 0.3,
		"scale": 30.0,
		"max_img_size": 224,
		"img_size_multiple": 28,
		"requires_fixed_size": True,
		"fixed_img_sizes": [224],
	},
	"models": [
		{
			"name": "swin_small",
			"backbone": "swin_small_patch4_window7_224",
			"pretrained_file": "swin-small-patch4-window7-224.bin",
			"drop_path": 0.18,
			"checkpoint": None,
			"pretrained_dir": "model/pretrained_model",
			"checkpoint_key": None,
			"head": "linear",
			"margin": 0.3,
			"scale": 30.0,
			"max_img_size": 224,
			"img_size_multiple": 28,
			"requires_fixed_size": True,
			"fixed_img_sizes": [224],
		},
		{
			"name": "maxvit_tiny",
			"backbone": "maxvit_tiny_tf_224.in1k",
			"pretrained_file": "maxvit_tiny_tf_224.in1k.bin",
			"drop_path": 0.12,
			"checkpoint": None,
			"pretrained_dir": "model/pretrained_model",
			"checkpoint_key": None,
			"head": "linear",
			"margin": 0.3,
			"scale": 30.0,
			"max_img_size": 448,
			"img_size_multiple": 112,
			"requires_fixed_size": False,
		},
	],
	"opt": {
		"name": "adamw",
		"lr_ref": 1.2e-4,
		"lr_min": 5.0e-6,
		"weight_decay": 0.05,
		"momentum": 0.9,
		"betas": [0.9, 0.999],
		"clip_grad": 1.0,
	},
	"train": {
		"epochs": 40,
		"batch_per_gpu": 64,
		"world_size": 1,
		"grad_accum": 1,
		"num_workers": 4,
		"pin_memory": True,
		"stages": [
			{"start_epoch": 0, "epochs": 15, "img_size": 224, "mixup": 0.2, "cutmix": 0.8},
			{"start_epoch": 15, "epochs": 25, "img_size": 448, "mixup": 0.2, "cutmix": 0.8, "batch_per_gpu": 16},
		],
		"label_smoothing": 0.05,
		"use_amp": True,
	},
	"sched": {
		"warmup_epochs": 5,
		"cosine_epochs": 35,
	},
	"loss": {
		"type": "lsce",  # {lsce, focal, balanced_softmax}
		"focal_gamma": 1.5,
	},
	"ema": {
		"enabled": True,
		"decay": 0.9995,
	},
	"robust": {
		"enabled": False,
		"kinds": ["gaussian_noise", "blur", "brightness", "jpeg", "occlusion"],
		"level": "low",
	},
	"es": {
		"enabled": True,
		"patience": 10,
		"min_delta": 0.001,
		"metric": "combo",
		"alpha": 0.7,
	},
	"logging": {
		"print_freq": 5,
		"log_interval_steps": 50,
	},
	"llrd": {
		"enabled": True,
		"multipliers": [0.25, 0.5, 1.0, 1.5],
	},
	"post_export": {
		"enabled": True,
		"dtype": "float16",
		"patterns": ["model/best_model_Fold*.pth"],
		"backup": False,
	},
}


@dataclass
class DistributedContext:
	is_distributed: bool = False
	rank: int = 0
	world_size: int = 1
	local_rank: int = 0

	@property
	def is_main_process(self) -> bool:
		return self.rank == 0


def is_distributed_active() -> bool:
	return dist.is_available() and dist.is_initialized()


def init_distributed_mode(cfg: Dict[str, Any]) -> DistributedContext:
	if not dist.is_available():
		cfg["train"]["world_size"] = cfg["train"].get("world_size", 1)
		return DistributedContext()
	required_env = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
	if not all(key in os.environ for key in required_env):
		cfg["train"]["world_size"] = cfg["train"].get("world_size", 1)
		return DistributedContext()
	rank = int(os.environ["RANK"])
	world_size = int(os.environ["WORLD_SIZE"])
	local_rank = int(os.environ["LOCAL_RANK"])
	backend = "nccl" if torch.cuda.is_available() else "gloo"
	cfg["train"]["world_size"] = world_size
	cfg["gpu"] = local_rank
	cfg["device"] = "cuda" if torch.cuda.is_available() else cfg.get("device", "cpu")
	if dist.is_initialized():
		if torch.cuda.is_available():
			torch.cuda.set_device(local_rank)
		return DistributedContext(True, rank, world_size, local_rank)
	dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=rank)
	if torch.cuda.is_available():
		torch.cuda.set_device(local_rank)
	return DistributedContext(True, rank, world_size, local_rank)


def cleanup_distributed(dist_ctx: DistributedContext) -> None:
	if dist_ctx.is_distributed and dist.is_initialized():
		try:
			dist.barrier()
		except RuntimeError:
			pass
		dist.destroy_process_group()


def unwrap_ddp(model: nn.Module) -> nn.Module:
	if isinstance(model, nn.parallel.DistributedDataParallel):
		return model.module
	return model


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train flower classification models")
	parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
	parser.add_argument("--override", type=str, default=None,
						help="JSON string to override config fields")
	parser.add_argument("--fold", type=int, default=None,
						help="Override fold index (0-based). Use -1 for all folds")
	parser.add_argument("--debug", action="store_true", help="Run in quick debug mode")
	parser.add_argument("--gpu", type=int, default=None,
					help="Specify CUDA device index to use (e.g., 0)")
	parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (automatically set by torchrun)")
	return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
	cfg = deepcopy(DEFAULT_CONFIG)
	if args.config:
		with open(args.config, "r", encoding="utf-8") as f:
			user_cfg = yaml.safe_load(f)
		cfg = deep_update(cfg, user_cfg)
	if args.override:
		override_cfg = json.loads(args.override)
		cfg = deep_update(cfg, override_cfg)
	if args.fold is not None:
		cfg["data"]["fold"] = args.fold
	if args.debug:
		cfg["train"]["epochs"] = 2
		cfg["train"]["batch_per_gpu"] = 8
		cfg["train"]["num_workers"] = 0
		cfg["sampler"]["type"] = "none"
	if args.gpu is not None:
		cfg["gpu"] = args.gpu
		cfg["device"] = "cuda"
	if not cfg.get("models"):
		cfg["models"] = [deepcopy(cfg["model"])]
	normalized_models = []
	for idx, model_cfg in enumerate(cfg["models"]):
		model_copy = deepcopy(cfg["model"])
		model_copy.update(model_cfg or {})
		model_copy.setdefault("name", f"model_{idx}")
		model_copy.setdefault("head", "linear")
		model_copy.setdefault("margin", 0.3)
		model_copy.setdefault("scale", 30.0)
		model_copy.setdefault("pretrained_dir", cfg["model"].get("pretrained_dir", "model/pretrained_model"))
		normalized_models.append(model_copy)
	cfg["models"] = normalized_models
	cfg["model"] = deepcopy(cfg["models"][0])
	warmup_epochs = cfg.get("sched", {}).get("warmup_epochs", 0)
	cosine_epochs = cfg.get("sched", {}).get("cosine_epochs")
	if cosine_epochs is None or cosine_epochs <= 0:
		cfg["sched"]["cosine_epochs"] = max(1, cfg["train"]["epochs"] - warmup_epochs)
	else:
		cfg["sched"]["cosine_epochs"] = max(1, min(cosine_epochs, cfg["train"]["epochs"] - warmup_epochs))
	return cfg


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
	for k, v in updates.items():
		if isinstance(v, dict) and isinstance(base.get(k), dict):
			base[k] = deep_update(base[k], v)
		else:
			base[k] = v
	return base


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int, rank: int = 0) -> None:
	combined_seed = seed + rank
	random.seed(combined_seed)
	np.random.seed(combined_seed)
	torch.manual_seed(combined_seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(combined_seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def get_device(cfg: Dict[str, Any]) -> torch.device:
	if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available():
		desired_gpu = cfg.get("gpu")
		if desired_gpu is not None:
			num_devices = torch.cuda.device_count()
			if desired_gpu < 0 or desired_gpu >= num_devices:
				raise ValueError(f"请求的GPU编号{desired_gpu}超出范围，当前可用设备数量为{num_devices}。")
			torch.cuda.set_device(desired_gpu)
			return torch.device(f"cuda:{desired_gpu}")
		return torch.device("cuda")
	return torch.device("cpu")


def auto_world_size(cfg: Dict[str, Any]) -> int:
	world = cfg["train"].get("world_size", 1)
	if world <= 0:
		world = torch.cuda.device_count() or 1
	return world


def compute_scaled_lr(cfg: Dict[str, Any]) -> float:
	train_cfg = cfg["train"]
	world = auto_world_size(cfg)
	global_batch = train_cfg["batch_per_gpu"] * world * train_cfg["grad_accum"]
	lr = cfg["opt"]["lr_ref"] * (global_batch / 256.0)
	return lr


def ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path) -> None:
	with path.open("w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False)


def resolve_pretrained_root(path_value: str) -> Path:
	if not path_value:
		return Path()
	raw_path = Path(path_value)
	if raw_path.is_absolute() and raw_path.exists():
		return raw_path
	candidates: List[Path] = []
	if raw_path.is_absolute():
		candidates.append(raw_path)
	else:
		candidates.append(Path.cwd() / raw_path)
		script_root = Path(__file__).resolve().parent.parent
		candidates.append(script_root / raw_path)
		candidates.append(raw_path)
	seen: Set[Path] = set()
	for candidate in candidates:
		if candidate in seen:
			continue
		seen.add(candidate)
		if candidate.exists():
			return candidate
	return candidates[0] if candidates else raw_path


def enumerate_pretrained_candidates(model_cfg: Dict[str, Any]) -> Tuple[Path, List[Path]]:
	pretrained_dir = resolve_pretrained_root(model_cfg.get("pretrained_dir", ""))
	backbone_name = model_cfg["backbone"]
	preferred_paths: List[Path] = []
	if model_cfg.get("pretrained_file"):
		preferred_paths.append(pretrained_dir / model_cfg["pretrained_file"])
	file_candidates: Set[Path] = set()
	file_candidates.add(pretrained_dir / f"{backbone_name}.bin")
	file_candidates.add(pretrained_dir / f"{backbone_name}.pth")
	name_alias = model_cfg.get("name")
	if name_alias:
		file_candidates.add(pretrained_dir / f"{name_alias}.bin")
		file_candidates.add(pretrained_dir / f"{name_alias}.pth")
	possible_dir_names: Set[Path] = set()
	possible_dir_names.add(pretrained_dir / backbone_name)
	if name_alias:
		possible_dir_names.add(pretrained_dir / name_alias)
	if "/" in backbone_name:
		possible_dir_names.add(pretrained_dir / backbone_name.replace("/", os.sep))
	for file_path in file_candidates:
		preferred_paths.append(file_path)
	for dir_path in possible_dir_names:
		if dir_path.is_file() and dir_path.name == "pytorch_model.bin":
			preferred_paths.append(dir_path)
		elif dir_path.is_dir():
			preferred_paths.append(dir_path / "pytorch_model.bin")
	preferred_paths.append(pretrained_dir / "pytorch_model.bin")
	return pretrained_dir, preferred_paths


def existing_pretrained_files(model_cfg: Dict[str, Any]) -> List[Path]:
	_, candidates = enumerate_pretrained_candidates(model_cfg)
	seen: Set[Path] = set()
	existing: List[Path] = []
	for candidate in candidates:
		candidate_resolved = candidate.resolve()
		if candidate_resolved in seen:
			continue
		seen.add(candidate_resolved)
		if candidate_resolved.exists() and candidate_resolved.is_file():
			existing.append(candidate_resolved)
	return existing


def load_weight_state(candidate: Path) -> Dict[str, Any]:
	def is_git_lfs_placeholder(path: Path) -> bool:
		try:
			with path.open("rb") as f:
				head = f.read(128)
		except OSError:
			return False
		return head.startswith(b"version https://git-lfs.github.com/spec/v1")

	if is_git_lfs_placeholder(candidate):
		raise RuntimeError(
			f"检测到 {candidate.name} 仍是 Git LFS 占位符，请先在本地同步真实权重文件后再重试。"
		)
	allowed_ext = {".bin", ".pth", ".pt"}
	if candidate.suffix.lower() not in allowed_ext:
		raise RuntimeError(
			f"仅支持加载后缀为 {sorted(allowed_ext)} 的本地权重文件，当前文件为 {candidate.name}"
		)
	try:
		return torch.load(candidate, map_location="cpu")
	except (pickle.UnpicklingError, RuntimeError, AttributeError, EOFError) as torch_exc:
		raise RuntimeError(f"无法用 torch.load 解析 {candidate.name}: {torch_exc}") from torch_exc


def sanitize_state_dict(state: Dict[str, Any], target_model: nn.Module) -> Tuple[Dict[str, Any], List[str]]:
	if not isinstance(state, dict):
		return state, []
	target_state = target_model.state_dict()
	filtered: Dict[str, Any] = {}
	removed_keys: List[str] = []
	for key, value in state.items():
		if not isinstance(value, torch.Tensor) or key not in target_state:
			filtered[key] = value
			continue
		target_tensor = target_state[key]
		if value.shape != target_tensor.shape:
			removed_keys.append(key)
			continue
		filtered[key] = value
	return filtered, removed_keys


class AverageMeter:
	def __init__(self) -> None:
		self.reset()

	def reset(self) -> None:
		self.sum = 0.0
		self.count = 0.0

	def update(self, value: float, n: int = 1) -> None:
		self.sum += value * n
		self.count += float(n)

	def sync(self, device: torch.device) -> None:
		if not is_distributed_active():
			return
		tensor = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
		dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
		self.sum = tensor[0].item()
		self.count = tensor[1].item()

	@property
	def avg(self) -> float:
		return self.sum / max(1, self.count)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


# ---------------------------------------------------------------------------
# Dataset and sampling
# ---------------------------------------------------------------------------


class FlowerDataset(Dataset):
	def __init__(self,
				 df: pd.DataFrame,
				 image_root: Path,
				 label_map: Dict[int, int],
				 transforms: Callable[[Any], Any]) -> None:
		self.df = df.reset_index(drop=True)
		self.image_root = image_root
		self.label_map = label_map
		self.transforms = transforms

	def set_transforms(self, transforms: Callable[[Any], Any]) -> None:
		self.transforms = transforms

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		row = self.df.iloc[idx]
		img_path = self.image_root / row["__image_name__"]
		image = default_loader(str(img_path))
		if self.transforms:
			image = self.transforms(image)
		label = self.label_map[row["__label_id__"]]
		return {"image": image, "label": label, "index": idx}


def prepare_dataframe(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
	data_cfg = cfg["data"]
	csv_path = Path(data_cfg["csv"])
	df = pd.read_csv(csv_path)
	image_col = data_cfg["image_col"]
	label_col = data_cfg["label_col"]
	if image_col not in df.columns or label_col not in df.columns:
		raise ValueError(f"Missing required columns: {image_col}, {label_col}")
	df = df.rename(columns={image_col: "__image_name__", label_col: "__label_id__"})
	unique_labels = sorted(df["__label_id__"].unique().tolist())
	label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
	index_to_label = {v: k for k, v in label_to_index.items()}
	return df, label_to_index, index_to_label


def compute_class_counts(df: pd.DataFrame) -> Dict[int, int]:
	counts = df["__label_id__"].value_counts().to_dict()
	return {int(k): int(v) for k, v in counts.items()}


def build_weighted_sampler(labels: List[int], beta: float) -> WeightedRandomSampler:
	labels_tensor = torch.tensor(labels, dtype=torch.long)
	label_unique, counts = torch.unique(labels_tensor, return_counts=True)
	counts = counts.float()
	beta_tensor = torch.tensor(beta, dtype=torch.float32)
	effective_num = (1.0 - beta_tensor ** counts) / (1.0 - beta_tensor)
	weights = (1.0 / effective_num)
	weights = weights / weights.sum() * len(label_unique)
	sample_weights = weights[labels_tensor]
	return WeightedRandomSampler(sample_weights.double(), len(sample_weights), replacement=True)


class ClassBalancedSampler(Sampler[int]):
	def __init__(self, labels: List[int], max_dup: float = 3.0) -> None:
		self.labels = labels
		self.max_dup = max_dup
		self.indices = self._build_indices()

	def _build_indices(self) -> List[int]:
		label_to_indices: Dict[int, List[int]] = defaultdict(list)
		for idx, label in enumerate(self.labels):
			label_to_indices[label].append(idx)
		max_count = max(len(v) for v in label_to_indices.values())
		all_indices: List[int] = []
		for label, idxs in label_to_indices.items():
			desired = min(int(math.ceil(max_count / len(idxs))), int(self.max_dup))
			replicated = idxs * desired
			all_indices.extend(replicated)
		random.shuffle(all_indices)
		return all_indices

	def __iter__(self) -> Iterable[int]:
		if not self.indices:
			self.indices = self._build_indices()
		random.shuffle(self.indices)
		return iter(self.indices)

	def __len__(self) -> int:
		return len(self.indices)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


class BalancedSoftmaxLoss(nn.Module):
	def __init__(self, class_counts: List[int]) -> None:
		super().__init__()
		counts = torch.tensor(class_counts, dtype=torch.float32)
		self.register_buffer("log_prior", torch.log(counts / counts.sum()))

	def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		balanced_logits = logits + self.log_prior
		loss = F.cross_entropy(balanced_logits, target)
		return loss


class FocalLoss(nn.Module):
	def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.0) -> None:
		super().__init__()
		self.gamma = gamma
		self.label_smoothing = label_smoothing

	def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		log_probs = F.log_softmax(logits, dim=-1)
		probs = log_probs.exp()
		target_one_hot = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
		if self.label_smoothing > 0:
			target_one_hot = target_one_hot * (1 - self.label_smoothing) + self.label_smoothing / target_one_hot.size(-1)
		loss = -(1 - probs) ** self.gamma * target_one_hot * log_probs
		return loss.sum(dim=1).mean()


def build_criterion(cfg: Dict[str, Any], class_counts: List[int]) -> nn.Module:
	loss_cfg = cfg["loss"]
	smoothing = cfg["train"]["label_smoothing"]
	if loss_cfg["type"].lower() == "lsce":
		return LabelSmoothingCrossEntropy(smoothing=smoothing)
	if loss_cfg["type"].lower() == "focal":
		return FocalLoss(gamma=loss_cfg.get("focal_gamma", 1.5), label_smoothing=smoothing)
	if loss_cfg["type"].lower() == "balanced_softmax":
		return BalancedSoftmaxLoss(class_counts)
	raise ValueError(f"Unknown loss type: {loss_cfg['type']}")


# ---------------------------------------------------------------------------
# EMA & Early Stopping
# ---------------------------------------------------------------------------


class ModelEMA:
	def __init__(self, model: nn.Module, decay: float) -> None:
		self.decay = decay
		self.ema_model = deepcopy(model).eval()
		device = next(model.parameters()).device
		self.ema_model.to(device)
		for p in self.ema_model.parameters():
			p.requires_grad_(False)

	def update(self, model: nn.Module) -> None:
		with torch.no_grad():
			msd = model.state_dict()
			for k, v in self.ema_model.state_dict().items():
				if k in msd:
					v.copy_(v * self.decay + msd[k].to(v.device) * (1.0 - self.decay))

	def state_dict(self) -> Dict[str, torch.Tensor]:
		return self.ema_model.state_dict()

	def eval_model(self) -> nn.Module:
		return self.ema_model


class EarlyStopping:
	def __init__(self, patience: int, min_delta: float) -> None:
		self.patience = patience
		self.min_delta = min_delta
		self.best_score = -float("inf")
		self.counter = 0

	def step(self, score: float) -> bool:
		if score > self.best_score + self.min_delta:
			self.best_score = score
			self.counter = 0
			return False
		self.counter += 1
		return self.counter >= self.patience


# ---------------------------------------------------------------------------
# Transforms and schedulers
# ---------------------------------------------------------------------------


class AddGaussianNoise:
	def __init__(self, std: float) -> None:
		self.std = std

	def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
		noise = torch.randn_like(tensor) * self.std
		return (tensor + noise).clamp(0.0, 1.0)


class BrightnessScale:
	def __init__(self, factor: float) -> None:
		self.factor = factor

	def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
		return (tensor * self.factor).clamp(0.0, 1.0)


class ClampTensor:
	def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
		return tensor.clamp(0.0, 1.0)


class JPEGCompression:
	def __init__(self, quality: int) -> None:
		self.quality = quality

	def __call__(self, image: Any) -> Any:
		if isinstance(image, torch.Tensor):
			image = TF.to_pil_image(image)
		if not isinstance(image, Image.Image):
			raise TypeError("JPEGCompression expects PIL image input")
		buffer = BytesIO()
		image.save(buffer, format="JPEG", quality=self.quality)
		buffer.seek(0)
		return Image.open(buffer).convert("RGB")


def build_corruption_transform(img_size: int, kind: str, level: str) -> Callable[[Any], Any]:
	resize_size = int(img_size * 1.14)
	level = level.lower()
	noise_std = {"low": 0.03, "medium": 0.06, "high": 0.1}
	blur_sigma = {"low": (0.5, 1.0), "medium": (0.8, 1.5), "high": (1.0, 2.0)}
	brightness_factor = {"low": 0.9, "medium": 0.8, "high": 0.7}
	jpeg_quality = {"low": 85, "medium": 65, "high": 45}
	base_ops: List[Callable[[Any], Any]] = [
		T.Resize(resize_size),
		T.CenterCrop(img_size),
	]
	tensor_ops: List[Callable[[Any], Any]] = [T.ToTensor()]
	kind_lower = kind.lower()
	if kind_lower == "gaussian_noise":
		std = noise_std.get(level, noise_std["low"])
		tensor_ops.append(AddGaussianNoise(std))
		tensor_ops.append(ClampTensor())
	elif kind_lower == "blur":
		sigma = blur_sigma.get(level, blur_sigma["low"])
		tensor_ops.append(T.GaussianBlur(kernel_size=5, sigma=sigma))
	elif kind_lower == "brightness":
		factor = brightness_factor.get(level, brightness_factor["low"])
		tensor_ops.append(BrightnessScale(factor))
		tensor_ops.append(ClampTensor())
	elif kind_lower == "jpeg":
		quality = jpeg_quality.get(level, jpeg_quality["low"])
		base_ops.insert(0, JPEGCompression(quality))
	elif kind_lower == "occlusion":
		tensor_ops.append(T.RandomErasing(p=1.0, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0.0))
	normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	return T.Compose(base_ops + tensor_ops + [normalize])


# ---------------------------------------------------------------------------
# Models and heads
# ---------------------------------------------------------------------------


class ArcMarginProduct(nn.Module):
	def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.3, easy_margin: bool = False) -> None:
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.s = s
		self.m = m
		self.easy_margin = easy_margin
		self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
		nn.init.xavier_uniform_(self.weight)
		self.cos_m = math.cos(m)
		self.sin_m = math.sin(m)
		self.th = math.cos(math.pi - m)
		self.mm = math.sin(math.pi - m) * m

	def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		cosine = F.linear(F.normalize(features), F.normalize(self.weight))
		sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=1e-6))
		phi = cosine * self.cos_m - sine * self.sin_m
		if not self.easy_margin:
			phi = torch.where(cosine > self.th, phi, cosine - self.mm)
		else:
			phi = torch.where(cosine > 0, phi, cosine)
		one_hot = torch.zeros_like(cosine)
		one_hot.scatter_(1, labels.view(-1, 1), 1.0)
		logits = one_hot * phi + (1.0 - one_hot) * cosine
		return logits * self.s

	def inference(self, features: torch.Tensor) -> torch.Tensor:
		cosine = F.linear(F.normalize(features), F.normalize(self.weight))
		return cosine * self.s


class CosMarginProduct(nn.Module):
	def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35) -> None:
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.s = s
		self.m = m
		self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
		nn.init.xavier_uniform_(self.weight)

	def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		cosine = F.linear(F.normalize(features), F.normalize(self.weight))
		one_hot = torch.zeros_like(cosine)
		one_hot.scatter_(1, labels.view(-1, 1), 1.0)
		logits = cosine - one_hot * self.m
		return logits * self.s

	def inference(self, features: torch.Tensor) -> torch.Tensor:
		cosine = F.linear(F.normalize(features), F.normalize(self.weight))
		return cosine * self.s


class MetricModelWrapper(nn.Module):
	def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
		super().__init__()
		self.backbone = backbone
		self.head = head
		self.requires_targets = True

	def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
		features = self.backbone.forward_features(x)
		features = self.backbone.forward_head(features, pre_logits=True)
		if isinstance(features, (list, tuple)):
			features = features[0]
		if features.ndim > 2:
			features = torch.flatten(features, 1)
		return features

	def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
		features = self._extract_features(x)
		if self.training or targets is not None:
			if targets is None:
				raise ValueError("Targets are required for metric head during training/evaluation.")
			return self.head(features, targets)
		return self.head.inference(features)


def model_requires_targets(model: nn.Module) -> bool:
	return bool(getattr(unwrap_ddp(model), "requires_targets", False))


def forward_model(model: nn.Module, images: torch.Tensor, targets: Optional[torch.Tensor]) -> torch.Tensor:
	if model_requires_targets(model):
		if targets is None:
			raise ValueError("Targets are required for metric head during training/evaluation.")
		return model(images, targets)
	return model(images)


def build_train_transform(img_size: int) -> Callable[[Any], Any]:
	return T.Compose([
		T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
		T.RandomHorizontalFlip(p=0.5),
		T.RandomVerticalFlip(p=0.1),
		T.RandomRotation(degrees=10),
		T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		T.RandomErasing(p=0.1, value="random"),
	])


def build_eval_transform(img_size: int) -> Callable[[Any], Any]:
	resize_size = int(img_size * 1.14)
	return T.Compose([
		T.Resize(resize_size),
		T.CenterCrop(img_size),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])


@dataclass
class Stage:
	start_epoch: int
	epochs: int
	img_size: int
	mixup: float
	cutmix: float
	batch_per_gpu: Optional[int] = None

	@property
	def end_epoch(self) -> int:
		return self.start_epoch + self.epochs


class StageScheduler:
	def __init__(self, stages_cfg: List[Dict[str, Any]]) -> None:
		self.stages = [Stage(**stage) for stage in stages_cfg]
		self.stages.sort(key=lambda s: s.start_epoch)

	def get_stage(self, epoch: int) -> Stage:
		for stage in reversed(self.stages):
			if epoch >= stage.start_epoch:
				return stage
		return self.stages[0]


def adjust_image_size_for_model(img_size: int, model_cfg: Dict[str, Any], dist_ctx: Optional[DistributedContext]) -> int:
	original_size = img_size
	max_size = model_cfg.get("max_img_size")
	if max_size is not None:
		img_size = min(img_size, int(max_size))
	if model_cfg.get("requires_fixed_size"):
		choices = model_cfg.get("fixed_img_sizes") or []
		choices = [int(choice) for choice in choices if choice]
		if choices:
			closest = min(choices, key=lambda x: abs(x - original_size))
			img_size = closest
	else:
		multiple = model_cfg.get("img_size_multiple")
		if multiple:
			multiple = int(multiple)
			if multiple > 0:
				remainder = img_size % multiple
				if remainder != 0 or img_size < multiple:
					lower = img_size - remainder if remainder != 0 else img_size
					upper = img_size + (multiple - remainder if remainder != 0 else 0)
					candidates: List[int] = []
					if lower >= multiple:
						candidates.append(lower)
					if upper != img_size and (max_size is None or upper <= max_size):
						candidates.append(upper)
					if not candidates:
						img_size = max(multiple, lower if lower >= multiple else upper)
					else:
						img_size = min(candidates, key=lambda x: abs(x - original_size))
	if img_size != original_size:
		should_log = dist_ctx is None or (not dist_ctx.is_distributed or dist_ctx.is_main_process)
		if should_log:
			model_name = model_cfg.get("name") or model_cfg.get("backbone")
			print(f"[img_size] 调整 {model_name} 阶段图像尺寸: {original_size} -> {img_size}")
	return img_size


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]) -> SequentialLR:
	opt_cfg = cfg["opt"]
	sched_cfg = cfg["sched"]
	total_epochs = cfg["train"]["epochs"]
	warmup_epochs = sched_cfg.get("warmup_epochs", 5)
	cosine_epochs = sched_cfg.get("cosine_epochs")
	if cosine_epochs is None or cosine_epochs <= 0:
		cosine_epochs = max(1, total_epochs - warmup_epochs)
	min_lr = opt_cfg.get("lr_min", 5e-6)
	warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
	cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=min_lr)
	scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
	return scheduler


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------


def build_model_instance(cfg: Dict[str, Any],
					 model_cfg: Dict[str, Any],
					 num_classes: int,
					 device: torch.device) -> nn.Module:
	backbone_name = model_cfg["backbone"]
	pretrained = model_cfg.get("pretrained", True)
	pretrained = False
	drop_path = model_cfg.get("drop_path", cfg.get("model", {}).get("drop_path", 0.0))
	global_pool = model_cfg.get("global_pool")
	create_kwargs: Dict[str, Any] = {
		"drop_path_rate": drop_path,
	}
	if global_pool is not None:
		create_kwargs["global_pool"] = global_pool
	head_type = (model_cfg.get("head") or "linear").lower()
	if head_type == "linear":
		model = create_model(backbone_name, pretrained=pretrained, num_classes=num_classes, **create_kwargs)
		setattr(model, "requires_targets", False)
	else:
		backbone = create_model(backbone_name, pretrained=pretrained, num_classes=0, **create_kwargs)
		feature_dim = getattr(backbone, "num_features", None)
		if feature_dim is None:
			raise ValueError(f"Backbone {backbone_name} does not expose num_features for metric head")
		scale = model_cfg.get("scale", 30.0)
		margin = model_cfg.get("margin", 0.3)
		if head_type == "arcface":
			head = ArcMarginProduct(feature_dim, num_classes, s=scale, m=margin)
		elif head_type == "cosface":
			head = CosMarginProduct(feature_dim, num_classes, s=scale, m=margin)
		else:
			raise ValueError(f"Unsupported head type: {head_type}")
		model = MetricModelWrapper(backbone, head)
	model.model_name = model_cfg.get("name", backbone_name)
	model.backbone_name = backbone_name
	model.to(device)
	return model


def build_param_groups(model: nn.Module,
					 base_lr: float,
					 cfg: Dict[str, Any],
					 opt_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
	llrd_cfg = cfg.get("llrd", {})
	weight_decay = opt_cfg.get("weight_decay", 0.0)
	if not llrd_cfg.get("enabled", False):
		params = [p for p in model.parameters() if p.requires_grad]
		return [{"params": params, "lr": base_lr, "weight_decay": weight_decay}]
	multipliers = llrd_cfg.get("multipliers", [0.25, 0.5, 1.0, 1.5])
	if not multipliers:
		multipliers = [1.0]
	body_mults = multipliers[:-1] if len(multipliers) > 1 else [multipliers[0]]
	head_mult = multipliers[-1]
	groups: List[Dict[str, Any]] = []
	assigned: Set[int] = set()

	def add_group(parameters: Iterable[torch.nn.Parameter], lr_mult: float) -> None:
		group_params: List[torch.nn.Parameter] = []
		for param in parameters:
			if not param.requires_grad:
				continue
			pid = id(param)
			if pid in assigned:
				continue
			assigned.add(pid)
			group_params.append(param)
		if group_params:
			groups.append({
				"params": group_params,
				"lr": base_lr * lr_mult,
				"weight_decay": weight_decay,
			})

	def resolve_head_parameters(mod: nn.Module) -> Iterable[torch.nn.Parameter]:
		if hasattr(mod, "get_classifier"):
			classifier = mod.get_classifier()
			if isinstance(classifier, nn.Module):
				return classifier.parameters()
			if isinstance(classifier, str):
				head_module = getattr(mod, classifier, None)
				if isinstance(head_module, nn.Module):
					return head_module.parameters()
		if hasattr(mod, "head") and isinstance(mod.head, nn.Module):
			return mod.head.parameters()
		return []

	body_source = model.backbone if isinstance(model, MetricModelWrapper) else model
	if isinstance(model, MetricModelWrapper):
		add_group(model.head.parameters(), head_mult)
	else:
		head_params = resolve_head_parameters(model)
		add_group(head_params, head_mult)

	stage_modules: List[nn.Module] = []
	if hasattr(body_source, "stages"):
		stage_modules = list(body_source.stages)
	elif hasattr(body_source, "blocks"):
		stage_modules = list(body_source.blocks)
	if stage_modules:
		segments = np.array_split(range(len(stage_modules)), max(1, len(body_mults)))
		for seg_idx, indices in enumerate(segments):
			params: List[torch.nn.Parameter] = []
			for idx in indices.tolist():
				params.extend(stage_modules[idx].parameters())
			lr_mult = body_mults[min(seg_idx, len(body_mults) - 1)]
			add_group(params, lr_mult)
	else:
		children = list(body_source.children())
		if children:
			segments = np.array_split(range(len(children)), max(1, len(body_mults)))
			for seg_idx, indices in enumerate(segments):
				params = []
				for idx in indices.tolist():
					params.extend(children[idx].parameters())
				lr_mult = body_mults[min(seg_idx, len(body_mults) - 1)]
				add_group(params, lr_mult)

	remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in assigned]
	if remaining:
		fallback_mult = body_mults[-1] if body_mults else 1.0
		add_group(remaining, fallback_mult)
	return groups


# ---------------------------------------------------------------------------
# Training and evaluation loops
# ---------------------------------------------------------------------------


def run_epoch(model: nn.Module,
			  loader: DataLoader,
			  criterion: nn.Module,
		  soft_criterion: Optional[nn.Module],
			  optimizer: torch.optim.Optimizer,
			  device: torch.device,
			  epoch: int,
			  scaler: Optional[GradScaler],
			  mixup_fn: Optional[Mixup],
			  ema: Optional[ModelEMA],
			  cfg: Dict[str, Any]) -> Dict[str, float]:
	model.train()
	meters = {
		"loss": AverageMeter(),
		"acc1": AverageMeter(),
	}
	grad_accum = cfg["train"]["grad_accum"]
	print_freq = cfg["logging"].get("print_freq", 50)
	optimizer.zero_grad(set_to_none=True)
	for step, batch in enumerate(loader):
		images = batch["image"].to(device)
		targets = batch["label"].to(device)
		targets_hard = targets.clone()
		if mixup_fn is not None:
			images, targets = mixup_fn(images, targets)
		with autocast(enabled=cfg["train"].get("use_amp", True) and device.type == "cuda"):
			outputs = forward_model(model, images, targets if model_requires_targets(model) else None)
			if mixup_fn is not None and targets.is_floating_point():
				if soft_criterion is None:
					raise RuntimeError("Soft target criterion required when mixup/cutmix is enabled.")
				loss = soft_criterion(outputs, targets)
			else:
				loss = criterion(outputs, targets)
			loss = loss / grad_accum
		if scaler is not None:
			scaler.scale(loss).backward()
		else:
			loss.backward()
		if (step + 1) % grad_accum == 0:
			if cfg["opt"].get("clip_grad", 0) > 0:
				if scaler is not None:
					scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["opt"]["clip_grad"])
			if scaler is not None:
				scaler.step(optimizer)
				scaler.update()
			else:
				optimizer.step()
			optimizer.zero_grad(set_to_none=True)
			if ema is not None:
				ema.update(unwrap_ddp(model))
		meters["loss"].update(loss.item() * grad_accum, targets_hard.size(0))
		acc1 = accuracy(outputs, targets_hard, topk=(1,))[0]
		meters["acc1"].update(acc1.item(), targets_hard.size(0))
		if (step + 1) % print_freq == 0 and (not is_distributed_active() or dist.get_rank() == 0):
			print(
				f"Epoch {epoch:03d} | step {step+1}/{len(loader)} | loss={meters['loss'].avg:.4f} | acc1={meters['acc1'].avg:.2f}"
			)
	for meter in meters.values():
		meter.sync(device)
	return {
		"loss": meters["loss"].avg,
		"acc1": meters["acc1"].avg,
	}


@torch.no_grad()
def evaluate(model: nn.Module,
			 loader: DataLoader,
			 criterion: nn.Module,
			 device: torch.device) -> Dict[str, float]:
	model.eval()
	loss_meter = AverageMeter()
	top1_meter = AverageMeter()
	for batch in loader:
		images = batch["image"].to(device)
		targets = batch["label"].to(device)
		outputs = forward_model(model, images, targets if model_requires_targets(model) else None)
		loss = criterion(outputs, targets)
		acc1 = accuracy(outputs, targets, topk=(1,))[0]
		loss_meter.update(loss.item(), targets.size(0))
		top1_meter.update(acc1.item(), targets.size(0))
	loss_meter.sync(device)
	top1_meter.sync(device)
	return {"loss": loss_meter.avg, "acc1": top1_meter.avg}


def build_mixup(stage: Stage, num_classes: int) -> Optional[Mixup]:
	if stage.mixup <= 0 and stage.cutmix <= 0:
		return None
	return Mixup(
		mixup_alpha=stage.mixup,
		cutmix_alpha=stage.cutmix,
		prob=1.0 if stage.mixup > 0 or stage.cutmix > 0 else 0.0,
		switch_prob=0.5,
		mode="batch",
		label_smoothing=0.0,
		num_classes=num_classes,
	)


def prepare_dataloader(dataset: FlowerDataset,
					   cfg: Dict[str, Any],
					   indices: Optional[List[int]],
					   sampler_config: Dict[str, Any],
				   is_train: bool,
			   batch_size_override: Optional[int] = None,
			   dist_ctx: Optional[DistributedContext] = None) -> Tuple[DataLoader, Optional[List[int]], Optional[Sampler[int]]]:
	batch_size = batch_size_override or cfg["train"]["batch_per_gpu"]
	num_workers = cfg["train"].get("num_workers", 4)
	pin_memory = cfg["train"].get("pin_memory", True)
	if indices is not None:
		sub_df = dataset.df.iloc[indices].reset_index(drop=True)
		sub_dataset = FlowerDataset(sub_df, dataset.image_root, dataset.label_map, dataset.transforms)
	else:
		sub_dataset = dataset
	sampler: Optional[Sampler[int]] = None
	shuffle = is_train
	sampler_cfg = sampler_config or {"type": "none"}
	sampler_type = sampler_cfg.get("type", "none").lower()
	if dist_ctx and dist_ctx.is_distributed:
		if is_train and sampler_type != "none" and dist_ctx.is_main_process:
			print(f"Distributed mode: sampler '{sampler_type}' is not supported; falling back to DistributedSampler.")
		sampler = DistributedSampler(
			sub_dataset,
			num_replicas=dist_ctx.world_size,
			rank=dist_ctx.rank,
			shuffle=is_train,
			drop_last=is_train,
		)
		shuffle = False
	elif is_train and sampler_type != "none":
		labels = sub_dataset.df["__label_id__"].map(dataset.label_map).tolist()
		if sampler_type == "weighted_en":
			sampler = build_weighted_sampler(labels, sampler_cfg.get("beta", 0.999))
		elif sampler_type == "class_balanced":
			sampler = ClassBalancedSampler(labels, max_dup=sampler_cfg.get("balanced_max_dup", 3.0))
		else:
			raise ValueError(f"Unknown sampler type: {sampler_cfg['type']}")
		shuffle = False
	loader = DataLoader(
		sub_dataset,
		batch_size=batch_size,
		shuffle=shuffle if sampler is None else False,
		sampler=sampler,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=is_train,
		collate_fn=None,
	)
	return loader, sub_dataset.df.index.tolist() if indices is not None else None, sampler


def save_checkpoint(state: Dict[str, Any], path: Path, is_best: bool = False) -> None:
	ensure_dir(path.parent)
	torch.save(state, path)
	if is_best:
		best_path = path.parent / f"best_{path.name}"
		shutil.copy2(path, best_path)



def cast_state_dict_precision(state_dict: Optional[Dict[str, Any]], dtype: Optional[torch.dtype]) -> Optional[Dict[str, Any]]:
	if state_dict is None or dtype is None:
		return state_dict
	converted: Dict[str, Any] = {}
	for key, value in state_dict.items():
		if torch.is_tensor(value):
			converted[key] = value.to(dtype)
		else:
			converted[key] = value
	return converted


def create_export_checkpoint(state: Dict[str, Any], include_ema_state: bool = False, storage_dtype: Optional[torch.dtype] = None) -> Dict[str, Any]:
	"""Create a lightweight checkpoint dictionary for inference only exports."""
	model_state = cast_state_dict_precision(state.get("model_state"), storage_dtype)
	export_state = {
		"epoch": state.get("epoch"),
		"model_name": state.get("model_name"),
		"model_state": model_state,
		"ema": state.get("ema", False),
		"metrics": state.get("metrics"),
		"label_to_index": state.get("label_to_index"),
		"index_to_label": state.get("index_to_label"),
		"config": state.get("config"),
	}
	if include_ema_state and state.get("ema_state") is not None:
		export_state["ema_state"] = cast_state_dict_precision(state.get("ema_state"), storage_dtype)
	if storage_dtype is not None:
		export_state["storage_dtype"] = str(storage_dtype).replace("torch.", "")
	return export_state



def convert_checkpoint_file(path: Path, dtype: torch.dtype, backup: bool) -> None:
	checkpoint = torch.load(path, map_location="cpu")
	checkpoint["model_state"] = cast_state_dict_precision(checkpoint.get("model_state"), dtype)
	if "ema_state" in checkpoint:
		checkpoint["ema_state"] = cast_state_dict_precision(checkpoint.get("ema_state"), dtype)
	checkpoint["storage_dtype"] = str(dtype).replace("torch.", "")
	temp_path = path.with_suffix(path.suffix + ".tmp")
	torch.save(checkpoint, temp_path)
	if backup:
		backup_path = path.with_suffix(path.suffix + ".bak")
		if not backup_path.exists():
			shutil.copy2(path, backup_path)
	shutil.move(str(temp_path), str(path))
	print(f"[post_export] Converted {path} to {checkpoint['storage_dtype']}")


def convert_checkpoints(patterns: Iterable[str], dtype: torch.dtype, backup: bool) -> None:
	pattern_list = list(patterns)
	matched: List[Path] = []
	for pattern in pattern_list:
		for path in sorted(Path().glob(pattern)):
			if path.is_file() and path not in matched:
				matched.append(path)
	if not matched:
		print(f"[post_export] 未找到需转换的模型权重，patterns={pattern_list}")
		return
	for path in matched:
		convert_checkpoint_file(path, dtype, backup)


def postprocess_exported_checkpoints(cfg: Dict[str, Any]) -> None:
	post_cfg = cfg.get("post_export") or {}
	if not post_cfg.get("enabled", False):
		return
	dtype_name = str(post_cfg.get("dtype", "float16")).lower()
	dtype_map = {
		"float16": torch.float16,
		"fp16": torch.float16,
		"half": torch.float16,
		"bfloat16": torch.bfloat16,
	}
	target_dtype = dtype_map.get(dtype_name)
	if target_dtype is None:
		print(f"[post_export] 未识别的 dtype: {dtype_name}，跳过转换。")
		return
	patterns = post_cfg.get("patterns") or ["model/best_model_Fold*.pth"]
	if isinstance(patterns, str):
		patterns = [patterns]
	backup = bool(post_cfg.get("backup", False))
	convert_checkpoints(patterns, target_dtype, backup)


def load_pretrained(model: nn.Module, model_cfg: Dict[str, Any], is_main_process: bool = True) -> None:
	checkpoint_path = model_cfg.get("checkpoint")
	target_model = model.backbone if isinstance(model, MetricModelWrapper) else model
	if checkpoint_path:
		state = torch.load(checkpoint_path, map_location="cpu")
		if model_cfg.get("checkpoint_key"):
			state = state[model_cfg["checkpoint_key"]]
		missing, unexpected = target_model.load_state_dict(state, strict=False)
		if is_main_process:
			print(f"Loaded checkpoint for {model_cfg.get('name', model_cfg['backbone'])}. Missing keys: {missing}. Unexpected: {unexpected}")
		return
	pretrained_dir, _ = enumerate_pretrained_candidates(model_cfg)
	backbone_name = model_cfg['backbone']
	for candidate in existing_pretrained_files(model_cfg):
		try:
			state = load_weight_state(candidate)
		except RuntimeError as load_exc:
			if is_main_process:
				print(load_exc)
			continue
		if model_cfg.get("checkpoint_key"):
			state = state[model_cfg["checkpoint_key"]]
		elif isinstance(state, dict):
			if "state_dict" in state and isinstance(state["state_dict"], dict):
				state = state["state_dict"]
			elif "model" in state and isinstance(state["model"], dict):
				state = state["model"]
		state, ignored_keys = sanitize_state_dict(state, target_model)
		ignored_key_set = set(ignored_keys)
		missing, unexpected = target_model.load_state_dict(state, strict=False)
		source_label = candidate.relative_to(pretrained_dir) if candidate.is_file() else candidate.name
		if is_main_process:
			print(f"Loaded local pretrained weights for {model_cfg.get('name', backbone_name)} from {source_label}.")
		residual_missing = [k for k in missing if k not in ignored_key_set]
		if is_main_process and ignored_key_set:
			print(f"忽略与当前模型结构不匹配的权重: {sorted(ignored_key_set)}")
		if is_main_process and residual_missing:
			print(f"Missing keys: {residual_missing}")
		if is_main_process and unexpected:
			print(f"Unexpected keys: {unexpected}")
		return
	raise FileNotFoundError(
		f"未在 {pretrained_dir} 找到 {model_cfg.get('name', backbone_name)} 的本地预训练权重，请检查文件是否存在并指向正确目录。"
	)


def train_fold(cfg: Dict[str, Any],
		   model_cfg: Dict[str, Any],
		   fold_id: int,
		   df: pd.DataFrame,
		   label_to_index: Dict[int, int],
		   index_to_label: Dict[int, int],
		   device: torch.device,
	   output_root: Path,
	   results_root: Path,
	   dist_ctx: DistributedContext) -> Dict[str, Any]:
	stages = StageScheduler(cfg["train"]["stages"])
	data_root = Path(cfg["data"]["root"])
	is_rank0 = not dist_ctx.is_distributed or dist_ctx.is_main_process
	stage0 = stages.get_stage(0)
	initial_img_size = adjust_image_size_for_model(stage0.img_size, model_cfg, dist_ctx)
	initial_transform = build_train_transform(initial_img_size)
	dataset = FlowerDataset(df, data_root, label_to_index, initial_transform)
	class_counts = compute_class_counts(df)
	if not dist_ctx.is_distributed or dist_ctx.is_main_process:
		print(f"Fold {fold_id} | class distribution snapshot: {dict(list(class_counts.items())[:8])} ...")
	class_counts_ordered = [class_counts[idx] for idx in sorted(label_to_index.keys())]
	criterion = build_criterion(cfg, class_counts_ordered).to(device)
	eval_criterion = nn.CrossEntropyLoss().to(device)
	soft_criterion = SoftTargetCrossEntropy().to(device)
	train_idx, val_idx = build_split(cfg, df, fold_id)
	initial_batch_size = stage0.batch_per_gpu or cfg["train"]["batch_per_gpu"]
	train_loader, _, train_sampler = prepare_dataloader(
		dataset,
		cfg,
		train_idx,
		cfg["sampler"],
		is_train=True,
		batch_size_override=initial_batch_size,
		dist_ctx=dist_ctx,
	)
	train_dataset = train_loader.dataset
	train_dataset.set_transforms(initial_transform)
	val_df = df.iloc[val_idx].reset_index(drop=True)
	val_dataset = FlowerDataset(val_df.copy(), data_root, label_to_index, build_eval_transform(initial_img_size))
	val_loader, _, val_sampler = prepare_dataloader(
		val_dataset,
		cfg,
		None,
		{"type": "none"},
		is_train=False,
		batch_size_override=cfg["train"]["batch_per_gpu"],
		dist_ctx=dist_ctx,
	)
	robust_cfg = cfg.get("robust", {})
	corruption_datasets: Dict[str, FlowerDataset] = {}
	corruption_loaders: Dict[str, DataLoader] = {}
	corruption_samplers: Dict[str, Optional[Sampler[int]]] = {}
	if robust_cfg.get("enabled", False):
		robust_level = robust_cfg.get("level", "low")
		for kind in robust_cfg.get("kinds", []):
			corr_dataset = FlowerDataset(val_df.copy(), data_root, label_to_index, build_corruption_transform(initial_img_size, kind, robust_level))
			corruption_datasets[kind] = corr_dataset
			corr_loader, _, corr_sampler = prepare_dataloader(
				corr_dataset,
				cfg,
				None,
				{"type": "none"},
				is_train=False,
				batch_size_override=cfg["train"]["batch_per_gpu"],
				dist_ctx=dist_ctx,
			)
			corruption_loaders[kind] = corr_loader
			corruption_samplers[kind] = corr_sampler
	model = build_model_instance(cfg, model_cfg, len(label_to_index), device)
	load_pretrained(model, model_cfg, dist_ctx.is_main_process)
	optimizer = build_optimizer(cfg, model, model_cfg)
	if dist_ctx.is_distributed:
		device_ids = [device.index] if device.type == "cuda" and device.index is not None else None
		output_device = device.index if device.type == "cuda" and device.index is not None else None
		model = nn.parallel.DistributedDataParallel(
			model,
			device_ids=device_ids,
			output_device=output_device,
			find_unused_parameters=False,
		)
	scheduler = build_scheduler(optimizer, cfg)
	scaler = GradScaler(enabled=cfg["train"].get("use_amp", True) and device.type == "cuda")
	ema = ModelEMA(unwrap_ddp(model), cfg["ema"]["decay"]) if cfg["ema"].get("enabled", True) else None
	early_stop = EarlyStopping(cfg["es"]["patience"], cfg["es"].get("min_delta", 0.0)) if cfg["es"].get("enabled", True) else None
	best = {
		"combo": {
			"score": -float("inf"),
			"clean_acc": 0.0,
			"corr_acc": 0.0,
			"ema": False,
			"epoch": -1,
			"name": "",
			"export_path": "",
		},
		"clean": {
			"score": -float("inf"),
			"ema": False,
			"epoch": -1,
			"name": "",
		},
		"ema": {
			"score": -float("inf"),
			"epoch": -1,
		},
	}
	fold_dir = output_root / f"fold{fold_id}"
	ensure_dir(fold_dir)
	ensure_dir(results_root)
	model_config_path = (results_root / "model_config.json").resolve()
	metrics_history: List[Dict[str, Any]] = []
	mixup_warning_emitted = False
	mixup_loss_warning_emitted = False
	robust_level = robust_cfg.get("level", "low") if robust_cfg.get("enabled", False) else "low"
	alpha = cfg["es"].get("alpha", 0.7)
	current_batch_size = initial_batch_size
	for epoch in range(cfg["train"]["epochs"]):
		stage = stages.get_stage(epoch)
		desired_img_size = stage.img_size
		stage_img_size = adjust_image_size_for_model(desired_img_size, model_cfg, dist_ctx)
		train_transform = build_train_transform(stage_img_size)
		dataset.set_transforms(train_transform)
		train_dataset.set_transforms(train_transform)
		desired_batch = stage.batch_per_gpu or cfg["train"]["batch_per_gpu"]
		if desired_batch != current_batch_size:
			train_loader, _, train_sampler = prepare_dataloader(
				dataset,
				cfg,
				train_idx,
				cfg["sampler"],
				is_train=True,
				batch_size_override=desired_batch,
				dist_ctx=dist_ctx,
			)
			train_dataset = train_loader.dataset
			train_dataset.set_transforms(train_transform)
			current_batch_size = desired_batch
		val_dataset.set_transforms(build_eval_transform(stage_img_size))
		for kind, corr_dataset in corruption_datasets.items():
			corr_dataset.set_transforms(build_corruption_transform(stage_img_size, kind, robust_level))
		if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
			train_sampler.set_epoch(epoch)
		if val_sampler is not None and hasattr(val_sampler, "set_epoch"):
			val_sampler.set_epoch(epoch)
		for sampler in corruption_samplers.values():
			if sampler is not None and hasattr(sampler, "set_epoch"):
				sampler.set_epoch(epoch)
		mixup_fn = build_mixup(stage, len(label_to_index))
		if mixup_fn is not None and cfg["loss"]["type"].lower() in {"balanced_softmax", "focal"}:
			mixup_fn = None
			if not mixup_loss_warning_emitted:
				if not dist_ctx.is_distributed or dist_ctx.is_main_process:
					print("Mixup/cutmix disabled because the selected loss expects hard labels.")
				mixup_loss_warning_emitted = True
		if mixup_fn is not None and model_requires_targets(model):
			mixup_fn = None
			if not mixup_warning_emitted:
				if not dist_ctx.is_distributed or dist_ctx.is_main_process:
					print("Metric head detected: mixup/cutmix disabled to keep hard targets.")
				mixup_warning_emitted = True
		train_metrics = run_epoch(model, train_loader, criterion, soft_criterion, optimizer, device, epoch, scaler, mixup_fn, ema, cfg)
		scheduler.step()
		candidates: Dict[str, Dict[str, Any]] = {}
		candidate_states: Dict[str, Dict[str, Any]] = {}

		def register_candidate(name: str, eval_model: nn.Module, ema_flag: bool, state_dict: Dict[str, Any]) -> None:
			clean_metrics = evaluate(eval_model, val_loader, eval_criterion, device)
			corr_metrics: Dict[str, Dict[str, float]] = {}
			if corruption_loaders:
				for ckind, loader in corruption_loaders.items():
					corr_metrics[ckind] = evaluate(eval_model, loader, eval_criterion, device)
			corr_acc_local = float(np.mean([m["acc1"] for m in corr_metrics.values()])) if corr_metrics else clean_metrics["acc1"]
			combo_local = alpha * clean_metrics["acc1"] + (1 - alpha) * corr_acc_local if corr_metrics else clean_metrics["acc1"]
			candidates[name] = {
				"clean": clean_metrics,
				"corr": corr_metrics,
				"corr_acc": corr_acc_local,
				"combo_score": combo_local,
				"ema": ema_flag,
			}
			candidate_states[name] = {
				"state_dict": state_dict,
			}

		register_candidate("raw", model, False, unwrap_ddp(model).state_dict())
		if ema is not None:
			ema_model = ema.eval_model()
			register_candidate("ema", ema_model, True, ema.state_dict())

		best_combo_name, best_combo = max(candidates.items(), key=lambda item: item[1]["combo_score"])
		best_clean_name, best_clean = max(candidates.items(), key=lambda item: item[1]["clean"]["acc1"])
		metrics = {
			"epoch": epoch,
			"train_loss": train_metrics["loss"],
			"train_acc": train_metrics["acc1"],
			"lr": optimizer.param_groups[0]["lr"],
			"selected": best_combo_name,
			"score": best_combo["combo_score"],
			"val_acc_clean": best_combo["clean"]["acc1"],
			"val_loss": best_combo["clean"]["loss"],
			"val_acc_corr": best_combo["corr_acc"],
			"corruptions": {name: {k: v for k, v in cand["corr"].items()} for name, cand in candidates.items()},
			"candidates": {
				name: {
					"clean_acc": cand["clean"]["acc1"],
					"clean_loss": cand["clean"]["loss"],
					"corr_acc": cand["corr_acc"],
					"combo_score": cand["combo_score"],
					"ema": cand["ema"],
				}
				for name, cand in candidates.items()
			},
		}
		metrics_history.append(metrics)
		corr_text = f" | corr_acc={best_combo['corr_acc']:.2f}" if corruption_loaders else ""
		selected_label = "EMA" if best_combo["ema"] else "RAW"
		if not dist_ctx.is_distributed or dist_ctx.is_main_process:
			print(
				f"[{model_cfg.get('name','model')}] Fold {fold_id} | Epoch {epoch:03d} | "
				f"train_loss={metrics['train_loss']:.4f} | train_acc={metrics['train_acc']:.2f} | "
				f"val_acc={metrics['val_acc_clean']:.2f}{corr_text} | score={metrics['score']:.2f} | sel={selected_label} | lr={metrics['lr']:.6f}"
			)
		checkpoint_state = {
			"epoch": epoch,
			"model_name": model_cfg.get("name"),
			"model_state": unwrap_ddp(model).state_dict(),
			"ema_state": ema.state_dict() if ema is not None else None,
			"optimizer_state": optimizer.state_dict(),
			"scaler_state": scaler.state_dict() if scaler is not None else None,
			"metrics": metrics,
			"label_to_index": label_to_index,
			"index_to_label": index_to_label,
			"config": cfg,
		}
		if is_rank0:
			save_checkpoint(checkpoint_state, fold_dir / "last.pth", is_best=False)
		combo_score = best_combo["combo_score"]
		if combo_score > best["combo"]["score"]:
			best["combo"].update({
				"score": combo_score,
				"clean_acc": best_combo["clean"]["acc1"],
				"corr_acc": best_combo["corr_acc"],
				"ema": best_combo["ema"],
				"epoch": epoch,
				"name": best_combo_name,
				"train_acc": metrics["train_acc"],
				"train_loss": metrics["train_loss"],
				"lr": metrics["lr"],
				"selected_candidate": best_combo_name,
				"path": str((fold_dir / "best_combo.pth").resolve()),
				"model_config_path": str(model_config_path),
			})
			best_state = dict(checkpoint_state)
			best_state["model_state"] = candidate_states[best_combo_name]["state_dict"]
			best_state["ema"] = best_combo["ema"]
			if best_combo["ema"]:
				best_state["ema_state"] = None
			export_path = (fold_dir / "best_combo_export.pth").resolve()
			best["combo"]["export_path"] = str(export_path)
			if is_rank0:
				save_checkpoint(best_state, fold_dir / "best_combo.pth", is_best=True)
				export_include_ema = bool(not best_combo["ema"] and best_state.get("ema_state"))
				export_state = create_export_checkpoint(best_state, include_ema_state=export_include_ema, storage_dtype=torch.float16)
				torch.save(export_state, export_path)
				best_export_path = cfg.get("best_model_path")
				if best_export_path:
					best_export_path = Path(best_export_path)
					ensure_dir(best_export_path.parent)
					torch.save(export_state, best_export_path)
					print(f"已将当前最优模型导出到 {best_export_path}。")
				print(f"[Best Combo] [{model_cfg.get('name','model')}] Fold {fold_id} | Epoch {epoch:03d} | score={combo_score:.2f} | sel={selected_label}")
		if best_clean["clean"]["acc1"] > best["clean"]["score"]:
			best["clean"].update({
				"score": best_clean["clean"]["acc1"],
				"ema": best_clean["ema"],
				"epoch": epoch,
				"name": best_clean_name,
			})
			best_clean_state = dict(checkpoint_state)
			best_clean_state["model_state"] = candidate_states[best_clean_name]["state_dict"]
			best_clean_state["ema"] = best_clean["ema"]
			if is_rank0:
				save_checkpoint(best_clean_state, fold_dir / "best_clean.pth", is_best=False)
		if "ema" in candidates and candidates["ema"]["clean"]["acc1"] > best["ema"]["score"]:
			best["ema"].update({
				"score": candidates["ema"]["clean"]["acc1"],
				"epoch": epoch,
			})
			ema_best_state = dict(checkpoint_state)
			ema_best_state["model_state"] = candidate_states["ema"]["state_dict"]
			ema_best_state["ema"] = True
			if is_rank0:
				save_checkpoint(ema_best_state, fold_dir / "ema_best.pth", is_best=False)
		if ema is not None:
			ema_last_state = dict(checkpoint_state)
			ema_last_state["model_state"] = ema.state_dict()
			ema_last_state["ema"] = True
			if is_rank0:
				save_checkpoint(ema_last_state, fold_dir / "ema_last.pth", is_best=False)
		if early_stop is not None and early_stop.step(combo_score):
			if is_rank0:
				print(f"Early stopping triggered at epoch {epoch} for fold {fold_id}")
			break
	metrics_path = results_root / f"fold{fold_id}_metrics.json"
	if is_rank0:
		save_json(metrics_history, metrics_path)
	return {
		"fold": fold_id,
		"model_name": model_cfg.get("name", model_cfg.get("backbone", "model")),
		"backbone": model_cfg.get("backbone"),
		"experiment_name": cfg.get("experiment_name"),
		"best_combo_score": best["combo"]["score"],
		"best_combo_clean_acc": best["combo"]["clean_acc"],
		"best_combo_corr_acc": best["combo"]["corr_acc"],
		"best_combo_epoch": best["combo"]["epoch"],
		"best_combo_model": best["combo"]["name"],
		"best_combo_ema": best["combo"]["ema"],
		"best_combo_train_acc": best["combo"].get("train_acc"),
		"best_combo_train_loss": best["combo"].get("train_loss"),
		"best_combo_lr": best["combo"].get("lr"),
		"best_combo_path": best["combo"].get("path", str((fold_dir / "best_combo.pth").resolve())),
		"best_combo_export_path": best["combo"].get("export_path"),
		"best_combo_selected": best["combo"].get("selected_candidate", ""),
		"model_config_path": best["combo"].get("model_config_path", str(model_config_path)),
		"fold_dir": str(fold_dir.resolve()),
		"best_clean_acc": best["clean"]["score"],
		"best_clean_epoch": best["clean"]["epoch"],
		"best_clean_model": best["clean"]["name"],
		"best_clean_ema": best["clean"]["ema"],
		"best_ema_acc": best["ema"]["score"],
		"best_ema_epoch": best["ema"]["epoch"],
		"metrics_path": str(metrics_path),
	}


def build_split(cfg: Dict[str, Any], df: pd.DataFrame, fold_id: int) -> Tuple[List[int], List[int]]:
	data_cfg = cfg["data"]
	n_splits = data_cfg["n_splits"]
	labels = df["__label_id__"].values
	splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=data_cfg.get("val_split_seed", 42))
	folds = list(splitter.split(np.zeros(len(df)), labels))
	if fold_id < 0 or fold_id >= len(folds):
		raise ValueError(f"Fold index {fold_id} out of range (0-{len(folds)-1})")
	train_idx, val_idx = folds[fold_id]
	return train_idx.tolist(), val_idx.tolist()



def build_optimizer(cfg: Dict[str, Any], model: nn.Module, model_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
	opt_cfg = cfg["opt"]
	lr = compute_scaled_lr(cfg)
	_ = model_cfg  # reserved for future per-model optimizer customisation
	param_groups = build_param_groups(model, lr, cfg, opt_cfg)
	if opt_cfg["name"].lower() == "adamw":
		optimizer = AdamW(param_groups, lr=lr, weight_decay=opt_cfg["weight_decay"], betas=tuple(opt_cfg.get("betas", [0.9, 0.999])))
	elif opt_cfg["name"].lower() == "sgd":
		optimizer = SGD(param_groups, lr=lr, momentum=opt_cfg.get("momentum", 0.9), weight_decay=opt_cfg["weight_decay"], nesterov=True)
	else:
		raise ValueError(f"Unsupported optimizer: {opt_cfg['name']}")
	return optimizer


def export_best_fold_checkpoints(cfg: Dict[str, Any], overall_results: Dict[str, List[Dict[str, Any]]]) -> None:
	n_splits = cfg.get("data", {}).get("n_splits", 1)
	best_by_fold: List[Optional[Dict[str, Any]]] = [None] * n_splits
	for model_name, fold_entries in overall_results.items():
		for entry in fold_entries:
			if entry is None:
				continue
			fold_idx = entry.get("fold")
			if fold_idx is None or fold_idx < 0 or fold_idx >= n_splits:
				continue
			candidate = dict(entry)
			candidate.setdefault("model_name", model_name)
			score = candidate.get("best_combo_score", float("-inf"))
			current_best = best_by_fold[fold_idx]
			if current_best is None or score > current_best.get("best_combo_score", float("-inf")):
				best_by_fold[fold_idx] = candidate
	best_root = Path("model")
	ensure_dir(best_root)
	info_root = best_root / "bestmodel"
	ensure_dir(info_root)
	for fold_idx, info in enumerate(best_by_fold):
		if info is None:
			continue
		src_path_value = info.get("best_combo_export_path") or info.get("best_combo_path")
		if not src_path_value:
			print(f"折 {fold_idx} 未提供最优模型路径，跳过导出。")
			continue
		src_path = Path(src_path_value)
		if not src_path.exists():
			print(f"折 {fold_idx} 的最优模型权重不存在: {src_path}")
			continue
		dest_path = best_root / f"best_model_Fold{fold_idx + 1}.pth"
		shutil.copy2(src_path, dest_path)
		metadata = {
			"fold": fold_idx,
			"fold_one_based": fold_idx + 1,
			"experiment_name": info.get("experiment_name", cfg.get("experiment_name")),
			"model_name": info.get("model_name"),
			"backbone": info.get("backbone"),
			"combo_score": info.get("best_combo_score"),
			"val_acc": info.get("best_combo_clean_acc"),
			"corr_acc": info.get("best_combo_corr_acc"),
			"train_acc": info.get("best_combo_train_acc"),
			"train_loss": info.get("best_combo_train_loss"),
			"learning_rate": info.get("best_combo_lr"),
			"best_epoch": info.get("best_combo_epoch"),
			"ema": info.get("best_combo_ema"),
			"selected_candidate": info.get("best_combo_selected"),
			"checkpoint_source": str(src_path.resolve()),
			"checkpoint_source_full": info.get("best_combo_path"),
			"checkpoint_destination": str(dest_path.resolve()),
			"metrics_summary_path": info.get("metrics_path"),
			"model_config_path": info.get("model_config_path"),
			"fold_output_dir": info.get("fold_dir"),
			"export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
		}
		info_path = info_root / f"fold{fold_idx + 1}.json"
		save_json(metadata, info_path)
		print(f"已将折 {fold_idx} 的最优模型复制到 {dest_path} 并写入 {info_path}。")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
	args = parse_args()
	cfg = load_config(args)
	dist_ctx = init_distributed_mode(cfg)
	try:
		set_seed(cfg["seed"], rank=dist_ctx.rank if dist_ctx.is_distributed else 0)
		device = get_device(cfg)
		df, label_to_index, index_to_label = prepare_dataframe(cfg)
		total_samples = len(df)
		num_classes = len(label_to_index)
		if not dist_ctx.is_distributed or dist_ctx.is_main_process:
			print("===== 数据概览 =====")
			print(f"样本总数: {total_samples}")
			print(f"类别数量: {num_classes}")
			print(f"类别示例: {list(label_to_index.keys())[:8]}")
		exp_name = cfg.get("experiment_name") or time.strftime("exp_%Y%m%d_%H%M%S")
		output_root = Path(cfg["output_dir"]) / exp_name
		ensure_dir(output_root)
		results_root = Path(cfg["results_dir"]) / exp_name
		ensure_dir(results_root)
		if not dist_ctx.is_distributed or dist_ctx.is_main_process:
			save_json(cfg, results_root / "config.json")
		fold_setting = cfg["data"].get("fold", -1)
		overall_results: Dict[str, Any] = {}
		for model_cfg in cfg["models"]:
			model_name = model_cfg.get("name", model_cfg.get("backbone", "model"))
			model_output_root = output_root / model_name
			model_results_root = results_root / model_name
			ensure_dir(model_output_root)
			ensure_dir(model_results_root)
			if not dist_ctx.is_distributed or dist_ctx.is_main_process:
				save_json(model_cfg, model_results_root / "model_config.json")
			folds_to_run = range(cfg["data"]["n_splits"]) if fold_setting == -1 else [fold_setting]
			fold_results: List[Dict[str, Any]] = []
			for fold_id in folds_to_run:
				if not dist_ctx.is_distributed or dist_ctx.is_main_process:
					print(f"========== Model {model_name} | Fold {fold_id} training ==========")
				fold_metrics = train_fold(
					cfg,
					model_cfg,
					fold_id,
					df,
					label_to_index,
					index_to_label,
					device,
					model_output_root,
					model_results_root,
					dist_ctx,
				)
				if not dist_ctx.is_distributed or dist_ctx.is_main_process:
					fold_results.append(fold_metrics)
			if not dist_ctx.is_distributed or dist_ctx.is_main_process:
				save_json({"folds": fold_results}, model_results_root / "summary.json")
				overall_results[model_name] = fold_results
		if not dist_ctx.is_distributed or dist_ctx.is_main_process:
			save_json({"models": overall_results}, results_root / "summary_all.json")
			export_best_fold_checkpoints(cfg, overall_results)
			postprocess_exported_checkpoints(cfg)
			print("Training completed. Summaries saved to", results_root)
	finally:
		cleanup_distributed(dist_ctx)


if __name__ == "__main__":
	main()

