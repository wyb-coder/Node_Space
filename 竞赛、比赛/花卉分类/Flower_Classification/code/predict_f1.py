#!/usr/bin/env python3
"""推理脚本：使用五折最优模型进行集成预测并输出提交文件。"""
from __future__ import annotations

# --- local third_party path bootstrap (must be before other imports) ---
import sys
from pathlib import Path
_THIRD_PARTY = Path(__file__).resolve().parent / "third_party"
if _THIRD_PARTY.exists():
    sys.path.insert(0, str(_THIRD_PARTY))
# ----------------------------------------------------------------------

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile
import pandas as pd  # 新增：用于后处理修正

try:
    # 优先使用顶层导入（某些版本可以）
    from timm import create_model  # type: ignore
except Exception:
    # 兼容有些版本只在 timm.models 下暴露
    from timm.models import create_model  # type: ignore

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.datasets.folder import default_loader

ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_TTA_SCALES: Tuple[int, int] = (320, 384)
DEFAULT_TEMPERATURE_MAP_CANDIDATES: Tuple[str, ...] = (
    "temperature_map.json",
    "model/temperature_map.json",
    "../model/temperature_map.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用五折最优模型进行花卉分类预测")
    parser.add_argument("test_dir", type=str, help="测试集图片所在目录")
    parser.add_argument("output_csv", type=str, help="预测结果输出路径，如 results/submission.csv")
    parser.add_argument("--batch-size", type=int, default=64, help="推理批大小")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 线程数")
    parser.add_argument("--device", type=str, default=None, help="强制指定推理设备，如 cuda:0 / cpu")
    parser.add_argument(
        "--model-pattern",
        type=str,
        default="model/best_model_Fold*.pth",
        help="最优模型权重查找通配符，可按需调整",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="禁用测试时增强（多尺度 + 水平翻转）。默认开启。",
    )
    parser.add_argument(
        "--tta-scales",
        type=int,
        nargs="+",
        default=None,
        help="指定 TTA 尺度列表（整数）。若未指定则使用默认尺度并包含模型自身分辨率。",
    )
    parser.add_argument(
        "--tta-no-hflip",
        action="store_true",
        help="开启 TTA 时禁用水平翻转视角。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="对融合后的 logits 进行全局温度缩放 (logits /= temperature)。",
    )
    parser.add_argument(
        "--temperature-map",
        type=str,
        default=None,
        help="JSON 文件，映射模型名/权重文件名到温度值，可包含 __default__ 作为兜底。",
    )
    parser.add_argument(
        "--ignore-ckpt-temperature",
        action="store_true",
        help="忽略权重文件中可能存储的温度信息，仅使用命令行提供的设置。",
    )
    parser.add_argument(
        "--no-temperature-scaling",
        action="store_true",
        help="禁用所有温度缩放。默认启用（优先使用权重/JSON中提供的温度）。",
    )
    return parser.parse_args()


def resolve_device(explicit: Optional[str]) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_image_paths(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"测试目录不存在: {root}")
    paths = sorted([p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    if not paths:
        raise RuntimeError(f"在 {root} 未找到可识别的图片文件")
    return paths


class PredictionDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], transform: Optional[T.Compose]) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self.image_paths[index]
        image = default_loader(str(path))
        if self.transform:
            image = self.transform(image)
        return image, index


def build_eval_transform(img_size: int, apply_hflip: bool = False) -> T.Compose:
    resize_size = int(math.ceil(img_size * 1.14))
    ops: List[Any] = [T.Resize(resize_size), T.CenterCrop(img_size)]
    if apply_hflip:
        ops.append(T.RandomHorizontalFlip(p=1.0))
    ops.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return T.Compose(ops)


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
        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-6))
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
                raise ValueError("Metric head requires targets during训练/评估")
            return self.head(features, targets)
        return self.head.inference(features)


def unwrap_ddp(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model


def model_requires_targets(model: nn.Module) -> bool:
    return bool(getattr(unwrap_ddp(model), "requires_targets", False))


def forward_model(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if model_requires_targets(model):
        return model(images, None)
    return model(images)


def build_model_instance(cfg: Dict[str, Any], model_cfg: Dict[str, Any], num_classes: int, device: torch.device) -> nn.Module:
    backbone_name = model_cfg["backbone"]
    pretrained = False
    drop_path = model_cfg.get("drop_path", cfg.get("model", {}).get("drop_path", 0.0))
    create_kwargs: Dict[str, Any] = {"drop_path_rate": drop_path}
    global_pool = model_cfg.get("global_pool")
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
            raise ValueError(f"Backbone {backbone_name} 未提供 num_features")
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
    model.eval()
    return model


def sanitize_state_dict_for_load(state: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v.float() if torch.is_tensor(v) else v for k, v in state.items()}


def determine_eval_img_size(cfg: Dict[str, Any], model_cfg: Dict[str, Any]) -> int:
    stages = cfg.get("train", {}).get("stages")
    img_size = None
    if isinstance(stages, list) and stages:
        stage_sizes = [s.get("img_size") for s in stages if isinstance(s, dict) and s.get("img_size")]
        if stage_sizes:
            img_size = max(stage_sizes)
    if img_size is None:
        candidate = cfg.get("data", {}).get("img_size")
        if isinstance(candidate, int):
            img_size = candidate
        elif isinstance(candidate, (list, tuple)) and candidate:
            img_size = max(candidate)
    if img_size is None:
        img_size = 288
    max_img_size = model_cfg.get("max_img_size")
    if max_img_size is not None:
        img_size = min(img_size, max_img_size)
    return int(img_size)


@dataclass
class LoadedModel:
    name: str
    model: nn.Module
    transform: T.Compose
    label_to_index: Dict[Any, int]
    index_to_label: Dict[int, Any]
    num_classes: int
    img_size: int
    checkpoint_path: Path
    temperature: Optional[float]


def extract_temperature(bundle: Dict[str, Any]) -> Optional[float]:
    candidates: List[Any] = [bundle.get("temperature")]
    calibration = bundle.get("calibration")
    if isinstance(calibration, dict):
        candidates.extend(
            calibration.get(key) for key in ("temperature", "temp", "T")
        )
    metrics = bundle.get("metrics")
    if isinstance(metrics, dict):
        candidates.extend(
            metrics.get(key) for key in ("temperature", "temp", "T")
        )
    for value in candidates:
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def normalize_tta_scales(default_scale: int, user_scales: Optional[Sequence[int]]) -> List[int]:
    ordered: List[int] = []

    def add(scale: int) -> None:
        if scale <= 0:
            raise ValueError(f"TTA 尺度必须为正整数，收到 {scale}")
        if scale not in ordered:
            ordered.append(scale)

    add(int(default_scale))
    candidates: Sequence[int] = DEFAULT_TTA_SCALES if user_scales is None else user_scales
    for raw in candidates:
        add(int(raw))
    return ordered


def prepare_tta_variants(
    default_scale: int,
    user_scales: Optional[Sequence[int]],
    include_hflip: bool,
) -> List[Tuple[T.Compose, Tuple[int, bool]]]:
    scales = normalize_tta_scales(default_scale, user_scales)
    variants: List[Tuple[T.Compose, Tuple[int, bool]]] = []
    for scale in scales:
        variants.append((build_eval_transform(scale, apply_hflip=False), (scale, False)))
        if include_hflip:
            variants.append((build_eval_transform(scale, apply_hflip=True), (scale, True)))
    return variants


def resolve_temperature_map_path(explicit: Optional[Union[str, Path]]) -> Optional[Path]:
    if explicit is not None:
        return Path(explicit)
    script_dir = Path(__file__).resolve().parent
    for relative in DEFAULT_TEMPERATURE_MAP_CANDIDATES:
        candidate = (script_dir / relative).resolve()
        if candidate.exists():
            return candidate
    return None


def load_temperature_overrides(path: Optional[Union[str, Path]]) -> Dict[str, float]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"温度映射文件不存在: {config_path}")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"无法解析温度映射 JSON: {config_path} ({exc})") from exc
    if isinstance(data, (int, float)):
        if data <= 0:
            raise ValueError("温度值必须为正数")
        return {"__default__": float(data)}
    if not isinstance(data, dict):
        raise ValueError("温度映射文件需为 JSON 对象或单个数字")
    overrides: Dict[str, float] = {}
    for key, value in data.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"温度映射 {key} 的值必须为数字")
        if value <= 0:
            raise ValueError(f"温度映射 {key} 的值必须为正数")
        overrides[key] = float(value)
    return overrides


def resolve_temperature_for_model(
    loaded: LoadedModel,
    overrides: Dict[str, float],
    ignore_checkpoint: bool,
) -> Optional[float]:
    temperature: Optional[float] = None if ignore_checkpoint else loaded.temperature
    keys = (
        loaded.name,
        loaded.checkpoint_path.stem,
        loaded.checkpoint_path.name,
        str(loaded.checkpoint_path),
    )
    for key in keys:
        if key in overrides:
            temperature = overrides[key]
            break
    if temperature is None and "__default__" in overrides:
        temperature = overrides["__default__"]
    if temperature is not None and temperature <= 0:
        raise ValueError(f"模型 {loaded.name} 的温度必须为正数，收到 {temperature}")
    return temperature


def align_label_mapping(bundle: Dict[str, Any]) -> Tuple[Dict[Any, int], Dict[int, Any]]:
    label_to_index = bundle.get("label_to_index") or {}
    index_to_label = bundle.get("index_to_label") or {}
    label_to_index = {int(k): int(v) for k, v in label_to_index.items()}
    index_to_label = {int(k): int(v) for k, v in index_to_label.items()}
    return label_to_index, index_to_label


def locate_model_cfg(cfg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    models = cfg.get("models") or []
    for mc in models:
        if mc.get("name") == model_name:
            return mc
    if cfg.get("model", {}).get("name") == model_name:
        return cfg["model"]
    if models:
        return models[0]
    return cfg.get("model", {})


def load_model_from_checkpoint(path: Path, device: torch.device) -> LoadedModel:
    bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise RuntimeError(f"无法解析模型文件: {path}")
    label_to_index, index_to_label = align_label_mapping(bundle)
    num_classes = len(index_to_label)
    config = bundle.get("config")
    if not config:
        raise RuntimeError(f"模型 {path} 未包含配置信息，无法构建骨干")
    model_name = bundle.get("model_name")
    model_cfg = locate_model_cfg(config, model_name)
    model_state = bundle.get("model_state")
    if model_state is None:
        raise RuntimeError(f"模型 {path} 未包含 model_state")
    storage_dtype = bundle.get("storage_dtype")
    if storage_dtype:
        model_state = sanitize_state_dict_for_load(model_state)
    model = build_model_instance(config, model_cfg, num_classes, device)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"加载 {path} 失败: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    img_size = determine_eval_img_size(config, model_cfg)
    transform = build_eval_transform(img_size)
    temperature = extract_temperature(bundle)
    return LoadedModel(
        name=model_cfg.get("name", model_cfg.get("backbone", "model")),
        model=model,
        transform=transform,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        num_classes=num_classes,
        img_size=img_size,
        checkpoint_path=path,
        temperature=temperature,
    )


from glob import glob

def collect_models(pattern: str, device: torch.device) -> List[LoadedModel]:
    here = Path(__file__).resolve().parent  # .../submission/code
    tried: List[str] = []

    # 1) 以当前工作目录(CWD)为基准(官方就是 cd 到 code 后调用)
    paths = sorted(Path('.').glob(pattern))
    tried.append(f"CWD:{Path('.').resolve()}/{pattern}")

    # 2) 以脚本所在目录为基准
    if not paths:
        paths = sorted(here.glob(pattern))
        tried.append(f"SCRIPT_DIR:{here}/{pattern}")

    # 3) 回退到上一级的 model 目录（构造绝对通配，必须用 glob.glob）
    if not paths:
        name_pattern = Path(pattern).name  # 只取文件名通配部分
        alt_glob = here.parent / 'model' / name_pattern  # .../submission/model/best_model_Fold*.pth
        paths = [Path(p) for p in glob(str(alt_glob))]
        tried.append(f"PARENT_MODEL:{alt_glob}")

    if not paths:
        msg = "未找到匹配的模型文件。尝试过：\n" + "\n".join(f" - {t}" for t in tried)
        raise RuntimeError(msg)

    loaded: List[LoadedModel] = []
    for path in sorted(set(paths)):
        loaded.append(load_model_from_checkpoint(path, device))
    return loaded


def ensure_same_label_mapping(models: List[LoadedModel]) -> LoadedModel:
    if not models:
        raise RuntimeError("模型列表为空")
    ref = models[0]
    for other in models[1:]:
        if other.index_to_label != ref.index_to_label:
            raise RuntimeError(f"模型 {other.name} 的标签映射与首个模型不一致，无法集成")
    return ref


def aggregate_predictions(
    models: List[LoadedModel],
    image_paths: Sequence[Path],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    enable_tta: bool = False,
    tta_scales: Optional[Sequence[int]] = None,
    tta_include_hflip: bool = True,
    temperature_overrides: Optional[Dict[str, float]] = None,
    ignore_checkpoint_temperature: bool = False,
    enable_temperature_scaling: bool = True,
) -> Tuple[torch.Tensor, List[Tuple[str, float, int, Optional[float]]]]:
    overrides = temperature_overrides or {}
    num_samples = len(image_paths)
    ref = ensure_same_label_mapping(models)
    ensemble_logits = torch.zeros(num_samples, ref.num_classes, dtype=torch.float32)
    per_model_stats: List[Tuple[str, float, int, Optional[float]]] = []
    with torch.no_grad():
        for loaded in models:
            start_ts = time.perf_counter()
            if enable_tta:
                variants = prepare_tta_variants(loaded.img_size, tta_scales, include_hflip=tta_include_hflip)
            else:
                variants = [(loaded.transform, (loaded.img_size, False))]
            if not variants:
                raise RuntimeError(f"模型 {loaded.name} 未生成任何推理视角")
            model_logits = torch.zeros(num_samples, ref.num_classes, dtype=torch.float32)
            for variant_idx, (transform, (scale, flipped)) in enumerate(variants):
                dataset = PredictionDataset(image_paths, transform)
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=device.type == "cuda",
                )
                if enable_tta and len(variants) > 1:
                    flip_tag = "-flip" if flipped else ""
                    desc = f"{loaded.name} 推理 TTA {variant_idx + 1}/{len(variants)} ({scale}px{flip_tag})"
                else:
                    desc = f"{loaded.name} 推理"
                progress = tqdm(
                    loader,
                    desc=desc,
                    total=len(loader),
                    leave=False,
                    unit="batch",
                )
                for images, indices in progress:
                    images = images.to(device, non_blocking=True)
                    logits = forward_model(loaded.model, images)
                    logits = logits.float().cpu()
                    model_logits[indices] += logits
            views = len(variants)
            model_logits /= float(views)
            model_temperature: Optional[float] = None
            if enable_temperature_scaling:
                model_temperature = resolve_temperature_for_model(
                    loaded,
                    overrides,
                    ignore_checkpoint_temperature,
                )
                if model_temperature is not None:
                    model_logits /= model_temperature
            ensemble_logits += model_logits
            elapsed = time.perf_counter() - start_ts
            per_model_stats.append((loaded.name, elapsed, views, model_temperature))
    ensemble_logits /= float(len(models))
    return ensemble_logits, per_model_stats


def logits_to_predictions(
    logits: torch.Tensor,
    index_to_label: Dict[int, Any],
) -> Tuple[List[Any], List[float]]:
    probs = torch.softmax(logits, dim=1)
    confidences, indices = probs.max(dim=1)
    labels = [index_to_label[int(idx)] for idx in indices.tolist()]
    return labels, confidences.tolist()


def save_submission(output_path: Path, image_paths: Sequence[Path], labels: Sequence[Any], confidences: Sequence[float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "category_id", "confidence"])
        for path, label, conf in zip(image_paths, labels, confidences):
            writer.writerow([path.name, label, f"{conf:.6f}"])


# ========== 新增：固定后处理修正 (1777→230，当置信度<0.60) ==========
def post_fix_submission(csv_path: Path, from_id: int = 1777, to_id: int = 230, conf_th: float = 0.60) -> None:
    """
    当预测类别为 `from_id` 且置信度 < `conf_th` 时，将该条目的类别修正为 `to_id`。
    会原地覆盖写回 csv_path，并打印修正条数。
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"读取提交文件失败: {csv_path} ({e})")

    if "category_id" not in df.columns or "confidence" not in df.columns:
        raise RuntimeError("提交文件缺少必要列: category_id / confidence")

    # 统一类型，避免字符串/浮点比较问题
    df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce").astype("Int64")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    mask = (df["category_id"] == from_id) & (df["confidence"] < conf_th)
    n = int(mask.sum())
    if n > 0:
        df.loc[mask, "category_id"] = to_id

    # 写回
    df.to_csv(csv_path, index=False)
    print(f"后处理：{n} 条 {from_id}→{to_id} (confidence<{conf_th}) 已修正。")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    image_paths = list_image_paths(Path(args.test_dir))
    models = collect_models(args.model_pattern, device)
    ref_model = ensure_same_label_mapping(models)
    inference_start = time.perf_counter()
    temperature_map_path = resolve_temperature_map_path(args.temperature_map)
    temperature_overrides = load_temperature_overrides(temperature_map_path)
    enable_tta = not args.no_tta
    logits, per_model_stats = aggregate_predictions(
        models,
        image_paths,
        args.batch_size,
        args.num_workers,
        device,
        enable_tta=enable_tta,
        tta_scales=args.tta_scales,
        tta_include_hflip=not args.tta_no_hflip,
        temperature_overrides=temperature_overrides,
        ignore_checkpoint_temperature=args.ignore_ckpt_temperature,
        enable_temperature_scaling=not args.no_temperature_scaling,
    )
    if args.temperature is not None:
        if args.temperature <= 0:
            raise ValueError("全局温度必须为正数")
        logits = logits / args.temperature
    total_duration = time.perf_counter() - inference_start
    labels, confidences = logits_to_predictions(logits, ref_model.index_to_label)
    out_path = Path(args.output_csv)
    save_submission(out_path, image_paths, labels, confidences)

    # 新增：固定后处理修正
    # post_fix_submission(out_path, from_id=1777, to_id=230, conf_th=0.60)

    num_images = len(image_paths)
    avg_per_image = total_duration / num_images if num_images else float("nan")
    throughput = num_images / total_duration if total_duration > 0 else float("inf")
    print(f"预测完成，共处理 {num_images} 张图片，结果已写入 {args.output_csv}")
    print(f"总耗时: {total_duration:.2f} 秒，平均每张: {avg_per_image * 1000:.2f} 毫秒，吞吐量: {throughput:.2f} 张/秒")
    if enable_tta:
        scales_repr = args.tta_scales if args.tta_scales else list(DEFAULT_TTA_SCALES)
        hflip_status = "关闭" if args.tta_no_hflip else "开启"
        print(f"TTA 已启用: scales={scales_repr}, 水平翻转={hflip_status}")
    if not args.no_temperature_scaling:
        if temperature_map_path is not None:
            print(f"温度映射文件: {temperature_map_path}")
        else:
            print("温度缩放：使用 checkpoint 中的温度或默认值")
    else:
        print("温度缩放已禁用")
    if args.temperature is not None:
        print(f"全局温度缩放: T={args.temperature:.4g}")
    for name, seconds, views, temperature in per_model_stats:
        details = f"{seconds:.2f} 秒"
        if views > 1:
            details += f", TTA x{views}"
        if temperature is not None:
            details += f", T={temperature:.4g}"
        print(f" - {name}: {details}")


if __name__ == "__main__":
    main()
