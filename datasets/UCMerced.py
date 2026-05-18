# -*- coding: utf-8 -*-

from typing import Tuple, List
from os import path
import torch
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms
from .DatasetWrapper import DatasetWrapper


class UCMerced_LandUse_(DatasetWrapper[Tuple[torch.Tensor, int]]):
    num_classes = 21
    # TODO:修改mean和std
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image_size = 384  # matches vit_b_16 in models/__init__.py

    basic_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

    augment_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.TrivialAugmentWide(
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

    def __init__(
        self,
        root: str,
        train: bool,
        base_ratio: float,
        num_phases: int,
        augment: bool = False,
        inplace_repeat: int = 1,
        shuffle_seed: int | None = None,
    ) -> None:
        root = path.expanduser(root)
        image_root = path.join(root, "Images")
        full = ImageFolder(image_root)

        class_to_indices: List[List[int]] = [[] for _ in range(self.num_classes)]
        for idx, (_, label) in enumerate(full.samples):
            class_to_indices[label].append(idx)

        for label in range(self.num_classes):
            class_to_indices[label] = sorted(
                class_to_indices[label], key=lambda i: full.samples[i][0]
            )
            if len(class_to_indices[label]) < 100:
                raise ValueError(
                    f"Class {label} has only {len(class_to_indices[label])} images, expected 100."
                )

        indices: List[int] = []
        for label in range(self.num_classes):
            cls = class_to_indices[label]
            split = cls[:80] if train else cls[80:100]
            indices.extend(split)

        self.dataset = Subset(full, indices)
        labels = [full.targets[i] for i in indices]
        super().__init__(
            labels,
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )