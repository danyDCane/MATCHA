import os
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def _resolve_domain_dir(vlcs_root: str, dataset_name: str) -> str:
    """
    Resolve domain directory name with some case-insensitivity fallback.

    Your local data commonly uses uppercase names (e.g. SUN, PASCAL).
    """
    candidates = [
        os.path.join(vlcs_root, dataset_name),
        os.path.join(vlcs_root, dataset_name.upper()),
        os.path.join(vlcs_root, dataset_name.lower()),
        os.path.join(vlcs_root, dataset_name.capitalize()),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # If nothing matches, return the first candidate so the error message is informative.
    return candidates[0]


def _is_digit_class_names(classes: List[str]) -> bool:
    if not classes:
        return False
    return all(isinstance(c, str) and c.isdigit() for c in classes)


def _remap_label_from_imagefolder(imfolder: ImageFolder, label_idx: int) -> int:
    # If class folder names are digits like "0","1","2","3","4", remap to numeric labels.
    if hasattr(imfolder, "classes") and _is_digit_class_names(list(imfolder.classes)):
        return int(imfolder.classes[label_idx])
    return int(label_idx)


class VLCSFullTestDataset(Dataset):
    """
    VLCS wrapper:
      trainset = full + test for each source domain
      target testset = full + test for the leave-out domain

    Expected folder (per domain):
      VLCS/<domain>/
        full/0..4
        test/0..4

    Your requirement: 5 classes, folder names are "0","1","2","3","4".
    """

    def __init__(
        self,
        root: str,
        dataset_name: str,
        transform=None,
        full_dir_name: str = "full",
        test_dir_name: str = "test",
    ):
        self.dataset_name = dataset_name
        self.num_class = 5

        vlcs_root = os.path.join(root, "VLCS")
        domain_dir = _resolve_domain_dir(vlcs_root, dataset_name)
        if not os.path.exists(domain_dir):
            raise FileNotFoundError(f'VLCS domain path not found: {domain_dir}')

        print(f'[VLCSFullTestDataset] Loading domain "{dataset_name}" from: {domain_dir}')

        # Full split
        full_dir = os.path.join(domain_dir, full_dir_name)
        if os.path.exists(full_dir):
            print(f'[VLCSFullTestDataset] Using full split from: {full_dir}')
            self._full_sources = [ImageFolder(root=full_dir, transform=transform)]
        else:
            # Fallback to Dassl naming: train + crossval => full
            train_dir = os.path.join(domain_dir, "train")
            crossval_dir = os.path.join(domain_dir, "crossval")
            print(
                f'[VLCSFullTestDataset] full dir "{full_dir_name}" not found; '
                f'fallback to "{train_dir}" + "{crossval_dir}"'
            )
            if not os.path.exists(train_dir) or not os.path.exists(crossval_dir):
                raise FileNotFoundError(
                    f'Cannot find "{full_dir_name}" or fallback "{train_dir}" + "{crossval_dir}".'
                )
            self._full_sources = [
                ImageFolder(root=train_dir, transform=transform),
                ImageFolder(root=crossval_dir, transform=transform),
            ]

        # Test split
        test_dir = os.path.join(domain_dir, test_dir_name)
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f'VLCS test path not found: {test_dir}')
        print(f'[VLCSFullTestDataset] Using test split from: {test_dir}')
        self._test_source = ImageFolder(root=test_dir, transform=transform)

        # Expose targets for samplers (RandomClassSampler relies on `.targets`)
        self.targets: List[int] = []
        self._full_lengths: List[int] = []
        for ds in self._full_sources:
            self._full_lengths.append(len(ds))
            for y in ds.targets:
                self.targets.append(_remap_label_from_imagefolder(ds, int(y)))

        for y in self._test_source.targets:
            self.targets.append(_remap_label_from_imagefolder(self._test_source, int(y)))

        # Cache total length for __len__ / __getitem__
        self._full_total = sum(self._full_lengths)
        self._test_total = len(self._test_source)

        print(
            f'[VLCSFullTestDataset] Successfully loaded: '
            f'full={self._full_total} samples, test={self._test_total} samples '
            f'from domain "{dataset_name}"'
        )
        print(f'[VLCSFullTestDataset] Number of classes: {self.num_class}')

    def __len__(self) -> int:
        return self._full_total + self._test_total

    def _getitem_full(self, idx: int) -> Tuple[torch.Tensor, int]:
        # idx is in [0, full_total)
        for ds_idx, ds in enumerate(self._full_sources):
            L = self._full_lengths[ds_idx]
            if idx < L:
                x, y = ds[idx]
                return x, _remap_label_from_imagefolder(ds, int(y))
            idx -= L
        raise IndexError("Index out of range in full sources")

    def __getitem__(self, index: int):
        if index < self._full_total:
            return self._getitem_full(index)
        x, y = self._test_source[index - self._full_total]
        return x, _remap_label_from_imagefolder(self._test_source, int(y))

