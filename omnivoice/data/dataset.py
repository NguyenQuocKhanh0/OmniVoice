#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset and data-loading utilities for training and evaluation.

Provides WebDataset-based iterable datasets, manifest parsing, and audio/token
loading. Used by ``omnivoice.training.builder.build_dataloaders()`` to construct
train and eval data loaders.
"""

import io
import json
import logging
import os
import random
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torchaudio
import webdataset as wds
from torch.utils.data import IterableDataset


def _broadcast_seed(seed: int) -> int:
    """
    Broadcast one integer seed from rank 0 to all ranks so every process
    in the same distributed job uses the same session seed.
    """
    if dist.is_available() and dist.is_initialized():
        backend = dist.get_backend()
        device = "cuda" if backend == "nccl" and torch.cuda.is_available() else "cpu"
        seed_tensor = torch.tensor(
            [seed if dist.get_rank() == 0 else 0],
            dtype=torch.long,
            device=device,
        )
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())
    return seed


def _system_time_seed() -> int:
    """
    Create a seed from system time. In distributed training, sync it across ranks.
    """
    # keep seed in a reasonable integer range
    seed = int(time.time_ns() % (2**31 - 1))
    return _broadcast_seed(seed)


def load_audio_webdataset(data, sample_rate: int = 24000, device="cpu"):
    """
    Load audio from bytes data and resample to the target sample rate if needed.
    Return a tensor of shape (1, num_samples)
    """
    audio, sr = torchaudio.load(io.BytesIO(data))
    audio = audio.to(device)
    if audio.size(dim=0) > 1:
        audio = torch.mean(audio, dim=0)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio


def prepare_data_manifests_from_json(
    data_config: str,
) -> Tuple[List[Tuple[str, str, int, float]], List[Tuple[str, str, int, float]]]:
    train_manifests = []
    dev_manifests = []
    with open(data_config, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data["train"]:
            manifest_paths = item["manifest_path"]
            repeat = item.get("repeat", 1)
            for manifest_path in manifest_paths:
                assert os.path.isfile(manifest_path), f"{manifest_path} is not a file."
                train_manifests.extend(
                    webdataset_manifest_reader(manifest_path) * repeat
                )
        if "dev" in data:
            for item in data["dev"]:
                manifest_paths = item["manifest_path"]
                repeat = item.get("repeat", 1)
                for manifest_path in manifest_paths:
                    dev_manifests.extend(
                        webdataset_manifest_reader(manifest_path) * repeat
                    )
    return train_manifests, dev_manifests


def webdataset_manifest_reader(
    manifest_path: str,
) -> List[Tuple[str, str]]:
    manifests = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid manifest line: {line}. "
                    f"Each line must contain "
                    "tar_path, label_jsonl_path, num_items, num_seconds."
                )
            tar_path, label_jsonl_path, num_items, num_seconds = (
                parts[0],
                parts[1],
                int(parts[2]),
                float(parts[3]),
            )
            manifests.append((tar_path, label_jsonl_path, num_items, num_seconds))
    return manifests


class SampleDecoder:
    def __init__(
        self,
        tar_to_label: Dict,
        sample_rate: int = 24000,
        audio_format: Optional[Tuple[str]] = None,
        normalize_audio: bool = True,
    ):
        self.tar_to_label = tar_to_label
        self.sample_rate = sample_rate
        self.label_dataset = None
        if audio_format is None:
            self.audio_format = ("flac", "wav", "mp3")
        else:
            self.audio_format = audio_format
        self.normalize_audio = normalize_audio

    def __call__(self, sample):
        return_dict = {}
        src = sample["__url__"]
        key = sample["__key__"]
        if (
            self.label_dataset is None
            or self.label_dataset.path != self.tar_to_label[src]
        ):
            self.label_dataset = LabelDataset(self.tar_to_label[src])

        audio = torch.empty(0)
        if "npy" in sample:
            audio_tokens = torch.from_numpy(sample["npy"])
            return_dict["audio_tokens"] = audio_tokens
        else:
            for ext in self.audio_format:
                if ext in sample:
                    audio = load_audio_webdataset(
                        sample[ext], sample_rate=self.sample_rate
                    )
                    if self.normalize_audio:
                        audio = (audio / (audio.abs().max() + 1e-7)) * 0.9
                    break
            return_dict["audio"] = audio
            return_dict["audio_duration"] = audio.size(-1) / self.sample_rate

        label = self.label_dataset[key]
        return_dict["label"] = label
        return return_dict


class LabelDataset:
    def __init__(self, jsonl_path: str):
        self._labels = {}
        self.path = jsonl_path
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Label jsonl file {jsonl_path} does not exist.")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "id" in item:
                    self._labels[item["id"]] = item

    def __getitem__(self, key):
        return self._labels[key]


class IterableDataReader:
    sample_rate: int

    def set_epoch(self, epoch: int):
        raise NotImplementedError

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class WrappedIterableDataset(IterableDataset):
    def set_epoch(self, epoch: int):
        raise NotImplementedError

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        raise NotImplementedError


class WebDatasetReader(IterableDataReader):
    def __init__(
        self,
        manifests: List[Tuple[str, str, int, float]],
        evaluation: bool = False,
        shuffle_buffer_size: int = 20000,
        sample_rate: int = 24000,
    ):
        self.shuffle_buffer_size = shuffle_buffer_size
        self.evaluation = evaluation
        self.epoch = 0
        self.base_seed: Optional[int] = None
        self.current_seed: int = 0

        self.orig_urls = []
        self.tar_to_label = {}
        self.num_items = 0
        self.num_seconds = 0.0
        for tar_path, label_jsonl_path, num_items, num_seconds in manifests:
            self.orig_urls.append(tar_path)
            self.tar_to_label[tar_path] = label_jsonl_path
            self.num_items += num_items
            self.num_seconds += num_seconds
        self.urls = self.orig_urls.copy()
        self.sample_decoder = SampleDecoder(
            tar_to_label=self.tar_to_label,
            sample_rate=sample_rate,
        )
        self.sample_rate = sample_rate

    def _ensure_base_seed(self):
        if self.base_seed is None:
            self.base_seed = _system_time_seed()
            logging.info(
                f"WebDatasetReader session base_seed={self.base_seed} "
                f"(generated from system time)"
            )

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._ensure_base_seed()
        self.current_seed = self.base_seed + epoch

        self.urls = self.orig_urls.copy()
        if not self.evaluation:
            random.Random(self.current_seed).shuffle(self.urls)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self._ensure_base_seed()
        if not self.evaluation and self.current_seed == 0:
            self.current_seed = self.base_seed + self.epoch

        dataset = wds.WebDataset(
            self.urls,
            shardshuffle=False,
            workersplitter=wds.split_by_worker,
            nodesplitter=wds.split_by_node,
        )

        pipeline = dataset.decode().map(self.sample_decoder)
        if not self.evaluation:
            pipeline = pipeline.shuffle(
                self.shuffle_buffer_size,
                seed=self.current_seed,
            )
        return iter(pipeline)

    def __len__(self) -> int:
        return self.num_items


class JsonlDatasetReader(IterableDataReader):
    """Read raw JSONL and load audio files, matching WebDatasetReader output format."""

    def __init__(
        self,
        jsonl_path: str,
        sample_rate: int = 24_000,
        shuffle: bool = True,
        shuffle_seed: Optional[int] = None,
        normalize_audio: bool = True,
    ):
        self.jsonl_path = jsonl_path
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.normalize_audio = normalize_audio

        # nếu user truyền seed thủ công thì giữ seed đó,
        # nếu không thì lấy seed từ thời gian hệ thống
        self.base_seed = shuffle_seed
        self.current_epoch = 0
        self.current_seed = 0

    def _ensure_base_seed(self):
        if self.base_seed is None:
            self.base_seed = _system_time_seed()
            logging.info(
                f"JsonlDatasetReader session base_seed={self.base_seed} "
                f"(generated from system time)"
            )

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        self._ensure_base_seed()
        self.current_seed = self.base_seed + epoch

    def _read_lines(self) -> list[dict]:
        entries = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        if self.shuffle:
            self._ensure_base_seed()
            if self.current_seed == 0:
                self.current_seed = self.base_seed + self.current_epoch

            rng = random.Random(self.current_seed)
            rng.shuffle(entries)
            logging.info(
                f"Shuffled {len(entries)} JSONL entries "
                f"(base_seed={self.base_seed}, epoch={self.current_epoch}, "
                f"seed={self.current_seed})"
            )
        return entries

    def _stream_lines(self):
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def __iter__(self):
        source = self._read_lines() if self.shuffle else self._stream_lines()

        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            source = [item for i, item in enumerate(source) if i % world_size == rank]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            source = (
                item
                for i, item in enumerate(source)
                if i % worker_info.num_workers == worker_info.id
            )

        for meta in source:
            audio_path = meta.get("audio_path")
            if not audio_path or not os.path.exists(audio_path):
                logging.warning(
                    f"Skipping {meta.get('id', '?')}: audio_path missing or not found"
                )
                continue
            try:
                waveform, sr = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != self.sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, sr, self.sample_rate
                    )
                if self.normalize_audio:
                    waveform = (waveform / (waveform.abs().max() + 1e-7)) * 0.9
                meta["audio_duration"] = waveform.shape[1] / self.sample_rate
                yield {"audio": waveform, "label": meta}
            except Exception as e:
                logging.warning(f"Skipping {meta.get('id', '?')}: {e}")


class MuxWebDatasetReader(IterableDataReader):
    def __init__(
        self,
        readers: List[WebDatasetReader],
        weights: Optional[List[float]] = None,
        stop_early: bool = False,
        seed: Optional[int] = None,
    ):
        self.readers = readers
        self.weights = weights
        self.stop_early = stop_early
        self.base_seed = seed
        self.epoch = 0
        self.current_seed = 0

    def _ensure_base_seed(self):
        if self.base_seed is None:
            self.base_seed = _system_time_seed()
            logging.info(
                f"MuxWebDatasetReader session base_seed={self.base_seed} "
                f"(generated from system time)"
            )

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._ensure_base_seed()
        self.current_seed = self.base_seed + epoch

        for reader in self.readers:
            reader.set_epoch(epoch)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self._ensure_base_seed()
        if self.current_seed == 0:
            self.current_seed = self.base_seed + self.epoch

        mux_iterator = LazyIteratorMultiplexer(
            *self.readers,
            stop_early=self.stop_early,
            weights=self.weights,
            seed=self.current_seed,
        )
        return iter(mux_iterator)


class LazyIteratorMultiplexer:
    def __init__(
        self,
        *iterators: IterableDataReader,
        stop_early: bool = False,
        weights: Optional[List[float]] = None,
        seed: int = 0,
    ) -> None:
        self.iterators = list(iterators)
        self.stop_early = stop_early
        self.seed = seed

        assert (
            len(self.iterators) > 1
        ), "There have to be at least two iterables to multiplex."

        if weights is None:
            if all(hasattr(it, "__len__") for it in self.iterators):
                lengths = [len(it) for it in self.iterators]
                total_length = sum(lengths)
                self.weights = [length / total_length for length in lengths]
            else:
                self.weights = [1] * len(self.iterators)
        else:
            self.weights = weights

        assert len(self.iterators) == len(self.weights)

    def __iter__(self):
        rng = random.Random(self.seed)
        iters = [iter(it) for it in self.iterators]
        exhausted = [False for _ in range(len(iters))]

        def should_continue():
            if self.stop_early:
                return not any(exhausted)
            else:
                return not all(exhausted)

        while should_continue():
            active_indexes, active_weights = zip(
                *[
                    (i, w)
                    for i, (is_exhausted, w) in enumerate(zip(exhausted, self.weights))
                    if not is_exhausted
                ]
            )
            idx = rng.choices(active_indexes, weights=active_weights, k=1)[0]
            selected = iters[idx]
            try:
                item = next(selected)
                yield item
            except StopIteration:
                exhausted[idx] = True
                continue

    def __len__(self) -> int:
        return sum(len(iterator) for iterator in self.iterators)
