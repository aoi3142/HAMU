
from pathlib import Path

from transformers import AutoImageProcessor, BaseImageProcessor
import torch
from datasets import Array3D, Dataset, Features, Value, load_dataset
from typing import Any, Optional
from ast import literal_eval
from transformers import set_seed
from hamu.models.resnet20 import ResNet20ForCIFAR, config as resnet20_config, image_processor as cifar_image_processor
from functools import partial
import hashlib
import numpy as np
import gc


def load_model_and_processor(
    model_name: str = "microsoft/resnet-18",
    seed: int = 42,
    num_labels: int = 10,
    freeze_batchnorm: bool = False,
    **kwargs,
) -> tuple[torch.nn.Module, BaseImageProcessor]:
    set_seed(seed)
    if model_name == "resnet20-cifar":
        model = ResNet20ForCIFAR(resnet20_config)
        image_processor = cifar_image_processor
    else:
        model = ResNet20ForCIFAR.from_pretrained(model_name)
        image_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    if freeze_batchnorm:
        model.set_batchnorm_training(False)

    return model, image_processor

def filter_by_label(only: Optional[list[int]], exclude: Optional[list[int]], example_label: int | list[int], complement: bool = False) -> bool | list[bool]:
    if not isinstance(example_label, int):
        return [filter_by_label(only, exclude, l) for l in example_label]
    excluded = bool(exclude) and (example_label in exclude)
    included = (not only) or (bool(only) and (example_label in only))
    res = included and not excluded
    return complement ^ res

def _processor_pixel_shape(processor: BaseImageProcessor, dataset: Dataset, dataset_field: str) -> tuple[int, ...]:
    sample = processor(images=[dataset[0][dataset_field]], return_tensors="pt").pixel_values
    return tuple(int(dim) for dim in sample.shape[1:])


def my_load_dataset(
        dataset_name: str="ylecun/mnist",
        split: str="train",
        only: Optional[list[int] | int | str] = None,
        exclude: Optional[list[int] | int | str] = None,
        split_ratio: float = 0.1,
        hard_probability: float = 0.0,
        shuffle_seed: int = 42,
        processor: Optional[BaseImageProcessor] = None,
        complement: bool = False,
        dataset_cache_dir: str = "artifacts/dataset_cache",
        ) -> Dataset:
    dataset: Dataset = load_dataset(dataset_name, split=split)
    if isinstance(processor, BaseImageProcessor):
        Path(dataset_cache_dir).mkdir(parents=True, exist_ok=True)
        pixel_shape = _processor_pixel_shape(processor, dataset, "img")
        pixel_shape_tag = "x".join(str(dim) for dim in pixel_shape)
        features = Features(
            {
                "pixel_values": Array3D(shape=pixel_shape, dtype="float32"),
                "label": dataset.features.get("label", Value("int64")),
            }
        )
        cache_file_name = (
            f"{dataset_cache_dir}/cache_{dataset_name.replace('/', '_')}_{split}_"
            f"{hashlib.md5(str(processor.to_dict()).encode()).hexdigest()}_array3d_{pixel_shape_tag}.arrow"
        )
        dataset = dataset.map(
            partial(preprocess_function, processor, "img", None),
            batched=True,
            remove_columns=dataset.column_names,
            features=features,
            load_from_cache_file=True,
            cache_file_name=cache_file_name
        )
        dataset.set_format(type="torch", columns=["pixel_values"], output_all_columns=True)
    if hard_probability == 1.0:
        if split == "test":
            return dataset
        dataset_split = dataset.train_test_split(test_size=split_ratio, shuffle=True, seed=shuffle_seed)
        if complement:
            dataset = dataset_split["train"]
        else:
            dataset = dataset_split["test"]
        return dataset

    if isinstance(only, str):
        only: List[int] | int = literal_eval(only)
    if isinstance(exclude, str):
        exclude: List[int] | int = literal_eval(exclude)

    if isinstance(only, int):
        only = [only]
    if isinstance(exclude, int):
        exclude = [exclude]

    filtered_dataset_mask = np.array(filter_by_label(only, exclude, dataset["label"]))
    filtered_dataset_indices = np.where(filtered_dataset_mask)[0]
    if hard_probability > 0.0 and split != "test":
        # Mix in some random examples with proportion of `hard_probability`
        random_len = int(hard_probability * len(filtered_dataset_indices))
        rng = np.random.default_rng(shuffle_seed)

        # Select 1-hard_probability portion from filtered_dataset_indices
        filtered_dataset_indices = rng.permutation(filtered_dataset_indices)
        filtered_dataset_indices_selected = filtered_dataset_indices[random_len:]
        filtered_dataset_indices_remainder = filtered_dataset_indices[:random_len]

        # select hard_probability portion from remainder of dataset
        remainder_dataset_indices = np.concatenate([np.where(~filtered_dataset_mask)[0], filtered_dataset_indices_remainder])
        remainder_dataset_indices = rng.permutation(remainder_dataset_indices)
        remainder_dataset_indices_selected = remainder_dataset_indices[:random_len]

        # Combine
        filtered_dataset_indices = np.concatenate([filtered_dataset_indices_selected, remainder_dataset_indices_selected])
    if complement:
        filtered_dataset_indices = np.delete(np.arange(len(dataset)), filtered_dataset_indices)
    dataset = dataset.select(filtered_dataset_indices)
    return dataset

def preprocess_function(
    processor: BaseImageProcessor,
    dataset_field: str,
    dataset_response_field: Optional[str],
    example_batch: dict[str, Any],
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    images = example_batch[dataset_field]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values.cpu().numpy()
    if not isinstance(images, list):
        pixel_values = pixel_values[0]
    return {
        "pixel_values": pixel_values,
        "label": example_batch["label"]
    }

def load_datasets(
        processor: BaseImageProcessor,
        dataset_name: str = "ylecun/mnist", dataset_subset: str = "[0]", dataset_split: str = "train",
        dataset_prompt_field: str = "img", #dataset_response_field: str,
        duplicate_dataset_name: str = "ylecun/mnist", duplicate_dataset_subset: str = "[]", duplicate_dataset_split: str = "train",
        duplicate_dataset_prompt_field: str = "img", #duplicate_dataset_response_field: str,
        forget_dataset_name: str = "ylecun/mnist", forget_dataset_subset: str = "[0]", forget_dataset_split: str = "train",
        forget_dataset_prompt_field: str = "img", #forget_dataset_response_field: str,
        add_duplicate_to_retain: bool = False,
        shuffle_seed: int = 42,
        hard_probability: float = 0.0,
        split_ratio: float = 0.1,
        dataset_cache_dir: str = "artifacts/dataset_cache",
        **kwargs,
        ) -> tuple[Dataset, Dataset, Dataset]:
    # Load dataset
    dataset = my_load_dataset(
        dataset_name=dataset_name, split=dataset_split,
        only=dataset_subset,
        hard_probability=hard_probability, shuffle_seed=shuffle_seed, split_ratio=split_ratio, processor=processor,
        dataset_cache_dir=dataset_cache_dir,
        complement=True
        )
    duplicate_dataset = my_load_dataset(
        dataset_name=duplicate_dataset_name, split=duplicate_dataset_split,
        # only=duplicate_dataset_subset,    # No filtering for duplicate dataset
        hard_probability=hard_probability, shuffle_seed=shuffle_seed, split_ratio=split_ratio, processor=processor,
        dataset_cache_dir=dataset_cache_dir
        )
    forget_dataset = my_load_dataset(
        dataset_name=forget_dataset_name, split=forget_dataset_split,
        only=forget_dataset_subset,
        hard_probability=hard_probability, shuffle_seed=shuffle_seed, split_ratio=split_ratio, processor=processor,
        dataset_cache_dir=dataset_cache_dir
        )
    gc.collect()
    return dataset, duplicate_dataset, forget_dataset

def get_pad_dataset(pad_length: int, processor: BaseImageProcessor, features: dict[str, Any], **kwargs) -> Dataset:
    pad_pixel_values = processor(images=[np.zeros((32, 32, 3), dtype=np.uint8)], return_tensors="pt").pixel_values[0]
    pad_dataset = Dataset.from_dict({
        "pixel_values": [pad_pixel_values] * pad_length,
        "label": [-1] * pad_length
    }, features=features)
    return pad_dataset
