
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, set_seed
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftModelForCausalLM
import torch
from datasets import Dataset, load_dataset, load_from_disk
import os
from transformers.tokenization_utils_base import BatchEncoding
from typing import Any, Optional

def load_model_and_processor(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    seed: int = 42,
    add_lora: bool = False,
    lora_name: Optional[str] = None,
    load_original: bool = False,
    **kwargs,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    # Load base model
    # Use device_map='auto' only when not in distributed mode (not using accelerate)
    # Check for distributed environment via environment variables or world_size
    is_distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    ) or int(os.environ.get("WORLD_SIZE", "1")) > 1
    
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            device_map=None if is_distributed else 0,
            dtype="auto",
            trust_remote_code=True,
            is_trainable=True,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None if is_distributed else 0,
            dtype="auto",
            trust_remote_code=True,
        )
    if add_lora:
        # PEFT LoRA configuration
        peft_config = LoraConfig(
            **kwargs,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        
        set_seed(seed)
        if isinstance(model, PeftModelForCausalLM):
            model.add_adapter(lora_name, peft_config)
        else:
            model = get_peft_model(model, peft_config)
    if load_original:
        model.load_adapter(model_name, "original", is_trainable=False)
        model.set_adapter(lora_name if add_lora else "default")
    
    model.config.use_cache = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model.name_or_path,
        trust_remote_code=True,
        legacy=True # Original experiments in the paper used legacy tokenizers, but for future compatibility we switch to the new tokenizers. The new tokenizers should be mostly compatible, but may have some differences in tokenization which could lead to slightly different results compared to the paper. We will update the paper with results using the new tokenizers in the future.
        )
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

def my_load_dataset(
        dataset_name: str,
        subset: str,
        split: str
        ) -> Dataset:
    if os.path.exists(f"{dataset_name}/{subset}"):
        dataset = load_from_disk(f"{dataset_name}/{subset}")[split]
    else:
        dataset = load_dataset(dataset_name, subset, split=split)
    return dataset

def preprocess_function(
    tokenizer: PreTrainedTokenizer,
    dataset_prompt_field: str,
    dataset_response_field: Optional[str],
    example: dict[str, Any],
) -> dict[str, list[int]]:
    if (not dataset_response_field) or (dataset_prompt_field == dataset_response_field):
        # Completion task
        content = example[dataset_prompt_field]
        if isinstance(content, list):
            content = content[0]
        prompt_completion_ids = tokenizer.encode(content)
        completion_mask = [1] * len(prompt_completion_ids)
    else:
        # Chat task
        user_content = example[dataset_prompt_field]
        assistant_content = example[dataset_response_field]
        if isinstance(user_content, list):
            user_content = user_content[0]
        if isinstance(assistant_content, list):
            assistant_content = assistant_content[0]
        prompt_ids = tokenizer.apply_chat_template([
            {"role": "user", "content": user_content}
        ], tokenize=True, add_generation_prompt=True, return_dict=False)
        prompt_completion_ids = tokenizer.apply_chat_template([
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ], tokenize=True, return_dict=False)
        if isinstance(prompt_ids, (dict, BatchEncoding)):
            prompt_ids = prompt_ids["input_ids"]
        if isinstance(prompt_completion_ids, (dict, BatchEncoding)):
            prompt_completion_ids = prompt_completion_ids["input_ids"]
        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
    return {
        "input_ids": prompt_completion_ids,
        "completion_mask": completion_mask
    }
    # return {
    #     "input_ids": [idx],  # Placeholder: use index as input_id
    #     "completion_mask": [1]  # Placeholder: dummy mask
    # }

def load_datasets(
        tokenizer: PreTrainedTokenizer,
        dataset_name: str, dataset_subset: str, dataset_split: str,
        dataset_prompt_field: str, dataset_response_field: str,
        duplicate_dataset_name: str, duplicate_dataset_subset: str, duplicate_dataset_split: str,
        duplicate_dataset_prompt_field: str, duplicate_dataset_response_field: str,
        forget_dataset_name: str, forget_dataset_subset: str, forget_dataset_split: str,
        forget_dataset_prompt_field: str, forget_dataset_response_field: str,
        add_duplicate_to_retain: bool,
        dataset_cache_dir: str = "artifacts/dataset_cache",
        **kwargs,
        ) -> tuple[Dataset, Dataset, Dataset]:
    # Load dataset
    dataset = my_load_dataset(dataset_name, dataset_subset, split=dataset_split)
    if add_duplicate_to_retain:
        duplicate_dataset = my_load_dataset(duplicate_dataset_name, duplicate_dataset_subset, split=duplicate_dataset_split)
    else:
        duplicate_dataset = Dataset.from_dict({k: [] for k in dataset.column_names})
    forget_dataset = my_load_dataset(forget_dataset_name, forget_dataset_subset, split=forget_dataset_split)
    return dataset, duplicate_dataset, forget_dataset

def get_pad_dataset(pad_length: int, processor: AutoTokenizer, **kwargs) -> Dataset:
    pad_dataset = Dataset.from_dict({
        "input_ids": [[processor.eos_token_id]] * pad_length,
        "completion_mask": [[0]] * pad_length,
    })
    return pad_dataset
