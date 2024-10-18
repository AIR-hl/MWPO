import multiprocessing
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config
)
from trl.commands.cli_utils import DPOScriptArguments, TrlParser

from mwpo_config import MWPOConfig
from mwpo_trainer import MWPOTrainer


@dataclass
class ModelConfig(ModelConfig):
    use_rslora: bool = field(default=False, metadata={"help": ("Whether to use Rank-Stabilized LoRA.")},)

if __name__ == "__main__":
    parser = TrlParser([DPOScriptArguments, MWPOConfig, ModelConfig])
    args, training_args, model_config = parser.parse_args_and_config()

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)

    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path,
                                                 **model_kwargs)
    peft_config = get_peft_config(model_config)

    if peft_config is not None:
        ref_model = None
        peft_config.target_modules =["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",  "all-linear"]
        peft_config.use_rslora = model_config.use_rslora
        print(peft_config)
    elif training_args.precompute_ref_log_probs:
        ref_model = None
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path,
                                                         **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'left'

    train_dataset = load_dataset(args.dataset_name,
                                 split=args.dataset_train_split, )

    def process(row):
        row["prompt"] = tokenizer.apply_chat_template(row["chosen"][:-1], tokenize=False, add_generation_prompt=True)
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"][-1:], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"][-1:], tokenize=False)
        return row

    train_dataset = train_dataset.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )

    trainer = MWPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
