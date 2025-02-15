import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import multiprocessing
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from mwpo_config import MWPOConfig
from mwpo_trainer import MWPOTrainer

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, MWPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    quantization_config = get_quantization_config(model_config)

    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )

    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    dataset=load_dataset(script_args.dataset_name,
                           split=script_args.dataset_train_split)
    dataset=dataset.select_columns(['prompt', 'chosen', 'rejected'])

    # def process(row):
    #     row['prompt'] = [{'role': 'user', 'content': row['input']}] if row['system'] is None else \
    #         [{'role': 'system', 'content': row['system']},
    #          {'role': 'user', 'content': row['input']}]
    #     row['chosen'] = [{'role': 'assistant', 'content': row['chosen']}]
    #     row['rejected'] = [{'role': 'assistant', 'content': row['rejected']}]
    #     return row

    trainer = MWPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)