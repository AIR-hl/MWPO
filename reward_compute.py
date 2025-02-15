import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM


@torch.inference_mode()
def armorm_rw(model_path, dataset_path, split):
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                               device_map='auto',
                                                               trust_remote_code=True,
                                                               torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    dataset = load_dataset(dataset_path, split=split)
    avg_length = np.mean(dataset['response_len'])

    def process(row):
        row["messages"] = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        return row

    dataset = dataset.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    dataset = [tokenizer(row, return_tensors="pt", add_special_tokens=False) for row in dataset['messages']]
    dataset = [{'input_ids': row['input_ids'].cuda(), 'attention_mask': row['attention_mask'].cuda()} for row in dataset]

    rewards = []
    with torch.no_grad():
        for step, data in tqdm(enumerate(dataset), desc='reward computing', total=len(dataset)):
            reward = model(**data)
            rewards.append(reward.score.float().item())

    return rewards, avg_length


if __name__ == '__main__':
    # 路径设置
    rm_name = "ArmoRM-Llama3-8B-v0.1"
    rm_path = f"model/{rm_name}"
    reference_path = f""
    alpaca_eval_path = f"tatsu-lab/alpaca_eval" # for 'pair-preference-model-LLaMA3-8B'
    data_path = "" # the inference result
    output_dir = f"rewards/alpaca_eval-mistral-mwpo/{rm_name}"
    split = 'train'

    if rm_name == "ArmoRM-Llama3-8B-v0.1":
        rewards, avg_length = armorm_rw(rm_path, data_path, split)
    else:
        ValueError("the reward model doesn't exist")

    ref_rewards = pd.read_csv(f'{reference_path}/rewards.csv')

    avg_reward = np.mean(rewards)
    print(f"Mean Reward: {avg_reward}")

    win_rate = (rewards > ref_rewards['rewards']).mean()
    print(f"Win Rate: {win_rate} || Mean Reward: {avg_reward} || Average Length: {avg_length} || Annotator: {rm_name}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pd.DataFrame(rewards, columns=['rewards']).to_csv(f"{output_dir}/rewards.csv", index=False)
    pd.DataFrame({
        'win_rate': [win_rate],
        'avg_reward': [avg_reward],
        'avg_length': [avg_length],
        'annotator': [rm_name]
    }).to_csv(f'{output_dir}/statistics.csv', index=False)
