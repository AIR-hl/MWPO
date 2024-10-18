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

@torch.inference_mode()
def pairpm(model_path, dataset_path, reference_path, split):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.chat_template = "\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"

    prompt_template = "[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
    token_id_A = tokenizer.encode("A", add_special_tokens=False)
    token_id_B = tokenizer.encode("B", add_special_tokens=False)
    assert len(token_id_A) == 1 and len(token_id_B) == 1
    token_id_A = token_id_A[0]
    token_id_B = token_id_B[0]
    temperature = 1.0

    dataset = load_dataset(dataset_path, split=split, keep_in_memory=True)
    avg_length = np.mean(dataset['response_len'])
    reference = load_dataset(reference_path, "alpaca_eval_gpt4_baseline")['eval']
    wins = []

    model.eval()
    for i, (data, ref) in enumerate(tqdm(zip(dataset, reference), total=len(dataset))):
        prompt = ref['instruction']
        response_chosen = data['messages'][-1]['content']
        response_rejected = ref["output"]
        instruction = [{"role": "user", "content": prompt}]
        context = tokenizer.apply_chat_template(instruction)
        responses = [response_chosen, response_rejected]
        probs_chosen = []

        for chosen_position in [0, 1]:
            # Swap order to mitigate position bias
            response_A = responses[chosen_position]
            response_B = responses[1 - chosen_position]
            prompt = prompt_template.format(context=context, response_A=response_A, response_B=response_B)
            message = [{"role": "user", "content": prompt}]

            input_ids = tokenizer.apply_chat_template(message, return_tensors='pt').cuda()

            with torch.no_grad():
                output = model(input_ids)
            logit_A = output.logits[0, -1, token_id_A].item()
            logit_B = output.logits[0, -1, token_id_B].item()
            # Take softmax to get the probability
            Z = np.exp(logit_A / temperature) + np.exp(logit_B / temperature)
            logit_chosen = [logit_A, logit_B][chosen_position]
            prob_chosen = np.exp(logit_chosen / temperature) / Z
            probs_chosen.append(prob_chosen)

        avg_prob_chosen = np.mean(probs_chosen)
        correct = float(avg_prob_chosen > 0.5)
        wins.append(correct)

    return wins, avg_length


if __name__ == '__main__':
    # 路径设置
    rm_name = "ArmoRM-Llama3-8B-v0.1"
    rm_path = f"model/{rm_name}"
    reference_path = f"datasets/alpaca_eval_reference/ArmoRM-Llama3-8B-v0.1"
    alpaca_eval_path = f"datasets/alpaca_eval" # for 'pair-preference-model-LLaMA3-8B'
    data_path = "inference/alpaca_eval-mistral-mwpo"
    output_dir = f"rewards/alpaca_eval-mistral-mwpo/{rm_name}"
    split = 'train'

    if rm_name == "pair-preference-model-LLaMA3-8B":
        wins, avg_length = pairpm(rm_path, data_path, alpaca_eval_path, split)
        win_rate = np.mean(wins)
        print(f"Win Rate: {win_rate} || Average Length: {avg_length} || Annotator: {rm_name}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        pd.DataFrame(wins, columns=['wins']).to_csv(f"{output_dir}/rewards.csv", index=False)
        pd.DataFrame({
            'win_rate': [win_rate],
            'avg_length': [avg_length],
            'annotator': [rm_name]
        }).to_csv(f'{output_dir}/statistics.csv', index=False)
    else:
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
