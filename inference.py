import argparse
import multiprocessing
import os
from os import path
import json

import numpy as np
import pandas as pd
from vllm import LLM
from datasets import load_dataset
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer

if __name__ == '__main__':

    # 路径设置
    MODEL_PATH = "model/mistral-mwpo"
    DATA_PATH = "datasets/alpaca_eval"
    OUTPUT_PATH = "inference/alpaca_eval-mistral-mwpo"
    temperature = 0.7
    max_new_tokens = 2048

    dataset = load_dataset(DATA_PATH, "alpaca_eval", trust_remote_code=True)["eval"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = LLM(MODEL_PATH,
                tensor_parallel_size=1,
                dtype='bfloat16')

    def process(row):
        row['messages'] = [{"role": "user", "content": row['instruction']}]
        return row

    dataset = dataset.map(process,
                          num_proc=multiprocessing.cpu_count(),
                          load_from_cache_file=False, )

    def process(row):
        row['messages'] = tokenizer.apply_chat_template(row['messages'],
                                                        tokenize=False,
                                                        add_generation_prompt=True)
        return row


    test_dataset = dataset.map(process,
                               num_proc=multiprocessing.cpu_count(),
                               load_from_cache_file=False, )

    sampling_params = SamplingParams(max_tokens=max_new_tokens,
                                     temperature=temperature,
                                     logprobs=1,
                                     stop_token_ids=[tokenizer.eos_token_id])

    vllm_generations = model.generate(test_dataset['messages'],
                                      sampling_params)

    responses = []
    dataset = dataset.select_columns(['messages'])
    dataset = dataset.to_list()
    for data, response in zip(dataset, vllm_generations):
        data['messages'].append({'role': 'assistant', 'content': response.outputs[0].text})
        avg_logp=[]
        for idx, logp in zip(response.outputs[0].token_ids, response.outputs[0].logprobs):
            avg_logp.append(logp[idx].logprob)
        data['avg_logp'] = np.mean(avg_logp)
        data['response_len'] = len(response.outputs[0].token_ids)
        responses.append(data)

    # 将结果保存到文件
    if not path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    df = pd.DataFrame(responses)
    df.to_json(f"{OUTPUT_PATH}/inference.json", orient='records', lines=True)

    print(f"Average generation length: {np.mean(df['response_len'])}")
    print(f"Responses saved to {OUTPUT_PATH}/inference.json")