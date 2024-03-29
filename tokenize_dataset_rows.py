import argparse
import json
from tqdm import tqdm
import datasets
import transformers
from typing import Union
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", type=str, help="checkpoint, like `THUDM/chatglm-6b`")
parser.add_argument("--input_file", type=str, help="Instruction 数据文件地址，文件中每一行都是json格式，包含一个输出和一个输出")
parser.add_argument("--instruct_key", type=str, default="instruction", help="你的jsonl文件里，prompt的指令字段是什么")
parser.add_argument("--prompt_key", type=str, default=None, help="你的jsonl文件里，prompt的输入字段是什么")
parser.add_argument("--target_key", type=str, default="output", help="你的jsonl文件里，prompt的输出字段是什么")
parser.add_argument("--save_name", type=str, default=f"temp", help="经过tokenize之后的数据集的存放位置")
parser.add_argument("--template_name", type=str, default="alpaca", help="The name of prompt template.")
parser.add_argument("--max_seq_length", type=int, default=2048)
parser.add_argument("--skip_overlength", type=bool, default=False)
args = parser.parse_args()
model_checkpoint = args.model_checkpoint


class Prompter(object):
    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


def preprocess(tokenizer, prompter, config, example, max_seq_length, instruct_key, prompt_key, target_key):
    instruct = example[instruct_key]
    target = example[target_key]
    if prompt_key:
        prompt = example[prompt_key]
        instruction = prompter.generate_prompt(instruct, prompt)
    else:
        instruction = prompter.generate_prompt(instruct)
    prompt_ids = tokenizer.encode(instruction, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(target, max_length=max_seq_length, truncation=True, add_special_tokens=False)

    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "labels": [-100]*len(prompt_ids)+target_ids+[config.eos_token_id], "attention_mask": [1]*len(input_ids)}


def read_jsonl(path, max_seq_length, instruct_key, prompt_key, target_key, skip_overlength=False):
    prompter = Prompter(args.template_name, verbose=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_checkpoint, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_checkpoint, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, prompter, config, example, max_seq_length, instruct_key, prompt_key, target_key)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] += [tokenizer.pad_token_id] * (max_seq_length - len(feature["input_ids"]))
            feature["labels"] += [-100] * (max_seq_length - len(feature["labels"]))
            feature["attention_mask"] += [0] * (max_seq_length - len(feature["attention_mask"]))

            assert len(feature["input_ids"]) == len(feature["labels"])

            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            feature["labels"] = feature["labels"][:max_seq_length]
            feature["attention_mask"] = feature["attention_mask"][:max_seq_length]
            yield feature


input_file_path = f'data/{args.input_file}'
save_path = f"data/tokenized_data/{args.save_name}"
dataset = datasets.Dataset.from_generator(
    lambda: read_jsonl(input_file_path, args.max_seq_length, args.instruct_key, args.prompt_key, args.target_key, args.skip_overlength)
)
dataset.save_to_disk(save_path)
