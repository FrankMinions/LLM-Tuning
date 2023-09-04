from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers.modeling_utils import PreTrainedModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from peft.utils import CONFIG_NAME, WEIGHTS_NAME
from dataclasses import dataclass, field
import datasets
import os
from pprint import pprint as print
from typing import Union, List

model_name_or_path = "baichuan-inc/baichuan-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


@dataclass
class FinetuningArguments:
    tokenized_dataset: str = field(
        metadata={"help": "Dataset after tokenized."}
    )
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=32)
    finetuning_type: str = field(default="lora")
    lora_dropout: float = field(default=0.1)
    lora_target: Union[List[str], str] = field(default=None)


@dataclass
class ModelArguments:
    checkpoint_dir: str = field(default=None)   # Need to specify if you want to resume checkpoints.
    resume_lora_training: bool = field(default=False)   # Need to specify 'True' if you want to resume checkpoints.

    def __post_init__(self):
        if self.checkpoint_dir is not None:  # support merging multiple lora weights
            self.checkpoint_dir = [cd.strip() for cd in self.checkpoint_dir.split(",")]


def load_trainable_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    weights_file = os.path.join(checkpoint_dir, WEIGHTS_NAME)
    if not os.path.exists(weights_file):
        print("Provided path ({}) does not contain pre-trained weights.".format(checkpoint_dir))
        return False
    model_state_dict = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(model_state_dict, strict=False)  # skip missing keys
    return True


def print_trainable_params(model: torch.nn.Module) -> None:
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
                trainable_params, all_param, 100 * trainable_params / all_param))


def _init_adapter(
        model: PreTrainedModel,
        model_args: ModelArguments,
        finetuning_args: FinetuningArguments,
        is_trainable: bool,
        is_mergeable: bool
) -> PreTrainedModel:
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """

    if finetuning_args.finetuning_type == "none" and is_trainable:
        raise ValueError("You cannot use finetuning_type=none while training.")

    if finetuning_args.finetuning_type == "full":
        print("Fine-tuning method: Full")
        model = model.float()

    if finetuning_args.finetuning_type == "freeze":
        print("Fine-tuning method: Freeze")
        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in finetuning_args.trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

    if model_args.checkpoint_dir is not None:
        if finetuning_args.finetuning_type != "lora":
            assert is_mergeable and len(
                model_args.checkpoint_dir) == 1, "Only LoRA tuning accepts multiple checkpoints."
            assert load_trainable_params(model,
                                         model_args.checkpoint_dir[0]), "Model checkpoint is not correctly loaded."
        else:
            assert is_mergeable or len(
                model_args.checkpoint_dir) == 1, "Quantized model only accepts a single checkpoint."

    if finetuning_args.finetuning_type == "lora":
        print("Fine-tuning method: LoRA")
        lastest_checkpoint = None

        if model_args.checkpoint_dir is not None:
            if os.path.exists(os.path.join(model_args.checkpoint_dir[0], WEIGHTS_NAME)) and \
                    not os.path.exists(os.path.join(model_args.checkpoint_dir[0], CONFIG_NAME)):
                raise ValueError("The given checkpoint may be not a LoRA checkpoint, \
                                  please specify `--finetuning_type full/freeze` instead.")

            if (is_trainable and model_args.resume_lora_training) or (
            not is_mergeable):  # continually train on the lora weights
                checkpoints_to_merge, lastest_checkpoint = model_args.checkpoint_dir[:-1], model_args.checkpoint_dir[-1]
            else:
                checkpoints_to_merge = model_args.checkpoint_dir

            for checkpoint in checkpoints_to_merge:
                model = PeftModel.from_pretrained(model, checkpoint)
                model = model.merge_and_unload()

            if len(checkpoints_to_merge) > 0:
                print("Merged {} model checkpoint(s).".format(len(checkpoints_to_merge)))

            if lastest_checkpoint is not None:  # resume lora training or quantized inference
                model = PeftModel.from_pretrained(model, lastest_checkpoint, is_trainable=is_trainable)

        if is_trainable and lastest_checkpoint is None:  # create new lora weights while training
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=finetuning_args.lora_rank,
                lora_alpha=finetuning_args.lora_alpha,
                lora_dropout=finetuning_args.lora_dropout,
                target_modules=finetuning_args.lora_target
            )
            model = get_peft_model(model, lora_config)

    if model_args.checkpoint_dir is not None:
        print("Loaded fine-tuned model from checkpoint(s): {}".format(",".join(model_args.checkpoint_dir)))

    return model


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


tokenizer.pad_token = tokenizer.unk_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def main():
    writer = SummaryWriter()
    model_args, finetune_args, training_args = HfArgumentParser(
        (ModelArguments, FinetuningArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # load dataset
    dataset = datasets.load_from_disk('data/tokenized_data/' + finetune_args.tokenized_dataset)
    print(f'\n{len(dataset)}\n=')

    # init model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, load_in_8bit=False, trust_remote_code=True, device_map="auto"
    )
    print(model.hf_device_map)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)

    # setup peft
    # Default not to merge with the original model
    model = _init_adapter(model, model_args, finetune_args, is_trainable=True, is_mergeable=False)
    print_trainable_params(model)

    # start train
    model.save_pretrained(training_args.output_dir)
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
    trainer.train()
    writer.close()

    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
