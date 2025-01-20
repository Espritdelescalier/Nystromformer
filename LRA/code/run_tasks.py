from model_wrapper import ModelForSC, ModelForSCDual
from dataset import LRADataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import os
import json
import pickle
import numpy as np
import argparse
import math
import itertools
import lra_config

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model",
                    dest="model", required=True)
parser.add_argument("--task", type=str, help="task",
                    dest="task", required=True)
parser.add_argument("--skip_train", type=int,
                    help="skip_train", dest="skip_train", default=0)
parser.add_argument("--select_type", type=str, help="Cur selection type (random, step, abs, sum)",
                    required=False, default="random")
parser.add_argument("--select_mode", type=str, help="Cur selection mode (default, same_k, same_q)",
                    required=False, default="default")
parser.add_argument("--select_number", type=int, help="Cur selection number",
                    required=False, default=64)
parser.add_argument("--num_iter", type=int, help="Cur pseudo inverse num iterations",
                    required=False, default=4)
parser.add_argument("--copy_rv", type=bool, help="Cur enabling copy rv",
                    required=False, default=True)
args = parser.parse_args()

attn_type = args.model
task = args.task

# print(lra_config.config[task]["extra_attn_config"].keys(), flush=True)

model_config = lra_config.config[task]["model"]
model_config.update(lra_config.config[task]["extra_attn_config"][attn_type])

init_t = time.time()
date = time.strftime('%Y_%m_%d_%H_%M', time.gmtime())

checkpoint_dir = f"../logs/{date}_{task}"

if attn_type == "curformer":
    model_config.update({
        "select_number": args.select_number,
        "select_type": args.select_type,
        "select_mode": args.select_mode,
        "copy_rv": args.copy_rv,
        "num_iter": args.num_iter,
    })
    checkpoint_dir = f"{checkpoint_dir}_curformer_{args.select_type}{args.select_number}" \
                     f"{args.select_mode if args.select_mode != 'default' else ''}" \
                     + (f"_CopyRV" if args.copy_rv else "") \
                     + f"_{args.num_iter}Iter"

os.makedirs(checkpoint_dir, exist_ok=True)
log_f_path = os.path.join(checkpoint_dir, f"training.log")
checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint.model")
log_f = open(log_f_path, "a+")

model_config["mixed_precision"] = True
model_config["attn_type"] = attn_type
model_config["max_seq_len"] = int(2 ** math.ceil(math.log2(model_config["max_seq_len"])))

training_config = lra_config.config[task]["training"]
gpu_memory_config = lra_config.config[task]["gpu_memory"]
extra_config = lra_config.config[task]["extra_attn_config"]

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, training_config], indent=4))

if task == "retrieval":
    model = ModelForSCDual(model_config)
else:
    model = ModelForSC(model_config)

print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush=True)
print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush=True)

model = model.cuda()
model = nn.DataParallel(model, device_ids=device_ids)

ds_iter = {
    "train": enumerate(
        DataLoader(LRADataset(f"../datasets/{task}.train.pickle", True), batch_size=training_config["batch_size"],
                   drop_last=True)),
    "dev": enumerate(
        DataLoader(LRADataset(f"../datasets/{task}.dev.pickle", True), batch_size=training_config["batch_size"],
                   drop_last=True)),
    "test": enumerate(
        DataLoader(LRADataset(f"../datasets/{task}.test.pickle", False), batch_size=training_config["batch_size"],
                   drop_last=True)),
}

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=training_config["learning_rate"],
    betas=(0.9, 0.999), eps=1e-6, weight_decay=training_config["weight_decay"]
)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=training_config["learning_rate"],
    pct_start=training_config["warmup"] / training_config["num_train_steps"],
    anneal_strategy=training_config["lr_decay"],
    total_steps=training_config["num_train_steps"]
)

amp_scaler = torch.cuda.amp.GradScaler(
) if model_config["mixed_precision"] else None


def step(component, step_idx):
    t0 = time.time()

    optimizer.zero_grad()

    _, batch = next(ds_iter[component])
    for key in batch:
        batch[key] = batch[key].cuda()

    if component == "train":
        outputs = {}

        partial_inputs_list = [{} for _ in range(accumu_steps)]
        for key in batch:
            for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim=0)):
                partial_inputs_list[idx][key] = inp

        for partial_inputs in partial_inputs_list:
            partial_outputs = model(**partial_inputs)
            for key in partial_outputs:
                partial_outputs[key] = partial_outputs[key].mean() / \
                                       accumu_steps
                if key not in outputs:
                    outputs[key] = partial_outputs[key]
                else:
                    outputs[key] += partial_outputs[key]
            amp_scaler.scale(partial_outputs["loss"]).backward()

        amp_scaler.step(optimizer)
        amp_scaler.update()
        lr_scheduler.step()
        # print(torch.cuda.max_memory_allocated("cuda:0"))
    else:
        with torch.no_grad():
            outputs = {}

            partial_inputs_list = [{} for _ in range(accumu_steps)]
            for key in batch:
                for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim=0)):
                    partial_inputs_list[idx][key] = inp

            for partial_inputs in partial_inputs_list:
                partial_outputs = model(**partial_inputs)
                for key in partial_outputs:
                    partial_outputs[key] = partial_outputs[key].mean() / \
                                           accumu_steps
                    if key not in outputs:
                        outputs[key] = partial_outputs[key]
                    else:
                        outputs[key] += partial_outputs[key]

    t1 = time.time()

    batch_size = batch[list(batch.keys())[0]].size(0)
    t_escape = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    loss = outputs["loss"].data.item()
    accu = outputs["accu"].data.item()
    time_since_start = time.time() - init_t

    print(
        f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, accu={accu:.4f}\t\t\t\t",
        end="\r", flush=True)

    summary[component]["t"] += t_escape
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)


def print_summary(summary, save_if_improved, train_step_idx):
    summary["loss"] = np.mean(summary["loss"])
    summary["accu"] = np.mean(summary["accu"])

    print()
    if summary["accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["accu"]
        if save_if_improved:
            best_accu = summary["best_accu"]
            torch.save({"model_state_dict": model.module.state_dict()},
                       checkpoint_file)
            print(f"best_accu={best_accu}. Saved best model")

    summary_round = {"train_step_idx": train_step_idx}
    for key in summary:
        if type(summary[key]) is str:
            summary_round[key] = summary[key]
        else:
            summary_round[key] = round(summary[key], 4)

    print(summary_round, flush=True)
    log_f.write(json.dumps(summary_round, sort_keys=True) + "\n")
    log_f.flush()

    summary["t"] = 0
    summary["loss"] = []
    summary["accu"] = []


summary = {
    component: {"t": 0, "loss": [], "accu": [],
                "best_accu": 0, "component": component}
    for component in ["train", "dev", "test"]
}

accumu_steps = max(training_config["batch_size"] //
                   len(device_ids) // gpu_memory_config[attn_type], 1)
print(f"accumu_steps={accumu_steps}")

if args.skip_train == 0:
    try:
        model.train()
        for train_step_idx in range(training_config["num_train_steps"]):
            outputs = step("train", train_step_idx)

            if (train_step_idx + 1) % training_config["eval_frequency"] == 0:
                print_summary(summary["train"], False, train_step_idx)
                model.eval()
                for dev_step_idx in range(training_config["num_eval_steps"]):
                    outputs = step("dev", dev_step_idx)
                print_summary(summary["dev"], True, train_step_idx)
                model.train()
    except KeyboardInterrupt as e:
        print(e)

checkpoint = torch.load(checkpoint_file, map_location="cpu")
model.module.load_state_dict(checkpoint["model_state_dict"])
model.eval()
try:
    for test_step_idx in itertools.count():
        outputs = step("test", test_step_idx)
except StopIteration:
    print_summary(summary["test"], False, train_step_idx)
