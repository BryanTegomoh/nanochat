"""
Finetune a base model for public health surveillance.
This script specializes nanochat for epidemiological surveillance tasks.

Run on one GPU e.g. for debugging:

python -m scripts.surveillance_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb
from nanochat.checkpoint_manager import load_model
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.surveillance import PublicHealthSurveillance
from tasks.smoltalk import SmolTalk

# -----------------------------------------------------------------------------
# SFT Hyperparameters for Public Health Surveillance
run = "surveillance_v1" # wandb run name (use "dummy" to skip wandb logging)
# input model options
source = "mid" # base|mid , which checkpoint to load the model from
model_tag = None # model tag to load the model from
step = None # step to load the model from
# compute/precision
dtype = "bfloat16"
device_batch_size = 4 # max to avoid OOM
# optimization (tuned for medical/surveillance domain)
num_epochs = 2 # More epochs for smaller, specialized dataset
max_iterations = -1 # override number of iterations (-1 = use num_epochs * num_iterations)
target_examples_per_step = 32
unembedding_lr = 0.003 # Slightly lower for conservative updates
embedding_lr = 0.15 # Slightly lower for medical domain
matrix_lr = 0.015 # Slightly lower for safety-critical domain
weight_decay = 0.0
init_lr_frac = 0.02
# evaluation and logging
eval_every = 50 # More frequent evaluation for specialized training
eval_steps = 100
eval_metrics_every = 200
# now allow CLI to override the settings via the configurator
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-surveillance",
    name=run,
    config=user_config,
    save_code=True
)

# Load the model and tokenizer
print0("Loading base model for public health surveillance specialization...")
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
orig_model = model # original, uncompiled model
# model = torch.compile(model, dynamic=True) # doesn't work well with variable lengths
engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# Task data mixture for public health surveillance
# We mix surveillance-specific data with general conversation to maintain versatility
print0("Loading public health surveillance dataset...")

try:
    train_ds = TaskMixture([
        PublicHealthSurveillance(split="train"), # Primary surveillance training data
        SmolTalk(split="train", stop=2_000), # Small amount of general conversation
    ])
    val_ds = PublicHealthSurveillance(split="validation") # Validate on surveillance data only

    print0(f"Training dataset: {len(train_ds)} examples")
    print0(f"  - Public Health Surveillance: ~{len(train_ds) - 2000} examples")
    print0(f"  - General Conversation (SmolTalk): 2,000 examples")
    print0(f"Validation dataset: {len(val_ds)} examples")

except FileNotFoundError as e:
    print0(f"\n❌ Error: Surveillance dataset not found!")
    print0(f"{e}")
    print0(f"\n📝 Please generate the dataset first:")
    print0(f"   python -m scripts.generate_surveillance_dataset")
    print0(f"\nExiting...")
    exit(1)

# -----------------------------------------------------------------------------
# DataLoader

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    # prepares a list of tokenized conversations into a batch and yields
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets
        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets

    # iterates over the dataset in epochs, tokenizes
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = device_batch_size * ddp_world_size
print0(f"Target examples per step: {target_examples_per_step}")
print0(f"Device batch size: {device_batch_size}")
print0(f"Examples per step (device_batch_size * ddp_world_size): {examples_per_step}")
assert target_examples_per_step % examples_per_step == 0, \
    "Target examples per step must be divisible by examples per step"
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"=> Gradient accumulation steps: {grad_accum_steps}")

num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
if max_iterations >= 0 and num_iterations > max_iterations:
    print0(f"Number of iterations capped from {num_iterations} to {max_iterations}")
    num_iterations = max_iterations
print0(f"Total training iterations: {num_iterations}")

train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=device_batch_size)

# -----------------------------------------------------------------------------
# Initialize the Optimizer

optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
# Set the initial learning rate as a fraction of the base learning rate
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]

# -----------------------------------------------------------------------------
# Training loop

# Learning rate scheduler (linear decay)
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm

# Go!
print0("\n" + "="*80)
print0("🦠 Starting Public Health Surveillance Training")
print0("="*80 + "\n")

step = 0
train_iter = iter(train_loader)
for step in range(num_iterations):
    last_step = step == num_iterations - 1

    # evaluate the validation loss
    if last_step or step % eval_every == 0:
        model.eval()
        val_iter = iter(build_val_loader())
        losses = []
        for _ in range(eval_steps):
            val_inputs, val_targets = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean()
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss = val_loss.item()
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        wandb_run.log({
            "step": step,
            "val_loss": val_loss,
        })
        model.train()

    # evaluate on general knowledge tasks periodically
    if last_step or (step > 0 and step % eval_metrics_every == 0):
        model.eval()
        metrics = {}
        with torch.no_grad(), autocast_ctx:
            # Evaluate on general knowledge to ensure we didn't lose capabilities
            metrics["mmlu_acc"] = run_chat_eval("MMLU", model, tokenizer, engine,
                                                  batch_size=device_batch_size*2, max_problems=512)
            metrics["arc_easy_acc"] = run_chat_eval("ARC-Easy", model, tokenizer, engine,
                                                      batch_size=device_batch_size*2, max_problems=512)
        metrics_str = ', '.join(f'{k}: {v:.6f}' for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        wandb_run.log({
            "step": step,
            **metrics,
        })
        model.train()

    if last_step:
        break

    # evaluate the gradient
    num_tokens = torch.tensor(0, device=device)
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)

    # learning rate scheduler
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # step the optimizers
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    # logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    print0(f"Step {step:05d}/{num_iterations:05d} | Train loss: {train_loss_item:.6f} | lrm: {lrm:.6f} | tokens: {num_tokens_item:,}")
    wandb_run.log({
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
    })
    step += 1

# Save the model at the end of the run
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    model_tag = f"d{depth}-surveillance" # Tag with surveillance specialization
    checkpoint_dir = os.path.join(base_dir, "surveillance_checkpoints", model_tag)
    model_config_kwargs = model.config.__dict__
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        None, # we don't save optimizer state
        {
            "step": step,
            "val_loss": val_loss,
            **metrics,
            "model_config": model_config_kwargs,
            "specialization": "public_health_surveillance",
            "training_data": {
                "surveillance_examples": len(train_ds) - 2000,
                "general_examples": 2000,
                "total": len(train_ds)
            }
        }
    )
    print0(f"\n{'='*80}")
    print0(f"✅ Saved surveillance-specialized model checkpoint to:")
    print0(f"   {checkpoint_dir}")
    print0(f"{'='*80}\n")

# Log to report
from nanochat.report import get_report
get_report().log(section="Surveillance SFT", data=[
    user_config,
    {
        "Training rows": len(train_ds),
        "Surveillance examples": len(train_ds) - 2000,
        "General examples": 2000,
        "Number of iterations": num_iterations,
        "Training loss": train_loss_item,
        "Validation loss": val_loss,
        "Specialization": "Public Health Surveillance",
    },
])

print0("\n🦠 Public Health Surveillance Training Complete!")

# Cleanup
wandb_run.finish()
compute_cleanup()
