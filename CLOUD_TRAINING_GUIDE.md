# Cloud GPU Training Guide for Surveillance Model

This guide shows you how to train your surveillance-specialized nanochat model using cloud GPU services.

## Option 1: Google Colab (Free Tier)

### Pros
- ✅ Free GPU (T4) available
- ✅ No setup required
- ✅ Easy to use

### Cons
- ❌ Limited to ~12 hours per session
- ❌ May disconnect randomly
- ❌ Slower GPU (T4 vs H100)
- ❌ Training will take longer (~2-3 days for d26)

### Steps

1. **Create a Colab Notebook:**
   - Go to https://colab.research.google.com/
   - Click "New Notebook"

2. **Enable GPU:**
   - Runtime → Change runtime type → Hardware accelerator → GPU (T4)

3. **Clone your repo:**
   ```python
   !git clone https://github.com/BryanTegomoh/nanochat.git
   %cd nanochat
   ```

4. **Install dependencies:**
   ```python
   !pip install -r requirements.txt
   ```

5. **Check GPU:**
   ```python
   !nvidia-smi
   ```

6. **Start training (single GPU):**
   ```python
   !python -m scripts.surveillance_sft --max_iterations 1000
   ```

7. **Save checkpoints to Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   # After training completes
   !cp -r surveillance_checkpoints /content/drive/MyDrive/
   ```

### Limitations
- Single GPU only (no multi-GPU training)
- Need to restart training if disconnected
- Much slower than 8xH100

---

## Option 2: Lambda Labs (Recommended)

### Pros
- ✅ Affordable GPU instances ($0.50-$1.50/hour)
- ✅ H100 GPUs available
- ✅ SSH access
- ✅ Easy to use

### Cons
- ❌ Requires payment
- ❌ Need to manage instances

### Steps

1. **Sign up:**
   - Go to https://lambdalabs.com/
   - Create account and add payment method

2. **Launch instance:**
   - Click "Instances" → "Launch instance"
   - Choose GPU: **8x H100 (80GB)** for production training
   - Or **1x A10** ($0.60/hour) for testing
   - Select region with availability

3. **SSH into instance:**
   ```bash
   ssh ubuntu@<instance-ip>
   ```

4. **Clone and setup:**
   ```bash
   git clone https://github.com/BryanTegomoh/nanochat.git
   cd nanochat
   pip install -r requirements.txt
   ```

5. **Start training:**
   ```bash
   # For 8x H100
   torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft

   # For single GPU
   python -m scripts.surveillance_sft --max_iterations 1000
   ```

6. **Monitor training:**
   ```bash
   # In another terminal window
   watch -n 10 tail -50 nohup.out
   ```

7. **Download checkpoints when done:**
   ```bash
   # From your local machine
   scp -r ubuntu@<instance-ip>:~/nanochat/surveillance_checkpoints ./
   ```

### Cost Estimate
- **1x A10 (testing):** $0.60/hour × 100 hours = $60
- **8x H100 (production):** $18/hour × 20 hours = $360

---

## Option 3: RunPod (Budget Option)

### Pros
- ✅ Very affordable
- ✅ Hourly and spot pricing
- ✅ Easy to use

### Cons
- ❌ Spot instances can be interrupted
- ❌ Less reliable than Lambda

### Steps

1. **Sign up:**
   - Go to https://www.runpod.io/
   - Create account

2. **Deploy Pod:**
   - Click "Deploy"
   - Choose GPU (RTX 4090 is good value at ~$0.40/hour)
   - Select PyTorch template

3. **Connect:**
   - Use JupyterLab or SSH

4. **Training:**
   Same as Lambda Labs steps above

### Cost Estimate
- **1x RTX 4090:** $0.40/hour × 100 hours = $40

---

## Option 4: Your Institution's Cluster

If you have access to university or hospital computing resources:

### Steps

1. **Check available resources:**
   ```bash
   # On cluster
   nvidia-smi
   squeue  # if using SLURM
   ```

2. **Create SLURM job script** (if applicable):
   ```bash
   #!/bin/bash
   #SBATCH --job-name=surveillance_training
   #SBATCH --nodes=1
   #SBATCH --gres=gpu:8
   #SBATCH --time=24:00:00
   #SBATCH --mem=200GB

   module load python/3.10
   module load cuda/12.1

   cd ~/nanochat
   torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft
   ```

3. **Submit job:**
   ```bash
   sbatch train_surveillance.sh
   ```

4. **Monitor:**
   ```bash
   squeue -u $USER
   tail -f slurm-<job-id>.out
   ```

---

## Comparison Table

| Provider | GPU | Cost/hour | Total Cost (d26) | Setup Time | Reliability |
|----------|-----|-----------|------------------|------------|-------------|
| **Google Colab** | T4 | Free | $0 | 5 min | Low |
| **Lambda Labs** | 8x H100 | $18 | $360 | 10 min | High |
| **RunPod** | RTX 4090 | $0.40 | $40-80 | 10 min | Medium |
| **Institution** | Varies | Free | $0 | Varies | High |

---

## Recommended Approach

### For Your Situation:

**Best option: Lambda Labs 8x H100**
- Professional-grade training
- Complete in 15-20 hours
- Total cost: ~$300-400
- Most reliable

**Budget alternative: RunPod RTX 4090**
- Single GPU training
- Will take 80-100 hours
- Total cost: ~$40-60
- Good for testing

**Free option: Google Colab**
- Test the pipeline first
- Train with `--max_iterations 500` to verify everything works
- Then use paid service for full training

---

## Step-by-Step: Lambda Labs (Detailed)

### 1. Before You Start

```bash
# Verify your setup locally
cd nanochat
python -m scripts.test_surveillance_setup

# Make sure all tests pass
```

### 2. Launch Lambda Instance

1. Go to https://cloud.lambdalabs.com/instances
2. Click "Launch instance"
3. Select **8x H100 (80GB)**
4. Choose cheapest available region
5. Click "Launch instance"

**Cost alert:** Instance starts billing immediately!

### 3. Connect via SSH

```bash
# Lambda gives you SSH command like:
ssh ubuntu@<ip-address>

# Accept fingerprint (type 'yes')
```

### 4. Setup Environment

```bash
# Check GPU
nvidia-smi

# Should show 8x H100 GPUs

# Clone your repo
git clone https://github.com/BryanTegomoh/nanochat.git
cd nanochat

# Install dependencies
pip install -r requirements.txt

# Verify dataset exists
ls data/surveillance/
```

### 5. Start Training

```bash
# Use screen or tmux so training continues if disconnected
screen -S training

# Start training
torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r training
```

### 6. Monitor Progress

```bash
# In a new SSH session
tail -f <log file if created>

# Or check manually
screen -r training
```

### 7. Training Should Complete in ~15-20 hours

Output will be in:
- `surveillance_checkpoints/d26-surveillance/`

### 8. Download Checkpoints

```bash
# From your local machine
scp -r ubuntu@<ip>:~/nanochat/surveillance_checkpoints ./nanochat/

# This might take 10-20 minutes
```

### 9. Terminate Instance

**IMPORTANT:** Don't forget to terminate!

1. Go to Lambda console
2. Click "Terminate" on your instance
3. Verify it's terminated (billing stops)

---

## After Training

Once you have the checkpoints:

```bash
# Test locally (CPU)
python -m scripts.surveillance_chat_cpu --source sft --model_tag d26-surveillance

# Evaluate
python -m scripts.surveillance_eval --source sft --model_tag d26-surveillance
```

---

## Troubleshooting

### Out of Memory
```python
# In surveillance_sft.py, reduce:
device_batch_size = 2  # or 1
```

### Connection Lost
```bash
# Reconnect and check if training is still running
screen -r training

# If stopped, restart from last checkpoint (feature not implemented yet)
```

### Slow Training
```bash
# Monitor GPU utilization
nvidia-smi -l 1

# Should show ~95%+ GPU utilization
```

---

## Cost Optimization Tips

1. **Test first on Colab** (free) with `--max_iterations 500`
2. **Use spot instances** on RunPod (cheaper but can be interrupted)
3. **Train during off-peak** hours (some providers have cheaper rates)
4. **Set billing alerts** so you don't overspend
5. **Download checkpoints immediately** after training
6. **Terminate instances** as soon as done

---

## Questions?

- Check [SURVEILLANCE_README.md](SURVEILLANCE_README.md) for training details
- Check [QUICKSTART_SURVEILLANCE.md](QUICKSTART_SURVEILLANCE.md) for commands
- Lambda Labs docs: https://lambdalabs.com/docs
- RunPod docs: https://docs.runpod.io/

---

**Ready to train?** I recommend starting with a **Lambda Labs 1x H100** instance for 1 hour (~$2) to verify everything works, then scale up to 8x H100 for the full training run.
