# Next Steps: Training Your Surveillance Model

Your surveillance specialization is **100% ready** to train. Here are your options:

---

## ⚡ Quick Decision Guide

**Choose based on your priorities:**

| Priority | Recommended Option | Time | Cost |
|----------|-------------------|------|------|
| **Free** | Google Colab | 2-3 days | $0 |
| **Fast & Professional** | Lambda Labs 8x H100 | 15-20 hours | $300-400 |
| **Budget-friendly** | RunPod RTX 4090 | 80-100 hours | $40-60 |
| **No training needed** | Community model demo | Instant | $0 |

---

## Option 1: Try Community Model (No Training) ⚡ INSTANT

**Best for:** Testing the concept without training

### Online Demo
Just visit: **https://huggingface.co/spaces/sdobson/nanochat**

- No installation needed
- Runs in browser
- Free to use
- General chat model (not surveillance-specialized)

### Download Model
```bash
python -m scripts.download_pretrained
```

**Note:** This is a general chat model, not surveillance-specialized. You'll still need to train for surveillance expertise.

---

## Option 2: Google Colab (Free GPU) 🆓

**Best for:** Testing your pipeline works before investing

### Quick Start

1. Go to https://colab.research.google.com/
2. Create new notebook
3. Enable GPU: Runtime → Change runtime type → GPU (T4)
4. Run these cells:

```python
# Clone your repo
!git clone https://github.com/BryanTegomoh/nanochat.git
%cd nanochat

# Install
!pip install -r requirements.txt

# Verify
!python -m scripts.test_surveillance_setup

# Test training (500 iterations)
!python -m scripts.surveillance_sft --max_iterations 500

# Full training (will take ~2-3 days with disconnections)
!python -m scripts.surveillance_sft
```

### Pros & Cons
✅ Free
✅ Easy setup
❌ Slow (single T4 GPU)
❌ Disconnects after ~12 hours
❌ Need to monitor and restart

---

## Option 3: Lambda Labs (Recommended) 🚀

**Best for:** Professional training, fastest results

### Cost
- **Testing (1x H100):** $2/hour × 1-2 hours = $2-4
- **Full training (8x H100):** $18/hour × 20 hours = **$360**

### Steps

1. **Sign up:** https://lambdalabs.com/
2. **Launch instance:** 8x H100 (80GB)
3. **SSH and setup:**
   ```bash
   ssh ubuntu@<ip>
   git clone https://github.com/BryanTegomoh/nanochat.git
   cd nanochat
   pip install -r requirements.txt
   ```

4. **Start training:**
   ```bash
   screen -S training
   torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft
   # Ctrl+A, D to detach
   ```

5. **Wait ~15-20 hours** (check periodically)

6. **Download checkpoints:**
   ```bash
   # From local machine
   scp -r ubuntu@<ip>:~/nanochat/surveillance_checkpoints ./
   ```

7. **Terminate instance** (stop billing!)

### Pros & Cons
✅ Fast (15-20 hours)
✅ Reliable
✅ Professional setup
❌ Costs $360

**Detailed guide:** [CLOUD_TRAINING_GUIDE.md](CLOUD_TRAINING_GUIDE.md)

---

## Option 4: RunPod (Budget) 💰

**Best for:** Budget-conscious training

### Cost
- **1x RTX 4090:** $0.40/hour × 100 hours = **$40-60**

### Steps

1. **Sign up:** https://www.runpod.io/
2. **Deploy pod:** RTX 4090, PyTorch template
3. **Connect via SSH or JupyterLab**
4. **Train:** (same as Lambda, but single GPU)
   ```bash
   python -m scripts.surveillance_sft
   ```

### Pros & Cons
✅ Cheapest paid option
✅ Good value
❌ Slower (single GPU, ~100 hours)
❌ Spot instances can be interrupted

---

## Recommended Path for You

### **Phase 1: Validate (Today - Free)**
```bash
# Test on Google Colab with limited iterations
!python -m scripts.surveillance_sft --max_iterations 500
```

**Goal:** Verify everything works (2-3 hours on Colab)

### **Phase 2: Full Training (When Ready - $360)**
```bash
# Lambda Labs 8x H100
torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft
```

**Goal:** Production-quality surveillance model (15-20 hours)

### **Phase 3: Deploy**
```bash
# Test locally on CPU
python -m scripts.surveillance_chat_cpu --source sft --model_tag d26-surveillance

# Evaluate
python -m scripts.surveillance_eval --source sft --model_tag d26-surveillance
```

---

## What You'll Get After Training

### Model Checkpoints
Location: `surveillance_checkpoints/d26-surveillance/`

Contains:
- Model weights
- Training metadata
- Performance metrics

### Usage
```bash
# Interactive chat (CPU)
python -m scripts.surveillance_chat_cpu --source sft --model_tag d26-surveillance

# Ask questions like:
# "There are 150 cases of flu vs baseline 40. Is this an outbreak?"
# "What does R₀ = 3.5 mean for disease control?"
# "Assess risk of measles with 5% vaccination coverage"
```

### Expected Performance
Based on your evaluation metrics:
- **ROUGE-1:** > 0.3 (text similarity)
- **Concept Coverage:** > 0.5 (epidemiological terms)
- **Actionability:** > 0.6 (includes recommendations)
- **Composite Score:** > 0.5 (overall quality)

---

## Budget Comparison

| Approach | Cost | Time | Quality |
|----------|------|------|---------|
| **Colab Free** | $0 | 2-3 days | Good |
| **RunPod RTX 4090** | $40-60 | 4-5 days | Good |
| **Lambda 1x H100** | $120-180 | 1-2 days | Very Good |
| **Lambda 8x H100** | $300-400 | 15-20 hours | Excellent |

---

## Files Reference

All your documentation:

| File | Purpose |
|------|---------|
| [NEXT_STEPS.md](NEXT_STEPS.md) | This file - decision guide |
| [CLOUD_TRAINING_GUIDE.md](CLOUD_TRAINING_GUIDE.md) | Detailed cloud GPU instructions |
| [SURVEILLANCE_README.md](SURVEILLANCE_README.md) | Complete reference |
| [QUICKSTART_SURVEILLANCE.md](QUICKSTART_SURVEILLANCE.md) | Quick commands |
| [SURVEILLANCE_SUMMARY.md](SURVEILLANCE_SUMMARY.md) | Overview & status |

---

## FAQ

### Q: Can I train on my laptop?
**A:** Only if you have a powerful NVIDIA GPU. CPU training would take months.

### Q: What if I don't have $360?
**A:** Start with Colab (free) or RunPod ($40-60). Quality will be similar, just takes longer.

### Q: Can I pause and resume training?
**A:** Not currently implemented. Use `screen` or `tmux` to keep training running if disconnected.

### Q: How do I know training is working?
**A:** Watch for decreasing validation loss every 50 steps. MMLU/ARC scores should stay > 0.5.

### Q: What if I don't want to train at all?
**A:** Use the community demo: https://huggingface.co/spaces/sdobson/nanochat
(Note: It's not surveillance-specialized)

---

## Ready to Start?

### **Recommended: Test on Colab First**

1. Open https://colab.research.google.com/
2. New notebook → Enable GPU
3. Clone and test:
   ```python
   !git clone https://github.com/BryanTegomoh/nanochat.git
   %cd nanochat
   !python -m scripts.test_surveillance_setup
   !python -m scripts.surveillance_sft --max_iterations 500
   ```

4. If successful, move to Lambda Labs for full training

### **Then: Full Training on Lambda**

Follow [CLOUD_TRAINING_GUIDE.md](CLOUD_TRAINING_GUIDE.md) Lambda Labs section

---

## Questions?

- **Technical issues:** Check [SURVEILLANCE_README.md](SURVEILLANCE_README.md)
- **Training commands:** Check [QUICKSTART_SURVEILLANCE.md](QUICKSTART_SURVEILLANCE.md)
- **Cloud setup:** Check [CLOUD_TRAINING_GUIDE.md](CLOUD_TRAINING_GUIDE.md)

---

**You're all set!** 🎉 Everything is ready - just choose your training approach and go!
