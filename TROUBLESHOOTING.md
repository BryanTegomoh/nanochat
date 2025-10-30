# Troubleshooting Guide

Common issues and solutions for the surveillance specialization.

---

## Import Errors

### ❌ Error: `ModuleNotFoundError: No module named 'nanochat'`

```
File "scripts\surveillance_chat_cpu.py", line 13, in <module>
    from nanochat.checkpoint_manager import load_model
ModuleNotFoundError: No module named 'nanochat'
```

**Cause:** Running the script directly instead of using Python's module syntax.

**Solution:** Always run scripts from the repository root using `-m` flag:

```bash
# ✅ CORRECT
cd /path/to/nanochat
python -m scripts.surveillance_chat_cpu --source mid

# ❌ WRONG
python scripts/surveillance_chat_cpu.py
python ./scripts/surveillance_chat_cpu.py
```

**Why:** Nanochat uses relative imports. The `-m` flag adds the current directory to Python's path.

---

## Model Loading Errors

### ❌ Error: `No checkpoint directory found`

```
Error loading model: [WinError 3] The system cannot find the path specified:
'C:\\Users\\bryan\\.cache\\nanochat\\mid_checkpoints'
```

**Cause:** No trained model checkpoints available.

**Solutions:**

**Option 1: Download Community Model**
```bash
python -m scripts.download_pretrained
```

**Option 2: Train Your Own Model**
Follow [CLOUD_TRAINING_GUIDE.md](CLOUD_TRAINING_GUIDE.md)

**Option 3: Use Online Demo**
Visit: https://huggingface.co/spaces/sdobson/nanochat

---

## CUDA/GPU Errors

### ❌ Error: `CUDA is needed for a distributed run atm`

```
AssertionError: CUDA is needed for a distributed run atm
```

**Cause:** Script requires GPU but none is available.

**Solution:** Use CPU-compatible version:

```bash
# Use CPU version instead of GPU version
python -m scripts.surveillance_chat_cpu --source mid  # ✅ Works on CPU
python -m scripts.surveillance_chat --source mid      # ❌ Requires GPU
```

---

## Training Errors

### ❌ Error: `Out of memory`

```
RuntimeError: CUDA out of memory
```

**Cause:** Batch size too large for available GPU memory.

**Solution:** Reduce batch size in training script:

Edit `scripts/surveillance_sft.py`:
```python
device_batch_size = 2  # Reduce from 4 to 2 (or even 1)
```

Or run with configuration override:
```bash
python -m scripts.surveillance_sft --device_batch_size 2
```

---

### ❌ Error: `Dataset not found`

```
FileNotFoundError: Surveillance dataset file not found:
data/surveillance/train.json
```

**Cause:** Dataset hasn't been generated yet.

**Solution:** Generate the dataset first:

```bash
python -m scripts.generate_surveillance_dataset
```

This creates:
- `data/surveillance/train.json` (4,000 examples)
- `data/surveillance/validation.json` (500 examples)
- `data/surveillance/test.json` (500 examples)

---

## Evaluation Errors

### ❌ Error: `Model tag not found`

```
Error: Model checkpoint not found for tag 'd26-surveillance'
```

**Cause:** Trying to evaluate before training is complete.

**Solution:**

1. Check if training created checkpoints:
   ```bash
   ls surveillance_checkpoints/
   ```

2. If empty, complete training first:
   ```bash
   # Training creates: surveillance_checkpoints/d26-surveillance/
   torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft
   ```

3. Use correct model tag:
   ```bash
   # Match the tag name from surveillance_checkpoints/
   python -m scripts.surveillance_eval --source sft --model_tag d26-surveillance
   ```

---

## Setup Verification

### ✅ Run Setup Tests

Before starting, verify everything is configured:

```bash
python -m scripts.test_surveillance_setup
```

**Expected output:**
```
[OK] Dataset Files: PASSED
[OK] Task Loader: PASSED
[OK] Training Script: PASSED
[OK] Evaluation Script: PASSED
[OK] Chat Interface: PASSED

All tests passed! Setup is complete.
```

If any tests fail, follow the error messages to fix issues.

---

## Common Workflow Issues

### Issue: "I can't test the model without training first"

**Solutions:**

1. **Use community model (instant):**
   - Online: https://huggingface.co/spaces/sdobson/nanochat
   - Download: `python -m scripts.download_pretrained`
   - Note: General chat, not surveillance-specialized

2. **Test on Google Colab (free):**
   ```python
   # In Colab
   !git clone https://github.com/BryanTegomoh/nanochat.git
   %cd nanochat
   !python -m scripts.surveillance_sft --max_iterations 500
   ```

3. **Cloud GPU training:**
   - See [CLOUD_TRAINING_GUIDE.md](CLOUD_TRAINING_GUIDE.md)

---

### Issue: "Training is too slow on Colab"

**Problem:** Free Colab T4 GPU is much slower than 8xH100.

**Solutions:**

1. **Test with limited iterations:**
   ```bash
   python -m scripts.surveillance_sft --max_iterations 500
   ```
   Just to verify pipeline works (~1-2 hours)

2. **Upgrade to Lambda Labs:**
   - 8xH100: ~15-20 hours total
   - Cost: $360
   - See [CLOUD_TRAINING_GUIDE.md](CLOUD_TRAINING_GUIDE.md)

3. **Budget option - RunPod:**
   - RTX 4090: ~100 hours total
   - Cost: $40-60
   - Slower but much cheaper

---

### Issue: "How do I know if training is working?"

**Check these indicators:**

1. **Validation loss should decrease:**
   ```
   Step 00050 | Validation loss: 2.4567
   Step 00100 | Validation loss: 2.3456  ← Decreasing ✅
   Step 00150 | Validation loss: 2.2345  ← Still decreasing ✅
   ```

2. **MMLU/ARC scores should stay > 0.5:**
   ```
   Step 00200 | mmlu_acc: 0.6234, arc_easy_acc: 0.7123  ✅
   ```

3. **Training loss should decrease:**
   ```
   Step 00050 | Training loss: 2.5678
   Step 00051 | Training loss: 2.5432  ← Decreasing ✅
   ```

4. **No NaN losses:**
   ```
   Training loss: nan  ❌ Something is wrong
   ```

---

## Git/GitHub Issues

### Issue: ".claude folder keeps showing in git status"

**Solution:** Already fixed! `.claude/` is in `.gitignore`

To verify:
```bash
cat .gitignore | grep claude
# Should show: .claude/
```

If it's still showing as modified:
```bash
# Ignore local changes to .claude settings
git update-index --assume-unchanged .claude/settings.local.json
```

---

## Performance Issues

### Issue: "Chat responses are very slow on CPU"

**Expected:** CPU inference is 10-100x slower than GPU.

**Solutions:**

1. **Use smaller model:**
   - Train d20 (561M params) instead of d26 (1B params)
   - Faster on CPU

2. **Reduce max_length:**
   Edit chat script:
   ```python
   response = engine.generate(
       conversation,
       max_length=512,  # Reduce from 1024
       temperature=0.7
   )
   ```

3. **Use GPU for inference:**
   - Rent cheap GPU for inference only
   - Or use `surveillance_chat.py` (GPU version)

---

## Data/Dataset Issues

### Issue: "Dataset examples don't look right"

**Check:**

1. **View sample:**
   ```bash
   python -c "import json; print(json.dumps(json.load(open('data/surveillance/train.json'))[0], indent=2))"
   ```

2. **Regenerate if needed:**
   ```bash
   # Backup old dataset
   mv data/surveillance data/surveillance.backup

   # Generate fresh dataset
   python -m scripts.generate_surveillance_dataset
   ```

3. **Use custom data:**
   - Replace with your own surveillance conversations
   - Format as JSON: `[{"messages": [...], "metadata": {...}}]`

---

## Quick Commands Reference

### Setup & Verification
```bash
# Verify setup
python -m scripts.test_surveillance_setup

# Generate dataset
python -m scripts.generate_surveillance_dataset
```

### Training
```bash
# Full training (8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft

# Single GPU or CPU testing
python -m scripts.surveillance_sft --max_iterations 500
```

### Evaluation
```bash
python -m scripts.surveillance_eval --source sft --model_tag d26-surveillance
```

### Chat/Inference
```bash
# CPU version
python -m scripts.surveillance_chat_cpu --source sft --model_tag d26-surveillance

# GPU version
python -m scripts.surveillance_chat --source sft --model_tag d26-surveillance
```

---

## Getting Help

### Check These First:
1. [NEXT_STEPS.md](NEXT_STEPS.md) - What to do next
2. [QUICKSTART_SURVEILLANCE.md](QUICKSTART_SURVEILLANCE.md) - Quick commands
3. [SURVEILLANCE_README.md](SURVEILLANCE_README.md) - Full documentation
4. This file - Troubleshooting

### Still Stuck?

1. **Run verification:**
   ```bash
   python -m scripts.test_surveillance_setup
   ```

2. **Check error messages carefully** - they usually tell you what's wrong

3. **Review documentation** - Most issues are covered in the guides

4. **Check GitHub issues** - Others may have had similar problems

---

## Summary of Most Common Issues

| Error | Quick Fix |
|-------|-----------|
| `ModuleNotFoundError: nanochat` | Use `python -m scripts.XXX` not `python scripts/XXX` |
| `CUDA is needed` | Use `surveillance_chat_cpu.py` instead |
| `Dataset not found` | Run `python -m scripts.generate_surveillance_dataset` |
| `Model not found` | Train first or use community model |
| `Out of memory` | Reduce `device_batch_size` |

---

**Most issues are caused by:**
1. Running scripts incorrectly (not using `-m` flag)
2. Missing dataset (forgot to generate)
3. Missing model checkpoints (haven't trained yet)

**Solution:** Follow [QUICKSTART_SURVEILLANCE.md](QUICKSTART_SURVEILLANCE.md) step-by-step!
