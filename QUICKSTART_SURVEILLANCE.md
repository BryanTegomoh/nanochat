# Quick Start: Public Health Surveillance Training

This is a condensed guide to get you started training nanochat for public health surveillance. For detailed documentation, see [SURVEILLANCE_README.md](SURVEILLANCE_README.md).

## Prerequisites

- nanochat repository cloned and set up
- Python 3.8+ with dependencies installed
- CUDA-capable GPU(s) for training
- Access to compute resources (8xH100 recommended for full training)

## Step-by-Step Guide

### 1. Verify Setup (1 minute)

```bash
python -m scripts.test_surveillance_setup
```

All tests should pass. If not, follow error messages to fix issues.

### 2. Inspect the Dataset (optional)

The dataset was already generated. To regenerate or customize:

```bash
# Regenerate with default settings (5,000 examples)
python -m scripts.generate_surveillance_dataset

# View a sample
python -c "import json; data=json.load(open('data/surveillance/train.json')); print(data[0])"
```

**Dataset Overview:**
- 4,000 training examples
- 500 validation examples
- 500 test examples
- 10 surveillance categories (outbreak detection, trend analysis, risk assessment, etc.)

### 3. Train the Model

#### For Production (8 GPUs):

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft
```

**Estimated training time:**
- d20 (561M params): ~8-12 hours on 8xH100 ($150-200)
- d26 (1B params): ~15-20 hours on 8xH100 ($300-400)
- d32 (1.9B params): ~30-40 hours on 8xH100 ($600-800)

#### For Testing (Single GPU):

```bash
# Quick test with limited iterations
python -m scripts.surveillance_sft --max_iterations 100
```

**Training Output:**
- Checkpoints saved to: `surveillance_checkpoints/d26-surveillance/`
- Logs show training/validation loss every 50 steps
- Evaluation on MMLU/ARC every 200 steps (ensures general knowledge retained)

**Monitoring Training:**
- Watch for decreasing validation loss
- MMLU/ARC scores should stay > 0.5 (if dropping, reduce learning rates)
- Training loss should decrease steadily

### 4. Evaluate the Model

```bash
python -m scripts.surveillance_eval --model_tag d26-surveillance --source sft
```

**Evaluation Metrics:**
- **ROUGE scores:** Text similarity to reference answers (target: ROUGE-1 > 0.3)
- **Concept coverage:** Epidemiological concepts present (target: > 0.5)
- **Structure quality:** Professional formatting (target: > 0.7)
- **Actionability:** Includes recommendations (target: > 0.6)
- **Composite score:** Overall quality (target: > 0.5)

**Results saved to:**
- `eval_results/surveillance/eval_results_d26-surveillance_[timestamp].json`
- `eval_results/surveillance/sample_outputs_d26-surveillance_[timestamp].txt`

### 5. Interactive Testing

```bash
python -m scripts.surveillance_chat --model_tag d26-surveillance --source sft
```

**Try these example queries:**

```
You: There are 150 cases of influenza in our region compared to a baseline of 40 cases. Is this an outbreak?

You: What does an R₀ of 3.5 mean for disease control measures?

You: Assess the risk of measles in a population with only 5% vaccination coverage.

You: How would you use syndromic surveillance to detect outbreaks early?

You: What are the key steps in contact tracing for tuberculosis?
```

**Commands:**
- `quit` or `exit` - Exit chat
- `clear` - Clear conversation history
- `examples` - Show example queries

## File Structure

```
nanochat/
├── data/
│   └── surveillance/
│       ├── train.json              # 4,000 training examples
│       ├── validation.json         # 500 validation examples
│       ├── test.json               # 500 test examples
│       └── dataset_stats.json      # Dataset statistics
├── tasks/
│   └── surveillance.py             # Custom Task class for surveillance
├── scripts/
│   ├── generate_surveillance_dataset.py  # Dataset generator
│   ├── surveillance_sft.py              # Training script
│   ├── surveillance_eval.py             # Evaluation script
│   ├── surveillance_chat.py             # Interactive chat
│   └── test_surveillance_setup.py       # Setup verification
├── surveillance_checkpoints/       # Model checkpoints (after training)
│   └── d26-surveillance/
├── eval_results/                   # Evaluation results (after eval)
│   └── surveillance/
├── SURVEILLANCE_README.md          # Full documentation
└── QUICKSTART_SURVEILLANCE.md      # This file
```

## Common Issues & Solutions

### Issue: Out of memory during training

**Solution 1:** Reduce batch size in `scripts/surveillance_sft.py`:
```python
device_batch_size = 2  # or even 1
```

**Solution 2:** Use gradient checkpointing (add to training script):
```python
model.gradient_checkpointing_enable()
```

**Solution 3:** Train smaller model (d20 instead of d26/d32):
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft --model_tag d20
```

### Issue: Model not loading for evaluation

**Check:**
1. Training completed successfully (check for checkpoint files)
2. Model tag matches: `--model_tag d26-surveillance`
3. Checkpoint directory exists: `ls surveillance_checkpoints/d26-surveillance/`

### Issue: Poor evaluation scores

**Possible fixes:**
1. Train for more iterations (increase `num_epochs` from 2 to 3)
2. Reduce learning rates by 30%:
   ```python
   embedding_lr = 0.10  # was 0.15
   matrix_lr = 0.01     # was 0.015
   ```
3. Increase surveillance data proportion (reduce SmolTalk from 2,000 to 1,000)
4. Review sample outputs to identify specific issues

### Issue: Model loses general knowledge

**Symptoms:** MMLU/ARC scores drop significantly during training

**Fix:** Increase general conversation data:
```python
# In surveillance_sft.py
SmolTalk(split="train", stop=5_000),  # was 2_000
```

## Customization Examples

### Use Your Own Data

Replace synthetic data with real surveillance conversations:

```python
# Format your data as JSON
[
  {
    "messages": [
      {"role": "user", "content": "Your surveillance question"},
      {"role": "assistant", "content": "Expert epidemiological answer"}
    ],
    "metadata": {"category": "outbreak_detection"}
  },
  ...
]
```

Save to `data/surveillance/train.json`, `validation.json`, `test.json`

### Adjust Dataset Size

Edit `scripts/generate_surveillance_dataset.py`:

```python
# Generate 10,000 examples instead of 5,000
train_data, val_data, test_data = generate_dataset(
    num_examples=10_000,  # was 5_000
    train_split=0.8,
    val_split=0.1
)
```

Then regenerate:
```bash
python -m scripts.generate_surveillance_dataset
```

### Train on Specific Categories

Modify `scripts/surveillance_sft.py` to focus on specific categories:

```python
from tasks.surveillance import SurveillanceCategory

train_ds = TaskMixture([
    SurveillanceCategory(split="train", category="outbreak_detection"),
    SurveillanceCategory(split="train", category="risk_assessment"),
    SmolTalk(split="train", stop=1_000),
])
```

## Training Tips

1. **Monitor validation loss** - If it stops decreasing, training is likely done
2. **Check MMLU/ARC scores** - Should stay > 0.5 throughout training
3. **Review sample outputs** - Manually inspect a few responses each eval cycle
4. **Start small** - Test with `--max_iterations 500` before full training
5. **Save checkpoints often** - Adjust `eval_every` if needed
6. **Use wandb** - Set `run="surveillance_exp1"` to log to Weights & Biases

## Next Steps After Training

1. **Production Deployment:**
   - Wrap in REST API ([FastAPI](https://fastapi.tiangolo.com/) or [Flask](https://flask.palletsprojects.com/))
   - Add authentication and rate limiting
   - Implement logging and monitoring
   - Add safety filters for inappropriate queries

2. **Continuous Improvement:**
   - Collect user queries and expert responses
   - Retrain periodically with new data
   - A/B test different model versions
   - Track key metrics (accuracy, response time, user satisfaction)

3. **Domain Expansion:**
   - Add more categories (AMR, food safety, environmental health)
   - Support multiple languages
   - Integrate real-time data sources (CDC APIs, WHO feeds)
   - Add tool use (calculators, database queries)

## Getting Help

1. **Test your setup:** `python -m scripts.test_surveillance_setup`
2. **Check logs:** Training script outputs detailed progress
3. **Review samples:** Look at generated dataset examples in `data/surveillance/`
4. **Read full docs:** [SURVEILLANCE_README.md](SURVEILLANCE_README.md)
5. **Consult experts:** For domain-specific questions, work with epidemiologists

## Important Disclaimers

- This model provides **decision support**, not autonomous medical advice
- Always **verify with official sources** (CDC, WHO, local health departments)
- **Consult senior epidemiologists** for critical public health decisions
- Model outputs should be **reviewed by experts** before operational use
- This is **not FDA-approved** medical software

## Resources

- **Full Documentation:** [SURVEILLANCE_README.md](SURVEILLANCE_README.md)
- **nanochat Main README:** [README.md](README.md)
- **CDC Field Epidemiology Manual:** https://www.cdc.gov/eis/field-epi-manual/
- **WHO Surveillance Standards:** https://www.who.int/teams/integrated-health-services/infection-prevention-control/surveillance

---

**Ready to start?** Run the verification script and begin training!

```bash
# 1. Verify everything is set up
python -m scripts.test_surveillance_setup

# 2. Start training (8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft

# 3. Evaluate when complete
python -m scripts.surveillance_eval --model_tag d26-surveillance --source sft

# 4. Test interactively
python -m scripts.surveillance_chat --model_tag d26-surveillance --source sft
```
