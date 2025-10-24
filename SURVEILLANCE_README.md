# Public Health Surveillance Training for nanochat

This guide explains how to train nanochat specifically for public health surveillance applications. The surveillance-specialized model is designed for epidemiologists, public health officials, and disease surveillance professionals.

## Overview

The surveillance specialization covers:

- **Outbreak Detection & Investigation** - Identifying disease outbreaks and coordinating responses
- **Epidemiological Trend Analysis** - Analyzing disease patterns and temporal trends
- **Public Health Risk Assessment** - Evaluating population health risks
- **Surveillance Reporting** - Creating professional surveillance reports
- **Vaccination Coverage Analysis** - Monitoring immunization programs
- **Data Interpretation** - Understanding surveillance metrics (R₀, incidence rates, etc.)
- **Syndromic Surveillance** - Early detection systems using syndromic data
- **Contact Tracing Protocols** - Disease contact tracing procedures
- **Zoonotic Surveillance** - Animal-human disease surveillance (One Health)
- **Global Health Security** - International outbreak coordination

## Files Created

### Core Training Infrastructure

1. **`scripts/generate_surveillance_dataset.py`** - Dataset generator
   - Creates 5,000 synthetic surveillance conversations
   - 10 categories covering all surveillance domains
   - Splits: 80% train, 10% validation, 10% test

2. **`tasks/surveillance.py`** - Custom Task class
   - Loads surveillance data for training
   - Implements evaluation methods
   - Supports category-specific filtering

3. **`scripts/surveillance_sft.py`** - Training pipeline
   - Supervised fine-tuning for surveillance
   - Optimized hyperparameters for medical domain
   - Saves to `surveillance_checkpoints/`

4. **`scripts/surveillance_eval.py`** - Evaluation script
   - Comprehensive metrics (ROUGE, concept coverage, structure, actionability)
   - Per-category performance analysis
   - Saves results and sample outputs

5. **`scripts/surveillance_chat.py`** - Interactive chat interface
   - Specialized surveillance chatbot
   - Includes system prompt and example queries
   - CLI interface for testing

## Quick Start

### Step 1: Generate the Dataset

```bash
python -m scripts.generate_surveillance_dataset
```

This creates:
- `data/surveillance/train.json` - 4,000 training examples
- `data/surveillance/validation.json` - 500 validation examples
- `data/surveillance/test.json` - 500 test examples
- `data/surveillance/dataset_stats.json` - Dataset statistics

**Dataset Statistics:**
- Total: 5,000 conversations
- Training: 4,000 (80%)
- Validation: 500 (10%)
- Test: 500 (10%)

**Categories:**
- Outbreak detection: 750 examples (15%)
- Trend analysis: 750 examples (15%)
- Risk assessment: 750 examples (15%)
- Surveillance reports: 500 examples (10%)
- Vaccination coverage: 500 examples (10%)
- Data interpretation: 500 examples (10%)
- Syndromic surveillance: 500 examples (10%)
- Contact tracing: 250 examples (5%)
- Zoonotic surveillance: 250 examples (5%)
- Global surveillance: 250 examples (5%)

### Step 2: Train the Surveillance Model

**For training on 8 GPUs (recommended):**

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft
```

**For single GPU (debugging/testing):**

```bash
python -m scripts.surveillance_sft
```

**Training Configuration:**
- Base model: Loads from midtraining checkpoint (`source=mid`)
- Training data: ~4,000 surveillance + 2,000 general conversation examples
- Epochs: 2 (tuned for specialized domain)
- Learning rates: Conservative (embedding_lr=0.15, matrix_lr=0.015)
- Evaluation: Every 50 steps (frequent monitoring)
- Output: `surveillance_checkpoints/d26-surveillance/` (or d20, d32 depending on model depth)

**Training Time Estimates:**
- **d20 (561M params):** ~8-12 hours on 8xH100 (~$150-200)
- **d26 (1B params):** ~15-20 hours on 8xH100 (~$300-400)
- **d32 (1.9B params):** ~30-40 hours on 8xH100 (~$600-800)

### Step 3: Evaluate the Model

```bash
python -m scripts.surveillance_eval --model_tag d26-surveillance --source sft
```

**Evaluation Metrics:**
- **ROUGE-1/ROUGE-2:** Text similarity to reference answers
- **Concept Coverage:** Presence of key epidemiological concepts
- **Structure Quality:** Use of headers, lists, paragraphs
- **Actionability:** Inclusion of recommendations and action items
- **Composite Score:** Weighted combination of all metrics

**Output:**
- `eval_results/surveillance/eval_results_d26-surveillance_[timestamp].json` - Full results
- `eval_results/surveillance/sample_outputs_d26-surveillance_[timestamp].txt` - Sample responses

### Step 4: Interactive Testing

```bash
python -m scripts.surveillance_chat --model_tag d26-surveillance --source sft
```

**Example Queries:**
- "Detect outbreak: 150 cases of influenza vs baseline of 40 cases"
- "Analyze: Measles cases increased 200% in unvaccinated children"
- "Risk assessment: COVID-19 in a city with 5% vaccination coverage"
- "Interpret: What does R₀ = 3.5 mean for disease control?"
- "Protocol: Contact tracing steps for tuberculosis exposure"

## Dataset Details

### Data Format

Each conversation follows this structure:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "There are 150 reported cases of influenza in the Northeast region..."
    },
    {
      "role": "assistant",
      "content": "Yes, this appears to be an outbreak. The current number of 150 cases..."
    }
  ],
  "metadata": {
    "category": "outbreak_detection",
    "domain": "public_health_surveillance"
  }
}
```

### Sample Categories

#### 1. Outbreak Detection
Identifies when disease cases exceed expected levels and triggers investigation.

**Example:**
```
Q: There are 150 cases of influenza vs baseline of 40. Is this an outbreak?
A: Yes, this appears to be an outbreak. The 275% increase significantly exceeds
   the baseline. Immediate investigation warranted to identify source...
```

#### 2. Trend Analysis
Analyzes epidemiological patterns over time and across populations.

**Example:**
```
Q: Influenza cases increased by 45% among elderly during winter. What does this indicate?
A: This 45% increase suggests enhanced transmission in this vulnerable population...
```

#### 3. Risk Assessment
Evaluates threat levels and population health impacts.

**Example:**
```
Q: Assess public health risk of measles with 5% vaccination coverage.
A: Overall Risk Level: HIGH. With 95% susceptible population, high attack rate
   expected. Urgent vaccination campaign required...
```

#### 4. Data Interpretation
Explains surveillance metrics and statistical concepts.

**Example:**
```
Q: What does R₀ = 3.5 mean for transmission?
A: R₀ = 3.5 indicates each infected person causes ~3.5 secondary infections.
   Herd immunity threshold: ~71% of population needs immunity...
```

## Training Hyperparameters

The surveillance training uses more conservative hyperparameters compared to general chat fine-tuning:

```python
# Conservative learning rates for medical domain
embedding_lr = 0.15      # (default: 0.2)
matrix_lr = 0.015        # (default: 0.02)
unembedding_lr = 0.003   # (default: 0.004)

# More epochs for specialized, smaller dataset
num_epochs = 2           # (default: 1)

# Frequent evaluation
eval_every = 50          # (default: 100)
```

**Rationale:**
- Lower learning rates prevent catastrophic forgetting of base knowledge
- More epochs ensure thorough learning of specialized domain
- Frequent evaluation catches overfitting early in medical applications

## Customization

### Adding Real Data

To replace synthetic data with real surveillance data:

1. **Format your data** as JSON conversations:
```python
{
  "messages": [
    {"role": "user", "content": "Your question"},
    {"role": "assistant", "content": "Expert answer"}
  ]
}
```

2. **Save** to `data/surveillance/train.json`, `validation.json`, `test.json`

3. **Update** `tasks/surveillance.py` if needed for custom fields

### Adjusting Dataset Size

Edit `scripts/generate_surveillance_dataset.py`:

```python
# Generate 10,000 examples instead of 5,000
train_data, val_data, test_data = generate_dataset(
    num_examples=10_000,
    train_split=0.8,
    val_split=0.1
)
```

### Adding New Categories

To add a new surveillance category:

1. **Create generator function** in `generate_surveillance_dataset.py`:
```python
def generate_antimicrobial_resistance():
    """Generate AMR surveillance questions"""
    templates = [...]
    return random.choice(templates)
```

2. **Add to generator list**:
```python
generators = [
    # ... existing ...
    (generate_antimicrobial_resistance, 0.05),  # 5% of dataset
]
```

### Training on Custom Model Size

```bash
# Train d20 (smaller, faster)
torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft --model_tag d20

# Train d32 (larger, more capable)
torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft --model_tag d32
```

## Evaluation Metrics Explained

### ROUGE Scores (0-1, higher is better)
- **ROUGE-1:** Unigram overlap with reference answer
- **ROUGE-2:** Bigram overlap (phrase-level similarity)
- Good scores: ROUGE-1 > 0.3, ROUGE-2 > 0.15

### Concept Coverage (0-1, higher is better)
- Checks for presence of key epidemiological terms
- Category-specific keywords (outbreak, baseline, risk, etc.)
- Good score: > 0.5

### Structure Quality (0-1, higher is better)
- Headers, bullet points, numbered lists
- Appropriate response length (100-800 words)
- Good score: > 0.7

### Actionability (0-1, higher is better)
- Includes recommendations and action items
- Uses action verbs (recommend, implement, monitor)
- Good score: > 0.6

### Composite Score (0-1, higher is better)
- Weighted combination of all metrics
- Overall quality indicator
- Good score: > 0.5

## Best Practices

### For Training

1. **Start with midtraining checkpoint** - Provides better base for chat
2. **Use conservative learning rates** - Prevents forgetting in medical domain
3. **Mix with general conversation** - Maintains versatility (we include 2,000 SmolTalk examples)
4. **Monitor validation loss closely** - Early stopping if overfitting
5. **Evaluate on general benchmarks** - Ensure MMLU/ARC scores don't collapse

### For Dataset Quality

1. **Use evidence-based information** - All surveillance guidance should be accurate
2. **Include structured responses** - Headers, bullets, numbered steps
3. **Add actionable recommendations** - Public health requires action
4. **Vary complexity** - Mix simple and complex queries
5. **Cover diverse scenarios** - Different diseases, regions, populations

### For Deployment

1. **Add disclaimers** - "Consult official sources", "Not medical advice"
2. **Implement safety checks** - Filter inappropriate medical recommendations
3. **Monitor model outputs** - Regular quality review by epidemiologists
4. **Update regularly** - Surveillance practices evolve
5. **Track usage** - Which categories are queried most?

## Safety & Ethical Considerations

### Data Privacy
- **Never train on identifiable patient data** (HIPAA/GDPR violations)
- Use only de-identified, synthetic, or public surveillance data
- Implement data use agreements for institutional data

### Clinical Validity
- All training data should reflect **evidence-based** surveillance practices
- Review outputs with **subject matter experts** (epidemiologists)
- Regular validation against current CDC/WHO guidelines

### Liability & Disclaimers
- Model outputs are **decision support**, not autonomous recommendations
- Always include disclaimer: "Consult with senior epidemiologists"
- Document model limitations and known failure modes
- Maintain audit trails for critical decisions

### Bias & Equity
- Test performance across diverse populations and diseases
- Ensure training data includes underrepresented diseases
- Monitor for geographic or demographic biases

## Troubleshooting

### Dataset not found
```
FileNotFoundError: Surveillance dataset file not found
```
**Solution:** Run `python -m scripts.generate_surveillance_dataset` first

### Out of memory during training
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce `device_batch_size` in `surveillance_sft.py`:
```python
device_batch_size = 2  # or even 1
```

### Model not loading for chat
```
Error loading model: d26-surveillance
```
**Solution:** Check model was saved to correct directory:
```bash
ls surveillance_checkpoints/d26-surveillance/
```

### Poor evaluation scores
**Possible causes:**
1. Training not complete (check if `val_loss` was still decreasing)
2. Learning rates too high (try reducing by 50%)
3. Dataset quality issues (review sample outputs)
4. Need more training iterations

## Next Steps

### Short-term Enhancements

1. **Add real-world data** - Replace synthetic with institutional data
2. **Expand categories** - Add AMR surveillance, food safety, environmental health
3. **Implement RAG** - Retrieve current guidelines from CDC/WHO databases
4. **Add citations** - Include source references in responses

### Medium-term Goals

1. **Multi-turn conversations** - Support follow-up questions
2. **Tool use** - Calculator for rates, database queries, plotting
3. **Multi-lingual** - Support Spanish, French, Portuguese for global health
4. **Mobile deployment** - Lightweight model for field use

### Production Deployment

1. **API wrapper** - REST API for integration with surveillance systems
2. **Web interface** - User-friendly dashboard for epidemiologists
3. **Monitoring** - Track usage, errors, and performance
4. **Continuous updates** - Regular retraining with new data
5. **Evaluation pipeline** - Automated testing of model updates

## Resources

### Public Health Data Sources

- **CDC Wonder:** https://wonder.cdc.gov/ (US surveillance data)
- **WHO Global Outbreak Alert:** https://www.who.int/emergencies/disease-outbreak-news
- **ECDC Surveillance:** https://www.ecdc.europa.eu/en/surveillance-and-disease-data
- **GIDEON Database:** Global infectious disease database
- **ProMED-mail:** Disease outbreak reports

### Surveillance Training Materials

- **CDC Field Epidemiology Manual:** https://www.cdc.gov/eis/field-epi-manual/
- **WHO Surveillance Standards:** https://www.who.int/teams/integrated-health-services/infection-prevention-control/surveillance
- **CSTE Position Statements:** https://www.cste.org/
- **Johns Hopkins OpenCourseWare:** Epidemiology courses

### Related Projects

- **EpiEstim:** R₀ estimation package
- **Epiweeks:** Epidemiological week calculations
- **Surveillance R package:** Outbreak detection algorithms
- **OpenEpi:** Epidemiologic calculators

## Citation

If you use this surveillance specialization in research or practice, please cite:

```bibtex
@software{nanochat_surveillance_2024,
  title={Public Health Surveillance Training for nanochat},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/nanochat-surveillance}
}
```

## Support

For questions or issues:
1. Check this README first
2. Review sample outputs in `data/surveillance/`
3. Test with `surveillance_chat.py` interactive mode
4. Consult with epidemiologists for domain-specific questions

## License

This surveillance extension follows the same license as nanochat.

---

**Note:** This system provides general epidemiological guidance for public health professionals. Always verify with official sources (CDC, WHO, local health departments) and consult senior epidemiologists for critical decisions. Not a replacement for professional judgment or official protocols.
