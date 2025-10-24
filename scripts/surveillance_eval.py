"""
Evaluate nanochat model on public health surveillance tasks.
This script provides comprehensive evaluation metrics for surveillance-specialized models.

Usage:
python -m scripts.surveillance_eval --source sft --model_tag d26-surveillance
"""

import os
import torch
import json
from pathlib import Path
from datetime import datetime

from nanochat.common import compute_init, compute_cleanup, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.surveillance import PublicHealthSurveillance

# Evaluation configuration
source = "sft" # base|mid|sft
model_tag = "d26-surveillance" # model tag for surveillance model
dtype = "bfloat16"
max_eval_examples = 500 # Limit evaluation for speed
batch_size = 1 # Generation is typically done one at a time
max_gen_length = 1024 # Maximum response length
temperature = 0.7 # Sampling temperature
output_dir = "eval_results/surveillance"

# Allow CLI overrides
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Evaluation Metrics
# -----------------------------------------------------------------------------

def compute_rouge_scores(reference, hypothesis):
    """
    Compute ROUGE scores for text similarity.
    Simple token-based implementation (for production, use rouge-score library).
    """
    def get_ngrams(text, n):
        tokens = text.lower().split()
        return set([' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

    # ROUGE-1 (unigram overlap)
    ref_unigrams = get_ngrams(reference, 1)
    hyp_unigrams = get_ngrams(hypothesis, 1)
    if len(hyp_unigrams) == 0:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    rouge1_overlap = len(ref_unigrams & hyp_unigrams)
    rouge1_precision = rouge1_overlap / len(hyp_unigrams) if hyp_unigrams else 0
    rouge1_recall = rouge1_overlap / len(ref_unigrams) if ref_unigrams else 0
    rouge1_f1 = 2 * rouge1_precision * rouge1_recall / (rouge1_precision + rouge1_recall) \
                if (rouge1_precision + rouge1_recall) > 0 else 0

    # ROUGE-2 (bigram overlap)
    ref_bigrams = get_ngrams(reference, 2)
    hyp_bigrams = get_ngrams(hypothesis, 2)
    rouge2_overlap = len(ref_bigrams & hyp_bigrams)
    rouge2_precision = rouge2_overlap / len(hyp_bigrams) if hyp_bigrams else 0
    rouge2_recall = rouge2_overlap / len(ref_bigrams) if ref_bigrams else 0
    rouge2_f1 = 2 * rouge2_precision * rouge2_recall / (rouge2_precision + rouge2_recall) \
                if (rouge2_precision + rouge2_recall) > 0 else 0

    return {
        "rouge1": round(rouge1_f1, 4),
        "rouge2": round(rouge2_f1, 4),
    }

def check_key_concepts(response, category):
    """
    Check for presence of key epidemiological concepts by category.
    Returns a score 0-1 based on concept coverage.
    """
    response_lower = response.lower()

    concept_keywords = {
        "outbreak_detection": [
            "outbreak", "baseline", "cases", "threshold", "investigation",
            "surveillance", "increase", "epidemiological", "public health"
        ],
        "trend_analysis": [
            "trend", "increase", "decrease", "pattern", "seasonal",
            "incidence", "prevalence", "demographic", "temporal"
        ],
        "risk_assessment": [
            "risk", "assessment", "population", "vulnerable", "impact",
            "probability", "severity", "mitigation", "prevention"
        ],
        "surveillance_report": [
            "report", "summary", "cases", "data", "findings",
            "week", "period", "statistics", "analysis"
        ],
        "vaccination_coverage": [
            "vaccination", "coverage", "immunization", "vaccine",
            "herd immunity", "uptake", "dose", "campaign"
        ],
        "data_interpretation": [
            "interpret", "data", "indicates", "suggests", "rate",
            "per 100,000", "statistically", "significance"
        ],
        "syndromic_surveillance": [
            "syndromic", "syndrome", "symptoms", "early detection",
            "aberration", "monitoring", "emergency department"
        ],
        "contact_tracing": [
            "contact", "tracing", "exposure", "quarantine", "isolation",
            "secondary cases", "transmission chain"
        ],
        "zoonotic_surveillance": [
            "zoonotic", "animal", "vector", "reservoir", "wildlife",
            "one health", "spillover", "transmission"
        ],
        "global_surveillance": [
            "international", "global", "WHO", "outbreak", "travel",
            "border", "alert", "coordination", "pandemic"
        ]
    }

    keywords = concept_keywords.get(category, [])
    if not keywords:
        return 0.5 # Unknown category

    present = sum(1 for keyword in keywords if keyword in response_lower)
    score = present / len(keywords)
    return round(score, 4)

def check_response_structure(response):
    """
    Evaluate response structure quality.
    Checks for:
    - Section headers (##, **)
    - Bullet points or numbered lists
    - Paragraph breaks
    - Length appropriateness
    """
    score = 0.0

    # Check for headers
    if '##' in response or '**' in response:
        score += 0.25

    # Check for lists
    if any(marker in response for marker in ['1.', '2.', '- ', '* ']):
        score += 0.25

    # Check for paragraph structure (multiple newlines)
    if '\n\n' in response:
        score += 0.25

    # Check length (not too short, not too long)
    word_count = len(response.split())
    if 100 < word_count < 800:
        score += 0.25

    return round(score, 4)

def check_actionability(response):
    """
    Check if response includes actionable recommendations.
    Returns score 0-1.
    """
    response_lower = response.lower()

    action_indicators = [
        "recommend", "should", "must", "need to", "important to",
        "implement", "establish", "conduct", "initiate", "activate",
        "monitor", "investigate", "enhance", "improve", "strengthen",
        "immediate", "urgent", "priority", "action", "step"
    ]

    # Check for recommendations section
    has_recommendations_section = any(
        marker in response_lower
        for marker in ["recommendation", "action", "next step", "priority"]
    )

    # Count action indicators
    action_count = sum(1 for indicator in action_indicators if indicator in response_lower)

    score = 0.0
    if has_recommendations_section:
        score += 0.5
    if action_count >= 3:
        score += 0.5
    elif action_count >= 1:
        score += 0.25

    return round(score, 4)

def comprehensive_evaluation(reference, response, category):
    """
    Comprehensive evaluation combining multiple metrics.
    """
    rouge = compute_rouge_scores(reference, response)
    concepts = check_key_concepts(response, category)
    structure = check_response_structure(response)
    actionability = check_actionability(response)

    # Composite score (weighted average)
    composite = (
        0.3 * rouge["rouge1"] +
        0.2 * rouge["rouge2"] +
        0.2 * concepts +
        0.15 * structure +
        0.15 * actionability
    )

    return {
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "concept_coverage": concepts,
        "structure_quality": structure,
        "actionability": actionability,
        "composite_score": round(composite, 4)
    }

# -----------------------------------------------------------------------------
# Main Evaluation
# -----------------------------------------------------------------------------

def main():
    print0("\n" + "="*80)
    print0("🦠 Public Health Surveillance Model Evaluation")
    print0("="*80 + "\n")

    # Setup
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    dtype_torch = torch.float32 if dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype_torch)

    # Load model
    print0(f"Loading model: source={source}, model_tag={model_tag}")
    model, tokenizer, meta = load_model(source, device, phase="test", model_tag=model_tag)
    model.eval()
    engine = Engine(model, tokenizer)
    print0("✅ Model loaded successfully\n")

    # Load test dataset
    print0("Loading test dataset...")
    try:
        test_ds = PublicHealthSurveillance(split="test")
        print0(f"✅ Loaded {len(test_ds)} test examples\n")
    except FileNotFoundError as e:
        print0(f"❌ Error: {e}")
        print0("\n📝 Please generate the dataset first:")
        print0("   python -m scripts.generate_surveillance_dataset")
        return

    # Limit evaluation examples
    num_eval = min(len(test_ds), max_eval_examples)
    print0(f"Evaluating on {num_eval} examples...\n")

    # Run evaluation
    all_results = []
    category_results = {}

    for i in range(num_eval):
        example = test_ds[i]
        messages = example["messages"]
        category = example.get("metadata", {}).get("category", "unknown")

        # Extract question and reference answer
        question = messages[0]["content"]
        reference_answer = messages[1]["content"]

        # Generate model response
        conversation = {"messages": [messages[0]]} # Just the user question
        with torch.no_grad(), autocast_ctx:
            response = engine.generate(
                conversation,
                max_length=max_gen_length,
                temperature=temperature
            )

        # Evaluate response
        metrics = comprehensive_evaluation(reference_answer, response, category)
        metrics["category"] = category
        metrics["question_length"] = len(question.split())
        metrics["response_length"] = len(response.split())

        all_results.append(metrics)

        # Aggregate by category
        if category not in category_results:
            category_results[category] = []
        category_results[category].append(metrics)

        # Progress update
        if (i + 1) % 50 == 0:
            print0(f"  Evaluated {i+1}/{num_eval} examples...")

    print0(f"\n✅ Evaluation complete!\n")

    # -----------------------------------------------------------------------------
    # Compute aggregate statistics
    # -----------------------------------------------------------------------------

    def compute_stats(results):
        """Compute mean statistics from list of result dicts"""
        keys = ["rouge1", "rouge2", "concept_coverage", "structure_quality",
                "actionability", "composite_score", "response_length"]
        stats = {}
        for key in keys:
            values = [r[key] for r in results if key in r]
            if values:
                stats[f"{key}_mean"] = round(sum(values) / len(values), 4)
        stats["num_examples"] = len(results)
        return stats

    # Overall statistics
    overall_stats = compute_stats(all_results)

    # Per-category statistics
    category_stats = {
        category: compute_stats(results)
        for category, results in category_results.items()
    }

    # -----------------------------------------------------------------------------
    # Display results
    # -----------------------------------------------------------------------------

    print0("="*80)
    print0("📊 OVERALL RESULTS")
    print0("="*80)
    print0(f"Number of examples: {overall_stats['num_examples']}")
    print0(f"ROUGE-1:            {overall_stats['rouge1_mean']:.4f}")
    print0(f"ROUGE-2:            {overall_stats['rouge2_mean']:.4f}")
    print0(f"Concept Coverage:   {overall_stats['concept_coverage_mean']:.4f}")
    print0(f"Structure Quality:  {overall_stats['structure_quality_mean']:.4f}")
    print0(f"Actionability:      {overall_stats['actionability_mean']:.4f}")
    print0(f"Composite Score:    {overall_stats['composite_score_mean']:.4f}")
    print0(f"Avg Response Length: {overall_stats['response_length_mean']:.1f} words")
    print0("")

    print0("="*80)
    print0("📊 RESULTS BY CATEGORY")
    print0("="*80)
    for category, stats in sorted(category_stats.items()):
        print0(f"\n{category}:")
        print0(f"  Examples:         {stats['num_examples']}")
        print0(f"  ROUGE-1:          {stats['rouge1_mean']:.4f}")
        print0(f"  Concept Coverage: {stats['concept_coverage_mean']:.4f}")
        print0(f"  Composite Score:  {stats['composite_score_mean']:.4f}")

    # -----------------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------------

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"eval_results_{model_tag}_{timestamp}.json"

    output_data = {
        "model_config": {
            "source": source,
            "model_tag": model_tag,
            "dtype": dtype,
        },
        "eval_config": {
            "num_examples": num_eval,
            "max_gen_length": max_gen_length,
            "temperature": temperature,
        },
        "overall_stats": overall_stats,
        "category_stats": category_stats,
        "detailed_results": all_results[:100] # Save first 100 for review
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print0(f"\n✅ Results saved to: {results_file}")

    # Save sample outputs
    samples_file = output_path / f"sample_outputs_{model_tag}_{timestamp}.txt"
    with open(samples_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SAMPLE MODEL OUTPUTS\n")
        f.write("="*80 + "\n\n")

        for i in range(min(10, len(test_ds))):
            example = test_ds[i]
            messages = example["messages"]
            category = example.get("metadata", {}).get("category", "unknown")

            question = messages[0]["content"]
            reference = messages[1]["content"]

            conversation = {"messages": [messages[0]]}
            with torch.no_grad(), autocast_ctx:
                response = engine.generate(conversation, max_length=max_gen_length, temperature=temperature)

            f.write(f"Example {i+1} - Category: {category}\n")
            f.write("-"*80 + "\n")
            f.write(f"QUESTION:\n{question}\n\n")
            f.write(f"MODEL RESPONSE:\n{response}\n\n")
            f.write(f"REFERENCE:\n{reference[:500]}...\n\n")
            f.write("="*80 + "\n\n")

    print0(f"✅ Sample outputs saved to: {samples_file}\n")

    # Cleanup
    compute_cleanup()

if __name__ == "__main__":
    main()
