"""
Interactive surveillance chatbot using the fine-tuned surveillance model.
Specialized for public health surveillance queries.
**CPU-compatible version**

Usage:
python -m scripts.surveillance_chat_cpu --model_tag d26-surveillance
"""

import argparse
import torch

from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# Parse command line arguments
parser = argparse.ArgumentParser(description='Surveillance Chat (CPU-compatible)')
parser.add_argument('--model_tag', type=str, default=None, help='Model tag to load')
parser.add_argument('--source', type=str, default='mid', help='Source: base|mid|sft')
parser.add_argument('--step', type=int, default=None, help='Checkpoint step to load')
args = parser.parse_args()

# Configuration from args
model_tag = args.model_tag
source = args.source
step = args.step
dtype = "float32" # Use float32 for CPU

# -----------------------------------------------------------------------------
# Surveillance System Prompt
# -----------------------------------------------------------------------------

SURVEILLANCE_SYSTEM_PROMPT = """You are a public health surveillance AI assistant specialized in epidemiology and disease monitoring. Your expertise includes:

- Disease outbreak detection and investigation
- Epidemiological trend analysis
- Public health risk assessment
- Surveillance report generation
- Contact tracing protocols
- Vaccination program monitoring
- Syndromic surveillance
- Zoonotic disease surveillance
- Global health security

When responding to surveillance queries:
1. Provide evidence-based, actionable information
2. Use appropriate epidemiological terminology
3. Structure responses clearly (use headers, bullets, numbered lists)
4. Include specific recommendations when appropriate
5. Consider public health implications
6. Acknowledge uncertainty when data is insufficient

IMPORTANT DISCLAIMERS:
- This system provides general epidemiological guidance for public health professionals
- Always verify data with official sources (CDC, WHO, local health departments)
- For urgent outbreaks, follow established emergency protocols
- Consult with senior epidemiologists for critical decisions

You may be asked about:
- Outbreak detection: Identifying when case counts exceed expected levels
- Risk assessment: Evaluating threat levels and population impact
- Data interpretation: Explaining surveillance metrics (R₀, incidence rates, etc.)
- Control measures: Recommending evidence-based interventions
- Surveillance systems: Designing or evaluating surveillance programs
- Reporting: Creating surveillance reports and alerts

Provide thorough, professional responses appropriate for public health practitioners."""

# -----------------------------------------------------------------------------
# Example Queries for User Guidance
# -----------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "Detect outbreak: 150 cases of influenza vs baseline of 40 cases",
    "Analyze: Measles cases increased 200% in unvaccinated children",
    "Risk assessment: COVID-19 in a city with 5% vaccination coverage",
    "Interpret: What does R₀ = 3.5 mean for disease control?",
    "Protocol: Contact tracing steps for tuberculosis exposure",
    "Report: Summarize weekly surveillance findings for hepatitis A",
    "Syndromic: How to use ED visits for early outbreak detection?",
    "Global: Dengue outbreak in Brazil - international implications?",
]

# -----------------------------------------------------------------------------
# Main Chat Interface
# -----------------------------------------------------------------------------

def print_welcome():
    """Print welcome message and examples"""
    print("\n" + "="*80)
    print("PUBLIC HEALTH SURVEILLANCE CHATBOT (CPU Mode)")
    print("="*80)
    print("\nThis AI assistant is specialized in epidemiological surveillance and")
    print("disease monitoring. Ask questions about outbreaks, trends, risk assessment,")
    print("surveillance systems, and public health response.\n")
    print("NOTE: Running on CPU - responses may be slower than GPU mode.\n")
    print("Example queries:")
    for i, example in enumerate(EXAMPLE_QUERIES, 1):
        print(f"  {i}. {example}")
    print("\n" + "-"*80)
    print("Commands:")
    print("  'quit' or 'exit' - Exit the chat")
    print("  'clear' - Clear conversation history")
    print("  'examples' - Show example queries again")
    print("-"*80 + "\n")

def print_examples():
    """Print example queries"""
    print("\nExample Surveillance Queries:")
    for i, example in enumerate(EXAMPLE_QUERIES, 1):
        print(f"  {i}. {example}")
    print()

def main():
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_torch = torch.float32 if dtype == 'float32' else torch.bfloat16

    if device == "cpu":
        dtype_torch = torch.float32  # Force float32 on CPU
        print("\nRunning on CPU (no GPU detected)")
    else:
        print(f"\nRunning on GPU ({torch.cuda.get_device_name(0)})")

    # Load model
    print(f"Loading surveillance-specialized model: {model_tag}...")
    print("This may take a minute on CPU...\n")

    try:
        model, tokenizer, meta = load_model(source, device, phase="test", model_tag=model_tag, step=step)
        model.eval()
        engine = Engine(model, tokenizer)
        print(f"Model loaded successfully!")
    except Exception as e:
        print(f"\nError loading model: {e}")
        print(f"\nMake sure you've trained the surveillance model first:")
        print(f"  torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft")
        print(f"\nOr use a pre-existing model with --source mid or --model_tag <tag>")
        return

    print_welcome()

    # Initialize conversation with system prompt
    conversation = {
        "messages": [
            {"role": "system", "content": SURVEILLANCE_SYSTEM_PROMPT}
        ]
    }

    # Chat loop
    autocast_ctx = torch.amp.autocast(device_type=device, dtype=dtype_torch, enabled=(device=="cuda"))

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting surveillance chat. Stay safe!")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nExiting surveillance chat. Stay safe!")
            break

        if user_input.lower() == 'clear':
            conversation = {
                "messages": [
                    {"role": "system", "content": SURVEILLANCE_SYSTEM_PROMPT}
                ]
            }
            print("\nConversation history cleared.\n")
            continue

        if user_input.lower() == 'examples':
            print_examples()
            continue

        # Add user message to conversation
        conversation["messages"].append({
            "role": "user",
            "content": user_input
        })

        # Generate response
        print("\nAssistant: ", end="", flush=True)
        try:
            with torch.no_grad(), autocast_ctx:
                response = engine.generate(
                    conversation,
                    max_length=1024,
                    temperature=0.7
                )

            print(response)
            print()

            # Add assistant response to conversation history
            conversation["messages"].append({
                "role": "assistant",
                "content": response
            })

        except Exception as e:
            print(f"\nError generating response: {e}")
            # Remove the failed user message
            conversation["messages"].pop()
            continue

if __name__ == "__main__":
    main()
