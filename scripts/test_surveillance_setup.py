"""
Test script to verify surveillance training setup is complete and working.
Run this before starting training to ensure everything is configured correctly.

Usage:
python -m scripts.test_surveillance_setup
"""

import sys
from pathlib import Path
import json

def print_status(message, status="info"):
    """Print formatted status message"""
    symbols = {
        "info": "[INFO]",
        "success": "[OK]  ",
        "error": "[FAIL]",
        "warn": "[WARN]"
    }
    print(f"{symbols.get(status, '[INFO]')} {message}")

def test_dataset_files():
    """Test that dataset files exist and are valid"""
    print("\n" + "="*80)
    print("Testing Dataset Files")
    print("="*80)

    data_dir = Path("data/surveillance")
    required_files = ["train.json", "validation.json", "test.json", "dataset_stats.json"]

    all_exist = True
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            print_status(f"Found {filename}", "success")
            # Check if it's valid JSON
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print_status(f"  Valid JSON with {len(data) if isinstance(data, list) else 'N/A'} items", "success")
            except json.JSONDecodeError:
                print_status(f"  Invalid JSON format!", "error")
                all_exist = False
        else:
            print_status(f"Missing {filename}", "error")
            all_exist = False

    if not all_exist:
        print_status("\nPlease generate dataset first:", "error")
        print_status("  python -m scripts.generate_surveillance_dataset", "info")
        return False

    # Load and validate dataset structure
    print_status("\nValidating dataset structure...", "info")
    try:
        with open(data_dir / "train.json", 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        sample = train_data[0]
        assert "messages" in sample, "Missing 'messages' field"
        assert len(sample["messages"]) >= 2, "Not enough messages"
        assert sample["messages"][0]["role"] == "user", "First message should be user"
        assert sample["messages"][1]["role"] == "assistant", "Second message should be assistant"

        print_status("Dataset structure is valid", "success")
        return True

    except Exception as e:
        print_status(f"Dataset validation failed: {e}", "error")
        return False

def test_task_loader():
    """Test that the Task class can load the dataset"""
    print("\n" + "="*80)
    print("Testing Task Loader")
    print("="*80)

    try:
        from tasks.surveillance import PublicHealthSurveillance

        print_status("Imported PublicHealthSurveillance successfully", "success")

        # Try loading each split
        for split in ["train", "validation", "test"]:
            try:
                ds = PublicHealthSurveillance(split=split)
                print_status(f"Loaded {split} split: {len(ds)} examples", "success")

                # Test getting an example
                example = ds[0]
                assert "messages" in example
                print_status(f"  Can retrieve examples from {split}", "success")

            except Exception as e:
                print_status(f"Failed to load {split} split: {e}", "error")
                return False

        return True

    except ImportError as e:
        print_status(f"Failed to import Task class: {e}", "error")
        return False
    except Exception as e:
        print_status(f"Task loader test failed: {e}", "error")
        return False

def test_training_script():
    """Test that training script exists and is valid Python"""
    print("\n" + "="*80)
    print("Testing Training Script")
    print("="*80)

    script_path = Path("scripts/surveillance_sft.py")

    if not script_path.exists():
        print_status("Training script not found", "error")
        return False

    print_status("Found surveillance_sft.py", "success")

    # Try to compile the script (syntax check)
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, script_path, 'exec')
        print_status("Training script syntax is valid", "success")
        return True
    except SyntaxError as e:
        print_status(f"Training script has syntax errors: {e}", "error")
        return False

def test_evaluation_script():
    """Test that evaluation script exists"""
    print("\n" + "="*80)
    print("Testing Evaluation Script")
    print("="*80)

    script_path = Path("scripts/surveillance_eval.py")

    if not script_path.exists():
        print_status("Evaluation script not found", "error")
        return False

    print_status("Found surveillance_eval.py", "success")

    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, script_path, 'exec')
        print_status("Evaluation script syntax is valid", "success")
        return True
    except SyntaxError as e:
        print_status(f"Evaluation script has syntax errors: {e}", "error")
        return False

def test_chat_script():
    """Test that chat script exists"""
    print("\n" + "="*80)
    print("Testing Chat Interface")
    print("="*80)

    script_path = Path("scripts/surveillance_chat.py")

    if not script_path.exists():
        print_status("Chat script not found", "error")
        return False

    print_status("Found surveillance_chat.py", "success")

    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, script_path, 'exec')
        print_status("Chat script syntax is valid", "success")
        return True
    except SyntaxError as e:
        print_status(f"Chat script has syntax errors: {e}", "error")
        return False

def print_dataset_statistics():
    """Print detailed dataset statistics"""
    print("\n" + "="*80)
    print("Dataset Statistics")
    print("="*80)

    try:
        with open("data/surveillance/dataset_stats.json", 'r') as f:
            stats = json.load(f)

        print(f"\nTotal examples: {stats['total_examples']}")
        print(f"  Training:   {stats['train_size']}")
        print(f"  Validation: {stats['val_size']}")
        print(f"  Test:       {stats['test_size']}")

        print(f"\nGenerated: {stats['generated_date']}")

        print("\nCategories:")
        for category, description in stats['categories'].items():
            print(f"  - {category}: {description}")

    except Exception as e:
        print_status(f"Could not load statistics: {e}", "warn")

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)
    print("\n1. Train the surveillance model:")
    print("   torchrun --standalone --nproc_per_node=8 -m scripts.surveillance_sft")
    print("\n   Or for single GPU (debugging):")
    print("   python -m scripts.surveillance_sft")

    print("\n2. After training, evaluate the model:")
    print("   python -m scripts.surveillance_eval --model_tag d26-surveillance --source sft")

    print("\n3. Test interactively:")
    print("   python -m scripts.surveillance_chat --model_tag d26-surveillance --source sft")

    print("\n4. Read the documentation:")
    print("   See SURVEILLANCE_README.md for detailed instructions")

    print("\n" + "="*80)

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("Public Health Surveillance Setup Verification")
    print("="*80)

    tests = [
        ("Dataset Files", test_dataset_files),
        ("Task Loader", test_task_loader),
        ("Training Script", test_training_script),
        ("Evaluation Script", test_evaluation_script),
        ("Chat Interface", test_chat_script),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_status(f"Unexpected error in {name}: {e}", "error")
            results.append((name, False))

    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    all_passed = True
    for name, result in results:
        status = "success" if result else "error"
        print_status(f"{name}: {'PASSED' if result else 'FAILED'}", status)
        if not result:
            all_passed = False

    if all_passed:
        print_status("\nAll tests passed! Setup is complete.", "success")
        print_dataset_statistics()
        print_next_steps()
        return 0
    else:
        print_status("\nSome tests failed. Please fix the issues above.", "error")
        return 1

if __name__ == "__main__":
    sys.exit(main())
