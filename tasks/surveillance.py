"""
Public Health Surveillance Task for nanochat.
Loads and manages surveillance-specific conversational data for training and evaluation.
"""

import json
from pathlib import Path
from tasks.common import Task


class PublicHealthSurveillance(Task):
    """
    Public health surveillance dataset for training epidemiological AI.
    Covers outbreak detection, trend analysis, risk assessment, and surveillance reporting.
    """

    def __init__(self, split, data_dir="data/surveillance", **kwargs):
        """
        Initialize surveillance task.

        Args:
            split: One of 'train', 'validation', or 'test'
            data_dir: Directory containing the surveillance dataset JSON files
            **kwargs: Additional arguments for Task base class (start, stop, step)
        """
        super().__init__(**kwargs)
        assert split in ["train", "validation", "test"], \
            f"Surveillance split must be train|validation|test, got {split}"

        self.split = split
        self.data_dir = Path(data_dir)

        # Load the appropriate split
        data_file = self.data_dir / f"{split}.json"

        if not data_file.exists():
            raise FileNotFoundError(
                f"Surveillance dataset file not found: {data_file}\n"
                f"Please run: python -m scripts.generate_surveillance_dataset"
            )

        # Load data from JSON file
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.length = len(self.data)
        print(f"Loaded {self.length} {split} examples for Public Health Surveillance")

    @property
    def eval_type(self):
        """This is a generative task (not multiple choice)"""
        return 'generative'

    def num_examples(self):
        """Return total number of examples in this split"""
        return self.length

    def get_example(self, index):
        """
        Get a single conversation example.

        Returns:
            dict with 'messages' key containing list of message dicts
        """
        assert 0 <= index < self.length, \
            f"Index {index} out of range for surveillance dataset of size {self.length}"

        conversation = self.data[index]

        # Validate message structure
        messages = conversation["messages"]
        assert len(messages) >= 2, "Surveillance conversations must have at least 2 messages"
        assert messages[0]["role"] == "user", "First message must be from user"
        assert messages[-1]["role"] == "assistant", "Last message must be from assistant"

        # Ensure alternating roles (user, assistant, user, assistant, ...)
        for i, message in enumerate(messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, \
                f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"

        return conversation

    def evaluate(self, problem, completion):
        """
        Evaluate a model's completion for a surveillance problem.

        This is a complex generative task, so we use multiple evaluation criteria:
        - Presence of key epidemiological concepts
        - Structured response format
        - Actionable recommendations
        """
        # For now, this is a placeholder
        # In production, you might use:
        # - BLEU/ROUGE scores for text similarity
        # - Custom rubrics for key concepts
        # - LLM-as-judge evaluation
        # - Human expert evaluation

        metadata = problem.get("metadata", {})
        category = metadata.get("category", "unknown")

        # Basic evaluation: check if response is substantive
        score = {
            "category": category,
            "length": len(completion),
            "has_structure": self._check_structure(completion),
            "has_recommendations": self._check_recommendations(completion),
        }

        return score

    def _check_structure(self, text):
        """Check if response has structured format (bullets, sections, etc.)"""
        structure_indicators = ['**', '##', '1.', '2.', '- ', '\n\n']
        return any(indicator in text for indicator in structure_indicators)

    def _check_recommendations(self, text):
        """Check if response includes actionable recommendations"""
        recommendation_keywords = [
            'recommend', 'should', 'implement', 'action', 'strategy',
            'intervention', 'monitor', 'investigate', 'enhance', 'immediate'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in recommendation_keywords)


class SurveillanceCategory(Task):
    """
    Filtered surveillance task for a specific category.
    Useful for targeted training or evaluation on specific surveillance domains.
    """

    def __init__(self, split, category, data_dir="data/surveillance", **kwargs):
        """
        Initialize category-specific surveillance task.

        Args:
            split: One of 'train', 'validation', or 'test'
            category: Specific surveillance category to filter
                     (e.g., 'outbreak_detection', 'trend_analysis', etc.)
            data_dir: Directory containing the surveillance dataset JSON files
        """
        super().__init__(**kwargs)
        assert split in ["train", "validation", "test"], \
            f"Surveillance split must be train|validation|test, got {split}"

        self.split = split
        self.category = category
        self.data_dir = Path(data_dir)

        # Load the appropriate split
        data_file = self.data_dir / f"{split}.json"

        if not data_file.exists():
            raise FileNotFoundError(
                f"Surveillance dataset file not found: {data_file}\n"
                f"Please run: python -m scripts.generate_surveillance_dataset"
            )

        # Load and filter data by category
        with open(data_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        # Filter for specific category
        self.data = [
            item for item in all_data
            if item.get("metadata", {}).get("category") == category
        ]

        self.length = len(self.data)
        print(f"Loaded {self.length} {split} examples for category: {category}")

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.length

    def get_example(self, index):
        assert 0 <= index < self.length, \
            f"Index {index} out of range for category dataset of size {self.length}"

        conversation = self.data[index]
        messages = conversation["messages"]

        # Validate structure
        assert len(messages) >= 2
        assert messages[0]["role"] == "user"
        assert messages[-1]["role"] == "assistant"

        return conversation


if __name__ == "__main__":
    """Test the surveillance task loader"""

    # Test loading full dataset
    print("Testing PublicHealthSurveillance task loader...")
    try:
        train_ds = PublicHealthSurveillance(split="train")
        print(f"✅ Successfully loaded {len(train_ds)} training examples")

        # Get a sample
        sample = train_ds[0]
        print(f"\n{'='*80}")
        print("SAMPLE CONVERSATION:")
        print(f"{'='*80}")
        print(f"User: {sample['messages'][0]['content'][:200]}...")
        print(f"\nAssistant: {sample['messages'][1]['content'][:300]}...")
        print(f"{'='*80}\n")

        # Test slicing
        subset = PublicHealthSurveillance(split="train", start=0, stop=100)
        print(f"✅ Successfully created subset: {len(subset)} examples")

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\n📝 To generate the dataset, run:")
        print("   python -m scripts.generate_surveillance_dataset")
    except Exception as e:
        print(f"\n❌ Error: {e}")
