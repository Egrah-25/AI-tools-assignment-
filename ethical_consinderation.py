# Ethical considerations analysis
print("=== Ethical Considerations ===")

print("\n1. MNIST Model Potential Biases:")
mnist_biases = [
    "• Writing style variations: Different cultural writing styles may not be well represented",
    "• Image quality bias: Clean, centered digits vs. messy, rotated ones",
    "• Demographic bias: Dataset primarily contains Western/Arabic numerals",
    "• Accessibility bias: May not perform well on digits written by people with motor disabilities"
]

for bias in mnist_biases:
    print(bias)

print("\n2. Amazon Reviews Model Potential Biases:")
review_biases = [
    "• Language bias: Primarily English reviews, excluding non-English speakers",
    "• Cultural bias: Sentiment expressions vary across cultures",
    "• Product category bias: Some product types may have more negative reviews",
    "• Demographic bias: Younger users may review differently than older users"
]

for bias in review_biases:
    print(bias)

print("\n3. Mitigation Strategies:")
mitigation_strategies = [
    "• TensorFlow Fairness Indicators: Monitor metrics across different subgroups",
    "• Data augmentation: Add diverse writing styles to MNIST",
    "• spaCy's rule-based systems: Add custom rules for cultural expressions",
    "• Regular bias audits: Continuously monitor model performance",
    "• Diverse training data: Include multi-lingual and multi-cultural examples"
]

for strategy in mitigation_strategies:
    print(strategy)
