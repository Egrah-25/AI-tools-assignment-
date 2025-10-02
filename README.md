# AI Tools Machine Learning Assignment Portfolio

📋 Project Overview

This repository contains a comprehensive machine learning assignment covering classical ML, deep learning, and NLP tasks. The project demonstrates end-to-end implementation of various ML techniques with a focus on practical applications and ethical considerations.

🎯 Assignment Structure

Part 2: Practical Implementation (50%)

Task 1: Classical ML with Scikit-learn - Iris Species Classification

· Dataset: Iris Species Dataset
· Goal: Preprocess data and train a decision tree classifier
· Evaluation: Accuracy, Precision, Recall
· Key Features: Data preprocessing, model training, performance evaluation

Task 2: Deep Learning with TensorFlow - MNIST Handwritten Digits

· Dataset: MNIST Handwritten Digits
· Goal: Build CNN model with >95% test accuracy
· Key Features: CNN architecture, training visualization, prediction analysis

Task 3: NLP with spaCy - Amazon Reviews Analysis

· Data: Amazon Product Reviews
· Goal: Named Entity Recognition and sentiment analysis
· Key Features: NER extraction, rule-based sentiment analysis, entity visualization

Part 3: Ethics & Optimization (10%)

Ethical Considerations

· Bias analysis for MNIST and Amazon Reviews models
· Mitigation strategies using TensorFlow Fairness Indicators and spaCy

Troubleshooting Challenge

· Debugging and fixing buggy TensorFlow code
· Common issues: dimension mismatches, incorrect loss functions

Bonus Task (Extra 10%)

· Streamlit Web Interface for MNIST classifier deployment

🛠 Installation & Setup

Prerequisites

· Python 3.8+
· pip package manager

Required Packages

```bash
# Core ML libraries
pip install tensorflow scikit-learn spacy

# Data manipulation and visualization
pip install pandas numpy matplotlib seaborn

# Web deployment (Bonus task)
pip install streamlit pillow

# Download spaCy model
python -m spacy download en_core_web_sm
```
Installation Steps

1. Clone this repository:

```bash
git clone <repository-url>
cd ml-assignment-portfolio
```

1. Create virtual environment (recommended):

```bash
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

📁 Project Structure

```
ml-assignment/
│
├── task1_iris_classification.py
├── task2_mnist_cnn.py
├── task3_nlp_analysis.py
├── task4_ethics_troubleshooting.py
├── bonus_streamlit_app.py
├── requirements.txt
└── README.md
```

🚀 Running the Projects

Task 1: Iris Classification

```bash
python task1_iris_classification.py
```

Expected Output:

· Dataset analysis and preprocessing steps
· Decision tree model training
· Evaluation metrics (accuracy, precision, recall)
· Feature importance analysis

Task 2: MNIST CNN

```bash
python mnist_cnn.py
```

Expected Output:

· CNN model architecture summary
· Training progress with accuracy/loss plots
· Test accuracy (>95%)
· Sample predictions visualization

Task 3: NLP Analysis

```bash
python nlp_analysis.py
```

Expected Output:

· Named Entity Recognition results
· Sentiment analysis of reviews
· Entity visualization using spaCy
· Brand mention analysis

Task 4: Ethics & Troubleshooting

```bash
python ethics_troubleshooting.py
```

Expected Output:

· Bias analysis report
· Fixed TensorFlow code execution
· Performance comparison

Bonus: Streamlit Deployment

```bash
streamlit run streamlit_app.py
```

Access at: http://localhost:8501

📊 Results Summary

Task 1: Iris Classification

· Accuracy: ~97%
· Precision: ~97%
· Recall: ~97%
· Key Insight: Petal measurements are most important features

Task 2: MNIST CNN

· Test Accuracy: >95% (typically 98-99%)
· Model: 3-layer CNN with dropout
· Training: 10 epochs with data normalization

Task 3: NLP Analysis

· Entities Extracted: Brands, products, organizations
· Sentiment Analysis: Rule-based approach
· Visualization: Interactive entity displays

🔍 Key Learnings

Technical Skills Demonstrated:

1. Data Preprocessing: Handling missing values, label encoding, normalization
2. Model Selection: Decision trees, CNNs, rule-based NLP
3. Evaluation Metrics: Accuracy, precision, recall, confusion matrices
4. Deep Learning: CNN architecture design, hyperparameter tuning
5. NLP: Named Entity Recognition, sentiment analysis
6. Deployment: Web interface with Streamlit

Ethical Considerations:

· Identified potential biases in training data
· Proposed mitigation strategies using fairness tools
· Emphasized importance of diverse datasets

🐛 Troubleshooting Common Issues

General Issues:

1. Memory Errors: Reduce batch size in CNN training
2. spaCy Model Not Found: Run python -m spacy download en_core_web_sm
3. TensorFlow GPU Issues: Install GPU version or use CPU-only

Task-Specific Issues:

· Iris Dataset: Automatic loading via scikit-learn
· MNIST Download: Automatic download on first run
· spaCy NER: Ensure correct model version

📈 Performance Optimization Tips

For MNIST CNN:

· Use data augmentation to improve generalization
· Experiment with different architectures (ResNet, VGG)
· Implement learning rate scheduling
· Use early stopping to prevent overfitting

For NLP Tasks:

· Fine-tune spaCy models on domain-specific data
· Implement more sophisticated sentiment analysis
· Add custom entity rules for product names

🤝 Contributing

Feel free to contribute to this project by:

1. Reporting bugs and issues
2. Suggesting improvements to models
3. Adding new features or datasets
4. Improving documentation

📝 License

This project is for educational purposes as part of a machine learning assignment.

👨‍💻 Author

Egrah Savai 

🔗 Useful Resources

· Scikit-learn Documentation
· TensorFlow Guide
· spaCy Documentation
· Streamlit Documentation
