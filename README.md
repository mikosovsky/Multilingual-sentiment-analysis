# Multilingual Sentiment Analysis

A comprehensive sentiment analysis project using Transformer-based models to classify sentiments in multilingual text data from the internet. This project implements and fine-tunes pre-trained models to analyze user opinions across English, Spanish, and Chinese languages.

## 📋 Overview

This project focuses on sentiment analysis of internet texts using Transformer neural networks. It processes and analyzes multilingual data to classify user opinions into positive, negative, and neutral categories. The implementation leverages state-of-the-art NLP techniques and pre-trained models to achieve robust sentiment classification across different languages.

## ✨ Features

- **Multilingual Support**: Handles English (en), Spanish (es), and Chinese (zh) text
- **Sentiment Classification**: Classifies text into three categories:
  - Positive
  - Negative
  - Neutral
- **Data Preprocessing**: Comprehensive data cleaning and language detection
- **Visualization**: Includes distribution charts and word clouds for data exploration
- **Fine-tuning Pipeline**: Complete training pipeline with evaluation metrics
- **Pre-trained Models**: Utilizes transformer-based architectures for transfer learning

## 📊 Dataset

The project uses the [MultiLingualSentiment dataset](https://huggingface.co/datasets/clapAI/MultiLingualSentiment) from Hugging Face.

**Dataset Statistics:**
- Training set: ~22,500 samples
- Evaluation set: ~2,800 samples
- Test set: ~2,900 samples
- Languages: English, Spanish, Chinese
- Labels: positive (2), neutral (1), negative (0)

## 🤖 Model

**Base Model**: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

The project also experiments with BERT-based multilingual models (`bert-base-multilingual-cased`) for tokenization and classification.

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `datasets` - For loading and processing datasets
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch deep learning framework
- `accelerate` - For distributed training

### Additional Dependencies (for full notebook execution)

```bash
pip install langdetect scikit-learn seaborn pandas matplotlib wordcloud nltk spacy jieba
```

## 🚀 Usage

### Running the Notebooks

1. **Main Project Notebook** (`projekt.ipynb`):
   - Complete workflow from data loading to model training
   - Includes data exploration, cleaning, and visualization
   - Training and evaluation pipeline

2. **Google Colab Notebook** (`colab.ipynb`):
   - Optimized for Google Colab environment
   - Simplified version for cloud execution

### Basic Workflow

```python
# 1. Load the dataset
from datasets import load_dataset
dataset = load_dataset("clapAI/MultiLingualSentiment")

# 2. Preprocess and tokenize
# (See notebooks for detailed implementation)

# 3. Train the model
# (Training configuration in notebooks)

# 4. Evaluate and predict
# (Evaluation metrics: accuracy, F1-score)
```

## 📁 Project Structure

```
Multilingual-sentiment-analysis/
├── README.md                 # Project documentation
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── projekt.ipynb            # Main project notebook
├── colab.ipynb              # Google Colab version
├── datasets/                # Processed dataset files
│   ├── train_ds.csv        # Training data
│   ├── eval_ds.csv         # Evaluation data
│   └── test_ds.csv         # Test data
├── images/                  # Data distribution visualizations
│   ├── label_distribution_*.png
│   └── language_distribution_*.png
├── wordclouds/             # Word cloud visualizations
│   ├── wordcloud_en.png    # English word cloud
│   ├── wordcloud_es.png    # Spanish word cloud
│   └── wordcloud_zh.png    # Chinese word cloud
└── resources/              # Additional resources
    └── simhei.ttf         # Font for Chinese text rendering
```

## 📈 Visualizations

The project includes comprehensive data exploration visualizations:

### Label Distribution
- Distribution of sentiment labels across train/eval/test sets
- Visualizations available in `images/label_distribution_*.png`

### Language Distribution
- Distribution of languages in each dataset split
- Visualizations available in `images/language_distribution_*.png`

### Word Clouds
- Visual representation of most frequent words per language
- Available for English, Spanish, and Chinese
- Located in `wordclouds/` directory

## 🔧 Data Processing Pipeline

1. **Data Loading**: Load multilingual sentiment dataset from Hugging Face
2. **Data Reduction & Shuffling**: Balance and randomize dataset records
3. **Language Cleaning**: Remove incorrectly labeled language samples
4. **Language Selection**: Filter for English, Spanish, and Chinese texts
5. **Label Encoding**: Add numeric label IDs for model training
6. **Tokenization**: Process text using multilingual tokenizer
7. **Training**: Fine-tune transformer model with custom metrics
8. **Evaluation**: Assess performance using accuracy and F1-score

## 📊 Model Training

The training pipeline includes:
- Custom `compute_metrics` function for accuracy and F1-score calculation
- Validation after each epoch
- Support for distributed training via Accelerate
- Checkpoint saving and model export

## 🎯 Results

The model is trained to classify sentiment across three languages with metrics including:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Balanced measure of precision and recall

Detailed results and model performance can be found in the notebook outputs.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [clapAI/MultiLingualSentiment](https://huggingface.co/datasets/clapAI/MultiLingualSentiment)
- Base Model: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- Hugging Face Transformers library
- PyTorch framework

## 👤 Author

Mikołaj - [GitHub Profile](https://github.com/mikosovsky)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

**Note**: This project was created as part of a sentiment analysis study focusing on Transformer-based architectures for multilingual text classification.
