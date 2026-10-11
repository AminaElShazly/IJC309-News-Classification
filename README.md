# News Classification with DistilBERT

Fine-tuning `distilbert-base-uncased` to classify news articles from the [AG News](https://huggingface.co/datasets/ag_news) dataset into four categories: **World, Sports, Business, Sci/Tech**. Built in Google Colab (Tesla T4) with Hugging Face Transformers, with an interactive Gradio demo.

Originally developed for the IJC309 Building AI Applications module (University of Sheffield).

## Results

Evaluated on a held-out, class-balanced test set of 200 articles.

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| Baseline (DistilBERT, no fine-tuning) | 25.0% | 0.100 |
| **Fine-tuned DistilBERT** | **91.0%** | **0.910** |

- Best validation accuracy was 94.0% (epoch 5); the checkpoint from that epoch was used for the test evaluation above.
- Per-class test accuracy: Sports 98%, World 96%, Sci/Tech 86%, Business 84%.
- The most frequent errors are Sci/Tech ↔ Business confusions (11 of 18 errors), which is expected given the topical overlap between technology and business news.

## Method

- **Data:** stratified subsample of AG News (1,000 training articles split 800 train / 200 validation, plus 200 test), balanced across the four classes, to fit Colab limits.
- **Preprocessing:** `distilbert-base-uncased` tokenizer, max length 128 tokens (0.76% of articles are truncated).
- **Training:** 6 epochs, learning rate 3e-5, batch size 16, weight decay 0.01, 100 warmup steps, fp16, seed 42. Best checkpoint selected on validation accuracy.
- **Evaluation:** accuracy, precision, recall, F1, confusion matrices, misclassification analysis, comparison against the un-fine-tuned baseline.
- **Demo:** a Gradio app that returns class probabilities for any pasted headline or article text.

## Limitations

- The test set contains only 200 articles, so the 91% figure carries roughly ±4 points of uncertainty.
- Training used a small subsample (1,000 of 120,000 available training articles); using the full dataset would likely improve results.

## Running it

1. Open `IJC309_News_Classification.ipynb` in Google Colab and select a GPU runtime.
2. Run all cells. The first cell installs dependencies; see `requirements.txt` for the full list if running locally.
3. The final cell launches the Gradio demo.

The first Drive-mount cell is only needed in Colab and can be skipped if you are not saving to Google Drive.
