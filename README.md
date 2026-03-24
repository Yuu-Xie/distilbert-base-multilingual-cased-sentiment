# Multilingual Sentiment Analysis Based on DistilBERT

本项目基于 `clapAI/MultiLingualSentiment` 多语言情感分类数据集，使用蒸馏模型 `distilbert-base-multilingual-cased` 进行全量微调，实现三分类情感分析（Positive, Negative, Neutral）。项目目标是探索多语言场景下轻量级模型的情感分类能力，并为后续低资源语言的情感分析提供基线。

## Dataset

本项目使用发布于 Hugging Face 的 [clapAI/MultiLingualSentiment](https://huggingface.co/datasets/clapAI/MultiLingualSentiment) 数据集，该数据集包含约 315 万条多语言情感标注文本，涵盖英文、中文、日文等多种语言，由 clapAI 团队于 2024 年发布。

## Citation

```bibtex
@article{Sanh2019DistilBERTAD,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Victor Sanh and Lysandre Debut and Julien Chaumond and Thomas Wolf},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.01108}
}
@dataset{clapAI2024multilingualsentiment,
  title        = {MultilingualSentiment: A Multilingual Sentiment Classification Dataset},
  author       = {clapAI},
  year         = {2024},
  url          = {https://huggingface.co/datasets/clapAI/MultiLingualSentiment},
  description  = {A multilingual dataset for sentiment analysis with labels: positive, neutral, negative, covering diverse languages and domains.},
}
```
