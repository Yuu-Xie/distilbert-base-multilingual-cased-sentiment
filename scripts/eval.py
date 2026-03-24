import json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

# 加载配置和模型路径
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "train_config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_PATH = str(BASE_DIR / "models" / config["output_dir_name"] / "best-model")


def main():
    print(f"Loading best model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    print("Loading test dataset...")
    dataset = load_dataset("clapAI/MultiLingualSentiment", split="test")

    label2id = model.config.label2id

    def preprocess_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, max_length=config["max_length"])
        tokenized["label"] = [label2id[l] for l in examples["label"]]
        return tokenized

    print("Processing test dataset...")
    test_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 使用 Trainer 进行高速批量推理
    training_args = TrainingArguments(
        output_dir="./tmp_dir",
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        bf16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        processing_class=tokenizer
    )

    print("Running evaluation on the entire test set...")
    predictions_output = trainer.predict(test_dataset)

    preds = np.argmax(predictions_output.predictions, axis=1)
    labels_true = predictions_output.label_ids

    # 打印详细的分类报告 (精准度、召回率、F1 Score)
    target_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    print("\n" + "=" * 50)
    print("📊 FINAL EVALUATION REPORT (Test Set)")
    print("=" * 50)
    print(classification_report(labels_true, preds, target_names=target_names, digits=4))
    print(f"Overall Accuracy: {accuracy_score(labels_true, preds):.4f}")
    print("=" * 50)

    # 绘制混淆矩阵
    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    cm = confusion_matrix(labels_true, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Confusion matrix saved to: results/confusion_matrix.png")


if __name__ == "__main__":
    main()
