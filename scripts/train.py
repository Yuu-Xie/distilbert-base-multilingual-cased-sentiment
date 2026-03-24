import json
from pathlib import Path
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

# 1. 加载配置
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "train_config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

OUTPUT_DIR = str(BASE_DIR / "models" / config["output_dir_name"])


def main():
    print("🚀 [1/6] Loading dataset...")
    dataset = load_dataset("clapAI/MultiLingualSentiment")

    # 获取标签字典
    labels = dataset["train"].unique("label")
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    num_labels = len(labels)
    print(f"Labels detected: {labels}")

    print("🚀 [2/6] Loading Tokenizer and processing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])

    def preprocess_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, max_length=config["max_length"])
        tokenized["label"] = [label2id[l] for l in examples["label"]]
        return tokenized

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("🚀 [3/6] Setting up Evaluation Metrics...")
    metric_f1 = evaluate.load("f1")
    metric_acc = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
        acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
        return {"accuracy": acc, "f1": f1}

    print("🚀 [4/6] Initializing Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_checkpoint"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    print("🚀 [5/6] Configuring Trainer...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=True,  # RTX 5090
        logging_steps=config["logging_steps"],
        seed=config["seed"],
        report_to="tensorboard"
    )

    # 引入 Early Stopping 防止过拟合
    early_stopping = EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )

    print("🚀 [6/6] Starting full-scale training on RTX 5090...")
    trainer.train()

    # 保存最佳模型
    best_model_path = Path(OUTPUT_DIR) / "best-model"
    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))
    print(f"🎉 Training complete! Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
