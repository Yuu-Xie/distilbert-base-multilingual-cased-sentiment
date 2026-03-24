import json
from pathlib import Path
import gradio as gr
from transformers import pipeline
import torch

# 1. 配置路径加载
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "train_config.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

MODEL_PATH = str(BASE_DIR / "models" / config["output_dir_name"] / "best-model")

# 2. 初始化 Pipeline
print(f"Loading model from: {MODEL_PATH}...")

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    task="sentiment-analysis",
    model=MODEL_PATH,
    device=device,
    top_k=None
)


def predict_sentiment(text):
    if not text.strip():
        return "Please enter some text.", {}

    results = classifier(text)[0]

    # 格式化输出给 Gradio Label 组件: {"label_name": score}
    # 标签为 ['positive', 'neutral', 'negative']
    output_scores = {item['label']: float(item['score']) for item in results}

    top_prediction = max(output_scores, key=output_scores.get)

    return top_prediction, output_scores


with gr.Blocks(title="Multilingual Sentiment Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 🌍 多语言情感分析演示系统
    本项目基于 **DistilBERT-Base-Multilingual-Cased** 微调，支持多种语言的情感分类。
    - **当前模型位置**: `{config['output_dir_name']}/best-model`
    - **运行设备**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
    """)

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="输入待分析文本",
                placeholder="在此输入任何语言的句子，例如：'这个产品真的很棒！'...",
                lines=5
            )
            submit_btn = gr.Button("开始分析", variant="primary")

            gr.Examples(
                examples=[
                    ["A good environment with good food. Price is reasonable."],
                    ["El producto llegó a tiempo, pero el color es diferente al de la foto."],
                    ["The customer service was absolutely terrible and I will never return."],
                    ["コードレス設計で車内の掃除もできます。"],
                    ["这东西一般般吧，没想象中那么好。"]
                ],
                inputs=input_text
            )

        with gr.Column(scale=1):
            main_label = gr.Label(label="主要情感倾向")
            conf_bar = gr.Label(label="置信度分布", num_top_classes=3)

    submit_btn.click(
        fn=predict_sentiment,
        inputs=input_text,
        outputs=[main_label, conf_bar]
    )
    input_text.submit(
        fn=predict_sentiment,
        inputs=input_text,
        outputs=[main_label, conf_bar]
    )

    gr.Markdown("""
    ### 💡 信息说明
    - **Positive**: 正面评价。
    - **Neutral**: 中性陈述或包含轻微不满但整体温和。
    - **Negative**: 负面批评或强烈不满。
    - *置信度代表模型对该预测的笃定程度。*
    """)

if __name__ == "__main__":
    # To create a public link, set share=True
    demo.launch(server_name="localhost", server_port=7860, share=False)
