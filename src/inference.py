import gradio as gr
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

def run_gradio(config: dict):
    """
    Launches Gradio web interface for interactive QA functionality.
    """
    # Load tokenizer and QA model pipeline
    ckpt_path = config["model"]["ckpt_path"]
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(ckpt_path)
    qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)

    def answer_fn(context, question):
        """Logic handler: processes context & question to output the text answer."""
        out = qa_pipe({"context": context, "question": question})
        return f"{out['answer']} (score={out['score']:.3f})"

    # Enable public Gradio link sharing if indicated in configuration
    share_gradio = config.get("inference", {}).get("share_gradio", False)

    # UI setup and component mapping
    app = gr.Interface(
        fn=answer_fn,
        inputs=[
            gr.Textbox(lines=8, label="Đoạn văn (tiếng Việt)"),
            gr.Textbox(lines=2, label="Câu hỏi")
        ],
        outputs=gr.Textbox(label="Trả lời (Kết quả và độ tin cậy)"),
        title=f"Vietnamese QA – {config['model']['name']} (ViQuAD)"
    )
    
    app.launch(share=share_gradio)