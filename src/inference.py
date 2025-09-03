import gradio as gr
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

def main():
    ckpt_path = "models/xlmr-large-viquad-final"
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(ckpt_path)
    qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)

    def answer_fn(context, question):
        out = qa_pipe({"context": context, "question": question})
        return f"{out['answer']} (score={out['score']:.3f})"

    gr.Interface(
        fn=answer_fn,
        inputs=[
            gr.Textbox(lines=8, label="Đoạn văn (tiếng Việt)"),
            gr.Textbox(lines=2, label="Câu hỏi")
        ],
        outputs=gr.Textbox(label="Trả lời"),
        title="Vietnamese QA – xlm-roberta-large (ViQuAD)"
    ).launch(share=False)

if __name__ == "__main__":
    main()