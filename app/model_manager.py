import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import torch

MODEL_CONFIG = {
    "llm_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
}

class ModelManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.pipeline = None

    def initialize_model(self):
        if self.pipeline is None:
            with st.spinner("Loading DeepSeek model... This may take a few minutes..."):
                tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["llm_model"])
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_CONFIG["llm_model"],
                    device_map="cuda",
                    torch_dtype=torch.float16
                )
                pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,max_length=2048,
                temperature=0.01,
                top_p=0.95,
                repetition_penalty=1.15)
                self.pipeline = HuggingFacePipeline(pipeline=pipe)
                st.success("✨ Model loaded successfully!")
                st.balloons()
                st.snow()
        return self.pipeline
