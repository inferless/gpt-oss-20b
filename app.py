import torch
import os
import inferless
from typing import Optional
from pydantic import BaseModel, Field
from transformers import pipeline

@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default="Explain quantum mechanics clearly and concisely.")
    system_prompt: Optional[str] ="You are a helpful and knowledgeable assistant."
    max_new_tokens: Optional[int] = 256
    temperature: Optional[float] =0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    do_sample: Optional[bool] = True
    repetition_penalty: Optional[float] = 1.1

@inferless.response
class ResponseObjects(BaseModel):
    generated_text: str = Field(default="Generated text will appear here")

class InferlessPythonModel:
    def initialize(self):
        model_id = "Inferless/gpt-oss-20b"
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

    def infer(self, inputs: RequestObjects) -> ResponseObjects:
        # Prepare messages
        messages = [{"role": "system", "content": inputs.system_prompt},{"role": "user", "content": inputs.prompt}]
        generation_kwargs = {
            "max_new_tokens": inputs.max_new_tokens,
            "temperature": inputs.temperature,
            "top_p": inputs.top_p,
            "top_k": inputs.top_k,
            "do_sample": inputs.do_sample,
            "repetition_penalty": inputs.repetition_penalty,
            "return_full_text": False,
            "pad_token_id": self.pipe.tokenizer.eos_token_id,
        }
        
        # Generate text using pipeline
        with torch.inference_mode():
            outputs = self.pipe(
                messages,
                **generation_kwargs
            )
        
        # Get the last message (assistant's response) from the conversation
        generated_text = outputs[0]["generated_text"]
        return ResponseObjects(generated_text=generated_text)

    def finalize(self):
        self.pipe = None
