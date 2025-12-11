from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

router = APIRouter(
    prefix="/llm",
    tags=["LLM Generation API"],
)

# Load model once globally
MODEL_NAME = "tiiuae/falcon-rw-1b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"   # CPU-friendly
)

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50


@router.get("/health")
def health():
    return {"status": "llm ok"}


@router.post("/generate")
def generate_text(req: PromptRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    inputs = tokenizer(req.prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=True,
        temperature=0.7,
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"prompt": req.prompt, "response": text}
