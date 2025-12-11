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

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model (CPU only, no accelerate, no device_map)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)
model.to("cpu")  # force CPU mode for CI compatibility


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

    # Tokenize input text
    inputs = tokenizer(req.prompt, return_tensors="pt").to("cpu")

    # Generate continuations
    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=True,
        temperature=0.7,
    )

    # Decode output tokens
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"prompt": req.prompt, "response": text}
