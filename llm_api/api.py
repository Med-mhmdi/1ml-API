from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

router = APIRouter(
    prefix="/llm",
    tags=["LLM Generation API"],
)

# =========================
# Model configuration
# =========================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_NAME = "microsoft/phi-2"
# MODEL_NAME = "distilgpt2"

DEVICE = "cpu"

# =========================
# Load tokenizer & model
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model.to(DEVICE)
model.eval()

# =========================
# Schemas
# =========================

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(50, ge=1, le=300)
    temperature: float = Field(0.7, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)


class PromptResponse(BaseModel):
    prompt: str
    response: str
    model: str
    tokens_generated: int
    inference_time_sec: float

# =========================
# Internal function (FOR AGENT)
# =========================

def generate_text_internal(
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Internal LLM call for AI Agent (no HTTP).
    """

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# API endpoints
# =========================

@router.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE
    }


@router.post("/generate", response_model=PromptResponse)
def generate_text(req: PromptRequest):

    start_time = time.time()

    text = generate_text_internal(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    elapsed = round(time.time() - start_time, 3)

    tokens_generated = (
        len(tokenizer.encode(text))
        - len(tokenizer.encode(req.prompt))
    )

    return {
        "prompt": req.prompt,
        "response": text,
        "model": MODEL_NAME,
        "tokens_generated": tokens_generated,
        "inference_time_sec": elapsed
    }
