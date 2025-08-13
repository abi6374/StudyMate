import os
from typing import List, Optional
from threading import Lock

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:  # lightweight fallback if transformers not installed yet
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    torch = None  # type: ignore

WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")

# Local / HF Granite model configuration
GRANITE_MODEL_ID = os.getenv("GRANITE_MODEL_ID", "ibm-granite/granite-3.3-2b-instruct")
GEN_MAX_NEW_TOKENS = int(os.getenv("GEN_MAX_NEW_TOKENS", "512"))

_model = None
_tokenizer = None
_load_lock = Lock()

def _load_local_model(model_name: Optional[str] = None):
    """Lazy-load the Granite instruct model (CPU/GPU auto) only once."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    if AutoTokenizer is None:
        raise RuntimeError("transformers not installed. Please pip install transformers torch.")
    with _load_lock:
        if _model is None:
            name = model_name or GRANITE_MODEL_ID
            device_map = "auto" if torch and torch.cuda.is_available() else None
            torch_dtype = torch.float16 if torch and torch.cuda.is_available() else None
            _tokenizer = AutoTokenizer.from_pretrained(name)
            _model = AutoModelForCausalLM.from_pretrained(name, device_map=device_map, torch_dtype=torch_dtype)
    return _model, _tokenizer


def generate_answer(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Generate an answer using the local Granite instruct model.

    The incoming 'prompt' already includes context + question. We wrap it as a single
    user message for the model's chat template.
    """
    try:
        model, tokenizer = _load_local_model()
    except Exception as e:  # pragma: no cover - fallback path
        return f"[GENERATION ERROR] {e}\nPrompt preview: {prompt[:200]}..."

    messages = [
        {"role": "system", "content": "You are StudyMate, an academic assistant. Be concise, cite sources."},
        {"role": "user", "content": prompt},
    ]

    # Use tokenizer chat template to format input ids
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "to") and torch:
        inputs = inputs.to(model.device)

    gen_kwargs = {
        "max_new_tokens": min(max_tokens, GEN_MAX_NEW_TOKENS),
        "temperature": temperature,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():  # type: ignore
        output = model.generate(**inputs, **gen_kwargs)
    # Slice off the prompt tokens
    gen_tokens = output[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return text


def build_prompt(question: str, contexts: List[str]) -> str:
    header = (
        "Answer the academic question using ONLY the provided context. Cite sources as (filename p.#). "
        "If insufficient information, say so."
    )
    joined_ctx = "\n---\n".join(contexts[:10])
    return f"{header}\n\nContext:\n{joined_ctx}\n\nQuestion: {question}\nAnswer:"