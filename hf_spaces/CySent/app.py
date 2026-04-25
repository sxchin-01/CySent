"""
CySent Gradio Space App
Simple interface for cyber defense decision-making.
"""

import os
import gradio as gr
import torch
from typing import Dict, Any

# Try to load HF model/adapter
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


MODEL_ID = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_PATH = os.getenv("HF_ADAPTER_PATH", "sxchin01/CySent-adapter")
HF_TOKEN = os.getenv("HF_TOKEN", "")

model = None
tokenizer = None


def load_hf_model() -> bool:
    """Attempt to load HF base model, then optional adapter."""
    global model, tokenizer
    if not HF_AVAILABLE:
        return False
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN or None)
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
        if ADAPTER_PATH:
            model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, token=HF_TOKEN or None)
        else:
            model = base_model
        model.eval()
        return True
    except Exception as e:
        print(f"[CySent] HF load failed: {e}")
        return False


def get_action(state_text: str) -> Dict[str, Any]:
    """
    Infer defense action from state description.
    Returns action name and confidence.
    """
    valid_actions = [
        "patch_system",
        "isolate_host",
        "enable_mfa",
        "reset_credentials",
        "block_traffic",
        "alert_soc",
        "backup_data",
        "monitor_logs",
        "disable_service",
        "enable_firewall",
        "scan_malware",
        "investigate_incident",
    ]

    if not model or not tokenizer:
        # Fallback: simple heuristic
        if "compromised" in state_text.lower():
            return {"action": "isolate_host", "confidence": 0.6, "method": "heuristic"}
        if "threat" in state_text.lower():
            return {"action": "alert_soc", "confidence": 0.5, "method": "heuristic"}
        return {"action": "monitor_logs", "confidence": 0.4, "method": "heuristic"}

    # HF inference
    try:
        prompt = f"You are a cyber defense expert. Given the security state: {state_text}\nChoose ONE action from: {', '.join(valid_actions)}.\nAction:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.7, top_p=0.9)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract action name
        action = "monitor_logs"
        for valid_action in valid_actions:
            if valid_action in response.lower():
                action = valid_action
                break
        
        method = "hf_adapter" if ADAPTER_PATH else "hf_base"
        return {"action": action, "confidence": 0.85, "method": method}
    except Exception as e:
        print(f"[CySent] HF inference failed: {e}")
        return {"action": "monitor_logs", "confidence": 0.4, "method": "error_fallback"}


def predict_action(state_description: str) -> str:
    """Gradio interface: state text -> action + rationale."""
    result = get_action(state_description)
    action = result["action"]
    confidence = result["confidence"]
    method = result["method"]
    
    output = f"""
**Recommended Action:** {action.replace('_', ' ').title()}

**Confidence:** {confidence:.0%}  
**Method:** {method}

---
*CySent Defense System — Autonomous Cyber Defense*
"""
    return output


# Initialize model on startup
if load_hf_model():
    if ADAPTER_PATH:
        model_status = f"✓ HF Model loaded: {MODEL_ID} + {ADAPTER_PATH.split('/')[-1]}"
    else:
        model_status = f"✓ HF Base model loaded: {MODEL_ID}"
else:
    model_status = "⚠ HF Model unavailable. Using heuristic fallback."

# Gradio interface
demo = gr.Interface(
    fn=predict_action,
    inputs=gr.Textbox(
        label="Security State Description",
        placeholder="e.g., 'Risk=0.75, phishing attack on email, 2 hosts compromised, no mfa'",
        lines=4,
    ),
    outputs=gr.Markdown(label="Defense Action"),
    title="CySent — Autonomous Cyber Defense Command Center",
    description=f"{model_status}\n\nProvide a security state description. CySent recommends an immediate defense action.",
    examples=[
        ["Risk=0.8, ransomware detected on file server, spreading to backups"],
        ["Risk=0.5, phishing campaign targeting employees, 10 clicked"],
        ["Risk=0.3, unauthorized SSH access from overseas IP"],
        ["Risk=0.9, supply chain attack vector identified in vendor software"],
    ],
    theme=gr.themes.Soft(primary_hue="blue"),
)

if __name__ == "__main__":
    demo.launch()
