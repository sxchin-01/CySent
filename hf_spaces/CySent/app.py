"""
CySent — Autonomous Cyber Defense Command Center
Premium dark UI with Gradio Blocks + custom CSS.
"""

import os
import gradio as gr
import torch
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Model loading (UNCHANGED)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Inference (UNCHANGED)
# ---------------------------------------------------------------------------
VALID_ACTIONS = [
    "patch_system", "isolate_host", "enable_mfa", "reset_credentials",
    "block_traffic", "alert_soc", "backup_data", "monitor_logs",
    "disable_service", "enable_firewall", "scan_malware", "investigate_incident",
]

ACTION_ICONS = {
    "patch_system": "\u2699\ufe0f", "isolate_host": "\U0001f6e1\ufe0f",
    "enable_mfa": "\U0001f510", "reset_credentials": "\U0001f511",
    "block_traffic": "\U0001f6ab", "alert_soc": "\U0001f6a8",
    "backup_data": "\U0001f4be", "monitor_logs": "\U0001f50d",
    "disable_service": "\u26d4", "enable_firewall": "\U0001f525",
    "scan_malware": "\U0001f41b", "investigate_incident": "\U0001f575\ufe0f",
}


def get_action(state_text: str) -> Dict[str, Any]:
    if not model or not tokenizer:
        if "compromised" in state_text.lower():
            return {"action": "isolate_host", "confidence": 0.6, "method": "heuristic"}
        if "threat" in state_text.lower():
            return {"action": "alert_soc", "confidence": 0.5, "method": "heuristic"}
        return {"action": "monitor_logs", "confidence": 0.4, "method": "heuristic"}
    try:
        prompt = (
            f"You are a cyber defense expert. Given the security state: {state_text}\n"
            f"Choose ONE action from: {', '.join(VALID_ACTIONS)}.\nAction:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.7, top_p=0.9)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        action = "monitor_logs"
        for valid_action in VALID_ACTIONS:
            if valid_action in response.lower():
                action = valid_action
                break
        method = "hf_adapter" if ADAPTER_PATH else "hf_base"
        return {"action": action, "confidence": 0.85, "method": method}
    except Exception as e:
        print(f"[CySent] HF inference failed: {e}")
        return {"action": "monitor_logs", "confidence": 0.4, "method": "error_fallback"}


# ---------------------------------------------------------------------------
# UI output formatter (redesigned presentation only)
# ---------------------------------------------------------------------------
def _confidence_color(c: float) -> str:
    if c >= 0.8:
        return "#00ffc8"
    if c >= 0.6:
        return "#a78bfa"
    return "#f472b6"


def predict_action(state_description: str) -> str:
    if not state_description or not state_description.strip():
        return ""
    result = get_action(state_description)
    action = result["action"]
    confidence = result["confidence"]
    method = result["method"]
    icon = ACTION_ICONS.get(action, "\u26a1")
    color = _confidence_color(confidence)
    pct = f"{confidence:.0%}"
    label = action.replace("_", " ").title()
    method_badge = (
        '<span style="background:rgba(0,255,200,0.12);color:#00ffc8;'
        'padding:2px 10px;border-radius:20px;font-size:0.8em;'
        f'border:1px solid rgba(0,255,200,0.25)">{method}</span>'
    )
    return f"""
<div style="text-align:center;padding:12px 0 4px">
  <div style="font-size:2.8em;margin-bottom:4px">{icon}</div>
  <div style="font-size:1.5em;font-weight:700;color:#e2e8f0;letter-spacing:0.02em">{label}</div>
  <div style="margin:12px 0">
    <span style="font-size:2em;font-weight:800;color:{color}">{pct}</span>
    <span style="color:#94a3b8;font-size:0.85em;margin-left:6px">confidence</span>
  </div>
  <div>{method_badge}</div>
</div>
"""


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
_model_ok = load_hf_model()
if _model_ok:
    _adapter_short = ADAPTER_PATH.split("/")[-1] if ADAPTER_PATH else ""
    _status_html = (
        f'<span class="status-dot live"></span>'
        f'<span style="color:#00ffc8;font-weight:600">Model Online</span>'
        f'<span style="color:#64748b;margin-left:8px;font-size:0.85em">'
        f'{MODEL_ID.split("/")[-1]} + {_adapter_short}</span>'
    )
else:
    _status_html = (
        '<span class="status-dot off"></span>'
        '<span style="color:#f472b6;font-weight:600">Heuristic Fallback</span>'
    )


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CSS = """
/* ---- base ---- */
.gradio-container {
    background: linear-gradient(165deg, #0a0e1a 0%, #0f1629 40%, #0c1220 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    min-height: 100vh;
}
.main { max-width: 960px !important; margin: 0 auto !important; }

/* ---- hero ---- */
.hero { text-align: center; padding: 32px 16px 8px; }
.hero h1 {
    font-size: 2.2em; font-weight: 800;
    background: linear-gradient(135deg, #00ffc8 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0;
}
.hero .subtitle {
    color: #64748b; font-size: 0.95em; margin-top: 6px; letter-spacing: 0.03em;
}

/* ---- status badge ---- */
.status-bar {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    padding: 8px 20px; margin: 12px auto 20px; width: fit-content;
    background: rgba(15,22,41,0.7); border: 1px solid rgba(100,116,139,0.2);
    border-radius: 999px; backdrop-filter: blur(8px);
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 2px;
}
.status-dot.live {
    background: #00ffc8;
    box-shadow: 0 0 6px #00ffc8, 0 0 12px rgba(0,255,200,0.3);
    animation: pulse 2s ease-in-out infinite;
}
.status-dot.off { background: #f472b6; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* ---- glass cards ---- */
.glass-card {
    background: rgba(15,22,41,0.6) !important;
    border: 1px solid rgba(100,116,139,0.15) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    padding: 24px !important;
}

/* ---- input textarea ---- */
.glass-card textarea {
    background: rgba(10,14,26,0.8) !important;
    border: 1px solid rgba(100,116,139,0.2) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.95em !important;
    transition: border-color 0.2s !important;
}
.glass-card textarea:focus {
    border-color: rgba(0,255,200,0.4) !important;
    box-shadow: 0 0 0 2px rgba(0,255,200,0.08) !important;
}
.glass-card label span {
    color: #94a3b8 !important; font-weight: 600 !important; font-size: 0.85em !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
}

/* ---- result card ---- */
.result-card {
    background: rgba(15,22,41,0.6) !important;
    border: 1px solid rgba(100,116,139,0.15) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    min-height: 200px !important;
    display: flex; align-items: center; justify-content: center;
}

/* ---- buttons ---- */
.cysent-submit {
    background: linear-gradient(135deg, #00ffc8 0%, #00c9a0 100%) !important;
    color: #0a0e1a !important; font-weight: 700 !important;
    border: none !important; border-radius: 12px !important;
    padding: 10px 28px !important; font-size: 0.95em !important;
    letter-spacing: 0.03em !important; cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 12px rgba(0,255,200,0.2) !important;
}
.cysent-submit:hover {
    box-shadow: 0 4px 20px rgba(0,255,200,0.35) !important;
    transform: translateY(-1px) !important;
}
.cysent-clear {
    background: rgba(100,116,139,0.12) !important;
    color: #94a3b8 !important; font-weight: 600 !important;
    border: 1px solid rgba(100,116,139,0.2) !important;
    border-radius: 12px !important; padding: 10px 28px !important;
    font-size: 0.95em !important; cursor: pointer !important;
    transition: all 0.2s !important;
}
.cysent-clear:hover {
    background: rgba(100,116,139,0.2) !important;
    color: #e2e8f0 !important;
}

/* ---- example chips ---- */
.example-chip {
    background: rgba(15,22,41,0.5);
    border: 1px solid rgba(100,116,139,0.15);
    border-radius: 12px; padding: 10px 16px;
    color: #94a3b8; font-size: 0.85em; cursor: pointer;
    transition: all 0.2s; line-height: 1.4;
}
.example-chip:hover {
    border-color: rgba(0,255,200,0.3); color: #e2e8f0;
    background: rgba(0,255,200,0.04);
}

/* ---- section labels ---- */
.section-label {
    color: #475569; font-size: 0.75em; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 10px;
}

/* ---- footer ---- */
.footer {
    text-align: center; padding: 20px 0 8px; margin-top: 24px;
    border-top: 1px solid rgba(100,116,139,0.1);
}
.footer a {
    color: #475569; text-decoration: none; font-size: 0.8em;
    margin: 0 12px; transition: color 0.2s;
}
.footer a:hover { color: #00ffc8; }

/* ---- Gradio overrides ---- */
footer { display: none !important; }
.gradio-container .prose { color: #94a3b8 !important; }
.gradio-container .prose h1, .gradio-container .prose h2 { color: #e2e8f0 !important; }
#component-0 { background: transparent !important; }
.block { background: transparent !important; border: none !important; box-shadow: none !important; }
"""

EXAMPLES = [
    "Risk=0.8, ransomware detected on file server, spreading to backups",
    "Risk=0.5, phishing campaign targeting employees, 10 clicked",
    "Risk=0.3, unauthorized SSH access from overseas IP",
    "Risk=0.9, supply chain attack vector identified in vendor software",
]


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------
with gr.Blocks(css=CSS, title="CySent — Cyber Defense AI") as demo:

    # Hero
    gr.HTML("""
    <div class="hero">
        <h1>CySent</h1>
        <div class="subtitle">Autonomous Cyber Defense Command Center</div>
    </div>
    """)

    # Status badge
    gr.HTML(f'<div class="status-bar">{_status_html}</div>')

    # Main two-column layout
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">Threat Intelligence Input</div>')
            with gr.Group(elem_classes="glass-card"):
                state_input = gr.Textbox(
                    label="Security State",
                    placeholder="Describe the current threat scenario...\ne.g. Risk=0.75, phishing attack on email, 2 hosts compromised",
                    lines=5,
                    max_lines=8,
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear", elem_classes="cysent-clear", scale=1)
                    submit_btn = gr.Button("Analyze Threat", elem_classes="cysent-submit", scale=2)

        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">Defense Recommendation</div>')
            result_output = gr.HTML(
                value='<div style="text-align:center;padding:40px 0;color:#475569">'
                      'Awaiting threat data...</div>',
                elem_classes="result-card",
            )

    # Examples
    gr.HTML('<div class="section-label" style="margin-top:20px">Example Scenarios</div>')
    with gr.Row():
        example_btns = []
        for ex in EXAMPLES:
            btn = gr.Button(ex, elem_classes="example-chip", size="sm")
            example_btns.append((btn, ex))

    # Footer
    gr.HTML("""
    <div class="footer">
        <a href="https://github.com/sxchin-01/CySent" target="_blank">GitHub</a>
        <a href="https://huggingface.co/spaces/sxchin01/CySent/blob/main/BLOG.md" target="_blank">Blog</a>
        <a href="https://huggingface.co/sxchin01/CySent-adapter" target="_blank">Model</a>
        <a href="https://huggingface.co/spaces/sxchin01/CySent" target="_blank">Space</a>
        <a href="https://github.com/sxchin-01/CySent/blob/main/notebooks/CySent_Unsloth_Train.ipynb" target="_blank">Colab</a>
    </div>
    """)

    # Wiring
    submit_btn.click(fn=predict_action, inputs=state_input, outputs=result_output)
    state_input.submit(fn=predict_action, inputs=state_input, outputs=result_output)
    clear_btn.click(
        fn=lambda: (
            "",
            '<div style="text-align:center;padding:40px 0;color:#475569">Awaiting threat data...</div>',
        ),
        outputs=[state_input, result_output],
    )
    for btn, ex_text in example_btns:
        btn.click(fn=lambda t=ex_text: t, outputs=state_input)

if __name__ == "__main__":
    demo.launch()
