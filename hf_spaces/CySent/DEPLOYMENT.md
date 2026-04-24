# HF Space Deployment Guide

## Quick Start

### 1. Create New Space on Hugging Face Hub

1. Visit https://huggingface.co/spaces
2. Click "Create new Space"
3. **Name:** `CySent` (or your username variant: `CySent-{username}`)
4. **License:** MIT (or your choice)
5. **SDK:** `Gradio`
6. **Visibility:** Public (recommended for credibility) or Private (for development)

### 2. Clone Space Locally

```bash
cd /tmp
git clone https://huggingface.co/spaces/{username}/CySent
cd CySent
```

### 3. Copy Files from This Folder

```bash
# From your local CySent repo:
cp hf_spaces/CySent/app.py /tmp/CySent/
cp hf_spaces/CySent/requirements.txt /tmp/CySent/
cp hf_spaces/CySent/README.md /tmp/CySent/
```

### 4. Add Model Adapter (If Using HF Model)

**Option A: Separate HF Model Card (Recommended)**

1. Upload your LoRA adapter to a separate HF model repo: `https://huggingface.co/{username}/CySent-adapter`
2. Set Space secrets:
   - `HF_MODEL_ID` = `Qwen/Qwen2.5-3B-Instruct`
   - `HF_ADAPTER_PATH` = `{username}/CySent-adapter`
   - `HF_TOKEN` = Your HF API token (if adapter is private)

**Option B: Commit Adapter to Space (Not Recommended for Large Models)**

If adapter is < 100MB:
```bash
cp -r outputs/cysent_unsloth_adapter /tmp/CySent/adapter
cd /tmp/CySent
git add adapter/
git commit -m "feat: add CySent LoRA adapter"
git push
```

Then set Space secrets:
- `HF_ADAPTER_PATH` = `./adapter`

### 5. Set Space Secrets (Settings → Repo Secrets)

```
HF_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
HF_ADAPTER_PATH={username}/CySent-adapter  # or ./adapter if local
HF_TOKEN=hf_xxxxxxxxxxxx  # Your HF API token
```

### 6. Test Locally First

```bash
cd /tmp/CySent
pip install -r requirements.txt
HF_MODEL_ID=Qwen/Qwen2.5-3B-Instruct \
  HF_ADAPTER_PATH={username}/CySent-adapter \
  HF_TOKEN=hf_xxxx \
  python app.py
```

Visit `http://127.0.0.1:7860`

### 7. Commit & Push to Space

```bash
cd /tmp/CySent
git add .
git commit -m "feat: initial Gradio Space for CySent autonomous defense"
git push
```

Space will auto-build and deploy. Check **App** tab in ~1-2 minutes.

---

## Uploading Adapter Model to HF Hub

### Create Model Card

1. Visit https://huggingface.co/new
2. **Name:** `CySent-adapter`
3. **Type:** Model
4. **Task:** Text Generation
5. Create repo

### Upload LoRA Weights

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import upload_folder
import os

token = os.getenv('HF_TOKEN')  # Set your HF token
adapter_path = 'outputs/cysent_unsloth_adapter'
repo_id = '{username}/CySent-adapter'

# Upload adapter folder
upload_folder(
    folder_path=adapter_path,
    repo_id=repo_id,
    repo_type='model',
    token=token,
    commit_message='Initial CySent LoRA adapter (Qwen2.5-3B-Instruct)',
)
print(f'Uploaded to https://huggingface.co/{repo_id}')
"
```

### Verify Upload

Visit `https://huggingface.co/{username}/CySent-adapter`  
Confirm `adapter_config.json`, `adapter_model.safetensors` are present.

---

## Files Included

| File | Purpose |
|------|---------|
| `app.py` | Gradio interface + inference logic |
| `requirements.txt` | Minimal deps (gradio, transformers, peft, torch) |
| `README.md` | Project overview, features, usage |
| `DEPLOYMENT.md` | This file — step-by-step deployment |

## Size Budget

- **Space Compute:** Gradio = ~2GB RAM (default CPU)
- **Requirements:** ~4GB download (torch + transformers)
- **Model (runtime):** Qwen2.5-3B-Instruct ≈ 7-15GB (4-bit quantized or float16)

**Recommendation:** Use 4-bit quantization on Qwen base for < 10GB Space storage.

## Troubleshooting

### App Won't Start
- Check **Logs** tab in Space settings.
- Ensure `HF_ADAPTER_PATH` is correct and accessible.
- Try with empty `HF_ADAPTER_PATH` to test fallback heuristic mode.

### Model Loads Slowly
- Qwen2.5-3B is medium-sized; first inference may take 30-60s.
- Use 4-bit quantization to reduce memory:
  ```python
  base_model = AutoModelForCausalLM.from_pretrained(
      MODEL_ID, 
      load_in_4bit=True,
      device_map="auto"
  )
  ```

### Adapter Path Not Found
- If private: ensure `HF_TOKEN` is set in Space secrets.
- If local (`./adapter`): ensure files committed to Space repo.
- If hub repo: verify repo name matches exactly.

---

## Next Steps

1. **Expand Examples:** Add more use-case examples to `app.py` examples list.
2. **Benchmark Table:** Populate `README.md` results after running eval.
3. **Screenshot Gallery:** Add incident scenario screenshots to Space README.
4. **Gradio Theming:** Customize colors/styling in `demo = gr.Interface(theme=...)`.

---

**Space URL:** `https://huggingface.co/spaces/{username}/CySent`  
**Adapter URL:** `https://huggingface.co/{username}/CySent-adapter`
