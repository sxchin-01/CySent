import os
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    token = _required_env("HF_TOKEN")
    base_id = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct").strip()
    adapter_id = os.getenv("HF_ADAPTER_PATH", "sxchin01/CySent-Qwen-RL").strip()
    repo_id = os.getenv("HF_MERGED_MODEL_ID", "sxchin01/CySent-Qwen-RL-merged").strip()
    output_dir = Path(os.getenv("HF_MERGED_OUTPUT_DIR", "merged_model").strip())

    print(f"[merge] base={base_id}")
    print(f"[merge] adapter={adapter_id}")
    print(f"[merge] target_repo={repo_id}")
    print(f"[merge] output_dir={output_dir}")

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token,
        trust_remote_code=True,
    )

    peft_model = PeftModel.from_pretrained(base, adapter_id, token=token)
    merged = peft_model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")

    tokenizer = AutoTokenizer.from_pretrained(base_id, token=token, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # Keep generation defaults aligned with the merged model artifacts.
    if hasattr(merged, "generation_config") and merged.generation_config is not None:
        merged.generation_config.save_pretrained(output_dir)

    files = sorted(p.name for p in output_dir.iterdir() if p.is_file())
    print("[merge] local_files=", files)

    must_have_any = ["model.safetensors", "model-00001-of-"]
    has_model_weights = any(name in f for f in files for name in must_have_any)
    if not has_model_weights:
        raise RuntimeError("Merged weights are missing in output directory.")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload merged Qwen2.5-3B + CySent RL adapter",
    )

    remote_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    print(f"[merge] remote_file_count={len(remote_files)}")
    print(f"[merge] remote_head={remote_files[:20]}")
    print(f"UPLOAD COMPLETE: {repo_id}")


if __name__ == "__main__":
    main()