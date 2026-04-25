from __future__ import annotations

import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None

try:
    import torch
except ImportError:  # pragma: no cover - optional when using remote-only HF
    torch = None

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional when using merged/local adapter paths
    PeftModel = None

try:
    from peft import AutoPeftModelForCausalLM
except ImportError:  # pragma: no cover - optional in older PEFT versions
    AutoPeftModelForCausalLM = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - optional when using remote-only HF
    AutoModelForCausalLM = None
    AutoTokenizer = None

from backend.env.security_env import ACTION_NAMES


VALID_ACTIONS = {action_name: action_id for action_id, action_name in ACTION_NAMES.items()}
ACTION_LIST = ", ".join(VALID_ACTIONS.keys())

ALIASES = {
    "isolate_host": "isolate_suspicious_host",
}


def _normalize_text(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def _is_merged_model_dir(path: str) -> bool:
    candidate = Path(path)
    return candidate.exists() and candidate.is_dir() and (candidate / "config.json").exists() and not (candidate / "adapter_config.json").exists()


def _looks_like_local_path(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False

    # Windows absolute paths (C:\...) or UNC paths.
    if ":" in text or text.startswith("\\\\"):
        return True

    # Explicit relative/absolute filesystem paths.
    if text.startswith((".", "..", "/", "~")):
        return True

    # Backslashes are filesystem separators; single forward slash may be a Hub repo id (owner/repo).
    return "\\" in text


def _read_adapter_base_model_name(adapter_path: str) -> Optional[str]:
    """Read adapter_config.json and return base_model_name_or_path when available."""
    try:
        cfg_path = Path(adapter_path) / "adapter_config.json"
        if not cfg_path.exists():
            return None
        loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
        base = str(loaded.get("base_model_name_or_path", "")).strip()
        return base or None
    except Exception:
        return None


class HFAgent:
    """HuggingFace LLM agent for action decision making."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        adapter_path: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
    ) -> None:
        self.model_id = model_id or os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
        self.adapter_path = adapter_path or os.getenv("HF_ADAPTER_PATH", "").strip()
        self.endpoint_url = (endpoint_url or os.getenv("HF_ENDPOINT_URL") or "").strip() or None
        self.token = (token or os.getenv("HF_TOKEN") or "").strip() or None
        if timeout is None:
            timeout = float(os.getenv("HF_TIMEOUT", "10.0"))
        self.timeout = float(timeout)
        self.max_retries = max_retries
        self.client: Optional[InferenceClient] = None
        self.model = None
        self.tokenizer = None
        self._using_local_model = False
        self._active_backend = "none"
        self._local_load_attempted = False
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize with precedence: hosted (if configured) -> local adapter -> unavailable."""
        hosted_configured = bool(str(self.endpoint_url or "").strip())
        hosted_model_configured = bool(str(self.token or "").strip()) and not bool(self.adapter_path)

        if hosted_configured or hosted_model_configured:
            if InferenceClient is None:
                raise ImportError("huggingface_hub not installed. Cannot use hosted HF agent.")
            target = self.endpoint_url if hosted_configured else self.model_id
            self.client = InferenceClient(model=target, token=self.token)
            self._using_local_model = False
            self._active_backend = "cloud"
            print(f"[HFAgent] Ready (cloud). target={target}")
            return

        if self.adapter_path and self.token and os.name == "nt" and InferenceClient is not None:
            # Local Qwen adapter loads can exceed Windows memory/paging limits; prefer hosted inference.
            self.client = InferenceClient(model=self.model_id, token=self.token)
            self._using_local_model = False
            self._active_backend = "cloud"
            print("[HFAgent] Using cloud inference on Windows to avoid local adapter memory pressure.")
            return

        if self.adapter_path:
            if AutoModelForCausalLM is None or AutoTokenizer is None:
                raise ImportError("transformers is required for local HF adapter usage.")
            # Defer local model load to first HF prediction to avoid startup OOM when PPO-only paths are used.
            self._active_backend = "local_deferred"
            print(f"[HFAgent] Deferred local adapter load. adapter={self.adapter_path}")
            return

        raise RuntimeError("HF is not configured: set HF_ENDPOINT_URL or HF_ADAPTER_PATH.")

    def _model_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "token": self.token,
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if torch is not None:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        return kwargs

    def _initialize_local_model(self) -> None:
        model_kwargs = self._model_kwargs()
        try:
            if self.adapter_path:
                if _looks_like_local_path(self.adapter_path) and not Path(self.adapter_path).exists():
                    raise FileNotFoundError(f"HF_ADAPTER_PATH does not exist: {self.adapter_path}")
                if _is_merged_model_dir(self.adapter_path):
                    self.model = AutoModelForCausalLM.from_pretrained(self.adapter_path, **model_kwargs)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path, token=self.token, trust_remote_code=True)
                else:
                    # Preferred path for Qwen LoRA: explicit base model + PeftModel.
                    standard_err: Optional[Exception] = None
                    if PeftModel is not None:
                        try:
                            print(f"[HFAgent] Loading LoRA with configured base model: {self.model_id}")
                            base_model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
                            try:
                                self.model = PeftModel.from_pretrained(base_model, self.adapter_path, token=self.token)
                            except TypeError:
                                self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
                        except Exception as exc:
                            standard_err = exc

                    # Fallback path: auto PEFT resolution from adapter metadata.
                    if self.model is None:
                        if AutoPeftModelForCausalLM is not None:
                            try:
                                adapter_base = _read_adapter_base_model_name(self.adapter_path)
                                if adapter_base:
                                    print(f"[HFAgent] Falling back to adapter-declared base model: {adapter_base}")
                                try:
                                    self.model = AutoPeftModelForCausalLM.from_pretrained(self.adapter_path, **model_kwargs)
                                except TypeError:
                                    self.model = AutoPeftModelForCausalLM.from_pretrained(self.adapter_path)
                            except Exception:
                                if standard_err is not None:
                                    raise standard_err
                                raise
                        elif standard_err is not None:
                            raise standard_err
                        else:
                            raise ImportError("peft is required to load LoRA adapters.")

                    tokenizer_source = self.adapter_path if Path(self.adapter_path).exists() else self.model_id
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, token=self.token, trust_remote_code=True)

                    if self.model is None:
                        raise RuntimeError("Failed to initialize adapter model.")
        except KeyError as exc:
            raise RuntimeError(
                "Adapter/base model mismatch while loading LoRA keys. "
                "Use the adapter's matching base model (from adapter_config.json) or a merged model export."
            ) from exc
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self._using_local_model = True

    def _build_prompt(self, state: Dict[str, Any]) -> str:
        """Build a compact prompt for the LLM."""
        scenario = state.get("scenario", "unknown")
        attacker = state.get("attacker", "unknown")
        risk = state.get("network_risk", 0.0)
        risk_breakdown = state.get("risk_breakdown", {})

        # Get compromised and infected assets
        assets = state.get("assets", [])
        compromised_assets = [a["name"] for a in assets if a.get("compromised", False)]
        infected_assets = [a["name"] for a in assets if a.get("infected", False)]

        # Get forecast
        intelligence = state.get("intelligence", {})
        forecast = intelligence.get("forecast", {})
        primary_attack = forecast.get("primary", {}).get("attack", "unknown")

        # Get recent events
        events = state.get("events", [])
        recent_events = events[-3:] if events else []  # Last 3 events
        event_summaries = [e.get("summary", str(e)) for e in recent_events]

        prompt = f"""You are an expert cybersecurity defender. Choose exactly one CySent action to protect this {scenario} network from a {attacker} attacker.

Current State:
- Network Risk: {risk:.2f}
- Risk Breakdown: {json.dumps(risk_breakdown, indent=2)}
- Compromised Assets: {compromised_assets or "None"}
- Infected Assets: {infected_assets or "None"}
- Primary Attack Forecast: {primary_attack}
- Recent Events: {event_summaries}

Available Actions:
- rotate_credentials: Reset all credentials to prevent unauthorized access
- patch_auth_server: Apply security patches to authentication server
- segment_finance_database: Isolate finance database from network threats
- investigate_top_alert: Examine the highest priority security alert
- isolate_suspicious_host: Quarantine the most suspicious host immediately
- patch_hr_systems: Apply security patches to HR systems
- patch_web_server: Apply security patches to web servers
- increase_monitoring: Increase monitoring and alerting intensity
- restore_backup: Restore from backup after destructive compromise
- deploy_honeypot: Deploy a decoy honeypot to absorb attacker attention
- phishing_training: Run phishing awareness training

Respond with ONLY one action name from this exact list: {ACTION_LIST}."""

        return prompt

    def _call_hf_api(self, prompt: str) -> str:
        """Call HuggingFace API with retry logic."""
        if self.client is None:
            raise RuntimeError("HF client not initialized")

        for attempt in range(self.max_retries):
            try:
                response = self.client.text_generation(
                    prompt,
                    max_new_tokens=16,
                    temperature=0.0,
                    do_sample=False,
                )

                return str(response).strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"HF API call failed after {self.max_retries} attempts: {e}")
                time.sleep(1)

        return ""

    def _call_local_model(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Local HF model not initialized")

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        model_device = getattr(self.model, "device", None)
        if model_device is None:
            model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        with torch.no_grad() if torch is not None else nullcontext():
            output = self.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return str(self.tokenizer.decode(output[0], skip_special_tokens=True)).strip()

    def _parse_action(self, response: str) -> Optional[int]:
        """Parse the LLM response to extract a valid action."""
        normalized_response = _normalize_text(response)

        for action_name, action_id in VALID_ACTIONS.items():
            if _normalize_text(action_name) in normalized_response:
                return action_id

        for alias, mapped in ALIASES.items():
            if _normalize_text(alias) in normalized_response:
                return VALID_ACTIONS.get(mapped)

        return None

    def _predict_action_impl(self, state: Dict[str, Any]) -> int:
        if self.adapter_path and not self._using_local_model and self.client is None and not self._local_load_attempted:
            self._local_load_attempted = True
            try:
                self._initialize_local_model()
                self._active_backend = "local"
                print(f"[HFAgent] Ready (local adapter). adapter={self.adapter_path}")
            except Exception as exc:
                if self.token and InferenceClient is not None:
                    # If local loading fails (e.g., memory pressure), fall back to hosted inference.
                    self.client = InferenceClient(model=self.model_id, token=self.token)
                    self._using_local_model = False
                    self._active_backend = "cloud"
                    print(f"[HFAgent] Local adapter load failed ({type(exc).__name__}: {exc}); using cloud fallback.")
                else:
                    raise

        prompt = self._build_prompt(state)
        response = self._call_local_model(prompt) if self._using_local_model else self._call_hf_api(prompt)
        action_id = self._parse_action(response)

        if action_id is None:
            raise ValueError(f"HF adapter returned an invalid CySent action: {response!r}")

        return action_id

    async def predict_action_async(self, state: Dict[str, Any]) -> int:
        """Predict action using HF LLM asynchronously."""
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._predict_action_impl, state), timeout=self.timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"HF adapter timed out after {self.timeout:.1f}s") from exc

    def predict_action(self, state: Dict[str, Any]) -> int:
        """Predict action using HF LLM (synchronous wrapper)."""
        try:
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self._predict_action_impl, state)
            try:
                return future.result(timeout=self.timeout)
            except FuturesTimeoutError as exc:
                future.cancel()
                raise TimeoutError(f"HF adapter timed out after {self.timeout:.1f}s") from exc
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            raise

    def is_available(self) -> bool:
        """Check if HF agent is available for use."""
        return bool(self._using_local_model or self.client is not None or bool(self.adapter_path))

    def deployment_label(self) -> str:
        if self._using_local_model:
            return "Colab LLM Defender"
        if self.client is not None:
            return "HF Cloud Defender"
        return "HF LLM Defender"