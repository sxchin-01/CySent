from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None


VALID_ACTIONS = {
    "rotate_credentials": 4,
    "patch_auth_server": 3,
    "segment_finance_database": 11,
    "investigate_top_alert": 10,
}


class HFAgent:
    """HuggingFace LLM agent for action decision making."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        self.model_id = model_id or os.getenv("HF_MODEL_ID", "microsoft/DialoGPT-medium")
        self.endpoint_url = endpoint_url or os.getenv("HF_ENDPOINT_URL")
        self.token = token or os.getenv("HF_TOKEN")
        self.timeout = timeout
        self.max_retries = max_retries
        self.client: Optional[InferenceClient] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the HuggingFace client."""
        if InferenceClient is None:
            raise ImportError("huggingface_hub not installed. Cannot use HF agent.")

        if self.endpoint_url:
            # Use custom endpoint (TGI, Serverless, etc.)
            self.client = InferenceClient(model=self.endpoint_url, token=self.token)
        else:
            # Use standard HF Inference API
            self.client = InferenceClient(model=self.model_id, token=self.token)

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

        prompt = f"""You are an expert cybersecurity defender. Choose the best action to protect this {scenario} network from a {attacker} attacker.

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

Respond with ONLY the action name (one of: rotate_credentials, patch_auth_server, segment_finance_database, investigate_top_alert)."""

        return prompt

    async def _call_hf_api(self, prompt: str) -> str:
        """Call HuggingFace API with retry logic."""
        if self.client is None:
            raise RuntimeError("HF client not initialized")

        for attempt in range(self.max_retries):
            try:
                if self.endpoint_url:
                    # TGI/Serverless endpoint
                    response = self.client.text_generation(
                        prompt,
                        max_new_tokens=50,
                        temperature=0.1,
                        do_sample=True,
                    )
                else:
                    # Standard HF Inference API
                    response = self.client.conversational(
                        {"inputs": {"past_user_inputs": [], "generated_responses": [], "text": prompt}},
                        parameters={"max_length": 100, "temperature": 0.1}
                    )
                    response = response["generated_text"]

                return str(response).strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"HF API call failed after {self.max_retries} attempts: {e}")
                await asyncio.sleep(1)  # Brief backoff

        return ""

    def _parse_action(self, response: str) -> Optional[int]:
        """Parse the LLM response to extract a valid action."""
        response_lower = response.lower().strip()

        for action_name, action_id in VALID_ACTIONS.items():
            if action_name.lower() in response_lower:
                return action_id

        return None

    async def predict_action_async(self, state: Dict[str, Any]) -> int:
        """Predict action using HF LLM asynchronously."""
        prompt = self._build_prompt(state)
        response = await self._call_hf_api(prompt)
        action_id = self._parse_action(response)

        if action_id is None:
            # Fallback to default action if parsing fails
            return 0  # do_nothing

        return action_id

    def predict_action(self, state: Dict[str, Any]) -> int:
        """Predict action using HF LLM (synchronous wrapper)."""
        try:
            # Try to run async in current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, we need to handle differently
                # For now, return fallback
                return 0
            else:
                return loop.run_until_complete(self.predict_action_async(state))
        except Exception:
            # Fallback on any error
            return 0

    def is_available(self) -> bool:
        """Check if HF agent is available for use."""
        return self.client is not None and bool(self.token)