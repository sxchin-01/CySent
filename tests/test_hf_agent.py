from __future__ import annotations

import unittest

import torch

from backend.agents.hf_agent import ACTION_LIST, HFAgent
from backend.env.security_env import ACTION_NAMES


class _FakeTokenizer:
    eos_token_id = 0

    def __init__(self, completion: str) -> None:
        self.completion = completion
        self.decoded_token_ids: list[int] = []

    def __call__(self, prompt: str, **kwargs):
        return {
            "input_ids": torch.tensor([[101, 102, 103]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def decode(self, token_ids, **kwargs) -> str:
        self.decoded_token_ids = token_ids.tolist()
        return self.completion


class _FakeLocalModel:
    device = torch.device("cpu")

    def generate(self, **kwargs):
        return torch.tensor([[101, 102, 103, 201, 202]], dtype=torch.long)


class _FakeCloudClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def text_generation(self, prompt: str, **kwargs) -> str:
        return self.response


def _local_agent(completion: str) -> HFAgent:
    agent = object.__new__(HFAgent)
    agent.model = _FakeLocalModel()
    agent.tokenizer = _FakeTokenizer(completion)
    agent.adapter_path = ""
    agent.client = None
    agent._using_local_model = True
    agent._local_load_attempted = True
    return agent


def _cloud_agent(response: str) -> HFAgent:
    agent = object.__new__(HFAgent)
    agent.model = None
    agent.tokenizer = None
    agent.adapter_path = ""
    agent.client = _FakeCloudClient(response)
    agent._using_local_model = False
    agent._local_load_attempted = True
    agent._cloud_adapter_id = None
    agent.max_retries = 1
    return agent


class HFAgentCompletionParsingTests(unittest.TestCase):
    def test_local_generation_parses_completion_not_prompt_actions(self) -> None:
        agent = _local_agent("investigate_top_alert")
        prompt = f"Available actions: {ACTION_LIST}"

        response = agent._call_local_model(prompt)

        self.assertEqual(agent.tokenizer.decoded_token_ids, [201, 202])
        self.assertEqual(agent._parse_action(response), 10)
        self.assertEqual(ACTION_NAMES[agent._parse_action(response)], "investigate_top_alert")

    def test_local_generation_maps_another_canonical_completion(self) -> None:
        agent = _local_agent("rotate_credentials")

        response = agent._call_local_model(f"Available actions: {ACTION_LIST}")

        self.assertEqual(agent._parse_action(response), 4)

    def test_invalid_local_completion_fails_truthfully(self) -> None:
        agent = _local_agent("not_a_cysent_action")
        agent._build_prompt = lambda state: f"Available actions: {ACTION_LIST}"

        with self.assertRaisesRegex(ValueError, "invalid CySent action"):
            agent._predict_action_impl({})

    def test_cloud_response_parsing_is_unchanged(self) -> None:
        agent = _cloud_agent("increase_monitoring")
        agent._build_prompt = lambda state: f"Available actions: {ACTION_LIST}"

        self.assertEqual(agent._predict_action_impl({}), 6)


if __name__ == "__main__":
    unittest.main()
