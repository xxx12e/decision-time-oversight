from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .schema import PublicMessage, sign_to_state


@dataclass
class LLMDecision:
    belief_state: str
    confidence: float
    cited_evidence: str = ""


class LocalLLMAdapter:
    """Minimal local-only Hugging Face causal LM adapter.

    This class never calls remote APIs and passes `local_files_only=True` to
    model/tokenizer loading. It is intended for small targeted runs after the
    scripted benchmark is validated.
    """

    def __init__(self, model_path: str, max_new_tokens: int = 96, temperature: float = 0.0):
        if not model_path or not os.path.exists(model_path):
            raise ValueError("A valid local_llm.model_path is required for model_type=local_llm.")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "model_type=local_llm requires local installations of transformers and torch."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def decide(
        self,
        private_signal: int,
        evidence_strength: float,
        current_belief: str,
        neighbor_messages: list[PublicMessage],
        defense_instruction: str = "",
    ) -> LLMDecision:
        prompt = self._prompt(private_signal, evidence_strength, current_belief, neighbor_messages, defense_instruction)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else None,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}
        with self.torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)
        text = self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return self._parse(text, current_belief)

    @staticmethod
    def _prompt(
        private_signal: int,
        evidence_strength: float,
        current_belief: str,
        neighbor_messages: list[PublicMessage],
        defense_instruction: str,
    ) -> str:
        messages = [
            {
                "claim": message.claim_state,
                "confidence": round(message.confidence, 3),
                "supported": message.supported,
                "source_group": message.source_group,
            }
            for message in neighbor_messages
        ]
        return (
            "You are an artificial benchmark agent in a synthetic hidden-state task.\n"
            "The hidden state is either A or B. This is not about people or the real world.\n"
            f"Your private signal is {sign_to_state(private_signal)} with reliability {evidence_strength:.2f}.\n"
            f"Your previous belief was {current_belief}.\n"
            f"Defense policy: {defense_instruction}\n"
            f"Neighbor messages: {json.dumps(messages)}\n"
            'Return only JSON like {"belief":"A","confidence":0.73,"cited_evidence":"private_signal:A; supported_neighbors:2"}.'
        )

    @staticmethod
    def _parse(text: str, fallback_state: str) -> LLMDecision:
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                belief = str(data.get("belief", fallback_state)).strip().upper()
                confidence = float(data.get("confidence", 0.55))
                if belief in {"A", "B"}:
                    cited = str(data.get("cited_evidence", "")).strip()
                    return LLMDecision(belief, max(0.51, min(0.99, confidence)), cited)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        text_upper = text.upper()
        if "B" in text_upper and "A" not in text_upper:
            return LLMDecision("B", 0.60, "")
        if "A" in text_upper and "B" not in text_upper:
            return LLMDecision("A", 0.60, "")
        return LLMDecision(fallback_state, 0.55, "")
