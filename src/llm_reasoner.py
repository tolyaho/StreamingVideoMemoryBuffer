from __future__ import annotations

from typing import Optional, Sequence

from .summary_builder import (
    _default_torch_device,
    _suppress_generate_warnings,
    _vlm_dtype_for_device,
)


SYSTEM_PROMPT = (
    "You are answering a question about a video.\n"
    "Use the evidence below. Each line includes a time range.\n"
    "When you answer, mention the time ranges you relied on."
)


def _user_block(llm_input: dict, options: Optional[Sequence[str]]) -> str:
    head = f"Evidence:\n{llm_input['text_context']}\n\nQuestion: {llm_input['query']}"
    if not options:
        return head
    return (
        f"{head}\n"
        f"Options:\n" + "\n".join(options) + "\n"
        "Answer with the single letter (A, B, C, or D) of the best option, "
        "followed by a one-sentence justification that cites the time range(s) you used."
    )


def build_prompt(
    llm_input: dict,
    *,
    options: Optional[Sequence[str]] = None,
    system: str = SYSTEM_PROMPT,
) -> str:
    return f"[system] {system}\n\n[user]\n{_user_block(llm_input, options)}"


class LLMReasoner:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 256,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = _default_torch_device(device)
        self.max_new_tokens = int(max_new_tokens)

        dtype = _vlm_dtype_for_device(self.device)
        print(f"Loading reasoner LLM {model_name} on {self.device} ({dtype})…")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                attn_implementation="sdpa" if self.device == "cuda" else "eager",
            )
            .to(self.device)
            .eval()
        )
        print("Reasoner LLM ready.")

    def answer(
        self,
        llm_input: dict,
        *,
        options: Optional[Sequence[str]] = None,
        max_new_tokens: Optional[int] = None,
        system: str = SYSTEM_PROMPT,
    ) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": _user_block(llm_input, options)},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad(), _suppress_generate_warnings():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
            )

        tail = out[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(tail, skip_special_tokens=True).strip()

    build_prompt = staticmethod(build_prompt)
