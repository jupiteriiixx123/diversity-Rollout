"""Check Qwen2.5 fragments against the whole chat template, byte for byte in token IDs."""

from pathlib import Path
from transformers import AutoTokenizer
from agentgym_adapter.babyai_loop import BabyAIAgentLoop

MODEL = Path("/public/home/hh_hh/AgentGym-RL/models/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)

# No environment/network involved. Access inherited helper without __init__.
loop = object.__new__(BabyAIAgentLoop)
loop.tokenizer = tokenizer

messages = [
    {"role": "user", "content": "Follow the BabyAI action format."},
    {"role": "assistant", "content": "OK."},
]
obs0 = "Your goal: go to the red ball."
action = "Thought:\nI will explore.\n\nAction:\nturn left"
obs1 = "The room looks different now."

prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
first_ids = loop._observation_ids(obs0, first=True, final=False)
whole_first = tokenizer.apply_chat_template(
    messages + [{"role": "user", "content": obs0}], tokenize=True, add_generation_prompt=True
)
assert prompt_ids + first_ids == whole_first, "Initial observation mismatches complete Qwen chat template"

action_ids = tokenizer.encode(action, add_special_tokens=False)
eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
last_ids = loop._observation_ids(obs1, first=False, final=True)
whole_last = tokenizer.apply_chat_template(
    messages + [
        {"role": "user", "content": obs0},
        {"role": "assistant", "content": action},
        {"role": "user", "content": obs1},
    ],
    tokenize=True,
    add_generation_prompt=False,
)
actual = prompt_ids + first_ids + action_ids + [eos_id] + last_ids
assert actual == whole_last, "Second observation mismatches complete Qwen chat template"

full_text = tokenizer.decode(actual, skip_special_tokens=False)
assert full_text.count("<|im_start|>system") == 1
assert full_text.count("<|im_start|>user") == 3
assert full_text.count("<|im_start|>assistant") == 2
print("ChatML two-turn token alignment: OK")
