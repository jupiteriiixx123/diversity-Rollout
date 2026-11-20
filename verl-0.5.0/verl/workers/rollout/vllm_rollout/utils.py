# Keep the same LoRA identity across weight updates and generation.
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"


def get_vllm_max_lora_rank(lora_rank: int) -> int:
    if lora_rank <= 0:
        raise ValueError("lora_rank must be positive")

    supported_ranks = [8, 16, 32, 64, 128, 256, 320, 512]

    for rank in supported_ranks:
        if lora_rank <= rank:
            return rank

    raise ValueError(
        f"vLLM max_lora_rank cannot exceed {supported_ranks[-1]}"
    )