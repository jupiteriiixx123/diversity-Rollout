"""Common AgentGym environment loop for official verl v0.5.0.

Model inference and RL training belong to verl; environment dynamics belong to
AgentGym. Subclasses provide environment-specific task IDs and terminal score
formatting. In particular, this class has no env-name switch statements.

The token-framing code below deliberately targets Qwen2.5 ChatML, matching the
project's Qwen2.5-7B-Instruct model. It is not a generic tokenizer adapter.
"""

import asyncio
import time
from abc import abstractmethod
from typing import Any, Mapping
from uuid import uuid4

from agentenv.controller.env import BaseEnvClient
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)


class BaseAgentGymLoop(AgentLoopBase):
    """Common rollout for AgentGym clients using verl-managed model inference.

    This class does not launch or manage environment server processes. Each
    subclass declares an EnvClient implementation; ``client_args`` is passed
    to that client unchanged, following AgentGym's BaseTask convention.
    """

    env_client_cls: type[BaseEnvClient] | None = None  # Client selected by subclass.
    env_name: str | None = None  # Human-readable environment name for diagnostics.

    def __init__(
        self,
        *args,
        client_args: Mapping[str, Any],
        max_rounds: int = 20,
        max_tokens_per_turn: int = 128,
        **kwargs,
    ):
        """Initialize a single rollout.

        Args:
            client_args: Constructor kwargs for the subclass's EnvClient.
                For BabyAI these include env_server_base (address of an
                already-running environment service) and data_len. They are
                *not* common arguments required by BaseEnvClient.
            max_rounds: Maximum number of environment actions in one rollout.
            max_tokens_per_turn: Maximum number of tokens generated per action.
        """
        super().__init__(*args, **kwargs)
        self.client_args = dict(client_args)
        self.max_rounds = int(max_rounds)
        self.max_tokens_per_turn = int(max_tokens_per_turn)
        self.response_length = int(self.config.actor_rollout_ref.rollout.response_length)
        if self.env_client_cls is None or not self.env_name:
            raise TypeError("A subclass must declare env_client_cls and env_name")
        if self.max_rounds < 1 or self.max_tokens_per_turn < 1:
            raise ValueError("max_rounds and max_tokens_per_turn must be positive")

    @abstractmethod
    def parse_task_id(self, messages: list[dict]) -> int:
        """Read the task ID from the non-model-visible prompt metadata."""

    @abstractmethod
    def terminal_feedback(self, step_result) -> str:
        """Return environment-specific terminal score text (including any prefix)."""

    def create_env(self) -> BaseEnvClient:
        """Create one EnvClient; never start an environment server process.

        AgentGym's concrete client handles any HTTP /create request itself.
        """
        return self.env_client_cls(**self.client_args)

    def reset_env(self, env, task_id):
        """Override only if a particular EnvClient requires special reset args."""
        env.reset(task_id)
        return env.observe()

    def step_env(self, env, answer: str):
        return env.step(answer)

    def close_env(self, env):
        # BaseEnvClient does not declare close(); concrete HTTP clients do.
        close = getattr(env, "close", None)
        if callable(close):
            close()

    def _observation_ids(self, observation: str, *, first: bool, final: bool) -> list[int]:
        """Qwen2.5 ChatML user message, never a new standalone conversation.

        The previous message is an assistant response ending with <|im_end|>.
        After the fixed initial assistant acknowledgement, a newline already
        exists; after vLLM generation, the newline is supplied here.
        """
        chat_fragment = (
            ("" if first else "\n")
            + "<|im_start|>user\n"
            + observation
            + "<|im_end|>\n"
            + ("" if final else "<|im_start|>assistant\n")
        )
        return self.tokenizer.encode(chat_fragment, add_special_tokens=False)

    async def run(self, messages: list[dict], sampling_params: dict) -> AgentLoopOutput:
        task_id = self.parse_task_id(messages)
        clean_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        # The initial environment observation is not known until reset().
        prompt_ids = self.tokenizer.apply_chat_template(
            clean_messages, tokenize=True, add_generation_prompt=False
        )
        eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos_id is None or eos_id == self.tokenizer.unk_token_id:
            raise ValueError("This adapter requires Qwen2.5's <|im_end|> delimiter")

        response_ids: list[int] = []
        response_mask: list[int] = []

        def append(ids, *, generated: bool):
            ids = list(ids)
            if len(response_ids) + len(ids) > self.response_length:
                raise ValueError(
                    f"{self.env_name} trajectory exceeds rollout.response_length="
                    f"{self.response_length}; increase it or shorten turns"
                )
            response_ids.extend(ids)
            response_mask.extend([int(generated)] * len(ids))

        env = None
        llm_time = 0.0
        env_time = 0.0
        num_steps = 0
        request_id = uuid4().hex
        try:
            t = time.perf_counter()
            env = await asyncio.to_thread(self.create_env)
            observation = await asyncio.to_thread(self.reset_env, env, task_id)
            env_time += time.perf_counter() - t
            append(self._observation_ids(observation, first=True, final=False), generated=False)

            for turn in range(self.max_rounds):
                params = dict(sampling_params)
                params["max_tokens"] = self.max_tokens_per_turn
                t = time.perf_counter()
                generated_ids = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids + response_ids,
                    sampling_params=params,
                )
                llm_time += time.perf_counter() - t
                if not generated_ids:
                    raise RuntimeError(f"Empty LLM response for {self.env_name}, task {task_id}")

                append(generated_ids, generated=True)
                if generated_ids[-1] != eos_id:
                    # Some inference backends omit the stop token. Synthetic
                    # tokens are never part of the policy loss.
                    append([eos_id], generated=False)

                answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                t = time.perf_counter()
                result = await asyncio.to_thread(self.step_env, env, answer)
                env_time += time.perf_counter() - t
                num_steps += 1

                final = bool(result.done) or turn == self.max_rounds - 1
                observation = str(result.state)
                if final:
                    # Only appended after the last generation, never shown to
                    # the model. The score therefore stays trajectory-specific.
                    observation += self.terminal_feedback(result)
                append(
                    self._observation_ids(observation, first=False, final=final),
                    generated=False,
                )
                if final:
                    break
        finally:
            if env is not None:
                await asyncio.to_thread(self.close_env, env)

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            num_turns=len(clean_messages) + 2 * num_steps + 1,
            metrics=AgentLoopMetrics(generate_sequences=llm_time, tool_calls=env_time),
        )