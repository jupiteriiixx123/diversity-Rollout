# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import pickle
from typing import Any, Callable, Optional

import ray
import zmq
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest

from verl.utils.fs import copy_to_local
from verl.workers.rollout.async_server import AsyncServerBase
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

logger = logging.getLogger(__file__)


def _get_model_runner_workers(vllm_config, init_ray: bool = True):
    assert vllm_config.instance_id is not None, "instance_id must be set for external ray actors."

    fields = vllm_config.instance_id.split(":")
    assert len(fields) == 4, (
        f"instance_id: {vllm_config.instance_id} must be in the format of "
        f"<namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
    )
    namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])

    # Make sure subprocess in same namespace as parent actor.
    # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
    if init_ray:
        ray.init(namespace=namespace)
    actor_names = [
        actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict")
    ]

    vllm_tp_size = vllm_config.parallel_config.tensor_parallel_size
    assert len(actor_names) == vllm_dp_size * vllm_tp_size, (
        f"instance_id: {vllm_config.instance_id} has {len(actor_names)} actors, but vllm_dp_size: "
        f"{vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected."
    )

    def get_pg_index_and_local_rank(actor_name) -> tuple[int, int]:
        fields = actor_name.split(":")
        assert len(fields) == 2, f"invalid actor name: {actor_name}"
        pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
        return pg_index, local_rank

    # sort actor names by pg_index and local_rank
    actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
    actor_names = actor_names[vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size]
    workers: list[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
    print(f"instance_id: {vllm_config.instance_id} initializes with external actors: {actor_names}")

    return workers


class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        self.workers = _get_model_runner_workers(vllm_config=self.vllm_config, init_ray=True)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")
        print(f"instance_id: {self.vllm_config.instance_id} initializes finished.")

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        # TODO(wuxibin): support ray compiled graph
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        # ~3ms overhead per schedule step due to SchedulerOutput/ModelRunnerOutput serialization/deserialization.
        outputs = ray.get(
            [worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers]
        )
        return outputs

    def check_health(self):
        return


class ExternalZeroMQDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        addresses = os.environ["VERL_VLLM_ZMQ_ADDRESSES"].split(",")
        self.context = zmq.Context()
        self.sockets = []
        for address in addresses:
            socket = self.context.socket(zmq.REQ)
            socket.connect(address)
            self.sockets.append(socket)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        message = pickle.dumps((sent_method, args, kwargs or {}))
        for socket in self.sockets:
            socket.send(message, zmq.DONTWAIT)

        outputs = []
        for socket in self.sockets:
            outputs.append(pickle.loads(socket.recv()))
        return outputs

    def check_health(self):
        return


@ray.remote(num_cpus=1)
class AsyncvLLMServer(AsyncServerBase):
    """
    AsyncvLLMServer is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers, i.e AsyncActorRolloutRefWorker.

    AsyncvLLMServer works as follows:
    1. Start FastAPI server first.
    2. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    3. AsyncLLM spawn EngineCore in subprocess.
    4. EngineCore initialize ExternalRayDistributedExecutor.
    5. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    6. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str):
        """
        Args:
            config: DictConfig.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        super().__init__()

        self.config = config.actor_rollout_ref
        self.lora_rank = int(
            self.config.model.get("lora_rank", 0)
        )
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.engine: AsyncLLM = None

        self._active_lora_id = None
        self._active_rollout_step = None

        self._lora_infer_count = 0
        self._lora_fallback_count = 0

    async def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        config = self.config
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get("trust_remote_code", False)
        config = config.rollout

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        self.max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            logprobs=0,
            repetition_penalty=1.0,
            max_new_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        print(f"override_generation_config: {kwargs}")

        backend = os.environ.get("VERL_VLLM_DISTRIBUTED_BACKEND", "zeromq")
        if backend == "zeromq":
            distributed_executor_backend = ExternalZeroMQDistributedExecutor
        elif backend == "ray":
            distributed_executor_backend = ExternalRayDistributedExecutor
        else:
            distributed_executor_backend = None

        lora_engine_kwargs = {}

        if self.lora_rank > 0:
            lora_engine_kwargs = {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": get_vllm_max_lora_rank(self.lora_rank),
            }
        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=config.free_cache_engine,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=self.max_model_len,
            load_format="auto",
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_engine_kwargs,
        )

        # init async llm engine
        vllm_config = self._create_engine_config(engine_args)
        self.engine = AsyncLLM.from_vllm_config(vllm_config)

        # build serving chat
        model_config = self.engine.model_config
        BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
        models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            model_config,
            models,
            "assistant",
            request_logger=RequestLogger(max_log_len=4096),
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=config.multi_turn.tool_config_path is not None,
            tool_parser=config.multi_turn.format,  # hermes, llama3_json, ...
        )

    def _create_engine_config(self, engine_args: AsyncEngineArgs):
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"

        # VERL_VLLM_ZMQ_ADDRESSES
        if engine_args.distributed_executor_backend == ExternalZeroMQDistributedExecutor:
            workers = _get_model_runner_workers(vllm_config=vllm_config, init_ray=False)
            zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in workers])
            print(f"VERL_VLLM_ZMQ_ADDRESSES: {zmq_addresses}")
            os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        return vllm_config

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str) -> list[int]:
        max_tokens = self.max_model_len - len(prompt_ids)
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        lora_request = None

        if self.lora_rank > 0:
            expected_id = self._active_lora_id

            if expected_id is None:
                raise RuntimeError(
                    "[LORA_INFER] No committed LoRA version"
                )

            loaded_ids = set(await self.engine.list_loras())

            self._lora_infer_count += 1

            if expected_id in loaded_ids:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME,
                    lora_int_id=expected_id,
                    lora_path=VLLM_LORA_PATH,
                )

                print(
                    "[LORA_INFER] USE_LORA "
                    f"rollout_step={self._active_rollout_step} "
                    f"request_id={request_id} "
                    f"lora_id={expected_id} "
                    "status=OK",
                    flush=True,
                )

            else:
                self._lora_fallback_count += 1

                logger.warning(
                    "[LORA_INFER] BASE_FALLBACK "
                    "step=%s expected_id=%s loaded_ids=%s "
                    "request_id=%s fallback_count=%d total=%d",
                    self._active_rollout_step,
                    expected_id,
                    sorted(loaded_ids),
                    request_id,
                    self._lora_fallback_count,
                    self._lora_infer_count,
                )

                # Strict by default; optional diagnostic fallback.
                policy = os.environ.get(
                    "VERL_LORA_MISSING_POLICY", "error"
                )

                if policy != "warn":
                    raise RuntimeError(
                        f"[LORA_INFER] LoRA {expected_id} "
                        "missing during inference"
                    )

        # if self.lora_rank > 0:
        #     loaded_loras = await self.engine.list_loras()

        #     if VLLM_LORA_INT_ID in loaded_loras:
        #         lora_request = LoRARequest(
        #             lora_name=VLLM_LORA_NAME,
        #             lora_int_id=VLLM_LORA_INT_ID,
        #             lora_path=VLLM_LORA_PATH,
        #         )

        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        )
        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        return final_res.outputs[0].token_ids

    async def wake_up(self, lora_id=None, rollout_step=None):
        """Prepare, load, verify and commit a LoRA version."""

        self._active_lora_id = None
        self._active_rollout_step = None

        # Non-LoRA execution keeps original behavior.
        if lora_id is None:
            if self.config.rollout.free_cache_engine:
                await self.engine.wake_up()
            return

        if not self.config.rollout.free_cache_engine:
            raise RuntimeError(
                "[LORA_SYNC] Dynamic synchronization requires "
                "free_cache_engine=True"
            )

        lora_id = int(lora_id)
        rollout_step = int(rollout_step)

        # Phase 1: distribute candidate ID to all TP workers.
        prepared = await self.engine.collective_rpc(
            "prepare_lora_version",
            args=(lora_id,),
        )

        expected_tp = int(
            self.config.rollout.tensor_model_parallel_size
        )

        if (
            len(prepared) != expected_tp
            or any(
                r["candidate_id"] != lora_id
                for r in prepared
            )
        ):
            raise RuntimeError(
                f"[LORA_SYNC] Prepare failed: {prepared}"
            )

        # Phase 2: wake up workers.
        # This triggers FSDP sharding_manager.__enter__()
        # and loads current LoRA weights.
        await self.engine.wake_up()

        # Phase 3: verify every TP worker.
        states = await self.engine.collective_rpc(
            "get_lora_sync_state"
        )

        all_workers_ok = (
            len(states) == expected_tp
            and len({
                r["rank"] for r in states
            }) == expected_tp
            and all(
                r["committed_id"] == lora_id
                and lora_id in r["loaded_ids"]
                for r in states
            )
        )

        engine_ids = set(await self.engine.list_loras())

        if not all_workers_ok or lora_id not in engine_ids:
            raise RuntimeError(
                "[LORA_SYNC] Global commit rejected: "
                f"expected={lora_id}, "
                f"states={states}, "
                f"engine_ids={sorted(engine_ids)}"
            )

        # Phase 4: global commit.
        # No inference request can use this version before here.
        self._active_lora_id = lora_id
        self._active_rollout_step = rollout_step

        print(
            "[LORA_SYNC] COMMIT_OK "
            f"rollout_step={rollout_step} "
            f"policy_step={rollout_step - 1} "
            f"dp_rank={self.vllm_dp_rank} "
            f"lora_id={lora_id} "
            f"tp_workers={len(states)} "
            "status=OK",
            flush=True,
        )

    # async def wake_up(self):
    #     if self.config.rollout.free_cache_engine:
    #         await self.engine.wake_up()

    # async def sleep(self):
    #     # TODO: https://github.com/vllm-project/vllm/issues/17103
    #     await self.engine.reset_prefix_cache()
    #     if self.config.rollout.free_cache_engine:
    #         await self.engine.sleep()
    async def sleep(self):
        await self.engine.reset_prefix_cache()

        if self.config.rollout.free_cache_engine:
            await self.engine.sleep()

        self._active_lora_id = None
        self._active_rollout_step = None
