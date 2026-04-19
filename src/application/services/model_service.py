from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import anyio
import numpy as np
from PIL import Image

from src.application.dataclasses.generation import (
    GenerationRequest,
    GenerationResult,
    SamplingProfile,
)

class ModelService:
    def __init__(
        self,
        model_name: str,
        hf_cache_dir: Path,
        provider: str,
        precision: str,
        verbose: bool,
        image_max_side: int,
        text_sampling: SamplingProfile,
        vl_sampling: SamplingProfile,
        vl_system_prompt: str = "",
        shared_kv: bool = False,
        max_context: int = 2048,
        cuda_graph: bool = True,
    ) -> None:
        self._model_name = model_name
        self._hf_cache_dir = hf_cache_dir
        self._provider = provider
        self._precision = precision
        self._verbose = verbose
        self._image_max_side = max(1, image_max_side)
        self._text_sampling = text_sampling
        self._vl_sampling = vl_sampling
        self._vl_system_prompt = vl_system_prompt
        self._shared_kv = shared_kv
        self._max_context = max_context
        self._cuda_graph = cuda_graph
        self._loaded: bool = False
        self._logger = logging.getLogger(__name__)

        self._vision_session: Any | None = None
        self._embed_session: Any | None = None
        self._decoder_session: Any | None = None
        # When shared_kv + cuda_graph is on this is a distinct session without
        # graph capture (for the prompt-length-dependent prefill pass).
        # Otherwise it aliases _decoder_session.
        self._prefill_decoder_session: Any | None = None

        self._processor: Any | None = None
        self._config: Any | None = None

        self._num_key_value_heads: int = 0
        self._head_dim: int = 0
        self._num_hidden_layers: int = 0
        self._eos_token_id: int = 0
        self._image_token_index: int = 0

        # dtypes discovered from the ONNX graphs; set during _load_sync.
        self._kv_dtype_np: np.dtype = np.dtype(np.float32)
        self._embed_dtype_np: np.dtype = np.dtype(np.float32)
        self._decoder_output_names: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        if self._loaded:
            return
        await anyio.to_thread.run_sync(self._load_sync)

    async def unload(self) -> None:
        if not self._loaded:
            return
        await anyio.to_thread.run_sync(self._unload_sync)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self._loaded:
            raise RuntimeError("model is not loaded")
        return await anyio.to_thread.run_sync(self._generate_sync, request)

    def _load_sync(self) -> None:
        import onnxruntime as ort
        from transformers import AutoConfig, AutoProcessor

        self._hf_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self._hf_cache_dir))

        self._logger.info(
            "loading model repo=%s provider=%s precision=%s",
            self._model_name,
            self._provider,
            self._precision,
        )
        start = time.perf_counter()

        self._config = AutoConfig.from_pretrained(self._model_name)
        self._processor = AutoProcessor.from_pretrained(self._model_name)

        text_config = self._config.text_config
        self._num_key_value_heads = text_config.num_key_value_heads
        self._head_dim = text_config.head_dim
        self._num_hidden_layers = text_config.num_hidden_layers
        self._eos_token_id = text_config.eos_token_id
        self._image_token_index = self._config.image_token_index

        suffix = f"_{self._precision}" if self._precision else ""
        vision_model_path = self._download_onnx_component(f"vision_encoder{suffix}")
        embed_model_path = self._download_onnx_component(f"embed_tokens{suffix}")
        decoder_model_path = self._download_onnx_component(f"decoder_model_merged{suffix}")

        # Optionally patch the decoder to use past_present_share_buffer on GQA
        # so KV cache shapes become static. Cached on disk; patches once.
        if self._shared_kv and self._provider == "cuda":
            decoder_model_path = self._ensure_shared_kv_decoder(decoder_model_path)

        providers = self._get_providers()
        provider_options = self._get_provider_options()

        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
        # Enable all graph optimizations
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._vision_session = ort.InferenceSession(
            vision_model_path, sess_options=sess_options, providers=providers, provider_options=provider_options
        )
        self._embed_session = ort.InferenceSession(
            embed_model_path, sess_options=sess_options, providers=providers, provider_options=provider_options
        )
        # For shared-kv + CUDA graph, the decoder needs its own provider
        # options with enable_cuda_graph=1. For prefill, the non-static prompt
        # length prevents CUDA graph capture, so we keep a separate decoder
        # session WITHOUT graph capture for the prefill step.
        decoder_provider_options = self._get_provider_options(
            for_decode=True if self._shared_kv else False
        )
        self._decoder_session = ort.InferenceSession(
            decoder_model_path, sess_options=sess_options, providers=providers, provider_options=decoder_provider_options
        )
        if self._shared_kv and self._cuda_graph and self._provider == "cuda":
            prefill_provider_options = self._get_provider_options(for_decode=False)
            self._prefill_decoder_session = ort.InferenceSession(
                decoder_model_path,
                sess_options=sess_options,
                providers=providers,
                provider_options=prefill_provider_options,
            )
        else:
            self._prefill_decoder_session = self._decoder_session

        # Discover tensor dtypes from the graphs so inputs we build match the
        # model exactly. Mismatched dtypes cause ORT to insert Cast/Memcpy
        # nodes (the "NNN Memcpy nodes ... negative impact on performance"
        # warning you saw on startup).
        self._kv_dtype_np = self._onnx_type_to_numpy(
            self._find_input_type(self._decoder_session, "past_key_values."),
            default=np.float16,
        )
        self._embed_dtype_np = self._onnx_type_to_numpy(
            self._find_input_type(self._decoder_session, "inputs_embeds"),
            default=np.float16,
        )
        self._decoder_output_names = [out.name for out in self._decoder_session.get_outputs()]

        self._logger.info(
            "dtypes kv=%s embed=%s decoder_outputs=%d",
            self._kv_dtype_np,
            self._embed_dtype_np,
            len(self._decoder_output_names),
        )

        self._loaded = True

        elapsed = time.perf_counter() - start
        self._logger.info("model loaded in %.1fs", elapsed)

    def _download_onnx_component(self, stem: str, subfolder: str = "onnx") -> str:
        """Download `<stem>.onnx` plus all `<stem>.onnx_data*` shards from HF.

        The number of external-data shards varies by precision variant
        (e.g. q4 decoder has `.onnx_data` + `.onnx_data_1`, q4f16 has only
        `.onnx_data`, plain fp16 has `.onnx_data` through `.onnx_data_3`).
        We probe the repo once and download whatever exists.
        """
        from huggingface_hub import hf_hub_download, list_repo_files

        main_path = hf_hub_download(
            self._model_name, f"{stem}.onnx", subfolder=subfolder
        )

        repo_files = list_repo_files(self._model_name)
        prefix = f"{subfolder}/{stem}.onnx_data"
        for f in repo_files:
            if f.startswith(prefix):
                filename = f[len(subfolder) + 1 :]  # strip "onnx/"
                hf_hub_download(self._model_name, filename, subfolder=subfolder)
        return main_path

    @staticmethod
    def _find_input_type(session: Any, name_prefix: str) -> str | None:
        for meta in session.get_inputs():
            if meta.name.startswith(name_prefix) or meta.name == name_prefix:
                return meta.type
        return None

    @staticmethod
    def _onnx_type_to_numpy(onnx_type: str | None, default: Any) -> np.dtype:
        if onnx_type is None:
            return np.dtype(default)
        mapping = {
            "tensor(float16)": np.float16,
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(bfloat16)": np.float32,  # numpy has no bf16; use fp32 + cast inside ORT
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
        }
        return np.dtype(mapping.get(onnx_type, default))

    def _get_providers(self) -> list[str]:
        """Get ONNX Runtime execution providers based on configuration."""
        if self._provider == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _get_provider_options(self, for_decode: bool = False) -> list[dict[str, Any]]:
        """Get provider-specific options for CUDA optimization.

        When `for_decode` is True and shared_kv + cuda_graph are enabled, this
        also turns on ORT's CUDA graph capture for the steady-state decode
        shape. The first run after session creation is the warmup; subsequent
        runs replay the captured graph which removes most per-step kernel
        launch and Python overhead.
        """
        if self._provider == "cuda":
            cuda_options: dict[str, Any] = {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            }
            if for_decode and self._shared_kv and self._cuda_graph:
                cuda_options["enable_cuda_graph"] = "1"
            return [cuda_options, {}]
        return [{}]

    def _unload_sync(self) -> None:
        import gc

        self._vision_session = None
        self._embed_session = None
        self._decoder_session = None
        self._prefill_decoder_session = None
        self._processor = None
        self._config = None
        self._decoder_output_names = []
        self._loaded = False
        gc.collect()

    def _ensure_shared_kv_decoder(self, source_path: str) -> str:
        """Placeholder for future shared-KV / CUDA-graph support.

        NOTE: as of ORT 1.24 the contrib GroupQueryAttention op does NOT
        accept `past_present_share_buffer` as a node attribute (schema error
        "Unrecognized attribute"). Shared-buffer KV is currently only wired
        up through the `onnxruntime-genai` runtime + its model_builder, which
        does not support the Mistral-3 vision architecture.

        Until a supported path exists (newer ORT that exposes the attribute,
        or genai support for this model family) we just return the source
        model unchanged. The rest of the shared_kv code path is harmless;
        it preallocates KV/mask buffers but without the kernel doing in-place
        writes it doesn't help performance. Keep shared_kv=False in settings.
        """
        self._logger.warning(
            "shared_kv requested but not supported on ORT %s for this model; "
            "using the unpatched decoder. See ModelService._ensure_shared_kv_decoder.",
            __import__("onnxruntime").__version__,
        )
        return source_path

    def _generate_sync(self, request: GenerationRequest) -> GenerationResult:
        assert self._processor is not None
        assert self._vision_session is not None
        assert self._embed_session is not None
        assert self._decoder_session is not None

        has_image = request.image_absolute_path is not None
        sampling = self._vl_sampling if has_image else self._text_sampling

        self._logger.info(
            "generate start max_new_tokens=%d has_image=%s profile=%s temp=%.2f top_p=%.2f top_k=%d",
            request.max_new_tokens,
            has_image,
            "vl" if has_image else "text",
            sampling.temperature,
            sampling.top_p,
            sampling.top_k,
        )

        start = time.perf_counter()

        messages: list[dict[str, Any]] = []

        system_text = request.system_prompt or (self._vl_system_prompt if has_image else None)

        if system_text:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_text}]})

        content: list[dict[str, Any]] = []
        if has_image:
            content.append({"type": "image", "url": request.image_absolute_path})
        content.append({"type": "text", "text": request.prompt})

        messages.append({"role": "user", "content": content})

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="np",
        )

        input_ids = np.asarray(inputs["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(inputs["attention_mask"], dtype=np.int64)
        batch_size = input_ids.shape[0]

        pixel_values = None
        if has_image:
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None:
                raise RuntimeError("image provided but pixel_values not returned by processor")

        position_ids = np.tile(
            np.arange(0, input_ids.shape[-1], dtype=np.int64), (batch_size, 1)
        )

        max_new_tokens = request.max_new_tokens

        if self._provider == "cuda" and self._shared_kv:
            generated_token_ids = self._generate_loop_cuda_shared_kv(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )
        elif self._provider == "cuda":
            generated_token_ids = self._generate_loop_cuda(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )
        else:
            generated_token_ids = self._generate_loop_cpu(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )

        latency_ms = int((time.perf_counter() - start) * 1000)

        generated_np = np.asarray([generated_token_ids], dtype=np.int64)
        text = self._processor.batch_decode(generated_np, skip_special_tokens=True)[0]

        new_token_count = generated_np.shape[1]
        tokens_per_second = (new_token_count / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0
        self._logger.info(
            "generate done new_tokens=%d latency_ms=%d tok_per_s=%.2f",
            new_token_count,
            latency_ms,
            tokens_per_second,
        )

        return GenerationResult(text=text.strip(), latency_ms=latency_ms)

    def _generate_loop_cuda(
        self,
        *,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        pixel_values: np.ndarray | None,
        batch_size: int,
        max_new_tokens: int,
    ) -> list[int]:
        """Greedy decode on GPU using IOBinding.

        Note: the IOBinding MUST be created fresh per inference step. In ORT
        1.24 the CUDA buffers that back `IOBinding.get_outputs()` are released
        when the originating IOBinding is destroyed, so handing OrtValues back
        through a function return or across a `clear_binding_outputs()` call
        leaves them as dangling references (you get "Invalid rank ... Got: 0"
        on the next use). The safe pattern - which is what this loop uses -
        is: each step creates its own IOBinding, uses its outputs as inputs
        of the NEXT step (by stashing them in `past_kv`), then lets the old
        IOBinding die AFTER the next one has already consumed its outputs.
        """
        from onnxruntime import OrtValue

        assert self._embed_session is not None
        assert self._decoder_session is not None

        device = "cuda"
        device_id = 0

        # ---- PREFILL (inlined so IOBinding stays in scope) -----------------
        t_prefill = time.perf_counter()

        prompt_len = int(input_ids.shape[-1])

        # 1. Text embed lookup on GPU.
        prefill_embed_io = self._embed_session.io_binding()
        prompt_ids_ov = OrtValue.ortvalue_from_numpy(input_ids, device, device_id)
        prefill_embed_io.bind_ortvalue_input("input_ids", prompt_ids_ov)
        prefill_embed_io.bind_output("inputs_embeds", device, device_id)
        self._embed_session.run_with_iobinding(prefill_embed_io)
        inputs_embeds_ov = prefill_embed_io.get_outputs()[0]

        # 2. Vision encode + embed merge if image present.
        if pixel_values is not None:
            assert self._vision_session is not None
            t_vis = time.perf_counter()
            image_features = self._vision_session.run(
                None, {"pixel_values": pixel_values}
            )[0]
            self._logger.info(
                "vision done image_features_shape=%s vision_ms=%.1f",
                image_features.shape,
                (time.perf_counter() - t_vis) * 1000.0,
            )
            image_token_mask = input_ids == self._image_token_index
            if image_token_mask.any():
                embeds_np = inputs_embeds_ov.numpy()
                feature_dim = image_features.shape[-1]
                embeds_np[image_token_mask] = image_features.reshape(
                    -1, feature_dim
                ).astype(embeds_np.dtype)
                inputs_embeds_ov = OrtValue.ortvalue_from_numpy(
                    embeds_np, device, device_id
                )

        # 3. Empty KV cache (past_sequence_length=0) on GPU in the model dtype.
        empty_shape = [batch_size, self._num_key_value_heads, 0, self._head_dim]
        past_kv: dict[str, Any] = {}
        for layer in range(self._num_hidden_layers):
            for kv in ("key", "value"):
                empty_np = np.zeros(empty_shape, dtype=self._kv_dtype_np)
                past_kv[f"past_key_values.{layer}.{kv}"] = OrtValue.ortvalue_from_numpy(
                    empty_np, device, device_id
                )

        # 4. Decoder prefill pass. The IOBinding MUST stay alive until after
        # the FIRST decode step has consumed its outputs (the past_kv we
        # harvest below), otherwise those OrtValues become invalid. We keep
        # a reference to `prefill_dec_io` in a list below.
        prefill_dec_io = self._decoder_session.io_binding()
        prefill_dec_io.bind_ortvalue_input("inputs_embeds", inputs_embeds_ov)
        prefill_dec_io.bind_ortvalue_input(
            "attention_mask", OrtValue.ortvalue_from_numpy(attention_mask, device, device_id)
        )
        prefill_dec_io.bind_ortvalue_input(
            "position_ids", OrtValue.ortvalue_from_numpy(position_ids, device, device_id)
        )
        for name, ov in past_kv.items():
            prefill_dec_io.bind_ortvalue_input(name, ov)
        for out_name in self._decoder_output_names:
            prefill_dec_io.bind_output(out_name, device, device_id)

        self._decoder_session.run_with_iobinding(prefill_dec_io)
        outputs = prefill_dec_io.get_outputs()

        logits_np = outputs[0].numpy()
        first_token = int(logits_np[0, -1].argmax())

        past_kv = {}
        for i, out_name in enumerate(self._decoder_output_names):
            if out_name.startswith("present."):
                parts = out_name.split(".")
                past_kv[f"past_key_values.{parts[1]}.{parts[2]}"] = outputs[i]

        prefill_ms = (time.perf_counter() - t_prefill) * 1000.0
        self._logger.info(
            "prefill done prompt_len=%d prefill_ms=%.1f first_tok=%d",
            prompt_len,
            prefill_ms,
            first_token,
        )

        generated: list[int] = [first_token]
        if first_token == self._eos_token_id or max_new_tokens <= 1:
            return generated

        # ---- DECODE LOOP ---------------------------------------------------
        t_decode_start = time.perf_counter()

        # Preallocated [1,1] GPU buffers we can update numpy-side via
        # update_inplace() each step - avoids per-step host->device alloc.
        input_ids_np = np.array([[first_token]], dtype=np.int64)
        input_ids_ov = OrtValue.ortvalue_from_numpy(input_ids_np, device, device_id)

        position_np = np.array([[prompt_len]], dtype=np.int64)
        position_ids_ov = OrtValue.ortvalue_from_numpy(position_np, device, device_id)

        # Host-side mask buffer: slice up to cur_total_len each step and
        # upload (tiny, <2KB). Avoids concatenating numpy arrays per step.
        max_ctx = prompt_len + max_new_tokens + 8
        mask_buffer = np.ones((batch_size, max_ctx), dtype=np.int64)

        # This list keeps the PREVIOUS step's IOBinding alive until AFTER
        # the current step has run. Without this the outputs from step N-1
        # that we stashed in past_kv get invalidated when step N starts.
        prev_binding_holder: list[Any] = [prefill_dec_io]

        for step in range(1, max_new_tokens):
            cur_total_len = prompt_len + step

            # Update small preallocated GPU buffers in place.
            input_ids_ov.update_inplace(input_ids_np)
            position_np[0, 0] = cur_total_len - 1
            position_ids_ov.update_inplace(position_np)
            attention_mask_np = mask_buffer[:, :cur_total_len]
            attention_mask_ov = OrtValue.ortvalue_from_numpy(
                attention_mask_np, device, device_id
            )

            # Fresh embed IOBinding per step (its single output is consumed
            # immediately by the decoder run below).
            embed_io = self._embed_session.io_binding()
            embed_io.bind_ortvalue_input("input_ids", input_ids_ov)
            embed_io.bind_output("inputs_embeds", device, device_id)
            self._embed_session.run_with_iobinding(embed_io)
            inputs_embeds_ov = embed_io.get_outputs()[0]

            # Fresh decoder IOBinding per step - REQUIRED so past_kv (which
            # holds OrtValues owned by the previous step's binding) stays
            # valid through this run.
            dec_io = self._decoder_session.io_binding()
            dec_io.bind_ortvalue_input("inputs_embeds", inputs_embeds_ov)
            dec_io.bind_ortvalue_input("attention_mask", attention_mask_ov)
            dec_io.bind_ortvalue_input("position_ids", position_ids_ov)
            for name, ov in past_kv.items():
                dec_io.bind_ortvalue_input(name, ov)
            for out_name in self._decoder_output_names:
                dec_io.bind_output(out_name, device, device_id)

            self._decoder_session.run_with_iobinding(dec_io)
            outputs = dec_io.get_outputs()

            # Last-position logits only (shape [1, 1, vocab] in decode).
            next_token = int(outputs[0].numpy()[0, -1].argmax())
            generated.append(next_token)
            if next_token == self._eos_token_id:
                break

            # Rotate present.* -> past_key_values.* (still GPU-resident).
            new_past: dict[str, Any] = {}
            for i, out_name in enumerate(self._decoder_output_names):
                if out_name.startswith("present."):
                    parts = out_name.split(".")
                    new_past[f"past_key_values.{parts[1]}.{parts[2]}"] = outputs[i]
            past_kv = new_past

            # Hand over IOBinding ownership: keep THIS step's binding alive,
            # release the previous one (whose outputs we no longer need).
            prev_binding_holder[0] = dec_io

            # Next iteration's input token.
            input_ids_np[0, 0] = next_token

        decode_ms = (time.perf_counter() - t_decode_start) * 1000.0
        decoded_tokens = len(generated) - 1
        self._logger.info(
            "decode done steps=%d decode_ms=%.1f per_step_ms=%.2f",
            decoded_tokens,
            decode_ms,
            decode_ms / max(1, decoded_tokens),
        )

        # prev_binding_holder goes out of scope here; its contents are freed.
        del prev_binding_holder
        return generated

    def _prefill_cuda(
        self,
        *,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        pixel_values: np.ndarray | None,
        batch_size: int,
        device: str,
        device_id: int,
    ) -> tuple[dict[str, Any], int, int]:
        """Single prompt-ingestion pass. Returns (past_kv, first_token, prompt_len)."""
        from onnxruntime import OrtValue

        assert self._embed_session is not None
        assert self._decoder_session is not None

        prompt_len = int(input_ids.shape[-1])

        # 1. Text embed lookup on GPU.
        embed_io = self._embed_session.io_binding()
        input_ids_ov = OrtValue.ortvalue_from_numpy(input_ids, device, device_id)
        embed_io.bind_ortvalue_input("input_ids", input_ids_ov)
        embed_io.bind_output("inputs_embeds", device, device_id)
        self._embed_session.run_with_iobinding(embed_io)
        inputs_embeds_ov = embed_io.get_outputs()[0]

        # 2. Vision encode + embed merge if image present.
        if pixel_values is not None:
            assert self._vision_session is not None
            t_vis = time.perf_counter()
            image_features = self._vision_session.run(
                None, {"pixel_values": pixel_values}
            )[0]
            vision_ms = (time.perf_counter() - t_vis) * 1000.0
            self._logger.info(
                "vision done image_features_shape=%s vision_ms=%.1f",
                image_features.shape,
                vision_ms,
            )

            image_token_mask = input_ids == self._image_token_index
            if image_token_mask.any():
                embeds_np = inputs_embeds_ov.numpy()
                feature_dim = image_features.shape[-1]
                embeds_np[image_token_mask] = image_features.reshape(
                    -1, feature_dim
                ).astype(embeds_np.dtype)
                inputs_embeds_ov = OrtValue.ortvalue_from_numpy(
                    embeds_np, device, device_id
                )

        # 3. Empty KV cache (past_sequence_length=0) on GPU in the model dtype.
        empty_shape = [batch_size, self._num_key_value_heads, 0, self._head_dim]
        past_kv: dict[str, Any] = {}
        for layer in range(self._num_hidden_layers):
            for kv in ("key", "value"):
                empty_np = np.zeros(empty_shape, dtype=self._kv_dtype_np)
                past_kv[f"past_key_values.{layer}.{kv}"] = OrtValue.ortvalue_from_numpy(
                    empty_np, device, device_id
                )

        # 4. Decoder prefill pass.
        dec_io = self._decoder_session.io_binding()
        dec_io.bind_ortvalue_input("inputs_embeds", inputs_embeds_ov)
        dec_io.bind_ortvalue_input(
            "attention_mask", OrtValue.ortvalue_from_numpy(attention_mask, device, device_id)
        )
        dec_io.bind_ortvalue_input(
            "position_ids", OrtValue.ortvalue_from_numpy(position_ids, device, device_id)
        )
        for name, ov in past_kv.items():
            dec_io.bind_ortvalue_input(name, ov)
        for out_name in self._decoder_output_names:
            dec_io.bind_output(out_name, device, device_id)

        self._decoder_session.run_with_iobinding(dec_io)
        outputs = dec_io.get_outputs()

        # First-token selection from last prompt position.
        logits_np = outputs[0].numpy()
        first_token = int(logits_np[0, -1].argmax())

        # Harvest present.* as new past_kv.
        new_past: dict[str, Any] = {}
        for i, out_name in enumerate(self._decoder_output_names):
            if out_name.startswith("present."):
                parts = out_name.split(".")
                new_past[f"past_key_values.{parts[1]}.{parts[2]}"] = outputs[i]

        return new_past, first_token, prompt_len

    def _generate_loop_cuda_shared_kv(
        self,
        *,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        pixel_values: np.ndarray | None,
        batch_size: int,
        max_new_tokens: int,
    ) -> list[int]:
        """Shared-KV decoder variant: preallocated KV at max_context, CUDA-graph friendly.

        Needs the decoder ONNX to have been patched so every GroupQueryAttention
        node has past_present_share_buffer=1. That lets us bind the same GPU
        buffer as both past_key_values.L.K (input) and present.L.K (output)
        for every layer, giving ORT fully static shapes per step and unlocking
        CUDA graph capture.
        """
        from onnxruntime import OrtValue

        assert self._embed_session is not None
        assert self._decoder_session is not None
        assert self._prefill_decoder_session is not None

        device = "cuda"
        device_id = 0
        MAX_CTX = self._max_context

        prompt_len = int(input_ids.shape[-1])
        if prompt_len + max_new_tokens > MAX_CTX:
            raise RuntimeError(
                f"prompt({prompt_len}) + max_new_tokens({max_new_tokens}) > model_max_context({MAX_CTX})"
            )

        # ---- Preallocate full-size GPU buffers once for the whole request. ----
        # KV cache is the big one: [1, kv_heads, MAX_CTX, head_dim] per layer per K/V.
        kv_shape = [batch_size, self._num_key_value_heads, MAX_CTX, self._head_dim]
        past_kv: dict[str, Any] = {}
        for layer in range(self._num_hidden_layers):
            for kv in ("key", "value"):
                zeros = np.zeros(kv_shape, dtype=self._kv_dtype_np)
                past_kv[f"past_key_values.{layer}.{kv}"] = OrtValue.ortvalue_from_numpy(
                    zeros, device, device_id
                )

        # Attention_mask is always length MAX_CTX; host-side buffer we mutate.
        mask_host = np.zeros((batch_size, MAX_CTX), dtype=np.int64)
        mask_host[:, :prompt_len] = 1  # prompt positions valid
        mask_ov = OrtValue.ortvalue_from_numpy(mask_host, device, device_id)

        # ---- 1) PREFILL via the non-graph-captured session. ----
        t_prefill = time.perf_counter()

        # Embed + vision merge (same as other path).
        embed_io = self._embed_session.io_binding()
        prompt_ids_ov = OrtValue.ortvalue_from_numpy(input_ids, device, device_id)
        embed_io.bind_ortvalue_input("input_ids", prompt_ids_ov)
        embed_io.bind_output("inputs_embeds", device, device_id)
        self._embed_session.run_with_iobinding(embed_io)
        inputs_embeds_ov = embed_io.get_outputs()[0]

        if pixel_values is not None:
            assert self._vision_session is not None
            t_vis = time.perf_counter()
            image_features = self._vision_session.run(None, {"pixel_values": pixel_values})[0]
            self._logger.info(
                "vision done image_features_shape=%s vision_ms=%.1f",
                image_features.shape,
                (time.perf_counter() - t_vis) * 1000.0,
            )
            image_token_mask = input_ids == self._image_token_index
            if image_token_mask.any():
                embeds_np = inputs_embeds_ov.numpy()
                feature_dim = image_features.shape[-1]
                embeds_np[image_token_mask] = image_features.reshape(-1, feature_dim).astype(
                    embeds_np.dtype
                )
                inputs_embeds_ov = OrtValue.ortvalue_from_numpy(embeds_np, device, device_id)

        # Prefill decoder call on the session WITHOUT CUDA graph capture,
        # because prompt_len (and thus the sequence dim of inputs_embeds) varies.
        prefill_io = self._prefill_decoder_session.io_binding()
        prefill_io.bind_ortvalue_input("inputs_embeds", inputs_embeds_ov)
        prefill_io.bind_ortvalue_input("attention_mask", mask_ov)
        prefill_io.bind_ortvalue_input(
            "position_ids", OrtValue.ortvalue_from_numpy(position_ids, device, device_id)
        )
        # Bind the same preallocated KV buffers as both past (inputs) and
        # present (outputs). past_present_share_buffer=1 means the op writes
        # in-place so these refer to the same memory.
        for layer in range(self._num_hidden_layers):
            for kv in ("key", "value"):
                in_name = f"past_key_values.{layer}.{kv}"
                out_name = f"present.{layer}.{kv}"
                ov = past_kv[in_name]
                prefill_io.bind_ortvalue_input(in_name, ov)
                prefill_io.bind_ortvalue_output(out_name, ov)
        prefill_io.bind_output("logits", device, device_id)

        self._prefill_decoder_session.run_with_iobinding(prefill_io)
        logits_ov = prefill_io.get_outputs()[0]
        # Only the LAST prompt position's logits matter for the first token.
        logits_np = logits_ov.numpy()
        first_token = int(logits_np[0, -1].argmax())
        prefill_ms = (time.perf_counter() - t_prefill) * 1000.0
        self._logger.info(
            "prefill done prompt_len=%d prefill_ms=%.1f first_tok=%d",
            prompt_len,
            prefill_ms,
            first_token,
        )

        generated: list[int] = [first_token]
        if first_token == self._eos_token_id or max_new_tokens <= 1:
            return generated

        # ---- 2) DECODE loop using the CUDA-graph-captured session. ----
        t_decode = time.perf_counter()

        # [1,1] static-shape buffers we update in place every step.
        one_id = np.array([[first_token]], dtype=np.int64)
        input_id_ov = OrtValue.ortvalue_from_numpy(one_id, device, device_id)
        one_pos = np.array([[prompt_len]], dtype=np.int64)
        position_ov = OrtValue.ortvalue_from_numpy(one_pos, device, device_id)

        # Also preallocate the embed output for [1,1,hidden_dim] so every step
        # writes into the same GPU buffer.
        hidden_dim = self._config.text_config.hidden_size if self._config is not None else 3072
        embed_out_shape = [1, 1, hidden_dim]
        embeds_ov = OrtValue.ortvalue_from_shape_and_type(
            embed_out_shape, self._embed_dtype_np, device, device_id
        )

        # Preallocate the decoder logits output on GPU as [1, 1, vocab_size].
        vocab_size = self._config.text_config.vocab_size if self._config is not None else 131072
        logits_dtype = self._onnx_type_to_numpy(
            self._find_input_type_output(self._decoder_session, "logits"),
            default=np.float32,
        )
        decode_logits_ov = OrtValue.ortvalue_from_shape_and_type(
            [1, 1, vocab_size], logits_dtype, device, device_id
        )

        # Reusable embed/decoder IOBindings.
        embed_io_d = self._embed_session.io_binding()
        dec_io = self._decoder_session.io_binding()

        # Bind the inputs/outputs of the decoder that don't change each step.
        # (KV buffers and attention_mask OrtValue are the same objects across
        # steps; only attention_mask CONTENTS change via update_inplace.)
        dec_io.bind_ortvalue_input("inputs_embeds", embeds_ov)
        dec_io.bind_ortvalue_input("attention_mask", mask_ov)
        dec_io.bind_ortvalue_input("position_ids", position_ov)
        for layer in range(self._num_hidden_layers):
            for kv in ("key", "value"):
                in_name = f"past_key_values.{layer}.{kv}"
                out_name = f"present.{layer}.{kv}"
                ov = past_kv[in_name]
                dec_io.bind_ortvalue_input(in_name, ov)
                dec_io.bind_ortvalue_output(out_name, ov)
        dec_io.bind_ortvalue_output("logits", decode_logits_ov)

        for step in range(1, max_new_tokens):
            cur_valid_len = prompt_len + step  # after appending the new token

            # Extend attention_mask on host + re-upload to the SAME GPU buffer.
            mask_host[:, cur_valid_len - 1] = 1
            mask_ov.update_inplace(mask_host)

            # Update input_id and position_id in place.
            one_id[0, 0] = generated[-1]
            input_id_ov.update_inplace(one_id)
            one_pos[0, 0] = cur_valid_len - 1  # 0-indexed position of the new token
            position_ov.update_inplace(one_pos)

            # Embed: bind input, run, output into embeds_ov (preallocated).
            embed_io_d.clear_binding_inputs()
            embed_io_d.clear_binding_outputs()
            embed_io_d.bind_ortvalue_input("input_ids", input_id_ov)
            embed_io_d.bind_ortvalue_output("inputs_embeds", embeds_ov)
            self._embed_session.run_with_iobinding(embed_io_d)

            # Decoder: all bindings already set above, just re-run. With CUDA
            # graph capture enabled, the first run warms up and subsequent
            # runs replay the captured graph.
            self._decoder_session.run_with_iobinding(dec_io)

            # Copy the small decode logits [1,1,vocab] to host and argmax.
            logits_step = decode_logits_ov.numpy()
            next_token = int(logits_step[0, 0].argmax())
            generated.append(next_token)
            if next_token == self._eos_token_id:
                break

        decode_ms = (time.perf_counter() - t_decode) * 1000.0
        decoded = len(generated) - 1
        self._logger.info(
            "decode done steps=%d decode_ms=%.1f per_step_ms=%.2f (shared_kv+graph)",
            decoded,
            decode_ms,
            decode_ms / max(1, decoded),
        )
        return generated

    @staticmethod
    def _find_input_type_output(session: Any, name: str) -> str | None:
        for meta in session.get_outputs():
            if meta.name == name:
                return meta.type
        return None

    def _generate_loop_cpu(
        self,
        *,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        pixel_values: np.ndarray | None,
        batch_size: int,
        max_new_tokens: int,
    ) -> list[int]:
        """Greedy decode on CPU. Plain `session.run` - no IOBinding needed."""
        assert self._embed_session is not None
        assert self._decoder_session is not None
        assert self._vision_session is not None

        past_key_values: dict[str, np.ndarray] = {
            f"past_key_values.{layer}.{kv}": np.zeros(
                [batch_size, self._num_key_value_heads, 0, self._head_dim],
                dtype=self._kv_dtype_np,
            )
            for layer in range(self._num_hidden_layers)
            for kv in ("key", "value")
        }

        generated: list[int] = []
        is_first_iteration = True

        for _ in range(max_new_tokens):
            inputs_embeds = self._embed_session.run(None, {"input_ids": input_ids})[0]

            if is_first_iteration and pixel_values is not None:
                image_features = self._vision_session.run(
                    None, {"pixel_values": pixel_values}
                )[0]
                image_token_mask = input_ids == self._image_token_index
                if image_token_mask.any():
                    feature_dim = image_features.shape[-1]
                    inputs_embeds[image_token_mask] = image_features.reshape(
                        -1, feature_dim
                    ).astype(inputs_embeds.dtype)

            outputs = self._decoder_session.run(
                None,
                {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    **past_key_values,
                },
            )

            logits = outputs[0]
            next_token = int(logits[0, -1].argmax())
            generated.append(next_token)

            if next_token == self._eos_token_id:
                break

            new_past: dict[str, np.ndarray] = {}
            for i, out_name in enumerate(self._decoder_output_names):
                if out_name.startswith("present."):
                    parts = out_name.split(".")
                    layer_idx, kv = parts[1], parts[2]
                    new_past[f"past_key_values.{layer_idx}.{kv}"] = outputs[i]
            past_key_values = new_past

            input_ids = np.asarray([[next_token]], dtype=np.int64)
            attention_mask = np.concatenate(
                [attention_mask, np.ones((batch_size, 1), dtype=attention_mask.dtype)],
                axis=-1,
            )
            position_ids = position_ids[:, -1:] + 1
            is_first_iteration = False

        return generated

    def _downscale_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        longest = max(width, height)
        if longest <= self._image_max_side:
            return image
        scale = self._image_max_side / longest
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(new_size, Image.Resampling.LANCZOS)
