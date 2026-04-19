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
    ) -> None:
        self._model_name = model_name
        self._hf_cache_dir = hf_cache_dir
        self._provider = provider
        self._precision = precision
        self._verbose = verbose
        self._image_max_side = max(1, image_max_side)
        self._text_sampling = text_sampling
        self._vl_sampling = vl_sampling
        self._loaded: bool = False
        self._logger = logging.getLogger(__name__)

        self._vision_session: Any | None = None
        self._embed_session: Any | None = None
        self._decoder_session: Any | None = None

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
        self._decoder_session = ort.InferenceSession(
            decoder_model_path, sess_options=sess_options, providers=providers, provider_options=provider_options
        )

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

    def _get_provider_options(self) -> list[dict[str, Any]]:
        """Get provider-specific options for CUDA optimization."""
        if self._provider == "cuda":
            # Tuned for small-VRAM GPUs (e.g. 6GB RTX 3050). kSameAsRequested
            # avoids doubling the arena on every allocation, which matters
            # because the q4 decoder + fp16 embed + vision encoder already
            # sit close to the VRAM ceiling on this card.
            cuda_options = {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            }
            return [cuda_options, {}]
        return [{}]

    def _unload_sync(self) -> None:
        import gc

        self._vision_session = None
        self._embed_session = None
        self._decoder_session = None
        self._processor = None
        self._config = None
        self._decoder_output_names = []
        self._loaded = False
        gc.collect()

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

        system_text = None
        if request.system_prompt:
            system_text = request.system_prompt
        elif has_image:
            system_text = (
                "You are a precise document extraction assistant. "
                "Extract ONLY the exact text and values visible in the image. "
                "Do NOT guess, infer, hallucinate, or invent any information. "
                "Do NOT repeat placeholder patterns like 'A1, A2, A3...' or similar sequences. "
                "For missing values, use proper JSON null: \"field_name\": null "
                "Do NOT invent field names with underscores like '__null__'. "
                "Read all numbers and text exactly as shown - verify each character. "
                "BAD EXAMPLE: 'A1, A2, A3...' is never correct. "
                "Return clean, valid JSON without invented placeholder keys."
            )

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

        if self._provider == "cuda":
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

        The KV cache lives entirely on device: each step's `present.*` outputs
        are rebound as the next step's `past_key_values.*` inputs, so nothing
        but the final-token logits ever travels GPU->CPU per step.
        """
        from onnxruntime import OrtValue

        assert self._embed_session is not None
        assert self._decoder_session is not None
        assert self._vision_session is not None

        device = "cuda"
        device_id = 0

        # Empty KV cache on GPU in the model's native dtype.
        past_kv_values: dict[str, Any] = {}
        empty_shape = [batch_size, self._num_key_value_heads, 0, self._head_dim]
        for layer in range(self._num_hidden_layers):
            for kv in ("key", "value"):
                name = f"past_key_values.{layer}.{kv}"
                empty_np = np.zeros(empty_shape, dtype=self._kv_dtype_np)
                past_kv_values[name] = OrtValue.ortvalue_from_numpy(empty_np, device, device_id)

        generated: list[int] = []
        is_first_iteration = True

        for _ in range(max_new_tokens):
            # --- 1. Embed lookup (tiny; stays on GPU). ---
            embed_io = self._embed_session.io_binding()
            input_ids_ov = OrtValue.ortvalue_from_numpy(input_ids, device, device_id)
            embed_io.bind_ortvalue_input("input_ids", input_ids_ov)
            embed_io.bind_output("inputs_embeds", device, device_id)
            self._embed_session.run_with_iobinding(embed_io)
            inputs_embeds_ov = embed_io.get_outputs()[0]

            # --- 2. Vision merge (first step only, when an image is present). ---
            if is_first_iteration and pixel_values is not None:
                image_features = self._vision_session.run(
                    None, {"pixel_values": pixel_values}
                )[0]
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

            # --- 3. Decoder step with IOBinding. Past KV stays on GPU. ---
            dec_io = self._decoder_session.io_binding()
            dec_io.bind_ortvalue_input("inputs_embeds", inputs_embeds_ov)
            dec_io.bind_ortvalue_input(
                "attention_mask",
                OrtValue.ortvalue_from_numpy(attention_mask, device, device_id),
            )
            dec_io.bind_ortvalue_input(
                "position_ids",
                OrtValue.ortvalue_from_numpy(position_ids, device, device_id),
            )
            for name, ov in past_kv_values.items():
                dec_io.bind_ortvalue_input(name, ov)
            for out_name in self._decoder_output_names:
                dec_io.bind_output(out_name, device, device_id)

            self._decoder_session.run_with_iobinding(dec_io)
            outputs = dec_io.get_outputs()

            # --- 4. Sample greedy from last position only. ---
            logits_np = outputs[0].numpy()  # [batch, seq, vocab]
            next_token = int(logits_np[0, -1].argmax())
            generated.append(next_token)

            if next_token == self._eos_token_id:
                break

            # --- 5. Rotate present.* -> past_key_values.* (still on GPU). ---
            new_past: dict[str, Any] = {}
            for i, out_name in enumerate(self._decoder_output_names):
                if out_name.startswith("present."):
                    parts = out_name.split(".")
                    layer_idx, kv = parts[1], parts[2]
                    new_past[f"past_key_values.{layer_idx}.{kv}"] = outputs[i]
            past_kv_values = new_past

            # --- 6. Prepare next step. ---
            input_ids = np.asarray([[next_token]], dtype=np.int64)
            attention_mask = np.concatenate(
                [attention_mask, np.ones((batch_size, 1), dtype=attention_mask.dtype)],
                axis=-1,
            )
            position_ids = position_ids[:, -1:] + 1
            is_first_iteration = False

        return generated

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
