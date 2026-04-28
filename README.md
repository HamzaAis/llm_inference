# llm-inferance

A clean-architecture FastAPI server that serves the multimodal `Qwen3.5-2B`
vision-language model on CPU via `llama.cpp` (GGUF). It accepts a text prompt
and an optional image, generates a text response, persists every inference in
SQLite, and stores uploaded images on disk.

## Architecture

```
client
  |
  v
presentation/endpoints   (FastAPI routers, multipart + JSON, Pydantic schemas)
  |
  v
presentation/schemas     (Pydantic request/response models)
  |
  v
application/services     (business logic, dataclass DTOs)
  |
  v
application/dataclasses  (frozen DTOs moving up/down the stack)
  |
  v
domain/repositories      (SQLAlchemy CRUD + filesystem ops)
  |
  v
domain/entities          (SQLAlchemy ORM)
  |
  v
data/llm_inferance.db    +    files/<uuid>.<ext>
```

Strict rules enforced in code:

- **Endpoints** depend only on **services** (via `infrastructure/config/dependency.py`).
- **Services** depend on other services and repositories.
- **Endpoints** never import repositories or ORM models.
- **Endpoints** speak Pydantic; **services** speak dataclasses; **repositories** speak ORM models.
- No comments, no docstrings, anywhere.

## Project Layout

```
llm_inferance/
  data/                     sqlite db (gitignored)
  files/                    uploaded images (gitignored)
  hf_cache/                 hugging face model cache (gitignored)
  pyproject.toml
  src/
    main.py                 fastapi app + lifespan
    application/
      dataclasses/          dtos
      services/             inference / image / model services
    domain/
      enums/
      entities/             sqlalchemy orm
      repositories/         sqlalchemy + file ops
    infrastructure/
      config/               settings, database, dependency injection
      middleware/           access log, rate limit, request id
    presentation/
      endpoints/            inferences
      schemas/              pydantic request/response models
```

## Setup

1. Install dependencies (pulls the `llama-cpp-python` CPU wheel and the HF Hub client):

   ```powershell
   uv sync
   ```

2. Run the server:

   ```powershell
   uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
   ```

   The first launch downloads the quantized GGUF (`Qwen_Qwen3.5-2B-Q4_K_M.gguf`,
   ~1.33 GB) plus the vision projector (`mmproj-Qwen_Qwen3.5-2B-f16.gguf`,
   ~0.67 GB) from `bartowski/Qwen_Qwen3.5-2B-GGUF` into `hf_cache/`. Subsequent
   launches reuse the cache; model load is typically under 10 seconds on CPU.

3. Open the interactive docs at <http://localhost:8000/docs>.

## Endpoints

| Method | Path                     | Description                                                                                                   |
| ------ | ------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `POST` | `/inferences`            | multipart: `prompt` (required), `image` (optional), `max_new_tokens` (optional). Generates text and persists. |
| `GET`  | `/inferences`            | Paginated list of past inferences (`?page=`, `?page_size=`).                                                  |
| `GET`  | `/inferences/{id}`       | Full inference detail.                                                                                        |
| `GET`  | `/inferences/{id}/image` | Returns the stored image (404 if no image).                                                                   |

### Quick examples

Text-only:

```powershell
curl.exe -X POST http://localhost:8000/inferences `
  -F "prompt=Give me a one-sentence intro to LLMs."
```

Image + text:

```powershell
curl.exe -X POST http://localhost:8000/inferences `
  -F "prompt=Describe what you see." `
  -F "image=@C:\path\to\photo.jpg"
```

List + detail:

```powershell
curl.exe http://localhost:8000/inferences?page=1
curl.exe http://localhost:8000/inferences/1
curl.exe http://localhost:8000/inferences/1/image -o downloaded.jpg
```

## Configuration

All settings can be overridden via environment variables (or a `.env` file
at the project root). Key defaults:

| Variable                 | Default                                                                  |
| ------------------------ | ------------------------------------------------------------------------ |
| `APP_HOST`               | `0.0.0.0`                                                                |
| `APP_PORT`               | `8000`                                                                   |
| `MODEL_NAME`             | `bartowski/Qwen_Qwen3.5-2B-GGUF`                                         |
| `MODEL_GGUF_FILENAME`    | `Qwen_Qwen3.5-2B-Q4_K_M.gguf`                                            |
| `MODEL_MMPROJ_FILENAME`  | `mmproj-Qwen_Qwen3.5-2B-f16.gguf`                                        |
| `MODEL_N_CTX`            | `8192` (context window)                                                  |
| `MODEL_N_THREADS`        | `0` (0 = all logical cores)                                              |
| `MODEL_N_GPU_LAYERS`     | `0` (increase when using a GPU-accelerated rebuild)                      |
| `MODEL_VERBOSE`          | `False` (set to `true` to see `llama.cpp` init/timing logs)              |
| `DEFAULT_MAX_NEW_TOKENS` | `512`                                                                    |
| `MAX_IMAGE_MB`           | `10`                                                                     |
| `IMAGE_MAX_SIDE`         | `768` (px on the longest side; safety net downscaling before the mmproj) |
| `ALLOWED_IMAGE_MIMES`    | `image/png, image/jpeg, image/webp`                                      |
| `RATE_LIMIT_PER_MINUTE`  | `30`                                                                     |
| `DATABASE_URL`           | `sqlite+aiosqlite:///./data/llm_inferance.db`                            |

### Sampling (Qwen-recommended, overridable)

The service picks a different sampling profile depending on whether the request
includes an image. These defaults are the **official Qwen3.5 non-thinking
settings** and are the single biggest lever against the 2B model's
repetition/thinking loops.

| Variable                                        | Default (text) | Default (VL) |
| ----------------------------------------------- | -------------- | ------------ |
| `TEXT_TEMPERATURE` / `VL_TEMPERATURE`           | `1.0`          | `0.7`        |
| `TEXT_TOP_P` / `VL_TOP_P`                       | `1.0`          | `0.8`        |
| `TEXT_TOP_K` / `VL_TOP_K`                       | `20`           | `20`         |
| `TEXT_MIN_P` / `VL_MIN_P`                       | `0.0`          | `0.0`        |
| `TEXT_PRESENCE_PENALTY` / `VL_PRESENCE_PENALTY` | `2.0`          | `1.5`        |
| `TEXT_REPEAT_PENALTY` / `VL_REPEAT_PENALTY`     | `1.0`          | `1.0`        |

The `VL_*` profile is used whenever an `image` multipart part is attached; the
`TEXT_*` profile is used otherwise. The selected profile and its temperature /
top-p / top-k / presence-penalty are logged per request.

## Notes

- Inference uses Qwen3.5's **official non-thinking sampling** (see the
  "Sampling" table above). Greedy decoding (`temperature=0.0`) is intentionally
  avoided on the 2B variant because Qwen's own model card warns it is prone to
  repetition / thinking loops under greedy.
- The model lives as a process-wide singleton owned by the FastAPI lifespan;
  all blocking `llama.cpp` calls run in a worker thread so the event loop
  stays responsive.
- Thinking mode is off by default (matches the Qwen3.5-2B default). The model
  still emits an empty `<think></think>` block before its answer — that is the
  documented non-thinking behaviour.
- Qwen3.5-2B's hybrid architecture (Gated DeltaNet + sparse Gated Attention)
  gives near-constant KV-cache growth with context length, so `MODEL_N_CTX`
  can be raised without the quadratic memory blow-up that a plain Qwen of the
  same size would have.

### One-time cleanup of old cached weights

If earlier experiments left weights in `hf_cache/` that the current backend no
longer uses, you can reclaim the disk. Check what is actually there first:

```powershell
Get-ChildItem .\hf_cache\ -Directory | Select-Object Name, @{Name="GB";Expression={[math]::Round((Get-ChildItem $_.FullName -Recurse | Measure-Object Length -Sum).Sum / 1GB, 2)}}
```

Then remove any folder you no longer need, for example:

```powershell
# Old transformers-backed weights from before the llama.cpp migration (~4.55 GB)
Remove-Item -Recurse -Force .\hf_cache\models--Qwen--Qwen3.5-2B

# Old unsloth GGUF from before the bartowski switch (~2 GB)
Remove-Item -Recurse -Force .\hf_cache\models--unsloth--Qwen3.5-2B-GGUF
```

### Optional: rebuilding `llama-cpp-python` with GPU acceleration

The default PyPI wheel is CPU-only and works everywhere (Intel / AMD / ARM).
If you want to offload layers to a GPU, reinstall with a backend-specific
build flag. All examples below are for PowerShell on Windows.

CUDA 12.x via prebuilt wheel (no local toolchain needed):

```powershell
uv pip install --force-reinstall --no-deps llama-cpp-python `
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

Build locally with CUDA (requires CUDA Toolkit + MSVC Build Tools):

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
uv pip install --force-reinstall --no-binary=:all: llama-cpp-python
```

Build locally with Vulkan (works on Intel Iris Xe + NVIDIA + AMD GPUs,
requires the Vulkan SDK):

```powershell
$env:CMAKE_ARGS = "-DGGML_VULKAN=on"
uv pip install --force-reinstall --no-binary=:all: llama-cpp-python
```

After reinstalling, raise `MODEL_N_GPU_LAYERS` (for example to `99`) to push
as many layers as fit onto the GPU. On a 2 GB dGPU only a handful of layers
will fit; on a shared-memory iGPU via Vulkan you can usually fit the whole
model.
