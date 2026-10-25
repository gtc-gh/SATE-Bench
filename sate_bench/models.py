"""Model backends: Gemini API, OpenAI API and local open-source models served with vLLM.

Every backend exposes the same two entry points:
  * `backend.new_chat(system_prompt)` -> chat session with `.send(image_paths, text) -> (answer, thought)`
    (multi-turn, used by the Active setting; the full interaction history is kept)
  * `backend.generate(system_prompt, image_paths, text) -> (answer, thought)`
    (single turn, used by the Passive setting)
"""

import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .config import API_MAX_RETRIES, API_RETRY_SLEEP, IMAGE_RESOLUTION, get_gemini_key, get_openai_key
from .data import image_to_base64, image_to_jpeg_bytes, load_image
from .parsing import split_thinking


class GenerationError(RuntimeError):
    """Raised when a backend fails to produce an output."""


def _with_retries(fn, what: str):
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - API clients raise many different error types
            print(f"[{what}] API error (attempt {attempt}/{API_MAX_RETRIES}): {e}")
            if attempt < API_MAX_RETRIES:
                time.sleep(API_RETRY_SLEEP)
    raise GenerationError(f"{what}: failed after {API_MAX_RETRIES} attempts")


# =============================================================================
# Gemini
# =============================================================================
class GeminiBackend:
    is_api = True

    def __init__(self, model: str, api_key: Optional[str] = None, temperature: float = 0.0):
        from google import genai
        api_key = api_key or get_gemini_key()
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY to evaluate Gemini models.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def _config(self, system_prompt: str):
        from google.genai import types
        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            thinking_config={"include_thoughts": True},
        )

    @staticmethod
    def _parts(image_paths: Sequence[str], text: str) -> list:
        from google.genai import types
        parts = [types.Part.from_bytes(data=image_to_jpeg_bytes(p, IMAGE_RESOLUTION), mime_type="image/jpeg")
                 for p in image_paths]
        return parts + [text]

    @staticmethod
    def _split(response) -> Tuple[str, str]:
        answer, thought = "", ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if getattr(part, "text", None):
                    if getattr(part, "thought", False):
                        thought += part.text
                    else:
                        answer += part.text
        return answer.strip(), thought.strip()

    def new_chat(self, system_prompt: str) -> "GeminiChat":
        return GeminiChat(self, system_prompt)

    def generate(self, system_prompt: str, image_paths: Sequence[str], text: str) -> Tuple[str, str]:
        contents = self._parts(image_paths, text)
        response = _with_retries(
            lambda: self.client.models.generate_content(
                model=self.model, contents=contents, config=self._config(system_prompt)),
            self.model)
        return self._split(response)


class GeminiChat:
    def __init__(self, backend: GeminiBackend, system_prompt: str):
        self.backend = backend
        self.chat = backend.client.chats.create(model=backend.model, config=backend._config(system_prompt))

    def send(self, image_paths: Sequence[str], text: str) -> Tuple[str, str]:
        message = self.backend._parts(image_paths, text) if image_paths else text
        response = _with_retries(lambda: self.chat.send_message(message), self.backend.model)
        return self.backend._split(response)


# =============================================================================
# OpenAI
# =============================================================================
class OpenAIBackend:
    is_api = True

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        from openai import OpenAI
        api_key = api_key or get_openai_key()
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY to evaluate OpenAI models.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @staticmethod
    def user_message(image_paths: Sequence[str], text: str) -> dict:
        if not image_paths:
            return {"role": "user", "content": text}
        content = [{"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(p, IMAGE_RESOLUTION)}",
                                  "detail": "high"}}
                   for p in image_paths]
        content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    def complete(self, messages: List[dict]) -> Tuple[str, str]:
        # GPT-5 family models only support the default temperature, so it is not set here.
        response = _with_retries(
            lambda: self.client.chat.completions.create(model=self.model, messages=messages), self.model)
        return (response.choices[0].message.content or "").strip(), ""

    def new_chat(self, system_prompt: str) -> "OpenAIChat":
        return OpenAIChat(self, system_prompt)

    def generate(self, system_prompt: str, image_paths: Sequence[str], text: str) -> Tuple[str, str]:
        return self.complete([{"role": "system", "content": system_prompt}, self.user_message(image_paths, text)])


class OpenAIChat:
    def __init__(self, backend: OpenAIBackend, system_prompt: str):
        self.backend = backend
        self.messages = [{"role": "system", "content": system_prompt}]

    def send(self, image_paths: Sequence[str], text: str) -> Tuple[str, str]:
        self.messages.append(self.backend.user_message(image_paths, text))
        try:
            answer, thought = self.backend.complete(self.messages)
        except GenerationError:
            self.messages.pop()
            raise
        self.messages.append({"role": "assistant", "content": answer})
        return answer, thought


# =============================================================================
# vLLM (open-source models)
# =============================================================================
class VLLMBackend:
    """Local inference with vLLM. Optionally applies a LoRA adapter (e.g. a GRPO checkpoint)."""

    is_api = False

    def __init__(self, model: str, max_images: int, tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9, max_model_len: int = 16384,
                 max_num_seqs: Optional[int] = None, trust_remote_code: bool = False,
                 temperature: float = 0.0, max_tokens: int = 8192, repetition_penalty: float = 1.05,
                 seed: int = 42, enable_thinking: Optional[bool] = None,
                 lora_path: Optional[str] = None, max_lora_rank: int = 64):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        llm_kwargs = dict(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": max_images},
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
        )
        if max_num_seqs:
            llm_kwargs["max_num_seqs"] = max_num_seqs
        self.lora_request = None
        if lora_path:
            from vllm.lora.request import LoRARequest
            llm_kwargs.update(enable_lora=True, max_lora_rank=max_lora_rank)
            self.lora_request = LoRARequest(lora_name="adapter", lora_int_id=1, lora_path=str(lora_path))

        print(f"Loading {model} with vLLM ...")
        self.llm = LLM(**llm_kwargs)
        # A GRPO checkpoint folder also stores the processor / chat template used during training.
        processor_source = lora_path if lora_path and _has_processor(lora_path) else model
        self.processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=trust_remote_code)
        self.sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, repetition_penalty=repetition_penalty, seed=seed)
        self.enable_thinking = enable_thinking
        self.model = model

    def build_request(self, messages: List[dict]) -> dict:
        template_kwargs = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
        images = [c["image"] for m in messages if isinstance(m["content"], list)
                  for c in m["content"] if c.get("type") == "image"]
        return {"prompt": prompt, "multi_modal_data": {"image": images} if images else {}}

    def run(self, requests: List[dict], use_tqdm: bool = False) -> List[str]:
        outputs = self.llm.generate(requests, sampling_params=self.sampling_params,
                                    lora_request=self.lora_request, use_tqdm=use_tqdm)
        return [o.outputs[0].text.strip() for o in outputs]

    @staticmethod
    def user_message(image_paths: Sequence[str], text: str) -> dict:
        content = [{"type": "image", "image": load_image(p, IMAGE_RESOLUTION)} for p in image_paths]
        content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    def new_chat(self, system_prompt: str) -> "VLLMChat":
        return VLLMChat(self, system_prompt)

    def generate(self, system_prompt: str, image_paths: Sequence[str], text: str) -> Tuple[str, str]:
        return self.generate_batch([(system_prompt, image_paths, text)])[0]

    def generate_batch(self, inputs: List[Tuple[str, Sequence[str], str]]) -> List[Tuple[str, str]]:
        """Batched single-turn generation. Falls back to one-by-one generation if the batch fails."""
        requests = [self.build_request([{"role": "system", "content": s}, self.user_message(imgs, t)])
                    for s, imgs, t in inputs]
        try:
            texts = self.run(requests, use_tqdm=True)
        except Exception as e:  # noqa: BLE001
            print(f"Batched generation failed ({e}); falling back to sample-by-sample generation.")
            texts = []
            for req in requests:
                try:
                    texts.append(self.run([req])[0])
                except Exception as e_single:  # noqa: BLE001
                    texts.append(f"ERROR: {e_single}")
        return [split_thinking(t) if not t.startswith("ERROR:") else (t, "") for t in texts]


class VLLMChat:
    def __init__(self, backend: VLLMBackend, system_prompt: str):
        self.backend = backend
        self.messages = [{"role": "system", "content": system_prompt}]

    def send(self, image_paths: Sequence[str], text: str) -> Tuple[str, str]:
        self.messages.append(self.backend.user_message(image_paths, text))
        try:
            raw = self.backend.run([self.backend.build_request(self.messages)])[0]
        except Exception as e:  # noqa: BLE001
            raise GenerationError(str(e)) from e
        # The full raw output (including any reasoning trace) is kept in the history.
        self.messages.append({"role": "assistant", "content": raw})
        return split_thinking(raw)


def _has_processor(path) -> bool:
    p = Path(path)
    return (p / "preprocessor_config.json").exists() or (p / "processor_config.json").exists()


def build_backend(args, setting: str):
    """Creates a backend from parsed command-line arguments (see evaluation/common.py)."""
    if args.backend == "gemini":
        return GeminiBackend(args.model)
    if args.backend == "openai":
        return OpenAIBackend(args.model, base_url=args.openai_base_url)
    if args.backend == "vllm":
        return VLLMBackend(
            model=args.model,
            max_images=10 if setting == "active" else 8,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            trust_remote_code=args.trust_remote_code,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            enable_thinking=args.enable_thinking,
            lora_path=args.lora_path,
            max_lora_rank=args.max_lora_rank,
        )
    raise ValueError(f"Unknown backend: {args.backend}")
