"""
Qwen3-VL-8B-Instruct inference wrapper.

Loads the model once and provides a simple interface for sending
image + text prompts and receiving text responses.
"""
import logging
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from src.config import MODEL_NAME, MODEL_LOCAL_PATH, MAX_NEW_TOKENS, TEMPERATURE, TOP_P

logger = logging.getLogger(__name__)


class VLMClient:
    """Wrapper around Qwen3-VL for image+text inference."""

    def __init__(self, model_path: str | None = None, device: str = "auto"):
        self.model_path = model_path or MODEL_LOCAL_PATH or MODEL_NAME
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        """Load model and processor into memory."""
        if self.model is not None:
            return

        logger.info(f"Loading model from {self.model_path}...")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
        )
        logger.info("Model loaded successfully.")

    def query(
        self,
        image_path: str | Path,
        prompt: str,
        example_images: list[tuple[str | Path, str]] | None = None,
        deterministic: bool = False,
    ) -> str:
        """
        Send an image + text prompt to the model and return the text response.

        Args:
            image_path: Path to the main image being analyzed.
            prompt: Text prompt to send alongside the image.
            example_images: Optional few-shot reference images as a list of
                (path, caption) tuples, prepended before the main image.
                Example: [("examples/elevator/ex1.png", "This is an elevator.")]
            deterministic: If True, use greedy decoding (do_sample=False).
                Recommended for classification / one-of-N answers.

        Returns:
            The model's text response.
        """
        self.load()

        content = []
        for ex_path, caption in example_images or []:
            content.append({"type": "image", "image": f"file://{Path(ex_path).resolve()}"})
            content.append({"type": "text", "text": caption})

        content.append({"type": "image", "image": f"file://{Path(image_path).resolve()}"})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS}
        if deterministic:
            # Greedy decoding — explicitly null the sampling params that the
            # model's generation_config sets by default, otherwise HF warns
            # "The following generation flags are not valid and may be ignored".
            gen_kwargs.update({
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            })
        else:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            })

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Strip input tokens from output
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response.strip()

    def query_text_only(self, prompt: str) -> str:
        """Send a text-only prompt (no image)."""
        self.load()

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response.strip()
