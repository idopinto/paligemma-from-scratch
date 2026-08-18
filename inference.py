import argparse
from PIL import Image
import torch

from paligemma.processor.processing_paligemma import PaliGemmaProcessor
from paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from paligemma.gemma.kv_cache import KVCache
from utils import load_hf_model


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def get_model_inputs(
    processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str
):
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="weights/paligemma-3b-pt-224")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the image: ",
        help="The user prompt.",
    )
    # This building is
    # test_images/pic1.png
    parser.add_argument("--image_file_path", type=str, default="test_images/dog.png")
    parser.add_argument("--max_tokens_to_generate", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--only_cpu", action="store_true")
    return parser.parse_args(args)


def get_device(only_cpu: bool):
    if not only_cpu:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def main(args):
    model_path = args.model_path
    prompt = args.prompt
    image_file_path = args.image_file_path
    max_tokens_to_generate = args.max_tokens_to_generate
    temperature = args.temperature
    top_p = args.top_p
    do_sample = args.do_sample
    device = get_device(args.only_cpu)

    print("Device in use: ", device)

    print(f"Loading model: {model_path}")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()
    print("Model loaded successfully")
    # print(model)
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference")
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    response = model.generate(
        model_inputs, processor, max_tokens_to_generate, temperature, top_p, do_sample
    )
    print("--------------------------------")
    print(prompt + response["response"])
    print(f"Generation time: {response['generation_time_s']:.2f}s")
    print(f"Tokens generated: {response['num_tokens_generated']}")
    print(f"Tokens per second: {response['tokens_per_second']:.2f}")


if __name__ == "__main__":
    main(parse_args())
