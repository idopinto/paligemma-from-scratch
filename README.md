# PaliGemma from scratch

A PyTorch re-implementation of [PaliGemma: A versatile 3B VLM for transfer](https://arxiv.org/abs/2407.07726) (Beyer et al., 2024) — Google's open vision-language model built from SigLIP and Gemma — written from the ground up to understand how multimodal inference actually works.

Inspired by [this walkthrough](https://www.youtube.com/watch?v=vAmKB7iPkWw&t=8310s), which builds the model layer by layer and walks through loading pretrained weights and running generation.

## What this is

PaliGemma takes an image and a text prompt, fuses them into a single sequence, and generates a text response. This repo implements that pipeline in plain PyTorch:

- **SigLIP** vision encoder — turns a 224×224 image into 256 patch tokens
- **Multimodal projector** — maps vision features into Gemma's embedding space
- **Gemma-2B** text decoder — causal LM with grouped-query attention, RoPE, and a KV cache for autoregressive decoding
- **Processor** — resizes/normalizes images and prepends `<image>` placeholders to the prompt

Weights are loaded from the official Hugging Face checkpoint ([`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224)); the model architecture itself is implemented here, not delegated to `transformers`.

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo>
cd paligemma-from-scratch
uv sync
```

### Download weights

The checkpoint is gated under the [Gemma license](https://huggingface.co/google/paligemma-3b-pt-224). Accept the terms on Hugging Face, then:

```bash
hf auth login
hf download google/paligemma-3b-pt-224 --local-dir weights/paligemma-3b-pt-224
```

## Usage

Run inference with a single image and prompt:

```bash
uv run inference.py \
  --model_path weights/paligemma-3b-pt-224 \
  --image_file_path test_images/dog.png \
  --prompt "describe this image" \
  --max_tokens_to_generate 20
```

Useful flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model_path` | `weights/paligemma-3b-pt-224` | Local path to the HF checkpoint |
| `--image_file_path` | `test_images/pic1.png` | Input image |
| `--prompt` | `This building is ` | Text prompt (prepended after image tokens) |
| `--max_tokens_to_generate` | `20` | Max new tokens |
| `--do_sample` | off | Enable temperature + top-p sampling |
| `--temperature` | `0.8` | Sampling temperature (only with `--do_sample`) |
| `--top_p` | `0.9` | Nucleus sampling threshold (only with `--do_sample`) |
| `--only_cpu` | off | Force CPU (otherwise uses CUDA or MPS if available) |

The script prints the generated text along with generation time and tokens/sec.

## Current limitations

**Single image + single prompt only.** Both the processor and `inference.py` assert one image per one text input. Batching, multiple images, and multi-turn conversation are not supported yet.

## TODO

- [ ] Support batched inference (multiple prompts and/or images in one forward pass)
- [ ] Support multi-image inputs
- [ ] Fine-tuning script (LoRA or full fine-tune on a custom dataset)
- [ ] Object detection — decode `<loc####>` bounding box tokens and render boxes on the image
- [ ] Segmentation — decode `<seg###>` tokens and reconstruct segmentation masks

## Project layout

```
paligemma/
  siglip/          # Vision encoder (patch embed → transformer → post-layernorm)
  gemma/           # Text decoder (attention, RoPE, MLP, KV cache)
  processor/       # Image preprocessing + prompt tokenization
  modeling_paligemma.py
inference.py       # CLI entry point
utils.py           # Load config + safetensors weights from a local checkpoint
```

## References

- **Paper:** [PaliGemma: A versatile 3B VLM for transfer](https://arxiv.org/abs/2407.07726) — Lucas Beyer et al., arXiv:2407.07726, 2024
- **Model:** [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) on Hugging Face
- **Tutorial:** [Video walkthrough that inspired this repo](https://www.youtube.com/watch?v=vAmKB7iPkWw&t=8310s)

## Citation

If you use the original PaliGemma model or paper, please cite:

```bibtex
@article{beyer2024paligemma,
  title={PaliGemma: A versatile 3B VLM for transfer},
  author={Beyer, Lucas and Steiner, Andreas and Pinto, Andr{\'e} Susano and Kolesnikov, Alexander and Wang, Xiao and Salz, Daniel and Neumann, Maxim and Alabdulmohsin, Ibrahim and Tschannen, Michael and Bugliarello, Emanuele and others},
  journal={arXiv preprint arXiv:2407.07726},
  year={2024}
}
```
