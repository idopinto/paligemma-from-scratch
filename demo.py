from paligemma.siglip.config import SiglipVisionConfig
from paligemma.siglip.modeling_siglip import SiglipVisionModel

config = SiglipVisionConfig()
print(config)
model = SiglipVisionModel(config)

print(model)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(num_params)
#  85,797,120, which is 85.8M.
# That is the usual ViT-B/16 size: D=768, 12 layers, MLP 3072, 224×224 with 16×16 patches. People often round it to 86M.
# It is not PaliGemma’s vision tower. PaliGemma uses SigLIP-so400m (~400M). This config is the smaller base SigLIP/ViT-B.
