from siglip.config import SiglipVisionConfig
from siglip.modeling_siglip import SiglipVisionModel

config = SiglipVisionConfig()
print(config)
model = SiglipVisionModel(config)

print(model)
