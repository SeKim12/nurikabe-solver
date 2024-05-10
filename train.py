from model import decision_transformer
from dataclasses import dataclass


cfg = decision_transformer.NurikabeDTConfigs()
model = decision_transformer.NurikabeDT.from_pretrained_gpt2(cfg, 'gpt2')

print(f'model parameters: {model.get_num_params()}')