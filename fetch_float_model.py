from asteroid.models import BaseModel
import torch


MODEL_NAME = "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k"
MODEL_SAVE_PATH = "ConvTasNet_Libri1Mix_enhsingle_16k.pth"

# -------------------------- #
# Model
# -------------------------- #
model = BaseModel.from_pretrained(MODEL_NAME)

# -------------------------- #
# Save weights
# -------------------------- #
torch.save(model.state_dict(), MODEL_SAVE_PATH)

