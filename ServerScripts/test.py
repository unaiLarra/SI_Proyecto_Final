import torch

model = torch.load('/home/unai/Documents/Uni/Año4/Deusto/SI/ExportedModels/v2.pth',
                   weights_only=False)
model.eval()