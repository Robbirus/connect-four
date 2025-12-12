# export_onnx.py
import torch
from train_model import SimpleCNN
model = SimpleCNN()
model.load_state_dict(torch.load("connect4_model.pth", map_location="cpu"))
model.eval()
dummy = torch.randn(1,2,6,7)
torch.onnx.export(model, dummy, "connect4.onnx", input_names=['input'], output_names=['output'], opset_version=11)
