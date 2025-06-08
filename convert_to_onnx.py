from pytorch_model import Classifier, BasicBlock
import torch

# load the PyTorch model
mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
mtailor.load_state_dict(torch.load("./pytorch_model_weights.pth"))
mtailor.eval()

# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
example_inputs = (torch.randn(1, 3, 224, 224),)
onnx_program = torch.onnx.export(mtailor, example_inputs, dynamo=True)


onnx_program.optimize()

# Save the ONNX model to a file
onnx_program.save("onnx_model_weights.onnx")