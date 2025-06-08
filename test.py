import os
import numpy as np
from PIL import Image
import torch
import onnx
import onnxruntime as ort
from model import  ONNX_MODEL
from pytorch_model import Classifier, BasicBlock
import onnxruntime

def test_inference_consistency():
    print("-------------- Testing inference consistency with sample image --------------")

    example_inputs = (torch.randn(1, 3, 224, 224),)
    onnx_model = onnx.load("./mtailor-assessment/onnx_model_weights.onnx")
    onnx.checker.check_model(onnx_model)

    # PyTorch Inference
    mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
    mtailor.load_state_dict(torch.load("./pytorch_model_weights.pth"))
    mtailor.eval()
    torch_outputs = mtailor(*example_inputs)


    onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]
    print(f"Input length: {len(onnx_inputs)}")
    print(f"Sample input: {onnx_inputs}")

    ort_session = onnxruntime.InferenceSession(
        "./mtailor-assessment/onnx_model_weights.onnx", providers=["CPUExecutionProvider"]
    )

    onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

    # ONNX Runtime returns a list of outputs
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]

    torch_outputs = mtailor(*example_inputs)

    assert len(torch_outputs) == len(onnxruntime_outputs)
    for torch_output, onnxruntime_output in zip(torch_outputs, onnxruntime_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnxruntime_output))

    print("PyTorch and ONNX Runtime output matched!")
    print(f"Output length: {len(onnxruntime_outputs)}")
    print(f"Sample output: {onnxruntime_outputs}")


def main():
    
    test_inference_consistency()

    print(" tests passed!")
        

if __name__ == "__main__":
    main()