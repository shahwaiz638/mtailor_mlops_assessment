import onnxruntime
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

class ONNX_MODEL():
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def predict(self, input_image):
        input_name = self.session.get_inputs()[0].name
        inputs = {input_name: input_image}
        outputs = self.session.run(None, inputs)
        return np.argmax(outputs[0], axis=1)[0]
    

    # Preprocessing function (same as in pytorch_model.py)
    def preprocess_image(self, img_path):
        img = Image.open(img_path)
        resize = transforms.Resize((224, 224))
        crop = transforms.CenterCrop((224, 224))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = resize(img)
        img = crop(img)
        img = to_tensor(img)
        img = normalize(img)
        return img.unsqueeze(0).numpy()  # Add batch dimension and convert to numpy

# Load ONNX model
def main():
    onnx_model_path = "./mtailor-assessment/onnx_model_weights.onnx"
    session = ONNX_MODEL(onnx_model_path)

    # Preprocess image
    input_image = session.preprocess_image("./n01667114_mud_turtle.JPEG")

    # Run prediction
    outputs = session.predict(input_image)
    print(f"Predicted class index: {outputs}")

if __name__ == "__main__":
    main()