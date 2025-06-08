
import onnxruntime
import numpy as np
import requests
from PIL import Image
from io import BytesIO



def run(input_image, run_id):  # run_id is optional, injected by Cerebrium at runtime
    
    resp = requests.get(input_image)
    img = Image.open(BytesIO(resp.content)).convert("RGB")

    # Preprocess: resize, normalize, CHW, batch dim
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC → CHW
    arr = np.expand_dims(arr, axis=0)  # add batch dimension

    # Load ONNX model
    onnx_model_path = "./onnx_model_weights.onnx"
    
    # Preprocess image
    session = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Run inference with NumPy tensor
    outputs = session.run(None, {input_name: arr})
    pred_class = int(np.argmax(outputs[0], axis=1)[0])

    return {"Class": pred_class, "status_code": 200} # return your results
    
# To deploy your app, run:
# cerebrium deploy


# # Load ONNX model
# onnx_model_path = "./onnx_model_weights.onnx"
# session = ONNX_MODEL(onnx_model_path)

# # Preprocess image
# input_image = session.preprocess_image("./n01667114_mud_turtle.JPEG")

# # Run prediction
# outputs = session.predict(input_image)
# print(f"Predicted class index: {outputs}")
