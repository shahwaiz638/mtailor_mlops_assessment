# MTailor MLOps Assessment

This repository provides an end-to-end machine learning deployment pipeline. It includes:

* Exporting a PyTorch model to ONNX
* Local consistency testing between PyTorch and ONNX
* Testing the deployed model on Cerebrium

---

## Repository Structure

```bash
.
├── model.py            # ONNX model loading, preprocessing, and inference
├── pytorch_model.py    # PyTorch model definition and preprocessing
├── convert_to_onnx.py  # Script to convert PyTorch model to ONNX format
├── test.py             # Local test: compares PyTorch and ONNX outputs
├── test_server.py      # Remote test: validates Cerebrium deployment
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## Workflow Guide

Run the full pipeline with these commands:

```bash
pip install -r requirements.txt
python model.py
python test.py
python test_server.py
```

---

## File Descriptions

### `convert_to_onnx.py`

* Converts trained PyTorch model to ONNX format using `torch.onnx.export`

### `model.py`

* Defines `ONNX_MODEL` class
* Loads ONNX model
* Fetches image from URL and preprocesses it
* Runs inference and returns predicted class


### `test.py`

* Performs inference with both PyTorch and ONNX models
* Validates consistency of outputs between both formats

### `test_server.py`

* Sends requests to the deployed Cerebrium endpoint
* Includes support for:

  * Inference from image URLs
  * concurrency and custom tests

---

## Running the Pipeline

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Convert PyTorch model to ONNX (if not already, Load the pth weights before execution) :

```bash
python convert_to_onnx.py
```

3. Run local tests:

```bash
python model.py
python test.py
```

4. Run remote tests (Cerebrium API key has been pushed on gitHub to avoid problems):

```bash
python test_server.py --run-tests
```

---

