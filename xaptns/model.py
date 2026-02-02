import torch
from transformers import AutoTokenizer, AutoModel
import openvino as ov
import numpy as np
import os
import sys
import onnxruntime as ort

class Embedder:
    def __init__(self, model_name="allenai/specter2_base"):
        self.model_name = model_name
        self.cache_dir = os.path.join(os.getcwd(), ".model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        try:
            print(f"Loading model {model_name}...", file=sys.stderr)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.core = ov.Core()
            self.device = self._detect_device()
            print(f"Detected device: {self.device}", file=sys.stderr)

            self.ov_compiled_model = None
            self.ort_session = None

            if self.device in ["GPU", "MYRIAD", "NPU"]:
                self._init_accel()
            else:
                self.model.to(self.device)
        except Exception as e:
            print(f"Error initializing model: {e}", file=sys.stderr)
            raise

    def _detect_device(self):
        devices = self.core.available_devices
        print(f"Available OpenVINO devices: {devices}", file=sys.stderr)

        # Priority:
        # 1. MYRIAD (NCS2)
        # 2. NPU
        # 3. GPU (Intel iGPU / AMD GPU via OpenCL)
        # 4. CUDA (NVIDIA or ROCm)
        # 5. CPU

        if "MYRIAD" in devices:
            return "MYRIAD"
        if "NPU" in devices:
            return "NPU"
        if "GPU" in devices:
            return "GPU"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _init_accel(self):
        """Initializes OpenVINO or ONNX acceleration."""
        onnx_path = os.path.join(self.cache_dir, "specter2.onnx")
        ov_path = os.path.join(self.cache_dir, "specter2.xml")

        try:
            # Try ONNX Runtime with OpenVINO EP first for NPU/GPU flexibility
            if "OpenVinoExecutionProvider" in ort.get_available_providers():
                if not os.path.exists(onnx_path):
                    print(f"Exporting model to ONNX...", file=sys.stderr)
                    dummy_input = self.tokenizer("test", return_tensors="pt")
                    torch.onnx.export(self.model,
                                      (dummy_input["input_ids"], dummy_input["attention_mask"]),
                                      onnx_path,
                                      input_names=["input_ids", "attention_mask"],
                                      output_names=["last_hidden_state"],
                                      dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                                                   "attention_mask": {0: "batch_size", 1: "sequence_length"}},
                                      opset_version=14)

                print(f"Initializing ONNX Runtime with OpenVinoExecutionProvider on {self.device}...", file=sys.stderr)
                self.ort_session = ort.InferenceSession(onnx_path, providers=['OpenVinoExecutionProvider'],
                                                        provider_options=[{'device_type': self.device}])
                self.model = None # Free memory
                return

            # Fallback to direct OpenVINO
            if os.path.exists(ov_path):
                ov_model = self.core.read_model(ov_path)
            else:
                print(f"Converting model to OpenVINO IR...", file=sys.stderr)
                inputs = self.tokenizer("test", return_tensors="pt")
                ov_model = ov.convert_model(self.model, example_input={
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                })
                ov.save_model(ov_model, ov_path)

            self.ov_compiled_model = self.core.compile_model(ov_model, self.device)
            self.model = None
            print(f"OpenVINO compiled for {self.device} successfully.", file=sys.stderr)

        except Exception as e:
            print(f"Acceleration setup failed: {e}. Falling back to CPU.", file=sys.stderr)
            self.device = "cpu"
            self.model.to("cpu")

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

        if self.ort_session:
            ort_inputs = {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy()
            }
            res = self.ort_session.run(None, ort_inputs)
            embeddings = res[0][:, 0, :] # CLS token
        elif self.ov_compiled_model:
            ov_inputs = {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy()
            }
            res = self.ov_compiled_model(ov_inputs)
            output_node = self.ov_compiled_model.output(0)
            embeddings = res[output_node][:, 0, :]
        else:
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                # CLS token
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Check for NaNs
        if np.isnan(embeddings).any():
            print("Warning: Inference produced NaNs. This may be due to hardware issues.", file=sys.stderr)

        return embeddings

if __name__ == "__main__":
    embedder = Embedder()
    test_text = "The Semantic Scholar Open Data Platform"
    emb = embedder.embed(test_text)
    print(f"Embedding shape: {emb.shape}")
    print(f"First 5 values: {emb[0][:5]}")
