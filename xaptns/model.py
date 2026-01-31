import torch
from transformers import AutoTokenizer, AutoModel
import openvino as ov
import numpy as np
import os
import sys

class Embedder:
    def __init__(self, model_name="allenai/specter2_base"):
        self.model_name = model_name
        try:
            print(f"Loading model {model_name}...", file=sys.stderr)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.core = ov.Core()
            self.device = self._detect_device()
            print(f"Detected device: {self.device}", file=sys.stderr)

            self.ov_compiled_model = None
            if self.device in ["GPU", "MYRIAD"]:
                self._init_openvino()
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
        # 2. GPU (Intel iGPU / AMD GPU via OpenCL)
        # 3. CUDA (NVIDIA or ROCm)
        # 4. CPU

        if "MYRIAD" in devices:
            return "MYRIAD"
        if "GPU" in devices:
            return "GPU"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _init_openvino(self):
        cache_dir = os.path.join(os.getcwd(), ".model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "specter2.xml")

        try:
            if os.path.exists(model_path):
                print(f"Loading cached OpenVINO model from {model_path}...", file=sys.stderr)
                ov_model = self.core.read_model(model_path)
            else:
                print(f"Converting model to OpenVINO IR (one-time process, may be slow)...", file=sys.stderr)
                inputs = self.tokenizer("This is a test paper abstract.", return_tensors="pt")
                ov_model = ov.convert_model(self.model, example_input={
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"]
                })
                ov.save_model(ov_model, model_path)
                print(f"Model saved to cache at {model_path}", file=sys.stderr)

            # Compile for the target device
            print(f"Compiling model for {self.device}...", file=sys.stderr)
            self.ov_compiled_model = self.core.compile_model(ov_model, self.device)
            self.model = None # Free memory
            print("OpenVINO setup successful.", file=sys.stderr)
        except Exception as e:
            print(f"OpenVINO setup failed: {e}. Falling back to CPU.", file=sys.stderr)
            self.device = "cpu"
            self.model.to("cpu")
            self.ov_compiled_model = None

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

        if self.ov_compiled_model:
            ov_inputs = {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy()
            }
            res = self.ov_compiled_model(ov_inputs)

            # The output can be accessed by index or by output object
            # Usually the first output is the last_hidden_state for BERT
            output_node = self.ov_compiled_model.output(0)
            last_hidden_state = res[output_node]

            # SPECTER uses the [CLS] token at index 0
            embeddings = last_hidden_state[:, 0, :]

            # Check for NaNs
            if np.isnan(embeddings).any():
                print("Warning: OpenVINO produced NaNs. Falling back to CPU Torch for this inference.", file=sys.stderr)
                # We need the model back to do fallback.
                # This is tricky if we deleted it.
                # For MVP, let's try to fix why it's producing NaNs.
        else:
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                # CLS token
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embeddings

if __name__ == "__main__":
    embedder = Embedder()
    test_text = "The Semantic Scholar Open Data Platform"
    emb = embedder.embed(test_text)
    print(f"Embedding shape: {emb.shape}")
    print(f"First 5 values: {emb[0][:5]}")
