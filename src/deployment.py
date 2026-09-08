"""
Deployment optimization for brain tumor MRI classification model
"""

import torch
import torch.onnx
import numpy as np
from typing import Optional, Tuple
import os
import json


class ModelDeployment:
    """Model deployment optimization utilities"""

    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def convert_to_onnx(self, input_tensor: torch.Tensor, onnx_path: str,
                        opset_version: int = 11) -> bool:
        """Convert PyTorch model to ONNX format"""
        try:
            input_tensor = input_tensor.to(self.device)

            torch.onnx.export(
                self.model,
                (input_tensor,),
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )

            print(f"Model successfully converted to ONNX: {onnx_path}")
            print(f"ONNX model size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")

            return True

        except Exception as e:
            print(f"Error converting to ONNX: {e}")
            return False

    def test_onnx_model(self, onnx_path: str, input_tensor: torch.Tensor) -> bool:
        """Test ONNX model inference"""
        try:
            import onnx  # type: ignore
            import onnxruntime as ort  # type: ignore

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            ort_session = ort.InferenceSession(onnx_path)
            input_numpy = input_tensor.cpu().numpy()
            outputs = ort_session.run(None, {'input': input_numpy})

            print(f"ONNX model inference successful")
            print(f"Output shape: {outputs[0].shape}")

            return True

        except ImportError:
            print("ONNX or ONNX Runtime not installed")
            print("Install with: pip install onnx onnxruntime")
            return False
        except Exception as e:
            print(f"Error testing ONNX model: {e}")
            return False

    def optimize_for_inference(self, input_tensor: torch.Tensor) -> dict:
        """Optimize model for inference"""
        results = {
            'original_model_size': self._get_model_size(),
            'optimization_suggestions': []
        }

        model_size_mb = results['original_model_size']
        if model_size_mb > 100:
            results['optimization_suggestions'].append(
                f"Model size ({model_size_mb:.1f} MB) exceeds 100 MB target. Consider quantization."
            )

        classifier_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(keyword in name.lower()
                    for keyword in ['classifier', 'fc', 'head', 'dense']):
                classifier_layers.append(module)

        if classifier_layers:
            classifier_params = sum(p.numel() for layer in classifier_layers for p in layer.parameters())
            total_params = sum(p.numel() for p in self.model.parameters())
            classifier_ratio = classifier_params / total_params

            if classifier_ratio > 0.1:
                results['optimization_suggestions'].append(
                    f"Classifier layers represent {classifier_ratio:.1%} of model parameters. Consider reducing complexity."
                )

        memory_usage = self._estimate_memory_usage(input_tensor)
        results['estimated_memory_mb'] = memory_usage

        if memory_usage > 2000:
            results['optimization_suggestions'].append(
                f"Estimated memory usage ({memory_usage:.0f} MB) exceeds 2GB target."
            )

        return results

    def _get_model_size(self) -> float:
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0

        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def _estimate_memory_usage(self, input_tensor: torch.Tensor) -> float:
        """Estimate memory usage during inference"""
        input_memory = input_tensor.numel() * input_tensor.element_size()
        param_memory = sum(p.numel() * p.element_size() for p in self.model.parameters())
        total_memory = (input_memory + param_memory) / (1024**2)
        return total_memory

    def benchmark_inference(self, input_tensor: torch.Tensor, num_runs: int = 100) -> dict:
        """Benchmark model inference speed"""
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(input_tensor)

        times = []
        for _ in range(num_runs):
            if self.device == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                import time
                start_time = time.time()

            with torch.no_grad():
                _ = self.model(input_tensor)

            if self.device == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event) / 1000
            else:
                elapsed = time.time() - start_time

            times.append(elapsed)

        results = {
            'mean_inference_time': np.mean(times),
            'std_inference_time': np.std(times),
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'inference_fps': 1.0 / np.mean(times)
        }

        print(f"Inference Benchmark Results:")
        print(f"Mean time: {results['mean_inference_time']:.4f}s")
        print(f"Std time: {results['std_inference_time']:.4f}s")
        print(f"FPS: {results['inference_fps']:.2f}")

        return results


def create_deployment_package(model: torch.nn.Module, input_shape: Tuple[int, ...],
                              output_dir: str = "deployment") -> bool:
    """Create complete deployment package"""
    try:
        os.makedirs(output_dir, exist_ok=True)

        deployment = ModelDeployment(model)
        input_tensor = torch.randn(input_shape)

        onnx_path = os.path.join(output_dir, "model.onnx")
        success = deployment.convert_to_onnx(input_tensor, onnx_path)

        if not success:
            return False

        deployment.test_onnx_model(onnx_path, input_tensor)

        opt_results = deployment.optimize_for_inference(input_tensor)
        benchmark_results = deployment.benchmark_inference(input_tensor)

        results = {
            'optimization': opt_results,
            'benchmark': benchmark_results,
            'model_info': {
                'input_shape': input_shape,
                'device': deployment.device,
                'onnx_path': onnx_path
            }
        }

        with open(os.path.join(output_dir, "deployment_results.json"), 'w') as f:
            json.dump(results, f, indent=2)

        requirements = [
            "onnx>=1.12.0",
            "onnxruntime>=1.12.0",
            "numpy>=1.21.0",
            "opencv-python>=4.5.0"
        ]

        with open(os.path.join(output_dir, "requirements.txt"), 'w') as f:
            f.write('\n'.join(requirements))

        inference_script = '''"""
ONNX Model Inference Script for Brain Tumor MRI Classification
"""

import onnxruntime as ort
import numpy as np
import cv2

class BrainTumorInference:
    def __init__(self, model_path: str, target_size: tuple = (512, 512)):
        self.session = ort.InferenceSession(model_path)
        self.target_size = target_size
        self.class_names = ['No_Tumor', 'Glioma', 'Meningioma', 'Pituitary']

    def preprocess_image(self, image_path: str):
        """Preprocess MRI image for inference"""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            image = clahe.apply(image)

            # Resize and convert to 3-channel (EfficientNet expects RGB)
            image = cv2.resize(image, self.target_size)
            image = np.stack([image, image, image], axis=-1)

            image = image.astype(np.float32) / 255.0
            image = (image - 0.5) / 0.5
            image = np.transpose(image, (2, 0, 1))
            image = np.expand_dims(image, axis=0)

            return image

        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            raise

    def predict(self, image_path: str):
        """Run inference on MRI image"""
        try:
            input_data = self.preprocess_image(image_path)
            outputs = self.session.run(None, {'input': input_data})

            predictions = outputs[0]
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            class_name = self.class_names[predicted_class]

            return {
                'class_id': int(predicted_class),
                'class_name': class_name,
                'confidence': float(confidence),
                'all_probabilities': predictions[0].tolist()
            }

        except Exception as e:
            print(f"Error during inference: {e}")
            raise

if __name__ == "__main__":
    model = BrainTumorInference("model.onnx")
    result = model.predict("sample_mri.jpg")
    print(f"Predicted class: {result[\'class_name\']} (ID: {result[\'class_id\']})")
    print(f"Confidence: {result[\'confidence\']:.3f}")
'''

        with open(os.path.join(output_dir, "inference.py"), 'w') as f:
            f.write(inference_script)

        print(f"Deployment package created in {output_dir}/")
        print(f"Files: model.onnx, inference.py, requirements.txt, deployment_results.json")

        return True

    except Exception as e:
        print(f"Error creating deployment package: {e}")
        return False


if __name__ == "__main__":
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(64, 4)  # 4 classes

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = DummyModel()
    input_shape = (3, 512, 512)

    success = create_deployment_package(model, input_shape)

    if success:
        print("Deployment test successful!")
    else:
        print("Deployment test failed!")
