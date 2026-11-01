"""Exporta a LeNet-5 (model.safetensors) para ONNX e confere a paridade com o PyTorch.

Uso (a partir da raiz do repositorio):
    python hf/export_onnx.py [--mnist-dir portfolio-4-machine-learning/mnist_data]

Gera hf/model.onnx (entrada 1x1x28x28, batch dinamico) e compara PyTorch x ONNX Runtime
nas primeiras 1000 imagens do teste do MNIST: as classes devem ser iguais e a maior
diferenca entre probabilidades deve ser menor que 1e-4. Sai com codigo 1 se falhar.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from safetensors.torch import load_file
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lenet5 import MEAN, STD, LeNet5  # noqa: E402

OPSET = 17
N_IMAGES = 1000
TOLERANCE = 1e-4


def export(model: LeNet5, path: Path) -> None:
    dummy = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(
        model, dummy, str(path), opset_version=OPSET, dynamo=False,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-dir", default=str(HERE.parent / "portfolio-4-machine-learning" / "mnist_data"))
    args = parser.parse_args()

    model = LeNet5()
    model.load_state_dict(load_file(str(HERE / "model.safetensors")))
    model.eval()
    onnx_path = HERE / "model.onnx"
    export(model, onnx_path)
    print(f"Exportado: {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB, opset {OPSET})")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    test = datasets.MNIST(args.mnist_dir, train=False, download=True, transform=tfm)
    x = torch.stack([test[i][0] for i in range(N_IMAGES)])

    with torch.no_grad():
        p_torch = model(x).softmax(dim=1).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logits = session.run(None, {"input": x.numpy()})[0]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p_onnx = e / e.sum(axis=1, keepdims=True)

    same_class = int((p_torch.argmax(1) == p_onnx.argmax(1)).sum())
    max_diff = float(np.abs(p_torch - p_onnx).max())
    print(f"Imagens: {N_IMAGES} | classes iguais: {same_class}/{N_IMAGES} | "
          f"maior diferenca de probabilidade: {max_diff:.2e} (limite {TOLERANCE:.0e})")
    ok = same_class == N_IMAGES and max_diff < TOLERANCE
    print("PARIDADE OK" if ok else "PARIDADE FALHOU")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
