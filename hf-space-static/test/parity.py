"""Paridade do pre-processamento: JS (Node) x Python (hf-space/app.py) em 20 imagens do MNIST.

Uso: python hf-space-static/test/parity.py   (a partir da raiz do repositorio)
Gera 20 PNGs, calcula o tensor 28x28 nos dois lados e compara tensores e classes (via ONNX).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf-space"))
import app  # noqa: E402

ds = datasets.MNIST(ROOT / "portfolio-4-machine-learning" / "mnist_data", train=False)
session = ort.InferenceSession(str(ROOT / "hf" / "model.onnx"), providers=["CPUExecutionProvider"])

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    items, py_tensors, labels = [], [], []
    for k in range(20):
        img, y = ds[k]
        if k >= 10:  # variantes: fundo branco 140x140 (k<15) e 280x280 invertido, como o canvas
            img = img.resize((140, 140) if k < 15 else (280, 280), Image.BICUBIC)
            if k >= 15:
                img = Image.eval(img, lambda v: 255 - v)
        path = tmp / f"img_{k:02d}.png"
        img.save(path)
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32)  # decodifica o PNG
        items.append({"w": gray.shape[1], "h": gray.shape[0], "gray": gray.ravel().tolist()})
        py_tensors.append(app.preprocess(gray).numpy().ravel())
        labels.append(y)
    (tmp / "in.json").write_text(json.dumps(items))
    subprocess.run(["node", str(ROOT / "hf-space-static" / "test" / "run_js.js"), str(tmp / "in.json"), str(tmp / "out.json")], check=True)
    js_tensors = [np.array(t, dtype=np.float32) for t in json.loads((tmp / "out.json").read_text())]

diffs, same, correct = [], 0, 0
for k, (a, b) in enumerate(zip(py_tensors, js_tensors)):
    d = float(np.abs(a - b).max())
    diffs.append(d)
    pa = session.run(None, {"input": a.reshape(1, 1, 28, 28)})[0].argmax()
    pb = session.run(None, {"input": b.reshape(1, 1, 28, 28)})[0].argmax()
    same += int(pa == pb)
    correct += int(pb == labels[k])
    print(f"img {k:02d} | rotulo {labels[k]} | py {pa} | js {pb} | max|dif| tensor = {d:.2e}")
print(f"\nMaior diferenca de tensor: {max(diffs):.2e} | classes iguais: {same}/20 | JS acertou: {correct}/20")
sys.exit(0 if same == 20 and max(diffs) < 1e-2 else 1)
