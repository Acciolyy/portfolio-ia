"""Demo Gradio da LeNet-5 treinada no MNIST (Acciolyyy/lenet5-mnist-unb)."""

import os
import sys

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from scipy import ndimage

REPO = "Acciolyyy/lenet5-mnist-unb"

sys.path.insert(0, os.path.dirname(hf_hub_download(REPO, "lenet5.py")))
from lenet5 import MEAN, STD, LeNet5  # noqa: E402

model = LeNet5()
model.load_state_dict(load_file(hf_hub_download(REPO, "model.safetensors")))
model.eval()


def to_gray(img) -> np.ndarray:
    """Converte a entrada do Gradio (dict do editor, array ou PIL) em cinza float [0, 255]."""
    if isinstance(img, dict):
        img = img.get("composite")
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        base = Image.new("RGBA", rgba.size, (255, 255, 255, 255))  # transparente vira branco
        img = Image.alpha_composite(base, rgba)
    return np.asarray(img.convert("L"), dtype=np.float32)


def preprocess(gray: np.ndarray) -> torch.Tensor | None:
    """Mesmo formato do MNIST: dígito claro sobre fundo preto, 28x28, centrado, normalizado."""
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if border.mean() > 127:  # fundo branco: inverte
        gray = 255.0 - gray
    gray = gray - gray.min()
    if gray.max() < 1e-3:
        return None
    gray = gray / gray.max() * 255.0
    ys, xs = np.where(gray > 30)
    if len(ys) == 0:
        return None
    gray = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]  # recorta o traço
    h, w = gray.shape
    scale = 20.0 / max(h, w)  # o MNIST cabe o dígito numa caixa de 20x20
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = np.asarray(Image.fromarray(gray.astype(np.uint8)).resize(new_size, Image.LANCZOS), dtype=np.float32)
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - small.shape[0]) // 2, (28 - small.shape[1]) // 2
    canvas[top : top + small.shape[0], left : left + small.shape[1]] = small
    cy, cx = ndimage.center_of_mass(canvas)  # centraliza pelo centro de massa
    canvas = ndimage.shift(canvas, (13.5 - cy, 13.5 - cx), order=1, mode="constant")
    x = torch.from_numpy(np.clip(canvas, 0, 255) / 255.0).float()
    return ((x - MEAN) / STD).reshape(1, 1, 28, 28)


LOW_CONFIDENCE = 0.70
WARNING = ("⚠️ **Baixa confiança:** a entrada pode não ser um dígito único "
           "ou estar fora do padrão do MNIST.")


def predict(sketch, upload):
    gray = to_gray(upload if upload is not None else sketch)
    x = preprocess(gray) if gray is not None else None
    if x is None:
        return "", None, None
    with torch.no_grad():
        probs = model(x).softmax(dim=1)[0].numpy()
    shown = (x[0, 0] * STD + MEAN).clamp(0, 1).numpy()
    warn = WARNING if probs.max() < LOW_CONFIDENCE else ""
    return warn, {str(i): float(p) for i, p in enumerate(probs)}, shown


with gr.Blocks(title="LeNet-5 no MNIST") as demo:
    gr.Markdown(
        "# Reconhecimento de dígitos (LeNet-5 / MNIST)\n"
        "**Desenhe um único dígito (0 a 9)** ou envie uma imagem de um único dígito. "
        "Projeto acadêmico de portfólio: "
        f"[modelo](https://huggingface.co/{REPO}). Desenhos fora do padrão do MNIST podem errar."
    )
    with gr.Row():
        with gr.Column():
            sketch = gr.Sketchpad(label="Desenhe aqui", type="numpy", canvas_size=(280, 280),
                                  brush=gr.Brush(default_size=18, colors=["#000000"], color_mode="fixed"))
            upload = gr.Image(label="Ou envie uma imagem", type="pil", sources=["upload"])
            btn = gr.Button("Classificar", variant="primary")
        with gr.Column():
            warning = gr.Markdown()
            label = gr.Label(num_top_classes=10, label="Probabilidades")
            seen = gr.Image(label="O que o modelo vê (28x28)", image_mode="L", height=200)
    btn.click(predict, [sketch, upload], [warning, label, seen])

if __name__ == "__main__":
    demo.launch()
