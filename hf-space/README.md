---
title: LeNet-5 MNIST Demo
emoji: 🔢
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# Demo: LeNet-5 no MNIST

Desenhe um único dígito (0 a 9) ou envie uma imagem de um único dígito e veja a classe prevista com as probabilidades.
O app aplica o mesmo pré-processamento do treino (tons de cinza, inversão se o fundo for branco,
recorte, 28×28 centralizado, normalização) e baixa os pesos de
[Acciolyyy/lenet5-mnist-unb](https://huggingface.co/Acciolyyy/lenet5-mnist-unb).

Projeto acadêmico de portfólio; não use em produção.
