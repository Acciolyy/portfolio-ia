---
title: LeNet-5 MNIST Demo
emoji: 🔢
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
license: mit
short_description: Desenhe um dígito e veja a LeNet-5 classificar no navegador
---

# Demo: LeNet-5 no MNIST (no navegador)

Desenhe **um único dígito (0 a 9)** e veja as probabilidades das 10 classes. O modelo roda
inteiro no seu navegador com [onnxruntime-web](https://onnxruntime.ai/), sem servidor: o
`model.onnx` é baixado do repositório de modelo no Hugging Face.

O pré-processamento (cinza, inversão se o fundo for branco, recorte do traço, 20 px dentro de 28×28,
centro de massa, normalização 0,1307/0,3081) reproduz em JavaScript o do app Gradio do repositório.
Se a maior probabilidade for menor que 0,70, aparece um aviso de baixa confiança, que é só uma
heurística: o modelo é de conjunto fechado e pode errar com confiança alta fora do padrão do MNIST.

- Modelo: <https://huggingface.co/Acciolyyy/lenet5-mnist-unb>
- Código: <https://github.com/Acciolyy/portfolio-ia> (pastas `hf-space-static` e `hf-space`)

Projeto acadêmico de portfólio; não use em produção.
