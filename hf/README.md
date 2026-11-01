---
license: mit
library_name: pytorch
pipeline_tag: image-classification
datasets:
- mnist
metrics:
- accuracy
language:
- pt
- en
model-index:
- name: lenet5-mnist-unb
  results:
  - task:
      type: image-classification
    dataset:
      name: MNIST
      type: mnist
    metrics:
    - type: accuracy
      value: 0.9864
      name: Test accuracy (seed 42)
---

# LeNet-5 no MNIST

**English summary.** A LeNet-5-style CNN (ReLU instead of tanh, 61,706 parameters) trained with PyTorch on MNIST digit classification. Split: 55k train / 5k validation / 10k test, with the test set evaluated only once per run. The released weights (seed 42) reach **98.64%** test accuracy; over 5 seeds (42–46) the mean is **98.74% ± 0.07** (sample std). An earlier version that monitored the test set during training reported 98.90%; that number was leaked and should not be trusted. It is a closed-set classifier: it always outputs one of the 10 digits and can be confidently wrong on out-of-distribution input (e.g. a drawn "X" predicted as 7 with 84%). This is an academic portfolio project, not meant for production.

## Descrição

Rede convolucional baseada na LeNet-5 (LeCun et al., 1998) que classifica imagens 28×28 em tons de cinza de dígitos manuscritos (0 a 9). Foi feita para a disciplina de Inteligência Artificial (FGA0221, UnB), no portfólio 4 de aprendizado de máquina.

## Como funciona

- **Convolução:** um pequeno filtro (5×5 pixels) desliza pela imagem e mede "quanto este trecho parece uma borda, uma curva, um traço". Cada filtro vira um mapa que mostra onde aquele padrão aparece. A primeira camada aprende 6 filtros simples; a segunda, 16 filtros que combinam os anteriores em formas maiores.
- **Pooling:** reduz cada mapa à metade, guardando só o maior valor de cada bloco 2×2. Assim a rede fica menos sensível a pequenos deslocamentos do traço e trabalha com menos números.
- **Camadas finais:** os mapas viram uma lista de 400 números, que passa por três camadas densas (120, 84 e 10 neurônios). A última dá uma pontuação para cada dígito, e a maior pontuação é a classe prevista.

## Arquitetura

| # | Camada | Saída | Parâmetros |
|---|---|---|---|
| 1 | Conv2d(1→6, kernel 5, padding 2) + ReLU | 6×28×28 | 156 |
| 2 | MaxPool2d(2) | 6×14×14 | 0 |
| 3 | Conv2d(6→16, kernel 5) + ReLU | 16×10×10 | 2.416 |
| 4 | MaxPool2d(2) | 16×5×5 | 0 |
| 5 | Flatten | 400 | 0 |
| 6 | Linear(400→120) + ReLU | 120 | 48.120 |
| 7 | Linear(120→84) + ReLU | 84 | 10.164 |
| 8 | Linear(84→10) | 10 | 850 |

Total: 61.706 parâmetros. A ativação original (tanh) foi trocada por ReLU.

## Dados e divisão

MNIST: 60.000 imagens de treino e 10.000 de teste. As de treino são divididas, com semente fixa (42), em **55.000 treino / 5.000 validação**. Os **10.000 de teste** são avaliados **uma única vez por execução**, no fim, sem influenciar nenhuma decisão de treino. Entrada normalizada com média 0,1307 e desvio 0,3081.

## Hiperparâmetros

Otimizador Adam, taxa de aprendizado 1e-3, 8 épocas, lote de 128, perda de entropia cruzada, sem aumento de dados. Treinado em CPU.

## Resultados

| | Acurácia de teste |
|---|---|
| Pesos publicados (semente 42) | **98,64%** (validação: 98,32%) |
| 5 sementes (42 a 46), média ± desvio amostral | **98,74% ± 0,07** (validação: 98,51% ± 0,19) |

As sementes individuais deram 98,64; 98,79; 98,76; 98,82 e 98,69%. Os pesos desta página são os de uma semente só, que ficou um pouco abaixo da média.

### Correção do vazamento de validação

A primeira versão do projeto treinava e acompanhava a validação sobre o próprio conjunto de teste e reportava **98,90%** de uma única execução. Como o teste orientou o treino, o número ficou otimista. Na revisão, separei 5 mil exemplos de validação dos dados de treino, passei a usar o teste uma só vez e repeti com 5 sementes: **98,74% ± 0,07**. Este é o valor confiável, porque vem de dados que não influenciaram o treino e não depende de uma execução sortuda.

## Como usar

```python
import os, sys, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

repo = "Acciolyyy/lenet5-mnist-unb"
sys.path.insert(0, os.path.dirname(hf_hub_download(repo, "lenet5.py")))
from lenet5 import LeNet5, MEAN, STD

model = LeNet5()
model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")))
model.eval()

x = torch.rand(1, 1, 28, 28)  # troque por uma imagem 28x28 em [0, 1], dígito claro sobre fundo preto
probs = model((x - MEAN) / STD).softmax(dim=1)
print(probs.argmax(dim=1).item(), probs.max().item())
```

## Formato ONNX

O repositório também traz `model.onnx` (opset 17, entrada `input` 1×1×28×28 com batch dinâmico, saída `logits`), gerado por `export_onnx.py` a partir de `model.safetensors`. Ele é usado pela demo estática no navegador (onnxruntime-web). Teste de paridade com o PyTorch nas primeiras 1000 imagens do teste do MNIST: as 1000 classes coincidem e a maior diferença entre probabilidades é de 1,07e-06 (limite do teste: 1e-4).

## Limitações

- MNIST é um problema simples: dígitos centralizados, de tamanho padronizado, em tons de cinza, fundo preto. Desenhos à mão livre, fora desse padrão (descentralizados, muito finos ou grossos, fundo diferente, outro estilo de escrita), podem errar.
- É um classificador de **conjunto fechado**: sempre escolhe um dos 10 dígitos e não tem a opção "não é um dígito". Entradas fora da distribuição (mais de um dígito, letras, rabiscos) podem receber confiança alta mesmo estando erradas; exemplo testado: um "X" desenhado foi classificado como 7 com 84%. O aviso de baixa confiança da demo (probabilidade máxima < 0,70) é só uma heurística e não resolve isso.
- Os hiperparâmetros são fixos, sem busca sistemática; o treino é curto e sem aumento de dados.
- Os 98,74% valem para o teste do MNIST, não para dígitos de outras fontes.

## Uso pretendido

Projeto acadêmico de portfólio, para estudo e demonstração. **Não** deve ser usado em produção nem em decisões reais.

## Repositório

Código de treino e análise: <https://github.com/Acciolyy/portfolio-ia> (pasta `portfolio-4-machine-learning`).
