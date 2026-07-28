# Portfólio 3 — Tratando Incerteza em IA: Redes Bayesianas, HMM e Filtro de Kalman

Três projetos que abordam formas diferentes de raciocinar sob incerteza, em complexidade crescente: um grafo causal estático (Rede Bayesiana), uma sequência temporal com estado oculto discreto (HMM), e um sistema dinâmico com estado contínuo (Filtro de Kalman). Implementação em Python orientado a objetos, com DTOs tipados e comentários em português.

## Projetos

### 1. Rede Bayesiana — Diagnóstico de qualidade de café (`projeto_01_rede_bayesiana.py`)
Modela três causas independentes (umidade do grão, temperatura de torra, tempo de torra) que convergem para a variável `DefeitoSabor`, com Tabela de Probabilidade Condicional definida manualmente. Inferência via algoritmo de Eliminação de Variáveis (`pgmpy`).

**Resultado:** na direção causal, a probabilidade de defeito sobe de 3% (nenhum fator de risco) para 74,6% (temperatura e tempo altos simultaneamente) — mais de 20x de diferença. Na direção diagnóstica, observado um defeito com tempo normal, a umidade alta se torna a causa mais provável (59,79%).

<p align="center">
  <img src="imagens/projeto-1-rede-bayesiana/figura_01_rede_bayesiana.png" width="45%">
  <img src="imagens/projeto-1-rede-bayesiana/figura_02_inferencias.png" width="45%">
</p>

### 2. HMM — Detecção de intenção do usuário (`projeto_02_hmm.py`)
Modelo Oculto de Markov implementado manualmente em `numpy` (sem bibliotecas prontas como `hmmlearn`), para forçar o entendimento de cada etapa — incluindo o uso de log-probabilidades para evitar underflow numérico no Viterbi. Três estados ocultos (Navegando / Comprando / Inativo) inferidos a partir de quatro tipos de evento observável.

**Resultado:** o algoritmo de Viterbi (decodificação global) atingiu 72,5% de acurácia contra o estado real, superando o Forward (filtragem online) com 70,0% — o Viterbi é mais robusto a eventos isolados que contradizem o estado atual, enquanto o Forward reage mais rápido a mudanças reais de comportamento.

<p align="center">
  <img src="imagens/projeto-2-hmm/figura_03_hmm_estados.png" width="45%">
  <img src="imagens/projeto-2-hmm/figura_04_hmm_posterior.png" width="45%">
</p>

### 3. Filtro de Kalman 2D — Posição de drone com GPS ruidoso (`projeto_03_kalman.py`)
Filtro implementado do zero em `numpy` (sem `filterpy`) para deixar explícito o cálculo do Ganho de Kalman. Estado de 4 dimensões (posição e velocidade em X/Y), observando apenas posição via GPS ruidoso.

**Resultado:** redução de 58,4% no erro de posição — RMSE caiu de 5,64m (GPS bruto) para 2,35m (estimativa filtrada) ao longo de 200 passos de missão simulada.

<p align="center">
  <img src="imagens/projeto-3-kalman/figura_05_trajetoria_drone.png" width="45%">
  <img src="imagens/projeto-3-kalman/figura_06_erro_covariancia.png" width="45%">
</p>

## Stack
Python orientado a objetos, `numpy`, `pgmpy` (Rede Bayesiana), `@dataclass` para DTOs, tipagem estática rigorosa.
