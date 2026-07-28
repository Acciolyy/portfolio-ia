# Portfólio 4 — Aprendizado de Máquina: Supervisionado, Não Supervisionado, Reforço e Deep Learning

Quatro projetos cobrindo os quatro grandes paradigmas de aprendizado de máquina. Os dois primeiros usam o mesmo dataset (composição química de vinhos) para comparar diretamente como a presença ou ausência de rótulos muda a abordagem ao mesmo problema.

## Projetos

### 1. Supervisionado — Classificação de cultivares de uva (`projeto_01_supervisionado.py`)
Gradient Boosting (ensemble sequencial que corrige resíduos das árvores anteriores) vs. baseline de Regressão Logística, no Wine recognition dataset (178 amostras, 13 atributos, 3 classes).

**Resultado:** a Regressão Logística (100% de acurácia) superou o Gradient Boosting (97,78%) — resultado que só faz sentido porque o dataset é linearmente bem separável após padronização, favorecendo o modelo mais simples. Validação cruzada de 5 folds confirmou estabilidade (acurácia média 95,49% ± 3,88%). Prolina, flavonoides e intensidade de cor foram os atributos mais discriminativos.

<p align="center">
  <img src="imagens/p1_confusion_matrix.png" width="30%">
  <img src="imagens/p1_model_comparison.png" width="30%">
  <img src="imagens/p1_feature_importance.png" width="30%">
</p>

### 2. Não supervisionado — Segmentação de amostras de vinho (`projeto_02_nao_supervisionado.py`)
Mesmo dataset do Projeto 1, mas sem acesso aos rótulos: K-Means (nas 13 dimensões originais padronizadas) + PCA (apenas para visualização em 2D), com k escolhido por varredura via método do cotovelo e coeficiente de silhueta.

**Resultado:** a silhueta atingiu seu máximo exatamente em k=3 — o mesmo número real de cultivares, descoberto sem qualquer rótulo. Validação externa via Adjusted Rand Index de 0,8975 (172 de 178 amostras corretamente agrupadas), mostrando que a estrutura química já era suficiente para recuperar quase toda a segmentação que o Projeto 1 tratou como problema supervisionado.

<p align="center">
  <img src="imagens/p2_elbow_silhouette.png" width="45%">
  <img src="imagens/p2_clusters_pca.png" width="45%">
</p>

### 3. Aprendizado por reforço — Navegação em grid de armazém (`projeto_03_reforco.py`)
Q-Learning (off-policy, TD-control) treinando um agente a navegar um grid 8x8 com obstáculos e zona de risco, sem conhecer o mapa previamente — apenas por recompensas. Exploração via epsilon-greedy com decaimento exponencial, 500 episódios.

**Resultado:** a política final convergiu para exatamente 14 passos — o mínimo teórico (distância de Manhattan) entre origem e destino — contornando obstáculos e a zona de risco sem nenhum desvio desnecessário. Recompensa média subiu de -195,50 (primeiros episódios) para 35,90 (últimos episódios), redução de ~90% no número de passos.

<p align="center">
  <img src="imagens/p3_learning_curve.png" width="45%">
  <img src="imagens/p3_policy_path.png" width="45%">
</p>

### 4. Deep Learning — Classificação de dígitos com CNN LeNet-5 (`projeto_04_deep_learning.py`)
Rede convolucional baseada na arquitetura LeNet-5 (1998), treinada em PyTorch no MNIST (60k treino / 10k teste), com ReLU no lugar da tanh original.

**Resultado:** 98,90% de acurácia no teste após apenas 8 épocas, sem sinais de overfitting (treino 99,28% vs. validação 98,90%). Maior taxa de erro no dígito 9 (2,97%), confundido principalmente com 4, 5 e 7 — ambiguidade visual esperada em traçados manuscritos.

<p align="center">
  <img src="imagens/p4_training_curves.png" width="30%">
  <img src="imagens/p4_confusion_matrix.png" width="30%">
  <img src="imagens/p4_sample_predictions.png" width="30%">
</p>

## Stack
Python, scikit-learn, NumPy (Projetos 1–3), PyTorch (Projeto 4, treinado no Google Colab), `@dataclass` para DTOs tipados em todos os projetos.
