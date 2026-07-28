# Portfólio 2 — Métodos Clássicos de IA: Busca, Otimização e Agentes Lógicos

Seis implementações de algoritmos clássicos de IA simbólica, cobrindo busca em espaço de estados, otimização, algoritmos genéticos, satisfação de restrições (CSP) e agentes baseados em conhecimento. Todo o código segue arquitetura orientada a objetos, com classes abstratas base por família de algoritmo e DTOs tipados (`@dataclass`) para isolar os resultados de cada execução.

## Projetos

### 1. Busca não informada — BFS vs. DFS (`projeto_01_busca_cega.py`)
Navegação em um grafo cíclico, comparando Busca em Largura (fila FIFO) e Busca em Profundidade (pilha LIFO), com controle de estados visitados para evitar loops infinitos.

**Resultado:** BFS visitou 9 nós (0,0140 ms); DFS visitou 8 nós (0,0084 ms) — ambos encontraram o mesmo caminho ótimo, confirmando na prática a diferença teórica de complexidade de espaço O(b^d) vs. O(b·m).

### 2. Busca informada — A* vs. Greedy (`informed_search.py`)
Otimização de rotas de pedestre no campus da UnB/FGA, usando heurística de distância euclidiana entre os prédios.

**Resultado:** ambos encontraram a rota de 250m em tempo equivalente neste cenário, mas o A* garante otimalidade por balancear custo acumulado g(n) e heurística h(n), enquanto o Greedy é suscetível a rotas subótimas.

### 3. Busca local — Hill Climbing vs. Simulated Annealing (`tsp_local_search.py`)
Problema do Caixeiro Viajante (TSP) aplicado a uma rota logística entre 10 Regiões Administrativas do DF, usando o operador de vizinhança 2-opt.

**Resultado:** após 50.000 iterações, ambos convergiram para 164,48 km (melhoria de 55,1% sobre a rota inicial). O Simulated Annealing é mais robusto contra ótimos locais por aceitar movimentos de piora durante o resfriamento, ao custo de mais tempo de execução (328,63 ms vs. 1,067 ms do Hill Climbing).

### 4. Algoritmo genético — N-Rainhas (`n_queens_genetic.py`)
Problema das 8 rainhas com codificação por permutação (elimina conflitos ortogonais por construção), crossover Order Crossover (OX1), mutação por troca, seleção por torneio e elitismo.

**Resultado:** solução ótima (fitness 28/28) encontrada em apenas 2 gerações, 3,91 ms.

### 5. CSP — Coloração de mapa (`csp_map_coloring.py`)
Coloração dos estados do Centro-Oeste (DF, GO, MT, MS) e fronteiriços via backtracking, com heurística MRV (Minimum Remaining Values) opcional.

**Resultado:** zero backtracks e 19 nós explorados em ambas as versões — o mapa era simples o suficiente para não precisar de correção de caminho, então a heurística MRV, mesmo sendo mais "inteligente", acabou sendo mais lenta (0,3527 ms vs. 0,0555 ms) por causa do overhead de calcular o domínio mais restrito a cada passo.

### 6. Agente baseado em conhecimento — Mundo do Wumpus (`wumpus_world.py`)
Agente que navega um grid 4x4 sob incerteza parcial, inferindo posições de ouro, poços e do Wumpus por lógica proposicional pura, a partir de percepções locais (Brisa, Fedor, Brilho) — sem heurísticas de busca em grafo.

**Resultado:** o agente encontrou o ouro em 10 passos lógicos (990 pontos), com encapsulamento estrito entre ambiente e agente (o agente nunca acessa o estado real, só percepções).

## Stack
Python puro, orientação a objetos, `@dataclass` para DTOs, tipagem estática (type hints), `collections.deque` e `heapq` para estruturas de dados.
