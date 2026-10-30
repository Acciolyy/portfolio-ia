"""
informed_search.py
==================
Algoritmos de busca informada — A* e Greedy Best-First Search —
aplicados a um mapa sintético do campus FGA (UnB Gama).

Visão geral da arquitetura
--------------------------
::

    InformedSearch   (ABC)
    ├── AStarSearch          — f(n) = g(n) + h(n)  → caminho ótimo
    └── GreedySearch         — f(n) = h(n)          → rápido, não garante ótimo

Representação do grafo
----------------------
Um grafo não-dirigido ponderado é armazenado como lista de adjacência:

    AdjacencyList = Dict[str, List[Tuple[str, float]]]

onde cada entrada ``(vizinho, peso)`` representa uma aresta bidirecional
cujo peso é a distância aproximada a pé em metros entre dois prédios do campus FGA.

Heurística
----------
A heurística ``h(n)`` estima a distância em linha reta (Euclidiana) do nó ``n``
até o nó objetivo (Restaurante Universitário — RU).
Os valores foram aproximados a partir do layout do campus; eles são admissíveis
(nunca superestimam) e consistentes, o que garante a otimalidade do A*.

A tabela ``FGA_HEURISTICS`` vale **somente para o objetivo RU**. A heurística
depende do objetivo: reutilizá-la para outro destino a torna inadmissível e
tira a garantia de otimalidade. Por isso a busca recusa objetivos cuja
estimativa não seja zero (``h(objetivo) = 0``), e :func:`inadmissible_nodes`
permite verificar uma tabela contra os custos reais.

Exemplo de uso
--------------
::

    from informed_search import AStarSearch, GreedySearch, FGA_GRAPH, FGA_HEURISTICS

    engine = AStarSearch(FGA_GRAPH, FGA_HEURISTICS)
    result = engine.execute_search(source_node="UAC", goal_node="RU")
    print(result)

"""

from __future__ import annotations

import heapq
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Aliases de tipo
# ---------------------------------------------------------------------------

#: Lista de adjacência não-dirigida e ponderada.
#: ``AdjacencyList["A"]`` retorna uma lista de pares ``(vizinho, distancia_m)``.
AdjacencyList = Dict[str, List[Tuple[str, float]]]

#: Mapeia o nome de cada nó para sua estimativa heurística h(n) em relação ao objetivo.
HeuristicMap = Dict[str, float]


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Objeto de valor imutável retornado por toda estratégia de busca.

    Atributos
    ---------
    algorithm_name:
        Identificador legível do algoritmo que produziu o resultado.
    source_node:
        Nó de onde a busca se originou.
    goal_node:
        Nó alvo que o algoritmo tentou alcançar.
    path:
        Lista ordenada de nomes de nós de *source_node* até *goal_node*.
        Lista vazia quando não existe caminho.
    total_cost_m:
        Peso acumulado das arestas (metros) ao longo de *path*.
        ``0.0`` quando não existe caminho.
    nodes_expanded:
        Número de nós removidos da fila de prioridade e totalmente processados.
        Proxy prático para a complexidade de tempo.
    elapsed_ms:
        Tempo de execução real medido com ``time.perf_counter``,
        expresso em milissegundos para facilitar leitura.
    goal_reached:
        ``True`` se e somente se *path* liga *source_node* a *goal_node*.
    """

    algorithm_name: str
    source_node: str
    goal_node: str
    path: List[str]
    total_cost_m: float
    nodes_expanded: int
    elapsed_ms: float
    goal_reached: bool = field(init=False)

    def __post_init__(self) -> None:
        self.goal_reached = bool(self.path) and self.path[-1] == self.goal_node

    def __str__(self) -> str:
        status = "✓ reached" if self.goal_reached else "✗ unreachable"
        path_str = " → ".join(self.path) if self.path else "—"
        return (
            f"[{self.algorithm_name}]\n"
            f"  Route          : {path_str}\n"
            f"  Total distance : {self.total_cost_m:.1f} m\n"
            f"  Nodes expanded : {self.nodes_expanded}\n"
            f"  Elapsed time   : {self.elapsed_ms:.4f} ms\n"
            f"  Goal ({self.goal_node})    : {status}"
        )


# ---------------------------------------------------------------------------
# Entrada da fila de prioridade
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _PriorityEntry:
    """Entrada do heap que evita comparação de fallback entre nomes de nós.

    O ``heapq`` do Python compara tuplas elemento a elemento. Se duas entradas
    compartilham a mesma prioridade, comparar strings (nomes de nós) pode lançar
    ``TypeError`` para nós não-string ou produzir ordenação inesperada.
    Usar um inteiro *insertion_index* como critério de desempate é a solução
    idiomática recomendada pela documentação do Python.

    Atributos
    ---------
    priority:
        O valor f usado para ordenar na fila de prioridade.
    insertion_index:
        Contador monotonicamente crescente que desempata prioridades iguais em
        ordem FIFO, evitando comparações entre objetos ``SearchState``.
    state:
        Estado da busca associado a esta entrada; excluído da ordenação.
    """

    priority: float
    insertion_index: int
    state: "_SearchState" = field(compare=False)


@dataclass
class _SearchState:
    """Estado interno de travessia carregado pela fila de prioridade.

    Atributos
    ---------
    current_node:
        O nó sendo considerado no momento.
    path_so_far:
        Lista ordenada de nós visitados para chegar a *current_node*.
    cost_so_far:
        g(n) acumulado — soma dos pesos de aresta da origem até *current_node*.
    """

    current_node: str
    path_so_far: List[str]
    cost_so_far: float


# ---------------------------------------------------------------------------
# Classe base abstrata
# ---------------------------------------------------------------------------

class InformedSearch(ABC):
    """Base abstrata para estratégias de busca heurística em grafos.

    Tanto o A* quanto o Greedy Best-First Search compartilham o mesmo esqueleto
    de fila de prioridade; eles diferem apenas em como calculam a prioridade
    atribuída a cada nó candidato:

    * **A\\*** :  ``f(n) = g(n) + h(n)``  — equilibra custo real e estimativa.
    * **Greedy**: ``f(n) = h(n)``          — ignora completamente o custo real.

    As subclasses devem implementar :meth:`_compute_priority` para expressar
    essa política, e :meth:`execute_search` para a API pública.

    Parâmetros
    ----------
    adjacency_list:
        Grafo não-dirigido ponderado como lista de vizinhos.
    heuristic_map:
        Estimativas h(n) pré-calculadas para cada nó do grafo.
    """

    def __init__(
        self,
        adjacency_list: AdjacencyList,
        heuristic_map: HeuristicMap,
    ) -> None:
        self._adjacency_list: AdjacencyList = adjacency_list
        self._heuristic_map: HeuristicMap = heuristic_map

    # ------------------------------------------------------------------
    # Interface abstrata
    # ------------------------------------------------------------------

    @abstractmethod
    def _compute_priority(self, cost_so_far: float, node: str) -> float:
        """Retorna o valor de prioridade usado para ordenar *node* no heap.

        Parâmetros
        ----------
        cost_so_far:
            Custo de caminho acumulado g(n) da origem até *node*.
        node:
            O nó candidato sendo avaliado.

        Retorna
        -------
        float
            Valores menores são desenfileirados primeiro (semântica de min-heap).
        """

    @abstractmethod
    def execute_search(self, source_node: str, goal_node: str) -> SearchResult:
        """Executa a busca de *source_node* em direção a *goal_node*.

        As implementações não devem produzir efeitos colaterais de I/O;
        toda saída é retornada exclusivamente por :class:`SearchResult`.

        Parâmetros
        ----------
        source_node:
            Nó onde a travessia começa.
        goal_node:
            Nó que o algoritmo tenta alcançar.

        Retorna
        -------
        SearchResult
            Resultado completamente preenchido — incluindo tempo, caminho, custo
            e contagem de expansões — independentemente de o objetivo ter sido encontrado.
        """

    # ------------------------------------------------------------------
    # Driver compartilhado da fila de prioridade
    # ------------------------------------------------------------------

    def _run_priority_queue_search(
        self,
        source_node: str,
        goal_node: str,
    ) -> Tuple[List[str], float, int]:
        """Loop central da fila de prioridade compartilhado por todas as subclasses.

        Implementa o template geral de busca best-first. As subclasses o
        especializam exclusivamente através de :meth:`_compute_priority`.

        Parâmetros
        ----------
        source_node:
            Nó inicial da travessia.
        goal_node:
            Condição de término — a busca para quando este nó é expandido.

        Retorna
        -------
        path : List[str]
            Sequência de nós da origem ao objetivo, ou lista vazia se inalcançável.
        total_cost : float
            Soma dos pesos de arestas ao longo de *path*.
        nodes_expanded : int
            Número de nós totalmente processados (desenfileirados e avaliados).

        Notas
        -----
        Um *conjunto fechado* (``expanded_nodes``) impede a re-expansão de nós
        já definidos, o que é correto para o A* com heurística consistente e
        também seguro para o Greedy.
        """
        if self._heuristic_map.get(goal_node, 0.0) != 0.0:
            raise ValueError(
                f"A heurística fornecida não é para o objetivo '{goal_node}': "
                f"h({goal_node}) = {self._heuristic_map[goal_node]} (deveria ser 0). "
                "Use uma tabela construída para este objetivo."
            )
        initial_state = _SearchState(
            current_node=source_node,
            path_so_far=[source_node],
            cost_so_far=0.0,
        )
        initial_priority = self._compute_priority(0.0, source_node)
        heap: List[_PriorityEntry] = [
            _PriorityEntry(
                priority=initial_priority,
                insertion_index=0,
                state=initial_state,
            )
        ]

        expanded_nodes: set[str] = set()
        insertion_counter: int = 1
        nodes_expanded: int = 0

        while heap:
            entry: _PriorityEntry = heapq.heappop(heap)
            current_state: _SearchState = entry.state
            current_node: str = current_state.current_node

            if current_node in expanded_nodes:
                continue

            expanded_nodes.add(current_node)
            nodes_expanded += 1

            if current_node == goal_node:
                return current_state.path_so_far, current_state.cost_so_far, nodes_expanded

            for neighbour, edge_weight in self._adjacency_list.get(current_node, []):
                if neighbour in expanded_nodes:
                    continue

                new_cost: float = current_state.cost_so_far + edge_weight
                new_priority: float = self._compute_priority(new_cost, neighbour)
                new_state = _SearchState(
                    current_node=neighbour,
                    path_so_far=current_state.path_so_far + [neighbour],
                    cost_so_far=new_cost,
                )
                heapq.heappush(
                    heap,
                    _PriorityEntry(
                        priority=new_priority,
                        insertion_index=insertion_counter,
                        state=new_state,
                    ),
                )
                insertion_counter += 1

        return [], 0.0, nodes_expanded

    @property
    def algorithm_name(self) -> str:
        """Nome legível da classe, usado em :class:`SearchResult`."""
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Busca A*
# ---------------------------------------------------------------------------

class AStarSearch(InformedSearch):
    """Algoritmo de busca informada A\\* (A-Star).

    O A\\* seleciona o próximo nó a expandir minimizando::

        f(n) = g(n) + h(n)

    onde

    * ``g(n)`` é o custo **exato** acumulado da origem até o nó *n*,
    * ``h(n)`` é a **estimativa heurística** do custo restante de *n* até o objetivo.

    Garantia de otimalidade
    -----------------------
    Quando a heurística é **admissível** (nunca superestima o custo real)
    *e* **consistente** (satisfaz a desigualdade triangular
    ``h(n) ≤ w(n, n') + h(n')`` para toda aresta ``(n, n')``), o A\\* garante
    encontrar um **caminho de custo mínimo**.

    Complexidade
    ------------
    * Tempo  : O(b^d) no pior caso, onde *b* é o fator de ramificação
      e *d* é a profundidade da solução ótima mais rasa.
    * Espaço : O(b^d) — todos os nós gerados são mantidos em memória.

    Parâmetros
    ----------
    adjacency_list:
        Grafo não-dirigido ponderado.
    heuristic_map:
        Valores heurísticos admissíveis e consistentes h(n) para cada nó.
    """

    def _compute_priority(self, cost_so_far: float, node: str) -> float:
        """Retorna ``f(n) = g(n) + h(n)``."""
        return cost_so_far + self._heuristic_map.get(node, 0.0)

    def execute_search(self, source_node: str, goal_node: str) -> SearchResult:
        """Encontra o caminho de menor custo de *source_node* a *goal_node* usando A\\*.

        Parâmetros
        ----------
        source_node:
            Nó de partida.
        goal_node:
            Nó alvo.

        Retorna
        -------
        SearchResult
            Contém o caminho ótimo, seu custo total em metros, o número de
            nós expandidos e o tempo de execução real em milissegundos.
        """
        start_time: float = time.perf_counter()
        path, total_cost, nodes_expanded = self._run_priority_queue_search(
            source_node, goal_node
        )
        elapsed_ms: float = (time.perf_counter() - start_time) * 1_000

        return SearchResult(
            algorithm_name=self.algorithm_name,
            source_node=source_node,
            goal_node=goal_node,
            path=path,
            total_cost_m=total_cost,
            nodes_expanded=nodes_expanded,
            elapsed_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Greedy Best-First Search
# ---------------------------------------------------------------------------

class GreedySearch(InformedSearch):
    """Algoritmo de busca informada Greedy Best-First Search.

    O Greedy seleciona o próximo nó a expandir minimizando apenas a heurística::

        f(n) = h(n)

    ignorando completamente o custo já pago para alcançar *n*. Isso o torna
    mais rápido que o A\\* na prática — expande menos nós — mas **não garante
    um caminho ótimo**.

    Comparativo
    -----------
    +---------------------+------------------+
    | Propriedade         | Valor            |
    +=====================+==================+
    | Completo            | Não (pode ciclar)|
    +---------------------+------------------+
    | Ótimo               | Não              |
    +---------------------+------------------+
    | Complexidade tempo  | O(b^m)           |
    +---------------------+------------------+
    | Complexidade espaço | O(b^m)           |
    +---------------------+------------------+

    onde *m* é a profundidade máxima do espaço de busca.

    Parâmetros
    ----------
    adjacency_list:
        Grafo não-dirigido ponderado.
    heuristic_map:
        Valores heurísticos h(n) para cada nó.
    """

    def _compute_priority(self, cost_so_far: float, node: str) -> float:
        """Retorna ``f(n) = h(n)``, ignorando *cost_so_far* completamente."""
        return self._heuristic_map.get(node, 0.0)

    def execute_search(self, source_node: str, goal_node: str) -> SearchResult:
        """Encontra um caminho de *source_node* a *goal_node* usando Greedy Best-First.

        Parâmetros
        ----------
        source_node:
            Nó de partida.
        goal_node:
            Nó alvo.

        Retorna
        -------
        SearchResult
            Contém o caminho encontrado (possivelmente sub-ótimo), seu custo total
            em metros, o número de nós expandidos e o tempo de execução.
        """
        start_time: float = time.perf_counter()
        path, total_cost, nodes_expanded = self._run_priority_queue_search(
            source_node, goal_node
        )
        elapsed_ms: float = (time.perf_counter() - start_time) * 1_000

        return SearchResult(
            algorithm_name=self.algorithm_name,
            source_node=source_node,
            goal_node=goal_node,
            path=path,
            total_cost_m=total_cost,
            nodes_expanded=nodes_expanded,
            elapsed_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Grafo do campus FGA
# ---------------------------------------------------------------------------

#: Grafo não-dirigido ponderado representando o campus FGA (UnB Gama).
#:
#: Nós
#: ---
#: - **UAC**   — Unidade Acadêmica (salas de aula principais)
#: - **UED**   — Unidade de Ensino e Docência
#: - **RU**    — Restaurante Universitário  *(nó objetivo)*
#: - **BCG**   — Biblioteca do Campus Gama
#: - **MEsp**  — Módulo Esportivo (quadras / academia)
#: - **Est**   — Estacionamento principal
#: - **Lab**   — Bloco de Laboratórios de Hardware/Software
#: - **Adm**   — Bloco Administrativo / Reitoria local
#: - **LTIA**  — Laboratório de Tecnologia e Inovação Aplicada
#: - **Gate**  — Portaria / Entrada principal do campus
#:
#: Os pesos das arestas são distâncias aproximadas a pé em metros, derivadas
#: do layout do campus. O grafo é não-dirigido, portanto cada aresta é
#: armazenada nos dois sentidos.
FGA_GRAPH: AdjacencyList = {
    "UAC": [
        ("UED",  80.0),
        ("BCG", 120.0),
        ("Lab",  95.0),
        ("Adm",  70.0),
    ],
    "UED": [
        ("UAC",  80.0),
        ("RU",  150.0),
        ("Lab",  60.0),
        ("LTIA", 110.0),
    ],
    "RU": [
        ("UED",  150.0),
        ("BCG",  200.0),
        ("MEsp", 180.0),
        ("Est",  250.0),
    ],
    "BCG": [
        ("UAC",  120.0),
        ("RU",   200.0),
        ("Adm",   90.0),
        ("Gate", 300.0),
    ],
    "MEsp": [
        ("RU",   180.0),
        ("Est",  100.0),
        ("Gate", 220.0),
    ],
    "Est": [
        ("RU",   250.0),
        ("MEsp", 100.0),
        ("Gate",  80.0),
        ("Adm",  200.0),
    ],
    "Lab": [
        ("UAC",  95.0),
        ("UED",  60.0),
        ("LTIA", 50.0),
    ],
    "Adm": [
        ("UAC",  70.0),
        ("BCG",  90.0),
        ("Est",  200.0),
    ],
    "LTIA": [
        ("UED",  110.0),
        ("Lab",   50.0),
    ],
    "Gate": [
        ("BCG",  300.0),
        ("MEsp", 220.0),
        ("Est",   80.0),
    ],
}

#: Distância em linha reta (metros) de cada nó do campus até o **RU**.
#:
#: Esses valores são heurísticas admissíveis: nunca excedem a distância real
#: do caminho mais curto, pois a distância Euclidiana é um limite inferior
#: para qualquer caminho por arestas. Também são consistentes (nenhum nó pode
#: "atalhar" a desigualdade triangular), o que garante a otimalidade do A*.
FGA_HEURISTICS: HeuristicMap = {
    "UAC":  200.0,   # distância moderada; dois saltos via UED
    "UED":  140.0,   # uma aresta direta para o RU
    "RU":     0.0,   # nó objetivo
    "BCG":  195.0,   # uma aresta direta, mas longa
    "MEsp": 175.0,   # uma aresta direta para o RU
    "Est":  245.0,   # ponto mais distante do RU
    "Lab":  175.0,   # próximo à UED, que é próxima do RU
    "Adm":  230.0,   # deve passar pela UAC ou BCG
    "LTIA": 185.0,   # próximo ao caminho Lab → UED
    "Gate": 270.0,   # entrada do campus, longe do RU
}


# ---------------------------------------------------------------------------
# Verificação de heurística
# ---------------------------------------------------------------------------

def inadmissible_nodes(
    adjacency_list: AdjacencyList,
    heuristic_map: HeuristicMap,
    goal_node: str,
) -> Dict[str, Tuple[float, float]]:
    """Retorna ``{nó: (h(nó), custo_real)}`` para cada nó com ``h > custo_real``.

    O custo real até *goal_node* vem de Dijkstra (o grafo é não-dirigido, então
    a partir do objetivo obtém-se a distância de todos os nós até ele). Um
    dicionário vazio significa que a heurística é admissível para esse objetivo.
    """
    best: Dict[str, float] = {goal_node: 0.0}
    heap: List[Tuple[float, str]] = [(0.0, goal_node)]
    while heap:
        cost, node = heapq.heappop(heap)
        if cost > best.get(node, float("inf")):
            continue
        for neighbour, weight in adjacency_list.get(node, []):
            new_cost = cost + weight
            if new_cost < best.get(neighbour, float("inf")):
                best[neighbour] = new_cost
                heapq.heappush(heap, (new_cost, neighbour))
    return {
        node: (heuristic_map.get(node, 0.0), real)
        for node, real in best.items()
        if heuristic_map.get(node, 0.0) > real + 1e-9
    }


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    separator: str = "─" * 55

    astar_engine: AStarSearch = AStarSearch(FGA_GRAPH, FGA_HEURISTICS)
    greedy_engine: GreedySearch = GreedySearch(FGA_GRAPH, FGA_HEURISTICS)

    scenarios: List[Tuple[str, str]] = [
        ("Gate", "RU"),    # entrada do campus → refeitório
        ("LTIA", "RU"),    # lab de inovação   → refeitório
        ("Est",  "RU"),    # estacionamento    → refeitório
        ("Adm",  "RU"),    # administrativo    → refeitório
    ]

    for source, goal in scenarios:
        print(separator)
        print(f"  {source}  →  {goal}")
        print(separator)
        print(astar_engine.execute_search(source_node=source, goal_node=goal))
        print()
        print(greedy_engine.execute_search(source_node=source, goal_node=goal))
        print()
