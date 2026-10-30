from __future__ import annotations

import argparse
import statistics
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Objeto de Transferência de Dados
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Encapsula o resultado de uma travessia de grafo."""

    traversal_order: List[str]
    total_nodes_visited: int
    elapsed_ms: float
    algorithm_name: str
    source_node: str
    target_node: Optional[str] = None
    path_to_target: Optional[List[str]] = field(default=None)

    def __str__(self) -> str:
        lines = [
            f"[{self.algorithm_name}]",
            f"  Source          : {self.source_node}",
            f"  Nodes visited   : {self.total_nodes_visited}",
            f"  Traversal order : {' → '.join(self.traversal_order)}",
            f"  Elapsed time    : {self.elapsed_ms:.4f} ms",
        ]
        if self.target_node is not None:
            found = self.path_to_target is not None
            lines.append(f"  Target '{self.target_node}' found : {found}")
            if found:
                lines.append(f"  Path            : {' → '.join(self.path_to_target)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aliases de tipo
# ---------------------------------------------------------------------------

AdjacencyList = Dict[str, List[str]]


# ---------------------------------------------------------------------------
# Classe base abstrata
# ---------------------------------------------------------------------------

class GraphSearch(ABC):
    """Base abstrata para estratégias de busca em grafos.

    As subclasses devem implementar :meth:`execute_search` e podem,
    opcionalmente, sobrescrever :meth:`find_path` para localizar um nó alvo.
    """

    def __init__(self, adjacency_list: AdjacencyList) -> None:
        self._adjacency_list: AdjacencyList = adjacency_list

    @abstractmethod
    def execute_search(
        self,
        source_node: str,
        target_node: Optional[str] = None,
    ) -> SearchResult:
        """Percorre o grafo a partir de *source_node*.

        Parâmetros
        ----------
        source_node:
            Nó de onde a travessia começa.
        target_node:
            Se informado, a travessia para assim que este nó é alcançado,
            e o caminho mais curto (BFS) ou o primeiro encontrado (DFS)
            é registrado.

        Retorna
        -------
        SearchResult
            Objeto de resultado completamente preenchido — nenhum efeito
            colateral de I/O ocorre dentro deste método.
        """

    def _reconstruct_path(
        self,
        parent_map: Dict[str, Optional[str]],
        target_node: str,
    ) -> List[str]:
        """Percorre *parent_map* de trás para frente para reconstruir o caminho da raiz ao alvo."""
        path: List[str] = []
        current: Optional[str] = target_node
        while current is not None:
            path.append(current)
            current = parent_map[current]
        path.reverse()
        return path

    @property
    def algorithm_name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Implementação BFS
# ---------------------------------------------------------------------------

class BreadthFirstSearch(GraphSearch):
    """Travessia em largura usando uma fila FIFO.

    Garante o *caminho mais curto* (por número de arestas) até qualquer alvo alcançável.
    """

    def execute_search(
        self,
        source_node: str,
        target_node: Optional[str] = None,
    ) -> SearchResult:
        start_time: float = time.perf_counter()

        visited_set: Set[str] = set()
        traversal_order: List[str] = []
        parent_map: Dict[str, Optional[str]] = {source_node: None}
        frontier: deque[str] = deque([source_node])
        visited_set.add(source_node)
        path_to_target: Optional[List[str]] = None

        while frontier:
            current_node: str = frontier.popleft()
            traversal_order.append(current_node)

            if current_node == target_node:
                path_to_target = self._reconstruct_path(parent_map, current_node)
                break

            for neighbour in self._adjacency_list.get(current_node, []):
                if neighbour not in visited_set:
                    visited_set.add(neighbour)
                    parent_map[neighbour] = current_node
                    frontier.append(neighbour)

        elapsed_ms: float = (time.perf_counter() - start_time) * 1_000

        return SearchResult(
            traversal_order=traversal_order,
            total_nodes_visited=len(traversal_order),
            elapsed_ms=elapsed_ms,
            algorithm_name=self.algorithm_name,
            source_node=source_node,
            target_node=target_node,
            path_to_target=path_to_target,
        )


# ---------------------------------------------------------------------------
# Implementação DFS
# ---------------------------------------------------------------------------

class DepthFirstSearch(GraphSearch):
    """Travessia em profundidade usando uma pilha LIFO explícita.

    Evita o limite de recursão do Python e mantém a pilha de chamadas
    previsível para grafos arbitrariamente grandes.
    """

    def execute_search(
        self,
        source_node: str,
        target_node: Optional[str] = None,
    ) -> SearchResult:
        start_time: float = time.perf_counter()

        visited_set: Set[str] = set()
        traversal_order: List[str] = []
        parent_map: Dict[str, Optional[str]] = {source_node: None}
        stack: List[str] = [source_node]
        path_to_target: Optional[List[str]] = None

        while stack:
            current_node: str = stack.pop()

            if current_node in visited_set:
                continue

            visited_set.add(current_node)
            traversal_order.append(current_node)

            if current_node == target_node:
                path_to_target = self._reconstruct_path(parent_map, current_node)
                break

            # Invertemos a ordem para que o vizinho mais à esquerda seja explorado primeiro
            for neighbour in reversed(self._adjacency_list.get(current_node, [])):
                if neighbour not in visited_set:
                    parent_map.setdefault(neighbour, current_node)
                    stack.append(neighbour)

        elapsed_ms: float = (time.perf_counter() - start_time) * 1_000

        return SearchResult(
            traversal_order=traversal_order,
            total_nodes_visited=len(traversal_order),
            elapsed_ms=elapsed_ms,
            algorithm_name=self.algorithm_name,
            source_node=source_node,
            target_node=target_node,
            path_to_target=path_to_target,
        )


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1,
                        help="com N > 1, repete a medição de tempo N vezes e reporta a mediana")
    args = parser.parse_args()

    # Grafo dirigido com ciclos — topologia realista para demonstrações acadêmicas.
    #
    #        ┌──────────────────────┐
    #        ▼                      │
    #   A ──► B ──► D ──► G        │
    #   │     │                     │
    #   ▼     ▼                     │
    #   C ──► E ──► F ──────────────┘
    #         │
    #         ▼
    #         H ──► I
    #
    cyclic_graph: AdjacencyList = {
        "A": ["B", "C"],
        "B": ["D", "E"],
        "C": ["E"],
        "D": ["G"],
        "E": ["F", "H"],
        "F": ["A"],          # aresta de retorno → cria um ciclo
        "G": [],
        "H": ["I"],
        "I": [],
    }

    separator = "─" * 50

    bfs_engine: BreadthFirstSearch = BreadthFirstSearch(cyclic_graph)
    dfs_engine: DepthFirstSearch = DepthFirstSearch(cyclic_graph)

    # --- Travessia completa (sem alvo definido) ---
    print(separator)
    print("FULL TRAVERSAL (no target node)")
    print(separator)
    print(bfs_engine.execute_search(source_node="A"))
    print()
    print(dfs_engine.execute_search(source_node="A"))

    # --- Busca com alvo definido ---
    print()
    print(separator)
    print("TARGETED SEARCH  (source=A  →  target=I)")
    print(separator)
    print(bfs_engine.execute_search(source_node="A", target_node="I"))
    print()
    print(dfs_engine.execute_search(source_node="A", target_node="I"))

    # --- Alvo inalcançável ---
    print()
    print(separator)
    print("UNREACHABLE TARGET  (source=G  →  target=A)")
    print(separator)
    print(bfs_engine.execute_search(source_node="G", target_node="A"))
    print()
    print(dfs_engine.execute_search(source_node="G", target_node="A"))
    print(separator)

    if args.repeat > 1:
        print()
        print(separator)
        print(f"TEMPO MEDIANO ({args.repeat} repetições, source=A → target=I)")
        print(separator)
        for engine in (bfs_engine, dfs_engine):
            runs = [engine.execute_search(source_node="A", target_node="I") for _ in range(args.repeat)]
            print(f"  {runs[0].algorithm_name:<22}: {statistics.median(r.elapsed_ms for r in runs):.4f} ms "
                  f"| nós visitados {runs[0].total_nodes_visited}")
