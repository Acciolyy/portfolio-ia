"""
tsp_local_search.py
===================
Algoritmos de busca local — Hill Climbing e Simulated Annealing — aplicados
ao Problema do Caixeiro Viajante (TSP) sobre um subconjunto de Regiões
Administrativas do Distrito Federal.

Definição do problema
---------------------
Dado um conjunto de cidades com coordenadas 2D, encontrar a permutação (tour)
que minimiza a distância Euclidiana total percorrida, partindo e retornando
à mesma cidade.

    TSP: minimizar  Σ dist(route[i], route[i+1 mod n])

Abordagem de busca local
------------------------
Ambos os algoritmos operam diretamente no *espaço de soluções* — partem de
um tour completo e o melhoram iterativamente aplicando pequenas perturbações
estruturais chamadas *movimentos*. O operador de movimento utilizado aqui é
o **2-opt**: seleciona duas arestas não-adjacentes, as remove e reconecta o
tour da única outra forma possível, que equivale a inverter a subsequência
entre os dois pontos de corte.

Arquitetura
-----------
::

    LocalSearchOptimizer   (ABC)
    ├── HillClimbing       — aceita apenas movimentos que melhoram a solução
    └── SimulatedAnnealing — aceita movimentos piores com probabilidade e^(-Δ/T)

Cada algoritmo retorna um dataclass :class:`OptimizationResult`.

Referências
-----------
* Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by
  Simulated Annealing. *Science*, 220(4598), 671–680.
* Applegate, D. L. et al. (2006). *The Traveling Salesman Problem*.
  Princeton University Press.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Modelo de domínio — Cidade
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class City:
    """Nó imutável de uma instância do TSP.

    Atributos
    ---------
    name:
        Nome legível (nome da Região Administrativa).
    x:
        Deslocamento longitudinal aproximado em quilômetros a partir de um
        ponto de referência interno próximo ao centro do Distrito Federal.
    y:
        Deslocamento latitudinal aproximado em quilômetros a partir da mesma origem.

    Notas
    -----
    As coordenadas são derivadas de posições geográficas reais do centro
    administrativo de cada RA e projetadas em um plano Cartesiano local
    via projeção equiretangular simplificada. As distâncias são portanto
    aproximadas, mas suficientes para demonstrar os algoritmos em uma
    instância geograficamente significativa.
    """

    name: str
    x: float   # km a leste da referência
    y: float   # km ao norte da referência

    def distance_to(self, other: City) -> float:
        """Retorna a distância Euclidiana em quilômetros até *other*.

        .. math::

            d(a, b) = \\sqrt{(x_a - x_b)^2 + (y_a - y_b)^2}

        Parâmetros
        ----------
        other:
            A cidade com a qual calcular a distância.

        Retorna
        -------
        float
            Distância Euclidiana em quilômetros.
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


# ---------------------------------------------------------------------------
# DTO de resultado
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Objeto de valor retornado por toda estratégia de busca local.

    Atributos
    ---------
    algorithm_name:
        Identificador do algoritmo que produziu este resultado.
    best_route:
        Sequência ordenada de cidades representando o melhor tour encontrado.
        O tour é implicitamente fechado: o viajante retorna a ``best_route[0]``
        após visitar ``best_route[-1]``.
    best_distance_km:
        Comprimento Euclidiano total de *best_route* em quilômetros.
    iterations:
        Número de movimentos candidatos avaliados durante a execução.
    elapsed_ms:
        Tempo de execução real medido com ``time.perf_counter``,
        expresso em milissegundos.
    initial_distance_km:
        Distância do tour inicial, útil para calcular o percentual de melhoria.
    """

    algorithm_name: str
    best_route: List[City]
    best_distance_km: float
    iterations: int
    elapsed_ms: float
    initial_distance_km: float
    improvement_pct: float = field(init=False)

    def __post_init__(self) -> None:
        if self.initial_distance_km > 0:
            self.improvement_pct = (
                (self.initial_distance_km - self.best_distance_km)
                / self.initial_distance_km
                * 100.0
            )
        else:
            self.improvement_pct = 0.0

    def __str__(self) -> str:
        route_str = " → ".join(c.name for c in self.best_route)
        route_str += f" → {self.best_route[0].name}"   # fecha o ciclo
        return (
            f"[{self.algorithm_name}]\n"
            f"  Route        : {route_str}\n"
            f"  Distance     : {self.best_distance_km:.2f} km\n"
            f"  Improvement  : {self.improvement_pct:.1f}% over initial tour\n"
            f"  Iterations   : {self.iterations:,}\n"
            f"  Elapsed time : {self.elapsed_ms:.3f} ms"
        )


# ---------------------------------------------------------------------------
# Funções utilitárias do TSP
# ---------------------------------------------------------------------------

def compute_tour_distance(route: List[City]) -> float:
    """Retorna o comprimento total de um tour TSP fechado.

    Soma os pesos das arestas do caminho
    ``route[0] → route[1] → … → route[n-1] → route[0]``.

    Parâmetros
    ----------
    route:
        Lista ordenada de cidades formando o tour.

    Retorna
    -------
    float
        Distância Euclidiana total em quilômetros.
    """
    total: float = 0.0
    n: int = len(route)
    for index in range(n):
        total += route[index].distance_to(route[(index + 1) % n])
    return total


def apply_two_opt_swap(route: List[City], i: int, k: int) -> List[City]:
    """Retorna um novo tour com o movimento 2-opt (i, k) aplicado.

    Um movimento 2-opt remove as arestas (i-1, i) e (k, k+1) e as substitui
    por (i-1, k) e (i, k+1), o que equivale a inverter a subsequência
    ``route[i : k+1]``.

    Parâmetros
    ----------
    route:
        Tour atual.
    i:
        Índice inicial do segmento invertido (inclusivo).
    k:
        Índice final do segmento invertido (inclusivo).

    Retorna
    -------
    List[City]
        Novo tour com o segmento ``route[i:k+1]`` invertido.

    Notas
    -----
    Criar uma nova lista em vez de mutar in-place preserva o tour do chamador,
    o que é fundamental para o Simulated Annealing, onde o algoritmo pode
    rejeitar o movimento e manter o estado anterior.
    """
    return route[:i] + route[i : k + 1][::-1] + route[k + 1 :]


# ---------------------------------------------------------------------------
# Classe base abstrata
# ---------------------------------------------------------------------------

class LocalSearchOptimizer(ABC):
    """Base abstrata para estratégias de busca local no TSP.

    Define a interface comum e a infraestrutura compartilhada de vizinhança
    2-opt. As subclasses especializam o *critério de aceitação* usado para
    decidir se devem ou não mover para uma solução vizinha.

    Parâmetros
    ----------
    cities:
        Conjunto completo de cidades a visitar exatamente uma vez.
    seed:
        Semente aleatória opcional para reprodutibilidade.
    """

    def __init__(self, cities: List[City], seed: Optional[int] = None) -> None:
        self._cities: List[City] = cities
        self._rng: random.Random = random.Random(seed)

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    @abstractmethod
    def optimize(self, max_iterations: int) -> OptimizationResult:
        """Executa a otimização e retorna o melhor tour encontrado.

        Parâmetros
        ----------
        max_iterations:
            Limite superior de movimentos candidatos a avaliar.
            Atua como critério de terminação principal.

        Retorna
        -------
        OptimizationResult
            DTO de resultado completamente preenchido — sem efeitos colaterais de I/O.
        """

    # ------------------------------------------------------------------
    # Métodos auxiliares compartilhados
    # ------------------------------------------------------------------

    def _build_initial_tour(self) -> List[City]:
        """Retorna uma permutação aleatória de todas as cidades como tour inicial."""
        tour: List[City] = list(self._cities)
        self._rng.shuffle(tour)
        return tour

    def _iter_two_opt_neighbours(
        self, route: List[City]
    ) -> List[Tuple[int, int]]:
        """Gera todos os pares de índices 2-opt válidos para *route*.

        Um movimento 2-opt nos índices ``(i, k)`` requer ``i < k`` e que o
        segmento contenha pelo menos 2 cidades (``k > i``). Inverter o tour
        inteiro ou um segmento de uma única cidade não produz mudança, portanto
        esses pares são excluídos.

        Retorna
        -------
        List[Tuple[int, int]]
            Lista embaralhada de pares ``(i, k)`` para aleatorizar a ordem de exploração.
        """
        n: int = len(route)
        pairs: List[Tuple[int, int]] = [
            (i, k)
            for i in range(1, n - 1)
            for k in range(i + 1, n)
        ]
        self._rng.shuffle(pairs)
        return pairs

    @property
    def algorithm_name(self) -> str:
        """Nome da classe usado no DTO de resultado."""
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Hill Climbing
# ---------------------------------------------------------------------------

class HillClimbing(LocalSearchOptimizer):
    """Hill Climbing de descida mais íngreme para o TSP via movimentos 2-opt.

    Estratégia
    ----------
    A cada iteração o algoritmo avalia um conjunto ordenado aleatoriamente de
    vizinhos 2-opt da solução atual. Ele aceita o *primeiro* vizinho que
    melhora estritamente a distância do tour (variante first-improvement).
    Quando nenhum vizinho melhora a solução, o algoritmo atingiu um *ótimo
    local* e termina antecipadamente.

    Limitação
    ---------
    O Hill Climbing não consegue escapar de ótimos locais. A qualidade da
    solução final depende fortemente do tour inicial. Para uma garantia mais
    robusta, prefira o :class:`SimulatedAnnealing` ou execute o Hill Climbing
    várias vezes com sementes aleatórias diferentes e mantenha o melhor resultado.

    Parâmetros
    ----------
    cities:
        Cidades a incluir no tour.
    seed:
        Semente aleatória opcional.
    """

    def optimize(self, max_iterations: int = 10_000) -> OptimizationResult:
        """Executa o Hill Climbing first-improvement com vizinhança 2-opt.

        Parâmetros
        ----------
        max_iterations:
            Número máximo de avaliações de vizinhos antes de parar.

        Retorna
        -------
        OptimizationResult
            Melhor tour encontrado, sua distância, contagem de iterações e tempo.
        """
        start_time: float = time.perf_counter()

        current_route: List[City] = self._build_initial_tour()
        current_distance: float = compute_tour_distance(current_route)
        initial_distance: float = current_distance
        iterations: int = 0
        improved: bool = True

        while improved and iterations < max_iterations:
            improved = False
            for i, k in self._iter_two_opt_neighbours(current_route):
                if iterations >= max_iterations:
                    break

                candidate_route: List[City] = apply_two_opt_swap(current_route, i, k)
                candidate_distance: float = compute_tour_distance(candidate_route)
                iterations += 1

                if candidate_distance < current_distance:
                    # Aceita o movimento de melhoria imediatamente (first-improvement)
                    current_route = candidate_route
                    current_distance = candidate_distance
                    improved = True
                    break   # reinicia a varredura da vizinhança a partir da nova solução

        elapsed_ms: float = (time.perf_counter() - start_time) * 1_000

        return OptimizationResult(
            algorithm_name=self.algorithm_name,
            best_route=current_route,
            best_distance_km=current_distance,
            iterations=iterations,
            elapsed_ms=elapsed_ms,
            initial_distance_km=initial_distance,
        )


# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------

class SimulatedAnnealing(LocalSearchOptimizer):
    """Simulated Annealing (SA) para o TSP via movimentos 2-opt.

    Fundamento matemático
    ---------------------
    O SA é inspirado no processo físico de recozimento em metalurgia, onde
    um material é aquecido e depois resfriado lentamente para que seus átomos
    se acomodem em uma estrutura cristalina de baixa energia.

    A cada iteração, uma solução vizinha ``s'`` é gerada a partir da solução
    atual ``s`` aplicando um movimento 2-opt aleatório. A variação de comprimento
    do tour é:

    .. math::

        \\Delta = d(s') - d(s)

    * Se ``Δ < 0`` (melhoria): o movimento é **sempre aceito**.
    * Se ``Δ ≥ 0`` (piora)   : o movimento é aceito com probabilidade

    .. math::

        P(\\text{aceitar}) = e^{-\\Delta / T}

    onde ``T`` é a *temperatura* atual. Essa expressão vem da distribuição
    de Boltzmann na mecânica estatística.

    Intuição por trás da probabilidade de aceitação
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * Quando ``T`` é **alta**, ``e^{-Δ/T} → 1`` para qualquer ``Δ`` finito:
      quase qualquer movimento pior é aceito, permitindo ampla exploração
      do espaço de soluções e escape de ótimos locais.
    * À medida que ``T`` **decresce** em direção a 0, ``e^{-Δ/T} → 0``:
      movimentos piores são rejeitados com probabilidade crescente, e a busca
      converge como o Hill Climbing em torno da melhor região encontrada.
    * Um ``Δ`` menor (movimentos levemente piores) é sempre mais provável de
      ser aceito do que grandes deteriorações, o que é geometricamente intuitivo.

    Esquema de temperatura (resfriamento geométrico)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A temperatura na iteração ``t`` segue um decaimento exponencial:

    .. math::

        T_t = T_0 \\cdot \\alpha^t, \\quad \\alpha \\in (0, 1)

    ``α`` (``cooling_rate``) controla a velocidade de queda da temperatura.
    Valores típicos variam de 0,99 (lento, alta qualidade) a 0,90 (rápido,
    qualidade menor). Um esquema mais lento geralmente produz soluções melhores
    ao custo de mais iterações.

    Parâmetros
    ----------
    cities:
        Cidades a incluir no tour.
    initial_temperature:
        Temperatura inicial ``T₀``. Deve ser grande o suficiente para que
        ``e^{-Δ_max / T₀}`` seja próximo de 1 para o pior caso esperado de Δ.
        Um valor em torno de 10× a distância média das arestas funciona bem.
    cooling_rate:
        O fator de decaimento geométrico ``α``. Deve estar no intervalo aberto
        (0, 1). O padrão 0,995 produz um recozimento lento e de alta qualidade.
    seed:
        Semente aleatória opcional para reprodutibilidade.
    """

    def __init__(
        self,
        cities: List[City],
        initial_temperature: float = 1_000.0,
        cooling_rate: float = 0.995,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(cities, seed)

        if not (0.0 < cooling_rate < 1.0):
            raise ValueError(
                f"cooling_rate deve estar em (0, 1); recebido {cooling_rate!r}."
            )
        if initial_temperature <= 0.0:
            raise ValueError(
                f"initial_temperature deve ser positivo; recebido {initial_temperature!r}."
            )

        self._initial_temperature: float = initial_temperature
        self._cooling_rate: float = cooling_rate

    # ------------------------------------------------------------------
    # Esquema de temperatura
    # ------------------------------------------------------------------

    def _temperature_at(self, iteration: int) -> float:
        """Retorna a temperatura para um dado índice de *iteration*.

        Implementa o resfriamento geométrico (exponencial):

        .. math::

            T_t = T_0 \\cdot \\alpha^t

        Parâmetros
        ----------
        iteration:
            Contador de iterações base zero ``t``.

        Retorna
        -------
        float
            Temperatura atual ``T_t``. Garantidamente positiva para qualquer
            valor finito de *iteration*.

        Notas
        -----
        Usar ``alpha ** t`` é matematicamente equivalente a multiplicar por
        ``alpha`` a cada iteração, mas calcular diretamente pelo índice evita
        o acúmulo de erro de ponto flutuante em execuções longas.
        """
        return self._initial_temperature * (self._cooling_rate ** iteration)

    def _acceptance_probability(
        self, delta: float, temperature: float
    ) -> float:
        """Calcula a probabilidade de aceitação de Boltzmann para um movimento que piora a solução.

        Para um movimento que piora a solução (``delta > 0``):

        .. math::

            P = e^{-\\Delta / T}

        Para um movimento que melhora a solução (``delta ≤ 0``), este método
        nunca é chamado; tais movimentos são aceitos incondicionalmente acima.

        Parâmetros
        ----------
        delta:
            Aumento de custo da solução candidata: ``d(s') - d(s) > 0``.
        temperature:
            Temperatura de recozimento atual ``T > 0``.

        Retorna
        -------
        float
            Valor em ``(0, 1)`` que representa a probabilidade de aceitar o
            movimento que piora a solução.

        Notas
        -----
        Quando ``T → 0``, ``P → 0``: o algoritmo para de aceitar soluções
        piores e se comporta como o Hill Climbing determinístico.
        Quando ``T → ∞``, ``P → 1``: o algoritmo realiza uma caminhada aleatória.
        """
        return math.exp(-delta / temperature)

    # ------------------------------------------------------------------
    # Loop principal de otimização
    # ------------------------------------------------------------------

    def optimize(self, max_iterations: int = 50_000) -> OptimizationResult:
        """Executa o Simulated Annealing com resfriamento geométrico e movimentos 2-opt.

        Parâmetros
        ----------
        max_iterations:
            Total de movimentos candidatos a avaliar. Ao contrário do Hill
            Climbing, o SA não termina antecipadamente em ótimos locais;
            sempre executa pelo orçamento completo.

        Retorna
        -------
        OptimizationResult
            Melhor tour já encontrado (não necessariamente o último aceito),
            sua distância, contagem de iterações e tempo de execução real.
        """
        start_time: float = time.perf_counter()

        current_route: List[City] = self._build_initial_tour()
        current_distance: float = compute_tour_distance(current_route)
        initial_distance: float = current_distance

        # O melhor global é rastreado separadamente: o SA pode se afastar dele.
        best_route: List[City] = list(current_route)
        best_distance: float = current_distance

        n: int = len(current_route)

        for iteration in range(max_iterations):
            temperature: float = self._temperature_at(iteration)

            # ── Gera um vizinho 2-opt aleatório ────────────────────────
            # Escolhe dois pontos de corte distintos uniformemente ao acaso.
            i: int = self._rng.randint(1, n - 2)
            k: int = self._rng.randint(i + 1, n - 1)

            candidate_route: List[City] = apply_two_opt_swap(current_route, i, k)
            candidate_distance: float = compute_tour_distance(candidate_route)

            delta: float = candidate_distance - current_distance

            # ── Critério de aceitação de Metropolis ────────────────────
            # Movimentos que melhoram são sempre aceitos.
            # Movimentos que pioram são aceitos com probabilidade e^(-Δ/T).
            if delta < 0 or self._rng.random() < self._acceptance_probability(delta, temperature):
                current_route = candidate_route
                current_distance = candidate_distance

                # Atualiza o melhor global sempre que encontramos uma melhoria
                # — mesmo que depois aceitemos soluções piores.
                if current_distance < best_distance:
                    best_route = list(current_route)
                    best_distance = current_distance

        elapsed_ms: float = (time.perf_counter() - start_time) * 1_000

        return OptimizationResult(
            algorithm_name=self.algorithm_name,
            best_route=best_route,
            best_distance_km=best_distance,
            iterations=max_iterations,
            elapsed_ms=elapsed_ms,
            initial_distance_km=initial_distance,
        )


# ---------------------------------------------------------------------------
# Dataset do DF — Regiões Administrativas
# ---------------------------------------------------------------------------

#: Coordenadas Cartesianas aproximadas (km) de 10 Regiões Administrativas
#: do Distrito Federal, projetadas em torno de um ponto de referência
#: próximo ao centro geográfico do DF (≈ 15,77°S, 47,93°O).
#:
#: x positivo → leste; y positivo → norte.
DF_CITIES: List[City] = [
    City("Brasília (Plano Piloto)",  x= 0.0,  y= 0.0),
    City("Taguatinga",               x=-19.5,  y=-7.0),
    City("Gama",                     x=-10.0,  y=-28.0),
    City("Ceilândia",                x=-26.0,  y=-5.5),
    City("Águas Claras",             x=-16.0,  y=-8.5),
    City("Samambaia",                x=-24.0,  y=-14.0),
    City("Sobradinho",               x=  9.0,  y= 19.5),
    City("Planaltina",               x= 22.0,  y= 27.0),
    City("São Sebastião",            x= 18.0,  y= -9.0),
    City("Recanto das Emas",         x=-20.0,  y=-22.0),
]


# ---------------------------------------------------------------------------
# Hill Climbing com reinícios aleatórios
# ---------------------------------------------------------------------------

class HillClimbingRestarts(HillClimbing):
    """Hill Climbing que recomeça de um tour aleatório ao parar num ótimo local.

    Repete :class:`HillClimbing` até gastar o orçamento de avaliações e devolve
    o melhor tour entre os reinícios. É a linha de base justa contra o
    Simulated Annealing quando ambos têm o mesmo número de avaliações.
    """

    def optimize(self, max_iterations: int = 10_000) -> OptimizationResult:
        start_time: float = time.perf_counter()
        remaining: int = max_iterations
        best: Optional[OptimizationResult] = None
        initial_distance: Optional[float] = None
        while remaining > 0:
            result = super().optimize(remaining)
            remaining -= max(result.iterations, 1)
            if initial_distance is None:
                initial_distance = result.initial_distance_km
            if best is None or result.best_distance_km < best.best_distance_km:
                best = result
        assert best is not None and initial_distance is not None
        return OptimizationResult(
            algorithm_name=self.algorithm_name,
            best_route=best.best_route,
            best_distance_km=best.best_distance_km,
            iterations=max_iterations - remaining,
            elapsed_ms=(time.perf_counter() - start_time) * 1_000,
            initial_distance_km=initial_distance,
        )


# ---------------------------------------------------------------------------
# Referência exata e avaliação com várias sementes
# ---------------------------------------------------------------------------

def solve_exact(cities: List[City]) -> Tuple[List[City], float]:
    """Tour ótimo por enumeração (força bruta), viável só para poucas cidades.

    Fixa a primeira cidade (o tour é cíclico) e descarta a metade das
    permutações por simetria (tour e seu reverso têm o mesmo custo), o que dá
    (n-1)!/2 candidatos: 181.440 para n = 10.
    """
    first, rest = cities[0], cities[1:]
    best_route: List[City] = list(cities)
    best_distance: float = float("inf")
    for perm in itertools.permutations(rest):
        if perm[0].name > perm[-1].name:   # visita só um dos dois sentidos
            continue
        route = [first, *perm]
        distance = compute_tour_distance(route)
        if distance < best_distance:
            best_route, best_distance = route, distance
    return best_route, best_distance


def make_random_cities(n: int, instance_seed: int, half_side_km: float = 30.0) -> List[City]:
    """Instância sintética: *n* cidades uniformes em [-L, L]², com semente própria.

    A semente da instância é independente das sementes dos algoritmos.
    """
    rng = random.Random(instance_seed)
    return [
        City(f"C{i + 1:02d}", x=rng.uniform(-half_side_km, half_side_km),
             y=rng.uniform(-half_side_km, half_side_km))
        for i in range(n)
    ]


def compare_algorithms(
    cities: List[City],
    seeds: range,
    budget: int,
    schedules: Dict[str, float],
    reference_km: Optional[float] = None,
) -> None:
    """Compara HC, HC com reinícios e SA (um ou mais cronogramas) com o mesmo
    orçamento de *budget* avaliações da função de custo, para cada semente.

    Sem *reference_km* (ótimo exato), o gap usa o melhor tour visto em todas
    as execuções. Imprime média ± desvio, melhor, pior, gap e tempo mediano.
    """
    factories: Dict[str, Callable[[int], OptimizationResult]] = {
        "HillClimbing": lambda sd: HillClimbing(cities, seed=sd).optimize(budget),
        "HillClimbing+reinícios": lambda sd: HillClimbingRestarts(cities, seed=sd).optimize(budget),
    }
    for label, rate in schedules.items():
        factories[f"SA ({label})"] = lambda sd, rate=rate: SimulatedAnnealing(
            cities, initial_temperature=800.0, cooling_rate=rate, seed=sd
        ).optimize(budget)

    runs = {name: [make(sd) for sd in seeds] for name, make in factories.items()}
    best_seen = min(r.best_distance_km for rs in runs.values() for r in rs)
    reference = reference_km if reference_km is not None else best_seen
    ref_label = "ótimo exato" if reference_km is not None else "melhor visto"

    print(f"  {len(cities)} cidades | {len(seeds)} sementes | orçamento {budget:,} avaliações | "
          f"referência ({ref_label}): {reference:.2f} km")
    for name, rs in runs.items():
        dist = [r.best_distance_km for r in rs]
        mean = statistics.mean(dist)
        std = statistics.stdev(dist) if len(dist) > 1 else 0.0
        hits = sum(d <= reference + 1e-6 for d in dist)
        print(
            f"  {name:<24}: {mean:.2f} ± {std:.2f} km | melhor {min(dist):.2f} | pior {max(dist):.2f} | "
            f"gap médio {max(0.0, mean / reference * 100 - 100):.2f}% | "
            f"na referência {hits}/{len(dist)} | "
            f"avaliações (mediana) {statistics.median(r.iterations for r in rs):,.0f} | "
            f"tempo mediano {statistics.median(r.elapsed_ms for r in rs):.1f} ms"
        )


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSP: Hill Climbing vs. Simulated Annealing")
    parser.add_argument("--seeds", type=int, default=1,
                        help="com N > 1, repete com N sementes e compara com o ótimo exato")
    args = parser.parse_args()

    SEED: int = 42
    INSTANCE_SEED: int = 2026   # semente da instância aleatória, separada das dos algoritmos
    SEP: str = "─" * 62

    hc_optimizer = HillClimbing(cities=DF_CITIES, seed=SEED)
    sa_optimizer = SimulatedAnnealing(
        cities=DF_CITIES,
        initial_temperature=800.0,
        cooling_rate=0.995,
        seed=SEED,
    )

    print(SEP)
    print("  TSP — Regiões Administrativas do Distrito Federal")
    print(SEP)

    hc_result: OptimizationResult = hc_optimizer.optimize(max_iterations=50_000)
    sa_result: OptimizationResult = sa_optimizer.optimize(max_iterations=50_000)

    print(hc_result)
    print()
    print(sa_result)
    print(SEP)

    # Comparativo lado a lado
    winner: str = (
        "SimulatedAnnealing"
        if sa_result.best_distance_km < hc_result.best_distance_km
        else "HillClimbing"
    )
    gap_km: float = abs(hc_result.best_distance_km - sa_result.best_distance_km)
    print(f"  Melhor algoritmo : {winner}")
    print(f"  Diferença        : {gap_km:.2f} km")
    print(SEP)

    if args.seeds > 1:
        seeds = range(SEED, SEED + args.seeds)
        # Cronograma atual (cooling 0.995) e um mais lento (0.9998), mostrados lado a lado:
        # com T0 = 800 e 50.000 avaliações, o atual congela após ~3.000 iterações.
        schedules = {"0.995": 0.995, "0.9998": 0.9998}
        _, optimal_km = solve_exact(DF_CITIES)
        print()
        print(SEP)
        print("  Instância 1 — 10 Regiões Administrativas do DF")
        print(SEP)
        compare_algorithms(DF_CITIES, seeds, 50_000, schedules, reference_km=optimal_km)
        print(SEP)
        print(f"  Instância 2 — 40 cidades aleatórias (semente da instância = {INSTANCE_SEED})")
        print(SEP)
        compare_algorithms(make_random_cities(40, INSTANCE_SEED), seeds, 50_000, schedules)
        print(SEP)
