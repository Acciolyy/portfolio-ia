import heapq
import itertools

import pytest

import csp_map_coloring as csp
import informed_search as inf
import n_queens_genetic as ga
import projeto_01_busca_cega as blind
import tsp_local_search as tsp
import wumpus_world as wumpus


# ---------------------------------------------------------------- blind search
GRAPH = {
    "A": ["B", "C"], "B": ["D", "E"], "C": ["E"], "D": ["G"],
    "E": ["F", "H"], "F": ["A"], "G": [], "H": ["I"], "I": [],
}


def test_bfs_finds_a_shortest_path_and_dfs_a_valid_one():
    bfs = blind.BreadthFirstSearch(GRAPH).execute_search("A", "I")
    dfs = blind.DepthFirstSearch(GRAPH).execute_search("A", "I")
    assert len(bfs.path_to_target) == 5          # A, B, E, H, I (4 edges is the minimum)
    for path in (bfs.path_to_target, dfs.path_to_target):
        assert path[0] == "A" and path[-1] == "I"
        assert all(b in GRAPH[a] for a, b in zip(path, path[1:]))


# ------------------------------------------------------------- A* and Greedy
def _dijkstra(graph, source):
    dist, heap = {source: 0.0}, [(0.0, source)]
    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist[node]:
            continue
        for nxt, w in graph[node]:
            if cost + w < dist.get(nxt, float("inf")):
                dist[nxt] = cost + w
                heapq.heappush(heap, (cost + w, nxt))
    return dist


def test_astar_is_optimal_from_every_source_and_never_worse_than_greedy():
    graph, h = inf.FGA_GRAPH, inf.FGA_HEURISTICS
    to_goal = _dijkstra(graph, "RU")   # undirected graph: distance to RU from each node
    astar, greedy = inf.AStarSearch(graph, h), inf.GreedySearch(graph, h)
    for source in graph:
        if source == "RU":
            continue
        a = astar.execute_search(source, "RU")
        g = greedy.execute_search(source, "RU")
        assert a.total_cost_m == pytest.approx(to_goal[source])
        assert a.total_cost_m <= g.total_cost_m


def test_ru_heuristic_is_admissible_and_consistent():
    graph, h = inf.FGA_GRAPH, inf.FGA_HEURISTICS
    assert inf.inadmissible_nodes(graph, h, "RU") == {}
    assert all(h[u] <= w + h[v] + 1e-9 for u in graph for v, w in graph[u])


def test_heuristic_for_another_goal_is_rejected():
    assert inf.inadmissible_nodes(inf.FGA_GRAPH, inf.FGA_HEURISTICS, "MEsp")   # not admissible there
    with pytest.raises(ValueError):
        inf.AStarSearch(inf.FGA_GRAPH, inf.FGA_HEURISTICS).execute_search("Adm", "MEsp")


# ----------------------------------------------------------------------- TSP
def test_tsp_exact_solver_matches_independent_enumeration():
    cities = tsp.make_random_cities(7, instance_seed=1)
    brute = min(
        tsp.compute_tour_distance([cities[0], *perm]) for perm in itertools.permutations(cities[1:])
    )
    _, exact = tsp.solve_exact(cities)
    assert exact == pytest.approx(brute)


@pytest.mark.parametrize("make", [
    lambda c: tsp.HillClimbing(c, seed=0),
    lambda c: tsp.HillClimbingRestarts(c, seed=0),
    lambda c: tsp.SimulatedAnnealing(c, initial_temperature=800.0, cooling_rate=0.995, seed=0),
])
def test_tsp_optimizers_return_a_valid_tour_no_worse_than_the_start(make):
    cities = tsp.make_random_cities(8, instance_seed=2)
    result = make(cities).optimize(3_000)
    assert sorted(c.name for c in result.best_route) == sorted(c.name for c in cities)
    assert result.best_distance_km <= result.initial_distance_km
    assert result.best_distance_km >= tsp.solve_exact(cities)[1] - 1e-9
    assert result.iterations <= 3_000


# -------------------------------------------------------------- N-Queens (GA)
def test_genetic_algorithm_solves_8_queens_with_a_valid_board():
    result = ga.GeneticAlgorithm(n=8, seed=42).executar()
    board = result.configuracao_tabuleiro
    assert result.solucao_encontrada and sorted(board) == list(range(8))
    for c1, c2 in itertools.combinations(range(8), 2):
        assert abs(board[c1] - board[c2]) != c2 - c1      # no shared diagonal


# ----------------------------------------------------------------------- CSP
@pytest.mark.parametrize("usar_mrv", [True, False])
def test_map_coloring_gives_neighbours_different_colors(usar_mrv):
    problema = csp.MapColoringProblem(csp.GRAFO_FRONTEIRAS_CENTRO_OESTE, csp.CORES_DISPONIVEIS)
    result = csp.BacktrackingSearch(csp=problema, usar_mrv=usar_mrv).resolver()
    assert result.solucao_encontrada
    assert set(result.atribuicao) == set(csp.GRAFO_FRONTEIRAS_CENTRO_OESTE)
    for estado, vizinhos in csp.GRAFO_FRONTEIRAS_CENTRO_OESTE.items():
        assert all(result.atribuicao[estado] != result.atribuicao[v] for v in vizinhos)


# -------------------------------------------------------------------- Wumpus
def test_wumpus_agent_gets_the_gold_without_entering_a_fatal_cell():
    world = wumpus.Environment(
        tamanho=4, posicao_wumpus=(3, 1), posicoes_pocos={(2, 3)},
        posicao_ouro=(1, 2), posicao_inicial=(0, 0),
    )
    result = wumpus.KnowledgeBasedAgent(environment=world).run_simulation(limite_passos=100)
    assert result.sobreviveu and result.pegou_ouro
    assert not any(world.e_fatal(pos) for _, pos in result.trilha_acoes)


def test_wumpus_agent_never_dies_in_any_world_with_one_pit_and_this_wumpus():
    cells = [(r, c) for r in range(4) for c in range(4) if (r, c) != (0, 0)]
    wumpus_pos = (3, 1)
    for pit in (c for c in cells if c != wumpus_pos):
        for gold in (c for c in cells if c not in (wumpus_pos, pit)):
            world = wumpus.Environment(
                tamanho=4, posicao_wumpus=wumpus_pos, posicoes_pocos={pit},
                posicao_ouro=gold, posicao_inicial=(0, 0),
            )
            result = wumpus.KnowledgeBasedAgent(environment=world).run_simulation(limite_passos=100)
            assert result.sobreviveu, (pit, gold)


def test_wumpus_agent_stays_alive_without_gold_when_no_move_is_provably_safe():
    # Pits on both neighbours of the start: a breeze at (0,0) cannot say which one is safe.
    world = wumpus.Environment(
        tamanho=4, posicao_wumpus=(3, 3), posicoes_pocos={(0, 1), (1, 0)},
        posicao_ouro=(2, 2), posicao_inicial=(0, 0),
    )
    result = wumpus.KnowledgeBasedAgent(environment=world).run_simulation(limite_passos=100)
    assert result.sobreviveu and not result.pegou_ouro
