"""
csp_map_coloring.py
===================
Infraestrutura Orientada a Objetos para resolução de Problemas de Satisfação
de Restrições (CSP — Constraint Satisfaction Problem), aplicada ao problema
clássico de Coloração de Grafos.

Problema alvo
-------------
Colorir o mapa da região Centro-Oeste do Brasil (Distrito Federal, Goiás,
Mato Grosso e Mato Grosso do Sul) junto com seus estados fronteiriços,
de forma que dois estados vizinhos (que compartilham fronteira) nunca
recebam a mesma cor, utilizando no máximo 4 cores.

O Teorema das Quatro Cores garante que qualquer mapa planar é colorível
com 4 cores — portanto a instância é sempre satisfatível.

Modelagem como CSP
------------------
Um CSP é definido formalmente pela tripla ``(X, D, C)``:

* **X** — conjunto de variáveis. Aqui, cada estado é uma variável.
* **D** — domínios. O domínio de cada variável é o conjunto de 4 cores.
* **C** — restrições. Para cada par de estados vizinhos existe a restrição
  binária de desigualdade: ``cor(A) ≠ cor(B)``.

Arquitetura
-----------
::

    ConstraintSatisfactionProblem  (ABC)
    └── MapColoringProblem         — instância concreta de coloração de mapa

    BacktrackingSearch             — solver com retrocesso + heurística MRV
    ResultadoCSP                   — DTO de saída (dataclass)

Decisões arquiteturais
----------------------
A separação entre a *definição do problema* (classe CSP) e o *algoritmo de
busca* (BacktrackingSearch) segue o princípio da Responsabilidade Única:
o mesmo solver pode resolver qualquer subclasse de CSP sem alteração, e o
mesmo problema pode ser atacado por solvers diferentes.
"""

from __future__ import annotations

import argparse
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Aliases de tipo
# ---------------------------------------------------------------------------

#: Uma variável do CSP é identificada por seu nome (ex.: sigla do estado).
Variavel = str

#: Um valor do domínio — neste problema, o nome de uma cor.
Valor = str

#: Uma atribuição parcial ou completa: mapeia variáveis a valores escolhidos.
Atribuicao = Dict[Variavel, Valor]


# ---------------------------------------------------------------------------
# DTO de resultado
# ---------------------------------------------------------------------------

@dataclass
class ResultadoCSP:
    """Objeto de valor retornado pelo solver ao fim da busca.

    Atributos
    ---------
    atribuicao:
        Mapa final de variável → valor. Vazio se nenhuma solução existe.
    backtracks:
        Número de retrocessos realizados — toda vez que o algoritmo desfaz
        uma atribuição por falha de consistência, este contador incrementa.
        É a métrica clássica de eficiência de um solver CSP.
    nos_explorados:
        Número total de tentativas de atribuição (chamadas recursivas).
    tempo_ms:
        Tempo de execução medido com ``time.perf_counter``, em milissegundos.
    solucao_encontrada:
        ``True`` se uma atribuição completa e consistente foi encontrada.
    """

    atribuicao: Atribuicao
    backtracks: int
    nos_explorados: int
    tempo_ms: float
    solucao_encontrada: bool = field(init=False)

    def __post_init__(self) -> None:
        self.solucao_encontrada = bool(self.atribuicao)

    def __str__(self) -> str:
        status = "✓ solução encontrada" if self.solucao_encontrada else "✗ sem solução"
        linhas = [
            "[BacktrackingSearch — Coloração de Mapa]",
            f"  Status         : {status}",
            f"  Backtracks     : {self.backtracks}",
            f"  Nós explorados : {self.nos_explorados}",
            f"  Tempo          : {self.tempo_ms:.4f} ms",
        ]
        if self.solucao_encontrada:
            linhas.append("  Atribuição     :")
            for variavel in sorted(self.atribuicao):
                linhas.append(f"    {variavel:<6} → {self.atribuicao[variavel]}")
        return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Classe base abstrata do CSP
# ---------------------------------------------------------------------------

class ConstraintSatisfactionProblem(ABC):
    """Base abstrata que define a estrutura formal de um CSP.

    Encapsula a tripla ``(X, D, C)``:

    * ``_variaveis``  — conjunto X de variáveis.
    * ``_dominios``   — função D que mapeia cada variável ao seu domínio.
    * As restrições C são expressas pelo método abstrato :meth:`restricao_satisfeita`.

    Manter as restrições como um método (em vez de uma estrutura de dados)
    permite que subclasses expressem restrições arbitrárias — binárias,
    n-árias ou globais — sem alterar a interface do solver.

    Parâmetros
    ----------
    variaveis:
        Lista de todas as variáveis do problema.
    dominios:
        Mapa de cada variável para a lista de valores possíveis.
    """

    def __init__(
        self,
        variaveis: List[Variavel],
        dominios: Dict[Variavel, List[Valor]],
    ) -> None:
        self._variaveis: List[Variavel] = variaveis
        self._dominios: Dict[Variavel, List[Valor]] = dominios

    @property
    def variaveis(self) -> List[Variavel]:
        """Retorna a lista de variáveis do CSP (conjunto X)."""
        return self._variaveis

    def dominio(self, variavel: Variavel) -> List[Valor]:
        """Retorna o domínio (lista de valores possíveis) de *variavel*."""
        return self._dominios[variavel]

    # ------------------------------------------------------------------
    # Interface abstrata de restrições
    # ------------------------------------------------------------------

    @abstractmethod
    def vizinhos(self, variavel: Variavel) -> Set[Variavel]:
        """Retorna o conjunto de variáveis restritas por *variavel*.

        Em coloração de grafos, são os estados que compartilham fronteira.
        O solver usa este método para verificar consistência apenas contra
        as variáveis relevantes, evitando checar todas as outras.
        """

    @abstractmethod
    def restricao_satisfeita(
        self,
        variavel_a: Variavel,
        valor_a: Valor,
        variavel_b: Variavel,
        valor_b: Valor,
    ) -> bool:
        """Verifica se a restrição entre duas variáveis é respeitada.

        Parâmetros
        ----------
        variavel_a, valor_a:
            A primeira variável e o valor a ela atribuído.
        variavel_b, valor_b:
            A segunda variável e o valor a ela atribuído.

        Retorna
        -------
        bool
            ``True`` se a atribuição conjunta não viola a restrição.
        """

    # ------------------------------------------------------------------
    # Verificação de consistência (lógica genérica)
    # ------------------------------------------------------------------

    def atribuicao_consistente(
        self,
        variavel: Variavel,
        valor: Valor,
        atribuicao: Atribuicao,
    ) -> bool:
        """Verifica se atribuir *valor* a *variavel* mantém a consistência.

        Percorre apenas os vizinhos já atribuídos de *variavel* e confirma
        que nenhuma restrição binária é violada. Vizinhos ainda não atribuídos
        são ignorados — eles serão verificados quando receberem valor.

        Parâmetros
        ----------
        variavel:
            A variável que se pretende atribuir.
        valor:
            O valor candidato.
        atribuicao:
            A atribuição parcial corrente.

        Retorna
        -------
        bool
            ``True`` se a atribuição candidata é consistente.
        """
        for vizinho in self.vizinhos(variavel):
            if vizinho not in atribuicao:
                continue   # vizinho ainda sem cor — nada a verificar
            if not self.restricao_satisfeita(
                variavel, valor, vizinho, atribuicao[vizinho]
            ):
                return False
        return True


# ---------------------------------------------------------------------------
# Instância concreta — Coloração de Mapa
# ---------------------------------------------------------------------------

class MapColoringProblem(ConstraintSatisfactionProblem):
    """CSP concreto para coloração de mapa via grafo de adjacência.

    Cada estado é uma variável; o domínio é o conjunto de cores; e existe
    uma restrição de desigualdade entre cada par de estados vizinhos.

    Parâmetros
    ----------
    grafo_fronteiras:
        Mapa de cada estado para o conjunto de estados com os quais faz
        fronteira. Deve ser simétrico: se B ∈ grafo[A], então A ∈ grafo[B].
    cores:
        Lista de cores disponíveis (o domínio compartilhado por todas as
        variáveis).
    """

    def __init__(
        self,
        grafo_fronteiras: Dict[Variavel, Set[Variavel]],
        cores: List[Valor],
    ) -> None:
        variaveis = list(grafo_fronteiras.keys())
        # Todas as variáveis compartilham o mesmo domínio: as 4 cores
        dominios = {estado: list(cores) for estado in variaveis}
        super().__init__(variaveis, dominios)
        self._grafo_fronteiras: Dict[Variavel, Set[Variavel]] = grafo_fronteiras

    def vizinhos(self, variavel: Variavel) -> Set[Variavel]:
        """Retorna os estados que fazem fronteira com *variavel*."""
        return self._grafo_fronteiras.get(variavel, set())

    def restricao_satisfeita(
        self,
        variavel_a: Variavel,
        valor_a: Valor,
        variavel_b: Variavel,
        valor_b: Valor,
    ) -> bool:
        """Restrição de coloração: estados vizinhos devem ter cores distintas.

        Retorna ``True`` quando as cores são diferentes — a única condição
        que a coloração de mapa exige.
        """
        return valor_a != valor_b


# ---------------------------------------------------------------------------
# Solver — Busca com Retrocesso (Backtracking)
# ---------------------------------------------------------------------------

class BacktrackingSearch:
    """Solver de Backtracking clássico com heurística MRV opcional.

    O backtracking é uma busca em profundidade sobre o espaço de atribuições
    parciais. Ele atribui valores a uma variável de cada vez; se nenhuma
    atribuição mantém a consistência, *retrocede* (backtrack) e tenta outro
    valor para a variável anterior.

    Heurística MRV (Minimum Remaining Values)
    ------------------------------------------
    Em vez de escolher a próxima variável em ordem fixa, a MRV seleciona a
    variável com o **menor número de valores legais restantes** em seu domínio.

    A intuição é o princípio "fail-first": atacar primeiro a variável mais
    restrita aumenta a chance de detectar becos sem saída cedo, podando ramos
    inteiros da árvore de busca e reduzindo drasticamente o número de
    backtracks. É uma das heurísticas mais eficazes para CSPs.

    Parâmetros
    ----------
    csp:
        A instância do problema a resolver (qualquer subclasse de
        :class:`ConstraintSatisfactionProblem`).
    usar_mrv:
        Se ``True``, usa a heurística MRV na seleção de variáveis.
        Se ``False``, usa ordem estática de declaração (útil para comparar
        o ganho da heurística).
    """

    def __init__(
        self,
        csp: ConstraintSatisfactionProblem,
        usar_mrv: bool = True,
    ) -> None:
        self._csp: ConstraintSatisfactionProblem = csp
        self._usar_mrv: bool = usar_mrv

        # Contadores de instrumentação, reiniciados a cada chamada de resolver()
        self._backtracks: int = 0
        self._nos_explorados: int = 0

    # ------------------------------------------------------------------
    # Seleção da próxima variável
    # ------------------------------------------------------------------

    def _selecionar_variavel(self, atribuicao: Atribuicao) -> Variavel:
        """Escolhe a próxima variável não atribuída a receber um valor.

        Com MRV ativada, retorna a variável não atribuída cujo domínio tem
        o menor número de valores ainda consistentes. Sem MRV, retorna a
        primeira variável não atribuída na ordem de declaração.

        Parâmetros
        ----------
        atribuicao:
            A atribuição parcial corrente.

        Retorna
        -------
        Variavel
            A variável selecionada para a próxima atribuição.
        """
        nao_atribuidas = [
            v for v in self._csp.variaveis if v not in atribuicao
        ]

        if not self._usar_mrv:
            # Ordem estática: simplesmente a primeira não atribuída
            return nao_atribuidas[0]

        # MRV — escolhe a variável com menor número de valores legais restantes
        return min(
            nao_atribuidas,
            key=lambda v: self._contar_valores_legais(v, atribuicao),
        )

    def _contar_valores_legais(
        self, variavel: Variavel, atribuicao: Atribuicao
    ) -> int:
        """Conta quantos valores do domínio de *variavel* ainda são consistentes.

        Este é o critério numérico da heurística MRV: quanto menor o retorno,
        mais restrita está a variável.

        Parâmetros
        ----------
        variavel:
            A variável a avaliar.
        atribuicao:
            A atribuição parcial corrente.

        Retorna
        -------
        int
            Número de valores legais (consistentes) restantes no domínio.
        """
        return sum(
            1
            for valor in self._csp.dominio(variavel)
            if self._csp.atribuicao_consistente(variavel, valor, atribuicao)
        )

    # ------------------------------------------------------------------
    # Recursão de backtracking
    # ------------------------------------------------------------------

    def _backtrack(self, atribuicao: Atribuicao) -> Optional[Atribuicao]:
        """Núcleo recursivo da busca com retrocesso.

        Caso base: se todas as variáveis estão atribuídas, a atribuição é
        completa e consistente — é uma solução.

        Passo recursivo: seleciona a próxima variável, e para cada valor
        de seu domínio que mantém a consistência, tenta estender a atribuição
        recursivamente. Se a recursão falha para todos os valores, retrocede
        retornando ``None``.

        Parâmetros
        ----------
        atribuicao:
            A atribuição parcial corrente.

        Retorna
        -------
        Optional[Atribuicao]
            A atribuição completa se uma solução for encontrada neste ramo,
            ou ``None`` se o ramo é um beco sem saída.
        """
        # Caso base — atribuição completa
        if len(atribuicao) == len(self._csp.variaveis):
            return atribuicao

        variavel = self._selecionar_variavel(atribuicao)

        for valor in self._csp.dominio(variavel):
            self._nos_explorados += 1

            if self._csp.atribuicao_consistente(variavel, valor, atribuicao):
                # Atribui tentativamente e desce na árvore de busca
                atribuicao[variavel] = valor
                resultado = self._backtrack(atribuicao)

                if resultado is not None:
                    return resultado   # solução propagada para cima

                # Falha — desfaz a atribuição (este é o "backtrack")
                del atribuicao[variavel]
                self._backtracks += 1

        # Nenhum valor funcionou para esta variável neste ramo
        return None

    # ------------------------------------------------------------------
    # Ponto de entrada público
    # ------------------------------------------------------------------

    def resolver(self) -> ResultadoCSP:
        """Executa a busca completa e retorna o DTO com o resultado.

        Reinicia os contadores de instrumentação, dispara a recursão de
        backtracking a partir de uma atribuição vazia e empacota o resultado.

        Retorna
        -------
        ResultadoCSP
            DTO com a atribuição final, número de backtracks, nós explorados
            e tempo de execução.
        """
        # Reinicia a instrumentação para esta execução
        self._backtracks = 0
        self._nos_explorados = 0

        inicio: float = time.perf_counter()
        solucao = self._backtrack({})
        tempo_ms: float = (time.perf_counter() - inicio) * 1_000

        return ResultadoCSP(
            atribuicao=solucao if solucao is not None else {},
            backtracks=self._backtracks,
            nos_explorados=self._nos_explorados,
            tempo_ms=tempo_ms,
        )


# ---------------------------------------------------------------------------
# Dataset — Mapa do Centro-Oeste e estados fronteiriços
# ---------------------------------------------------------------------------

#: Grafo de fronteiras entre os estados modelados.
#:
#: Núcleo Centro-Oeste : DF, GO, MT, MS
#: Estados fronteiriços : TO, BA, MG, SP, PR, RO
#:
#: O grafo é simétrico — toda fronteira é registrada nos dois sentidos —
#: pois a relação de adjacência em um mapa é mútua.
GRAFO_FRONTEIRAS_CENTRO_OESTE: Dict[Variavel, Set[Variavel]] = {
    "DF": {"GO"},
    "GO": {"DF", "MT", "MS", "TO", "BA", "MG"},
    "MT": {"GO", "MS", "TO", "RO"},
    "MS": {"GO", "MT", "SP", "PR"},
    "TO": {"GO", "MT", "BA"},
    "BA": {"GO", "TO", "MG"},
    "MG": {"GO", "BA", "SP"},
    "SP": {"MS", "MG", "PR"},
    "PR": {"MS", "SP"},
    "RO": {"MT"},
}

#: Domínio compartilhado — as 4 cores disponíveis.
CORES_DISPONIVEIS: List[Valor] = ["Vermelho", "Verde", "Azul", "Amarelo"]


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1,
                        help="com N > 1, repete a medição de tempo N vezes e reporta a mediana")
    args = parser.parse_args()

    SEP = "─" * 55

    problema = MapColoringProblem(
        grafo_fronteiras=GRAFO_FRONTEIRAS_CENTRO_OESTE,
        cores=CORES_DISPONIVEIS,
    )

    # --- Execução COM a heurística MRV ---
    solver_mrv = BacktrackingSearch(csp=problema, usar_mrv=True)
    resultado_mrv = solver_mrv.resolver()

    print(SEP)
    print("  COM heurística MRV")
    print(SEP)
    print(resultado_mrv)

    # --- Execução SEM MRV (ordem estática) para comparação ---
    solver_estatico = BacktrackingSearch(csp=problema, usar_mrv=False)
    resultado_estatico = solver_estatico.resolver()

    print()
    print(SEP)
    print("  SEM heurística MRV (ordem estática)")
    print(SEP)
    print(resultado_estatico)

    # --- Comparativo do ganho da heurística ---
    print()
    print(SEP)
    print("  Comparativo de eficiência")
    print(SEP)
    print(f"  Backtracks com MRV    : {resultado_mrv.backtracks}")
    print(f"  Backtracks sem MRV    : {resultado_estatico.backtracks}")
    print(f"  Nós explorados c/ MRV : {resultado_mrv.nos_explorados}")
    print(f"  Nós explorados s/ MRV : {resultado_estatico.nos_explorados}")
    print(SEP)

    if args.repeat > 1:
        print(f"  Tempo mediano ({args.repeat} repetições)")
        print(SEP)
        for rotulo, usar_mrv in (("com MRV", True), ("sem MRV", False)):
            runs = [BacktrackingSearch(csp=problema, usar_mrv=usar_mrv).resolver() for _ in range(args.repeat)]
            print(f"  {rotulo}: {statistics.median(r.tempo_ms for r in runs):.4f} ms "
                  f"| {runs[0].nos_explorados} nós | {runs[0].backtracks} backtracks")
        print(SEP)
