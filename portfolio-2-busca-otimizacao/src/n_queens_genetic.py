"""
n_queens_genetic.py
===================
Solução para o Problema das N-Rainhas utilizando Algoritmos Genéticos,
estruturada com Orientação a Objetos e pensada como peça de portfólio
de IA clássica.

Definição do problema
---------------------
Posicionar N rainhas em um tabuleiro N×N de forma que nenhum par de rainhas
se ataque mutuamente — ou seja, nenhuma rainha compartilha linha, coluna
ou diagonal com outra.

Representação cromossômica
--------------------------
Cada cromossomo é uma permutação de comprimento N. O índice representa a
coluna e o valor representa a linha em que a rainha está posicionada naquela
coluna. Essa codificação **elimina conflitos de coluna e linha por construção**
— só restam os conflitos diagonais para o fitness penalizar.

    Exemplo para N=4: [2, 0, 3, 1]
    Coluna 0 → rainha na linha 2
    Coluna 1 → rainha na linha 0
    Coluna 2 → rainha na linha 3
    Coluna 3 → rainha na linha 1

Fitness
-------
O fitness conta o número de pares de rainhas que **não** se atacam.
O valor máximo possível é C(N, 2) = N*(N-1)/2. Para N=8 → fitness_max = 28.

Arquitetura
-----------
::

    Cromossomo              — estado do tabuleiro + operadores genéticos
    GeneticAlgorithm        — controlador da população e do ciclo evolutivo
    ResultadoBusca          — DTO de saída (dataclass)

Parâmetros padrão
-----------------
- Tamanho da população  : 200 indivíduos
- Taxa de crossover     : 0,85 (85% dos pares geram filhos por crossover)
- Taxa de mutação       : 0,05 (5% de chance de trocar dois genes por filho)
- Seleção               : torneio de tamanho 5
- Elitismo              : os 2 melhores indivíduos são copiados diretamente
- Limite de gerações    : 2.000
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# DTO de resultado
# ---------------------------------------------------------------------------

@dataclass
class ResultadoBusca:
    """Objeto de valor retornado pelo :class:`GeneticAlgorithm` ao fim da execução.

    Atributos
    ---------
    configuracao_tabuleiro:
        Lista de comprimento N onde o índice é a coluna e o valor é a linha
        da rainha naquela coluna. Ex.: ``[2, 4, 6, 0, 3, 1, 7, 5]``.
    geracoes:
        Número de gerações evoluídas até encontrar a solução (ou atingir o limite).
    fitness_final:
        Número de pares de rainhas não-atacantes no melhor indivíduo encontrado.
    fitness_maximo:
        Valor ótimo teórico — C(N, 2) = N*(N-1)/2.
    solucao_encontrada:
        ``True`` se o fitness final atingiu o valor máximo (solução perfeita).
    tempo_ms:
        Tempo total de execução medido com ``time.perf_counter``, em milissegundos.
    """

    configuracao_tabuleiro: List[int]
    geracoes: int
    fitness_final: int
    fitness_maximo: int
    solucao_encontrada: bool = field(init=False)
    tempo_ms: float

    def __post_init__(self) -> None:
        self.solucao_encontrada = self.fitness_final == self.fitness_maximo

    def __str__(self) -> str:
        status = "✓ solução ótima encontrada" if self.solucao_encontrada else "✗ limite de gerações atingido"
        return (
            f"[GeneticAlgorithm — N-Rainhas]\n"
            f"  Tabuleiro    : {self.configuracao_tabuleiro}\n"
            f"  Fitness      : {self.fitness_final} / {self.fitness_maximo}\n"
            f"  Gerações     : {self.geracoes}\n"
            f"  Tempo        : {self.tempo_ms:.2f} ms\n"
            f"  Status       : {status}"
        )


# ---------------------------------------------------------------------------
# Cromossomo
# ---------------------------------------------------------------------------

class Cromossomo:
    """Representa um indivíduo da população — um estado completo do tabuleiro.

    A codificação usada é uma **permutação** dos inteiros [0, N-1]:
    o índice ``i`` da lista indica a coluna e o valor ``genes[i]`` indica
    a linha onde a rainha está posicionada na coluna ``i``. Por ser uma
    permutação, conflitos de linha e coluna são impossíveis por construção;
    o algoritmo precisa apenas minimizar conflitos diagonais.

    Atributos
    ---------
    genes:
        Permutação de comprimento N representando as posições das rainhas.
    _fitness:
        Cache do valor de fitness calculado em :meth:`calcular_fitness`.
        Usar ``None`` como sentinela evita recomputar quando os genes não
        foram alterados.

    Parâmetros
    ----------
    n:
        Tamanho do tabuleiro (número de rainhas).
    genes:
        Permutação inicial. Se ``None``, uma permutação aleatória é gerada.
    rng:
        Instância de ``random.Random`` injetada para garantir reprodutibilidade
        sem depender do estado global do módulo ``random``.
    """

    def __init__(
        self,
        n: int,
        genes: Optional[List[int]] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._n: int = n
        self._rng: random.Random = rng or random.Random()
        # Se nenhum gene foi fornecido, gera uma permutação aleatória
        self.genes: List[int] = genes if genes is not None else self._gerar_permutacao()
        self._fitness: Optional[int] = None   # calculado sob demanda (lazy)

    # ------------------------------------------------------------------
    # Geração de permutação aleatória
    # ------------------------------------------------------------------

    def _gerar_permutacao(self) -> List[int]:
        """Cria e embaralha uma permutação de [0, N-1].

        O embaralhamento de Fisher-Yates implementado por ``random.shuffle``
        garante distribuição uniforme sobre todas as N! permutações possíveis.
        """
        permutacao = list(range(self._n))
        self._rng.shuffle(permutacao)
        return permutacao

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def calcular_fitness(self) -> int:
        """Calcula e armazena em cache o número de pares não-atacantes.

        Para cada par de rainhas (i, j) com i < j, verifica se elas estão
        na mesma diagonal. Conflito diagonal ocorre quando:

            |genes[i] - genes[j]| == |i - j|

        O fitness é:

            pares_totais - pares_em_conflito
            = C(N,2) - numero_de_conflitos

        Um fitness igual a C(N,2) = N*(N-1)//2 indica solução perfeita.

        Retorna
        -------
        int
            Número de pares de rainhas que não se atacam mutuamente.
        """
        if self._fitness is not None:
            return self._fitness   # retorna o valor em cache

        conflitos = 0
        for i in range(self._n):
            for j in range(i + 1, self._n):
                # Verifica conflito diagonal: distância de linha == distância de coluna
                if abs(self.genes[i] - self.genes[j]) == abs(i - j):
                    conflitos += 1

        pares_totais = self._n * (self._n - 1) // 2
        self._fitness = pares_totais - conflitos
        return self._fitness

    def invalidar_cache(self) -> None:
        """Descarta o fitness em cache após uma mutação nos genes."""
        self._fitness = None

    # ------------------------------------------------------------------
    # Crossover — Order Crossover (OX1)
    # ------------------------------------------------------------------

    def crossover(self, outro: Cromossomo) -> Tuple[Cromossomo, Cromossomo]:
        """Aplica Order Crossover (OX1) entre este cromossomo e *outro*.

        O OX1 preserva a propriedade de permutação nos filhos:

        1. Seleciona um segmento contíguo aleatório do pai A e copia-o
           diretamente para o filho na mesma posição.
        2. Preenche as posições restantes do filho percorrendo o pai B na
           ordem circular, pulando os genes já presentes no segmento copiado.

        Esse operador é amplamente utilizado para problemas de permutação
        pois garante que cada valor aparece exatamente uma vez no filho,
        mantendo a codificação válida sem necessidade de reparo.

        Parâmetros
        ----------
        outro:
            O segundo pai para o crossover.

        Retorna
        -------
        Tuple[Cromossomo, Cromossomo]
            Dois filhos resultantes do cruzamento.
        """
        # Escolhe dois pontos de corte aleatórios (ponto1 < ponto2)
        ponto1 = self._rng.randint(0, self._n - 2)
        ponto2 = self._rng.randint(ponto1 + 1, self._n - 1)

        filho1_genes = self._ox1(self.genes, outro.genes, ponto1, ponto2)
        filho2_genes = self._ox1(outro.genes, self.genes, ponto1, ponto2)

        return (
            Cromossomo(self._n, filho1_genes, self._rng),
            Cromossomo(self._n, filho2_genes, self._rng),
        )

    @staticmethod
    def _ox1(
        pai_a: List[int],
        pai_b: List[int],
        ponto1: int,
        ponto2: int,
    ) -> List[int]:
        """Executa a lógica OX1 de pai_a → filho usando pai_b para o preenchimento.

        Parâmetros
        ----------
        pai_a:
            Pai do qual o segmento central é copiado diretamente.
        pai_b:
            Pai que fornece os genes restantes na ordem circular.
        ponto1:
            Início do segmento copiado (inclusivo).
        ponto2:
            Fim do segmento copiado (inclusivo).

        Retorna
        -------
        List[int]
            Genes do filho com permutação garantida.
        """
        n = len(pai_a)
        filho = [-1] * n

        # Passo 1 — copia o segmento de pai_a para o filho
        segmento = set()
        for idx in range(ponto1, ponto2 + 1):
            filho[idx] = pai_a[idx]
            segmento.add(pai_a[idx])

        # Passo 2 — preenche as lacunas com os genes de pai_b em ordem circular
        posicao_filho = (ponto2 + 1) % n
        for gene in pai_b[ponto2 + 1:] + pai_b[:ponto2 + 1]:
            if gene not in segmento:
                filho[posicao_filho] = gene
                posicao_filho = (posicao_filho + 1) % n

        return filho

    # ------------------------------------------------------------------
    # Mutação — troca de dois genes (swap mutation)
    # ------------------------------------------------------------------

    def mutar(self, taxa_mutacao: float) -> None:
        """Aplica mutação por troca (swap mutation) com probabilidade *taxa_mutacao*.

        Sorteia um número uniforme em [0, 1). Se for menor que *taxa_mutacao*,
        dois índices distintos são sorteados aleatoriamente e seus genes são
        trocados. A troca preserva a propriedade de permutação.

        Exemplo:
            genes antes : [2, 4, 1, 3]
            índices      : i=1, j=3
            genes depois : [2, 3, 1, 4]

        A taxa padrão de 0,05 equivale a uma mutação esperada a cada 20 filhos,
        o que é conservador o suficiente para não destruir boas soluções parciais,
        mas suficiente para manter diversidade genética na população.

        Parâmetros
        ----------
        taxa_mutacao:
            Probabilidade de mutação por indivíduo. Valor recomendado: 0,01–0,10.
        """
        if self._rng.random() < taxa_mutacao:
            # Sorteia dois índices distintos para a troca
            i, j = self._rng.sample(range(self._n), 2)
            self.genes[i], self.genes[j] = self.genes[j], self.genes[i]
            self.invalidar_cache()   # o fitness anterior não é mais válido

    def copiar(self) -> Cromossomo:
        """Retorna uma cópia profunda deste cromossomo (usado no elitismo)."""
        clone = Cromossomo(self._n, list(self.genes), self._rng)
        clone._fitness = self._fitness   # preserva o cache de fitness
        return clone


# ---------------------------------------------------------------------------
# Algoritmo Genético
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """Controlador do ciclo evolutivo para o Problema das N-Rainhas.

    Implementa o loop geracional clássico:
        1. Avaliação de fitness da população
        2. Seleção por torneio
        3. Crossover OX1 com taxa configurável
        4. Mutação por troca com taxa configurável
        5. Elitismo — preserva os melhores indivíduos sem alteração

    Parâmetros
    ----------
    n:
        Tamanho do tabuleiro (número de rainhas). Padrão: 8.
    tamanho_populacao:
        Número de indivíduos por geração. Populações maiores exploram mais
        o espaço de busca, mas aumentam o custo computacional por geração.
        Padrão: 200.
    taxa_crossover:
        Probabilidade de aplicar crossover a um par de pais. Se não aplicado,
        os pais são copiados diretamente para a próxima geração.
        Padrão: 0,85.
    taxa_mutacao:
        Probabilidade de mutação por indivíduo após o crossover.
        Padrão: 0,05.
    tamanho_torneio:
        Número de indivíduos que competem em cada seleção por torneio.
        Valores maiores aumentam a pressão seletiva.
        Padrão: 5.
    elitismo:
        Número de melhores indivíduos copiados diretamente para a próxima
        geração, garantindo que a qualidade nunca regrida entre gerações.
        Padrão: 2.
    limite_geracoes:
        Número máximo de gerações antes de encerrar sem solução ótima.
        Padrão: 2000.
    seed:
        Semente aleatória opcional para reprodutibilidade dos experimentos.
    """

    # Taxa de crossover padrão: 85% dos pares de pais geram filhos por cruzamento
    TAXA_CROSSOVER_PADRAO: float = 0.85

    # Taxa de mutação padrão: 5% de chance de swap por indivíduo
    TAXA_MUTACAO_PADRAO: float = 0.05

    def __init__(
        self,
        n: int = 8,
        tamanho_populacao: int = 200,
        taxa_crossover: float = TAXA_CROSSOVER_PADRAO,
        taxa_mutacao: float = TAXA_MUTACAO_PADRAO,
        tamanho_torneio: int = 5,
        elitismo: int = 2,
        limite_geracoes: int = 2_000,
        seed: Optional[int] = None,
    ) -> None:
        self._n: int = n
        self._tamanho_populacao: int = tamanho_populacao
        self._taxa_crossover: float = taxa_crossover
        self._taxa_mutacao: float = taxa_mutacao
        self._tamanho_torneio: int = tamanho_torneio
        self._elitismo: int = elitismo
        self._limite_geracoes: int = limite_geracoes
        self._rng: random.Random = random.Random(seed)

        # Valor de fitness que representa a solução perfeita para N rainhas:
        # C(N, 2) = N*(N-1)/2 pares de rainhas, todos não-atacantes
        self._fitness_maximo: int = n * (n - 1) // 2

    # ------------------------------------------------------------------
    # Inicialização da população
    # ------------------------------------------------------------------

    def _inicializar_populacao(self) -> List[Cromossomo]:
        """Gera a população inicial com indivíduos aleatórios.

        Retorna
        -------
        List[Cromossomo]
            Lista de *tamanho_populacao* cromossomos com permutações aleatórias.
        """
        return [
            Cromossomo(self._n, rng=self._rng)
            for _ in range(self._tamanho_populacao)
        ]

    # ------------------------------------------------------------------
    # Seleção por torneio
    # ------------------------------------------------------------------

    def _selecao_torneio(self, populacao: List[Cromossomo]) -> Cromossomo:
        """Seleciona um indivíduo via torneio de tamanho *tamanho_torneio*.

        Sorteia *k* candidatos aleatoriamente da população e retorna aquele
        com maior fitness. O torneio é mais simples de implementar do que
        a roleta e lida melhor com fitness não-uniforme, pois não requer
        normalização da distribuição de probabilidade.

        Uma pressão seletiva maior (torneio grande) converge mais rápido,
        mas pode reduzir diversidade e causar convergência prematura.
        Uma pressão menor (torneio pequeno) explora mais, porém converge
        mais devagar.

        Parâmetros
        ----------
        populacao:
            Lista de cromossomos da geração atual.

        Retorna
        -------
        Cromossomo
            O indivíduo vencedor do torneio.
        """
        candidatos = self._rng.sample(populacao, self._tamanho_torneio)
        return max(candidatos, key=lambda c: c.calcular_fitness())

    # ------------------------------------------------------------------
    # Formação da próxima geração
    # ------------------------------------------------------------------

    def _formar_proxima_geracao(
        self, populacao: List[Cromossomo]
    ) -> List[Cromossomo]:
        """Constrói a próxima geração aplicando elitismo, crossover e mutação.

        O fluxo é:
            1. Ordena a população por fitness (decrescente).
            2. Copia os *elitismo* melhores indivíduos sem alteração.
            3. Preenche o restante por pares selecionados via torneio:
               - Com probabilidade *taxa_crossover*, aplica OX1 e gera dois filhos.
               - Caso contrário, copia os pais diretamente.
            4. Aplica mutação em todos os indivíduos não-elites.

        Parâmetros
        ----------
        populacao:
            Geração atual.

        Retorna
        -------
        List[Cromossomo]
            Nova geração do mesmo tamanho.
        """
        # Ordena do melhor para o pior para facilitar o elitismo
        populacao_ordenada = sorted(
            populacao, key=lambda c: c.calcular_fitness(), reverse=True
        )

        proxima_geracao: List[Cromossomo] = []

        # Elitismo — copia os melhores sem modificação
        for elite in populacao_ordenada[: self._elitismo]:
            proxima_geracao.append(elite.copiar())

        # Preenche o restante até atingir o tamanho original
        while len(proxima_geracao) < self._tamanho_populacao:
            pai_a = self._selecao_torneio(populacao)
            pai_b = self._selecao_torneio(populacao)

            if self._rng.random() < self._taxa_crossover:
                # Gera dois filhos por crossover
                filho_a, filho_b = pai_a.crossover(pai_b)
            else:
                # Sem crossover: filhos são cópias diretas dos pais
                filho_a, filho_b = pai_a.copiar(), pai_b.copiar()

            # Aplica mutação nos filhos gerados
            filho_a.mutar(self._taxa_mutacao)
            filho_b.mutar(self._taxa_mutacao)

            proxima_geracao.append(filho_a)
            # Garante não ultrapassar o tamanho da população
            if len(proxima_geracao) < self._tamanho_populacao:
                proxima_geracao.append(filho_b)

        return proxima_geracao

    # ------------------------------------------------------------------
    # Método principal de execução
    # ------------------------------------------------------------------

    def executar(self) -> ResultadoBusca:
        """Executa o ciclo evolutivo até solução ótima ou limite de gerações.

        A cada geração, avalia todos os indivíduos e verifica se algum
        atingiu o fitness máximo. Se sim, encerra imediatamente. Caso
        contrário, forma a próxima geração e repete.

        Retorna
        -------
        ResultadoBusca
            DTO com a melhor configuração encontrada, métricas de desempenho
            e flag indicando se a solução ótima foi atingida.
        """
        inicio: float = time.perf_counter()

        populacao = self._inicializar_populacao()
        melhor_individuo: Cromossomo = max(
            populacao, key=lambda c: c.calcular_fitness()
        )

        for geracao in range(1, self._limite_geracoes + 1):
            # Verifica se o melhor atual já é a solução ótima
            if melhor_individuo.calcular_fitness() == self._fitness_maximo:
                break

            # Evolui para a próxima geração
            populacao = self._formar_proxima_geracao(populacao)

            # Atualiza o melhor indivíduo global
            melhor_geracao = max(populacao, key=lambda c: c.calcular_fitness())
            if melhor_geracao.calcular_fitness() > melhor_individuo.calcular_fitness():
                melhor_individuo = melhor_geracao.copiar()

        tempo_ms: float = (time.perf_counter() - inicio) * 1_000

        return ResultadoBusca(
            configuracao_tabuleiro=melhor_individuo.genes,
            geracoes=geracao,
            fitness_final=melhor_individuo.calcular_fitness(),
            fitness_maximo=self._fitness_maximo,
            tempo_ms=tempo_ms,
        )


# ---------------------------------------------------------------------------
# Visualização auxiliar do tabuleiro
# ---------------------------------------------------------------------------

def visualizar_tabuleiro(genes: List[int]) -> str:
    """Renderiza o tabuleiro em texto com ♛ nas posições das rainhas.

    Parâmetros
    ----------
    genes:
        Configuração do cromossomo (permutação de colunas → linhas).

    Retorna
    -------
    str
        Representação ASCII do tabuleiro N×N.
    """
    n = len(genes)
    linhas: List[str] = []
    separador = "  +" + ("---+" * n)

    for linha in range(n):
        linhas.append(separador)
        celulas = []
        for coluna in range(n):
            # genes[coluna] == linha significa que há rainha nessa célula
            celulas.append(" ♛ " if genes[coluna] == linha else "   ")
        linhas.append(f"{linha} |" + "|".join(celulas) + "|")

    linhas.append(separador)
    colunas_label = "  " + "".join(f"  {c} " for c in range(n))
    linhas.append(colunas_label)
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEP = "─" * 55

    # Configuração padrão: N=8, seed fixo para reprodutibilidade
    algoritmo = GeneticAlgorithm(
        n=8,
        tamanho_populacao=200,
        taxa_crossover=0.85,    # 85% dos pares fazem crossover
        taxa_mutacao=0.05,      # 5% de chance de swap por indivíduo
        tamanho_torneio=5,
        elitismo=2,
        limite_geracoes=2_000,
        seed=42,
    )

    resultado = algoritmo.executar()

    print(SEP)
    print(resultado)
    print(SEP)
    print("\nTabuleiro encontrado:\n")
    print(visualizar_tabuleiro(resultado.configuracao_tabuleiro))
