"""
wumpus_world.py
===============
Agente Baseado em Conhecimento (Knowledge-Based Agent) para o Mundo do Wumpus
em um grid 4×4 — implementação de portfólio de IA clássica.

O Mundo do Wumpus
-----------------
Um agente explora uma caverna em grade. Em algum lugar há:

* **Ouro**   — o objetivo; pegá-lo encerra a missão com sucesso.
* **Wumpus** — uma criatura mortal; entrar em sua célula mata o agente.
* **Poços**  — buracos mortais; cair em um encerra a missão em fracasso.

O agente percebe o ambiente apenas localmente, por meio de *percepções*:

* **Fedor (Stench)** — há um Wumpus em uma célula adjacente.
* **Brisa (Breeze)** — há um poço em uma célula adjacente.
* **Brilho (Glitter)** — há ouro na célula atual.

Arquitetura
-----------
A solução separa rigorosamente dois blocos de responsabilidade:

::

    Environment           — detém a verdade do mundo (posições reais) e
                            gera as percepções da célula consultada.
    KnowledgeBasedAgent   — não conhece o mapa; mantém crenças (seguro,
                            visitado, fronteira de risco) e decide a ação
                            por inferência lógica, sem busca informada.

Separação rigorosa Environment / Agent
--------------------------------------
O agente NUNCA consulta o ambiente sobre células onde não esteve. Toda
percepção sentida é registrada em sua própria base de conhecimento
(``_percepcoes_conhecidas``). A inferência opera exclusivamente sobre
essa estrutura interna — o ambiente é tocado apenas no instante em que
o agente fisicamente ocupa uma célula. Isso garante que o agente resolve
o problema com o que ele sabe, não com o que o mundo sabe.

A inferência é puramente proposicional e local — nada de A* ou busca
heurística. As regras são:

1. Se a célula atual NÃO tem brisa nem fedor → todos os vizinhos são
   100% seguros.
2. Se a célula atual TEM brisa → vizinhos não visitados tornam-se
   suspeitos de poço (entram na fronteira de risco).
3. Se a célula atual TEM fedor → vizinhos não visitados tornam-se
   suspeitos de Wumpus.
4. Uma célula suspeita pode ser *inocentada*: se ela é vizinha de pelo
   menos uma célula visitada e segura sem a percepção correspondente,
   ela não pode conter o perigo — é reclassificada como segura.

Pontuação
---------
* +1000 ao pegar o ouro.
* −1000 ao morrer (Wumpus ou poço).
* −1 por cada ação executada (incentiva trajetos curtos).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Aliases de tipo
# ---------------------------------------------------------------------------

#: Uma posição no grid, dada por (linha, coluna), ambas em [0, TAMANHO-1].
Posicao = Tuple[int, int]


# ---------------------------------------------------------------------------
# Enumerações
# ---------------------------------------------------------------------------

class Acao(Enum):
    """Ações que o agente pode registrar em sua trilha de execução."""

    MOVER = auto()
    PEGAR_OURO = auto()
    RECUAR = auto()        # volta a uma célula segura por falta de opções
    AGUARDAR = auto()      # sem movimento seguro disponível


class Percepcao(Enum):
    """Percepções sensoriais disponíveis ao agente em uma célula."""

    FEDOR = auto()    # Wumpus adjacente
    BRISA = auto()    # poço adjacente
    BRILHO = auto()   # ouro na célula atual


# ---------------------------------------------------------------------------
# DTO de resultado
# ---------------------------------------------------------------------------

@dataclass
class ResultadoSimulacao:
    """Objeto de valor retornado por :meth:`run_simulation`.

    Atributos
    ---------
    trilha_acoes:
        Sequência ordenada de pares ``(acao, posicao)`` descrevendo cada
        passo dado pelo agente, na ordem em que ocorreram.
    pontuacao_final:
        Pontuação acumulada ao fim da simulação (ver regras no cabeçalho).
    sobreviveu:
        ``True`` se o agente não morreu durante a exploração.
    pegou_ouro:
        ``True`` se o agente alcançou e coletou o ouro.
    passos:
        Número total de ações registradas na trilha.
    """

    trilha_acoes: List[Tuple[Acao, Posicao]]
    pontuacao_final: int
    sobreviveu: bool
    pegou_ouro: bool
    passos: int = field(init=False)

    def __post_init__(self) -> None:
        self.passos = len(self.trilha_acoes)

    def __str__(self) -> str:
        if self.pegou_ouro:
            desfecho = "✓ pegou o ouro"
        elif self.sobreviveu:
            desfecho = "~ sobreviveu sem o ouro"
        else:
            desfecho = "✗ morreu"

        linhas = [
            "[KnowledgeBasedAgent — Mundo do Wumpus]",
            f"  Desfecho   : {desfecho}",
            f"  Pontuação  : {self.pontuacao_final}",
            f"  Passos     : {self.passos}",
            "  Trilha     :",
        ]
        for indice, (acao, posicao) in enumerate(self.trilha_acoes, start=1):
            linhas.append(f"    {indice:>2}. {acao.name:<11} {posicao}")
        return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Ambiente
# ---------------------------------------------------------------------------

class Environment:
    """Detém o estado verdadeiro do mundo e gera percepções.

    O ambiente é a *única* fonte da verdade: ele conhece as posições reais
    do Wumpus, dos poços e do ouro. O agente nunca acessa esses dados
    diretamente — apenas consulta :meth:`perceber` para a célula em que está.

    Essa separação garante que o agente resolva o problema por inferência,
    sem "trapacear" lendo o mapa.

    Parâmetros
    ----------
    tamanho:
        Lado do grid quadrado. Padrão: 4 (grid 4×4).
    posicao_wumpus:
        Célula ocupada pelo Wumpus.
    posicoes_pocos:
        Conjunto de células que contêm poços.
    posicao_ouro:
        Célula que contém o ouro.
    posicao_inicial:
        Célula de partida do agente — assumida sempre segura.
    """

    def __init__(
        self,
        tamanho: int,
        posicao_wumpus: Posicao,
        posicoes_pocos: Set[Posicao],
        posicao_ouro: Posicao,
        posicao_inicial: Posicao = (0, 0),
    ) -> None:
        self._tamanho: int = tamanho
        self._posicao_wumpus: Posicao = posicao_wumpus
        self._posicoes_pocos: Set[Posicao] = posicoes_pocos
        self._posicao_ouro: Posicao = posicao_ouro
        self._posicao_inicial: Posicao = posicao_inicial

    # ------------------------------------------------------------------
    # Propriedades de leitura
    # ------------------------------------------------------------------

    @property
    def tamanho(self) -> int:
        """Retorna o lado do grid."""
        return self._tamanho

    @property
    def posicao_inicial(self) -> Posicao:
        """Retorna a célula de partida do agente."""
        return self._posicao_inicial

    # ------------------------------------------------------------------
    # Geometria do grid
    # ------------------------------------------------------------------

    def dentro_do_grid(self, posicao: Posicao) -> bool:
        """Verifica se *posicao* está dentro dos limites do grid."""
        linha, coluna = posicao
        return 0 <= linha < self._tamanho and 0 <= coluna < self._tamanho

    def vizinhos(self, posicao: Posicao) -> List[Posicao]:
        """Retorna as células ortogonalmente adjacentes e válidas de *posicao*.

        Movimento diagonal não é permitido no Mundo do Wumpus, portanto
        considera-se apenas norte, sul, leste e oeste.
        """
        linha, coluna = posicao
        candidatos: List[Posicao] = [
            (linha - 1, coluna),  # norte
            (linha + 1, coluna),  # sul
            (linha, coluna - 1),  # oeste
            (linha, coluna + 1),  # leste
        ]
        return [c for c in candidatos if self.dentro_do_grid(c)]

    # ------------------------------------------------------------------
    # Consultas de perigo / objetivo
    # ------------------------------------------------------------------

    def tem_wumpus(self, posicao: Posicao) -> bool:
        """Indica se o Wumpus está em *posicao*."""
        return posicao == self._posicao_wumpus

    def tem_poco(self, posicao: Posicao) -> bool:
        """Indica se há um poço em *posicao*."""
        return posicao in self._posicoes_pocos

    def tem_ouro(self, posicao: Posicao) -> bool:
        """Indica se o ouro está em *posicao*."""
        return posicao == self._posicao_ouro

    def e_fatal(self, posicao: Posicao) -> bool:
        """Indica se entrar em *posicao* mata o agente (Wumpus ou poço)."""
        return self.tem_wumpus(posicao) or self.tem_poco(posicao)

    # ------------------------------------------------------------------
    # Geração de percepções
    # ------------------------------------------------------------------

    def perceber(self, posicao: Posicao) -> Set[Percepcao]:
        """Retorna o conjunto de percepções sentidas em *posicao*.

        As percepções seguem as regras canônicas do Mundo do Wumpus:

        * FEDOR  — se algum vizinho contém o Wumpus.
        * BRISA  — se algum vizinho contém um poço.
        * BRILHO — se a própria célula contém o ouro.

        Parâmetros
        ----------
        posicao:
            A célula a ser sentida.

        Retorna
        -------
        Set[Percepcao]
            Conjunto (possivelmente vazio) de percepções ativas.
        """
        percepcoes: Set[Percepcao] = set()
        vizinhanca = self.vizinhos(posicao)

        if any(self.tem_wumpus(v) for v in vizinhanca):
            percepcoes.add(Percepcao.FEDOR)
        if any(self.tem_poco(v) for v in vizinhanca):
            percepcoes.add(Percepcao.BRISA)
        if self.tem_ouro(posicao):
            percepcoes.add(Percepcao.BRILHO)

        return percepcoes


# ---------------------------------------------------------------------------
# Agente Baseado em Conhecimento
# ---------------------------------------------------------------------------

class KnowledgeBasedAgent:
    """Agente que explora o mundo por inferência lógica, sem busca informada.

    A base de conhecimento (KB) do agente é composta por quatro estruturas
    de crenças, atualizadas a cada percepção:

    * ``_visitadas``  — células onde o agente já esteve (sempre seguras,
      pois ele sobreviveu a elas).
    * ``_seguras``    — células comprovadamente sem perigo, ainda que não
      visitadas.
    * ``_suspeitas``  — fronteira de risco: células que *podem* conter um
      poço ou o Wumpus, segundo as percepções até agora.
    * ``_percepcoes_conhecidas`` — registro das percepções que o próprio
      agente sentiu em cada célula visitada. É a memória sensorial da KB
      e a única fonte usada pela inferência — o ambiente nunca é
      reconsultado para células do passado.

    O ciclo de raciocínio é: perceber → registrar na KB → inferir → agir.
    O agente só se move para células seguras; se não houver nenhuma
    fronteira segura, ele recua para a última célula segura conhecida.

    Parâmetros
    ----------
    environment:
        O ambiente no qual o agente atua. O agente consulta apenas
        :meth:`Environment.perceber` para a célula atual e a geometria do
        grid — nunca as posições reais dos perigos.
    """

    # Pontuações conforme as regras canônicas do Mundo do Wumpus
    RECOMPENSA_OURO: int = 1000
    PENALIDADE_MORTE: int = -1000
    CUSTO_POR_ACAO: int = -1

    def __init__(self, environment: Environment) -> None:
        self._env: Environment = environment

        # Estado de localização do agente
        self._posicao_atual: Posicao = environment.posicao_inicial

        # ── Base de Conhecimento (crenças) ──────────────────────────────
        self._visitadas: Set[Posicao] = set()
        self._seguras: Set[Posicao] = {environment.posicao_inicial}
        self._suspeitas: Set[Posicao] = set()
        # Memória sensorial: percepções sentidas pelo agente em cada célula
        self._percepcoes_conhecidas: Dict[Posicao, Set[Percepcao]] = {}

        # ── Estado da missão ────────────────────────────────────────────
        self._vivo: bool = True
        self._tem_ouro: bool = False
        self._pontuacao: int = 0
        self._trilha: List[Tuple[Acao, Posicao]] = []

    # ------------------------------------------------------------------
    # Atualização da base de conhecimento
    # ------------------------------------------------------------------

    def _atualizar_conhecimento(
        self, posicao: Posicao, percepcoes: Set[Percepcao]
    ) -> None:
        """Registra as *percepcoes* sentidas em *posicao* e infere segurança.

        Primeiro grava a percepção na memória sensorial da KB. Depois aplica
        as regras de inferência local:

        * Sem brisa e sem fedor → todos os vizinhos não visitados são
          marcados como 100% seguros (Regra 1).
        * Com brisa ou fedor → vizinhos não visitados e ainda não provados
          seguros entram na fronteira de suspeitas (Regras 2 e 3).

        Em seguida, dispara :meth:`_inocentar_suspeitas` para reclassificar
        suspeitas que se tornaram comprovadamente seguras.

        Parâmetros
        ----------
        posicao:
            A célula onde o agente está e sentiu as percepções.
        percepcoes:
            O conjunto de percepções sentido na célula atual.
        """
        # Grava a percepção na memória sensorial — fonte única da inferência
        self._percepcoes_conhecidas[posicao] = set(percepcoes)

        vizinhos_atuais = self._env.vizinhos(posicao)
        ha_perigo_adjacente = (
            Percepcao.BRISA in percepcoes or Percepcao.FEDOR in percepcoes
        )

        if not ha_perigo_adjacente:
            # Regra 1 — ausência de percepção prova segurança dos vizinhos
            for vizinho in vizinhos_atuais:
                self._seguras.add(vizinho)
                self._suspeitas.discard(vizinho)
        else:
            # Regras 2 e 3 — percepção de perigo gera suspeitas na fronteira
            for vizinho in vizinhos_atuais:
                if vizinho not in self._visitadas and vizinho not in self._seguras:
                    self._suspeitas.add(vizinho)

        # Regra 4 — tenta inocentar suspeitas com a nova informação
        self._inocentar_suspeitas()

    def _inocentar_suspeitas(self) -> None:
        """Reclassifica como seguras as suspeitas comprovadamente inofensivas.

        Uma célula suspeita é *inocentada* quando possui ao menos um vizinho
        já visitado cuja percepção registrada na KB não acusou perigo. Se uma
        célula visitada não tem brisa, nenhum de seus vizinhos pode ter poço;
        analogamente para fedor e Wumpus. Logo, a suspeita cai.

        A verificação usa exclusivamente ``_percepcoes_conhecidas`` — a
        memória do próprio agente — e nunca reconsulta o ``Environment``,
        preservando a fronteira entre o que o agente sabe e a verdade do mundo.

        Esta é a etapa que permite ao agente expandir com segurança a área
        explorável sem recorrer a qualquer algoritmo de busca.
        """
        inocentadas: Set[Posicao] = set()

        for suspeita in self._suspeitas:
            for vizinho in self._env.vizinhos(suspeita):
                # Só células com percepção registrada na KB servem de prova
                percepcoes_vizinho = self._percepcoes_conhecidas.get(vizinho)
                if percepcoes_vizinho is None:
                    continue
                # Vizinho visitado sem perigo percebido → suspeita cai
                sem_perigo = (
                    Percepcao.BRISA not in percepcoes_vizinho
                    and Percepcao.FEDOR not in percepcoes_vizinho
                )
                if sem_perigo:
                    inocentadas.add(suspeita)
                    break

        for celula in inocentadas:
            self._suspeitas.discard(celula)
            self._seguras.add(celula)

    # ------------------------------------------------------------------
    # Decisão de movimento
    # ------------------------------------------------------------------

    def _escolher_proximo_destino(self) -> Optional[Posicao]:
        """Escolhe a próxima célula segura a visitar.

        A política de movimento prioriza, nesta ordem:

        1. Uma célula segura ainda não visitada e adjacente à posição atual
           — expande a exploração sem risco.
        2. Qualquer célula segura adjacente já visitada — permite recuar
           quando não há fronteira nova a explorar.

        O agente nunca escolhe uma célula suspeita: a inferência lógica
        garante exploração conservadora.

        Retorna
        -------
        Optional[Posicao]
            A célula de destino, ou ``None`` se nenhum movimento seguro
            for possível.
        """
        vizinhos_atuais = self._env.vizinhos(self._posicao_atual)

        # Prioridade 1 — fronteira segura inexplorada
        for vizinho in vizinhos_atuais:
            if vizinho in self._seguras and vizinho not in self._visitadas:
                return vizinho

        # Prioridade 2 — recuo para célula segura já visitada
        for vizinho in vizinhos_atuais:
            if vizinho in self._seguras:
                return vizinho

        # Nenhum movimento seguro disponível
        return None

    # ------------------------------------------------------------------
    # Ciclo de simulação
    # ------------------------------------------------------------------

    def run_simulation(self, limite_passos: int = 100) -> ResultadoSimulacao:
        """Executa o ciclo de exploração até o agente pegar o ouro ou morrer.

        O loop segue o ciclo perceber → raciocinar → agir:

        1. O agente percebe a célula atual.
        2. Se há BRILHO, pega o ouro e encerra com sucesso.
        3. Caso contrário, atualiza a base de conhecimento.
        4. Escolhe um destino seguro e se move; se não houver, aguarda.
        5. Ao entrar numa célula, verifica se ela é fatal.

        Para evitar laços de recuo improdutivos, se o agente só dispõe de
        destinos já visitados durante várias iterações consecutivas, a
        exploração encerra: não há fronteira nova a alcançar com segurança.

        Parâmetros
        ----------
        limite_passos:
            Número máximo de ações antes de encerrar a simulação.

        Retorna
        -------
        ResultadoSimulacao
            DTO com a trilha completa de ações, a pontuação e o desfecho.
        """
        # Conta recuos consecutivos sem nenhuma exploração nova
        recuos_improdutivos: int = 0

        for _ in range(limite_passos):
            # ── 1. Percepção da célula atual ────────────────────────────
            self._visitadas.add(self._posicao_atual)
            self._seguras.add(self._posicao_atual)
            percepcoes = self._env.perceber(self._posicao_atual)

            # ── 2. Brilho → pega o ouro e encerra ───────────────────────
            if Percepcao.BRILHO in percepcoes:
                self._tem_ouro = True
                self._pontuacao += self.RECOMPENSA_OURO + self.CUSTO_POR_ACAO
                self._trilha.append((Acao.PEGAR_OURO, self._posicao_atual))
                break

            # ── 3. Raciocínio — atualiza a base de conhecimento ─────────
            self._atualizar_conhecimento(self._posicao_atual, percepcoes)

            # ── 4. Ação — escolhe e executa o movimento ─────────────────
            destino = self._escolher_proximo_destino()

            if destino is None:
                # Sem movimento seguro: registra espera e encerra
                self._trilha.append((Acao.AGUARDAR, self._posicao_atual))
                self._pontuacao += self.CUSTO_POR_ACAO
                break

            # Classifica a ação: avanço para nova célula ou recuo
            if destino not in self._visitadas:
                acao = Acao.MOVER
                recuos_improdutivos = 0
            else:
                acao = Acao.RECUAR
                recuos_improdutivos += 1

            self._posicao_atual = destino
            self._pontuacao += self.CUSTO_POR_ACAO
            self._trilha.append((acao, destino))

            # ── 5. Verificação de perigo ao entrar na célula ────────────
            if self._env.e_fatal(self._posicao_atual):
                self._vivo = False
                self._pontuacao += self.PENALIDADE_MORTE
                break

            # Salvaguarda — recuos demais seguidos indicam fronteira esgotada
            if recuos_improdutivos > len(self._visitadas):
                self._trilha.append((Acao.AGUARDAR, self._posicao_atual))
                self._pontuacao += self.CUSTO_POR_ACAO
                break

        return ResultadoSimulacao(
            trilha_acoes=self._trilha,
            pontuacao_final=self._pontuacao,
            sobreviveu=self._vivo,
            pegou_ouro=self._tem_ouro,
        )


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------

def _gold_reachable(tamanho: int, fatais: Set[Posicao], inicio: Posicao, ouro: Posicao) -> bool:
    """True se existe caminho do início ao ouro sem passar por célula fatal (visão do observador)."""
    visitadas, fila = {inicio}, [inicio]
    while fila:
        atual = fila.pop()
        if atual == ouro:
            return True
        r, c = atual
        for viz in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= viz[0] < tamanho and 0 <= viz[1] < tamanho and viz not in fatais and viz not in visitadas:
                visitadas.add(viz)
                fila.append(viz)
    return False


def varrer_mundos(tamanho: int = 4) -> None:
    """Roda o agente em todos os mundos com 1 Wumpus e 1 poço (início em (0,0)).

    Mede segurança (o agente nunca deve morrer) e completude prática (quantas
    vezes pega o ouro quando ele é fisicamente alcançável sem passar por célula
    fatal, algo que o agente não sabe de antemão).
    """
    inicio = (0, 0)
    celulas = [(r, c) for r in range(tamanho) for c in range(tamanho) if (r, c) != inicio]
    total = mortes = alcancavel = pegou = pegou_alcancavel = 0
    for wumpus_pos in celulas:
        for poco in (c for c in celulas if c != wumpus_pos):
            for ouro in (c for c in celulas if c not in (wumpus_pos, poco)):
                mundo = Environment(tamanho=tamanho, posicao_wumpus=wumpus_pos, posicoes_pocos={poco},
                                    posicao_ouro=ouro, posicao_inicial=inicio)
                resultado = KnowledgeBasedAgent(environment=mundo).run_simulation(limite_passos=100)
                acessivel = _gold_reachable(tamanho, {wumpus_pos, poco}, inicio, ouro)
                total += 1
                mortes += not resultado.sobreviveu
                alcancavel += acessivel
                pegou += resultado.pegou_ouro
                pegou_alcancavel += resultado.pegou_ouro and acessivel
    print(f"  Mundos {tamanho}x{tamanho} com 1 Wumpus e 1 poço: {total}")
    print(f"  Mortes do agente        : {mortes}")
    print(f"  Ouro alcançável (sem células fatais): {alcancavel}")
    print(f"  Ouro coletado           : {pegou} ({pegou / total:.1%} dos mundos, "
          f"{pegou_alcancavel / alcancavel:.1%} dos alcançáveis)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true",
                        help="roda o agente em todos os mundos com 1 Wumpus e 1 poço e mede segurança e completude")
    args = parser.parse_args()
    if args.sweep:
        varrer_mundos()
        raise SystemExit(0)

    SEP = "─" * 55

    # Mundo 4×4 onde o ouro é alcançável por inferência puramente local.
    # Coordenadas em (linha, coluna), com (0, 0) no canto superior esquerdo.
    #
    #   (0,0) início   (0,1) seguro   (0,2) seguro   (0,3) brisa
    #   (1,0) seguro   (1,1) seguro   (1,2) ouro     (1,3) -
    #   (2,0) seguro   (2,1) seguro   (2,2) brisa    (2,3) poço
    #   (3,0) fedor    (3,1) Wumpus   (3,2) fedor    (3,3) brisa
    #
    # O corredor superior (linhas 0 e 1) está livre de perigos adjacentes,
    # de modo que a Regra 1 abre um caminho seguro de (0,0) até o ouro.
    mundo = Environment(
        tamanho=4,
        posicao_wumpus=(3, 1),
        posicoes_pocos={(2, 3)},
        posicao_ouro=(1, 2),
        posicao_inicial=(0, 0),
    )

    agente = KnowledgeBasedAgent(environment=mundo)
    resultado = agente.run_simulation(limite_passos=100)

    print(SEP)
    print(resultado)
    print(SEP)
