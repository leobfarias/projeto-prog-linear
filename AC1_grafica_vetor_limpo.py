"""
AC1 - Programacao Linear Aplicada (IBM0803)
Grafica Vetor - Mix otimo de producao semanal
Integrantes: Leonardo Farias, Gustavo Zocharato, Enzo Cardin

Problema: a grafica produz 5 tipos de produto e tem 5 recursos limitados
(horas de maquina/pessoas por semana). Queremos descobrir quantos lotes de
cada produto fabricar para ter o MAIOR lucro possivel sem estourar nenhum
recurso.
"""

import numpy as np                    # trabalha com vetores e matrizes
from scipy.optimize import linprog    # resolve problemas de programacao linear


# ---------------------------------------------------------------------------
# 1. DADOS DO PROBLEMA
# ---------------------------------------------------------------------------

# Os 5 produtos (variaveis de decisao x1 ... x5 = lotes por semana)
produtos = ["Cartao de visita", "Flyer A5", "Folder dobrado",
            "Banner em lona", "Catalogo grampeado"]

# Os 5 recursos limitados (cada um vira uma restricao)
recursos = ["Impressao", "Corte/refile", "Acabamento",
            "Arte/pre-impressao", "Montagem/expedicao"]

# Lucro (R$) por lote de cada produto -> coeficientes da funcao objetivo
lucro = np.array([90, 70, 130, 110, 260])

# Horas que cada produto consome de cada recurso.
# Linha = recurso, coluna = produto (mesma ordem das listas acima).
# Ex.: 1 lote de Cartao gasta 0.5h de Impressao, 0.3h de Corte, etc.
consumo_por_lote = np.array([
    [0.5, 0.8, 1.0, 1.5, 2.5],   # Impressao
    [0.3, 0.2, 0.3, 0.1, 0.4],   # Corte/refile
    [0.2, 0.1, 0.6, 0.3, 1.2],   # Acabamento
    [0.4, 0.5, 0.8, 1.0, 2.0],   # Arte/pre-impressao
    [0.2, 0.3, 0.3, 0.4, 0.8],   # Montagem/expedicao
])

# Horas disponiveis por semana de cada recurso
# (quantidade de maquinas/pessoas x horas por dia x 5 dias)
horas_disponiveis = np.array([
    2 * 16 * 5,   # Impressao:          2 maquinas, 16h/dia = 160h
    1 * 8 * 5,    # Corte/refile:       1 maquina,   8h/dia =  40h
    2 * 8 * 5,    # Acabamento:         2 pessoas,   8h/dia =  80h
    3 * 8 * 5,    # Arte/pre-impressao: 3 pessoas,   8h/dia = 120h
    2 * 8 * 5,    # Montagem/expedicao: 2 pessoas,   8h/dia =  80h
])

# Minimo e maximo de lotes de cada produto (demanda contratada / limite de mercado)
limites = [
    (20, 120),   # Cartao de visita
    (15, 100),   # Flyer A5
    (10, 60),    # Folder dobrado
    (8, 50),     # Banner em lona
    (5, 25),     # Catalogo grampeado
]


# ---------------------------------------------------------------------------
# 2. RESOLVER O MODELO
# ---------------------------------------------------------------------------

# O linprog so sabe MINIMIZAR. Para MAXIMIZAR o lucro, minimizamos o lucro
# negativo (-lucro). No final basta trocar o sinal de volta.
resultado = linprog(
    c=-lucro,                 # funcao objetivo (negativa para maximizar)
    A_ub=consumo_por_lote,    # lado esquerdo das restricoes:  consumo * x
    b_ub=horas_disponiveis,   # lado direito das restricoes:   <= horas
    bounds=limites,           # minimo <= x <= maximo de cada produto
    method="highs",           # algoritmo (simplex moderno do scipy)
)

# Se o modelo nao tiver solucao (ex.: restricoes impossiveis), avisa e para
if not resultado.success:
    print("Nao foi possivel resolver o modelo:", resultado.message)
    raise SystemExit

x = resultado.x                  # quantidade otima de lotes de cada produto
lucro_maximo = -resultado.fun    # troca o sinal de volta (era -lucro)


# ---------------------------------------------------------------------------
# 3. MOSTRAR OS RESULTADOS
# ---------------------------------------------------------------------------

print("=" * 62)
print("GRAFICA VETOR - MIX OTIMO DE PRODUCAO SEMANAL")
print("=" * 62)

# Quantos lotes produzir de cada produto
print("\nSOLUCAO OTIMA (lotes por semana)")
print("-" * 62)
for nome, quantidade in zip(produtos, x):
    print(f"  {nome:<22} {quantidade:10.4f} lotes")

# Valor da funcao objetivo no ponto otimo
print("\nVALOR OTIMO")
print("-" * 62)
print(f"  Lucro maximo semanal: R$ {lucro_maximo:,.2f}")

# Quanto de cada recurso foi usado na solucao otima.
# consumo_por_lote @ x = multiplicacao matriz x vetor (soma o consumo de
# todos os produtos para cada recurso).
horas_usadas = consumo_por_lote @ x
folga = horas_disponiveis - horas_usadas   # horas que sobraram

# Se a folga e zero, o recurso foi usado ate o limite -> e um GARGALO
print("\nUSO DOS RECURSOS")
print("-" * 62)
print(f"  {'Recurso':<20}{'Usado':>10}{'Total':>10}{'Folga':>10}   Situacao")
for nome, usado, total, sobra in zip(recursos, horas_usadas, horas_disponiveis, folga):
    situacao = "GARGALO" if abs(sobra) < 1e-6 else "folga"
    print(f"  {nome:<20}{usado:10.3f}{total:10.1f}{sobra:10.3f}   {situacao}")

# Preco-sombra = quanto o lucro aumentaria se tivessemos 1 hora a mais
# daquele recurso. Recursos com folga tem preco-sombra zero (nao adianta
# ter mais horas do que ja sobra). O sinal e trocado porque o modelo foi
# resolvido como minimizacao.
precos_sombra = -resultado.ineqlin.marginals

print("\nPRECOS-SOMBRA (R$ por hora adicional de recurso)")
print("-" * 62)
for nome, preco in zip(recursos, precos_sombra):
    print(f"  {nome:<24} R$ {preco:8.2f}")

print("\n" + "=" * 62)
