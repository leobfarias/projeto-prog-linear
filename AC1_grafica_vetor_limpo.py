"""
AC1 - Programacao Linear Aplicada (IBM0803)
Grafica Vetor - Mix otimo de producao semanal
Integrantes: Leonardo Farias, Gustavo Zocharato, Enzo Cardin
"""

import numpy as np  
from scipy.optimize import linprog 


produtos = ["Cartao de visita", "Flyer A5", "Folder dobrado",
            "Banner em lona", "Catalogo grampeado"]

recursos = ["Impressao", "Corte/refile", "Acabamento",
            "Arte/pre-impressao", "Montagem/expedicao"]

lucro = np.array([90, 70, 130, 110, 260], dtype=float)
c = -lucro

A_ub = np.array([
    [0.5, 0.8, 1.0, 1.5, 2.5],
    [0.3, 0.2, 0.3, 0.1, 0.4],
    [0.2, 0.1, 0.6, 0.3, 1.2],
    [0.4, 0.5, 0.8, 1.0, 2.0],
    [0.2, 0.3, 0.3, 0.4, 0.8],
], dtype=float)

b_ub = np.array([
    2 * 16 * 5,
    1 * 8 * 5,
    2 * 8 * 5,
    3 * 8 * 5,
    2 * 8 * 5,
], dtype=float)

bounds = [
    (20, 120),
    (15, 100),
    (10, 60),
    (8, 50),
    (5, 25),
]

resultado = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

print("=" * 62)
print("GRAFICA VETOR - MIX OTIMO DE PRODUCAO SEMANAL")
print("=" * 62)

if not resultado.success:
    print("Nao foi possivel resolver o modelo:", resultado.message)
    raise SystemExit

x = resultado.x
lucro_otimo = -resultado.fun

print("\nSOLUCAO OTIMA (lotes por semana)")
print("-" * 62)
for nome, quantidade in zip(produtos, x):
    print(f"  {nome:<22} {quantidade:10.4f} lotes")

print("\nVALOR OTIMO")
print("-" * 62)
print(f"  Lucro maximo semanal: R$ {lucro_otimo:,.2f}")

consumo = A_ub @ x
folga = b_ub - consumo

print("\nUSO DOS RECURSOS")
print("-" * 62)
print(f"  {'Recurso':<20}{'Usado':>10}{'Total':>10}{'Folga':>10}   Situacao")
for nome, usado, total, sobra in zip(recursos, consumo, b_ub, folga):
    situacao = "GARGALO" if abs(sobra) < 1e-6 else "folga"
    print(f"  {nome:<20}{usado:10.3f}{total:10.1f}{sobra:10.3f}   {situacao}")

precos_sombra = -resultado.ineqlin.marginals

print("\nPRECOS-SOMBRA (R$ por hora adicional de recurso)")
print("-" * 62)
for nome, preco in zip(recursos, precos_sombra):
    print(f"  {nome:<24} R$ {preco:8.2f}")

print("\n" + "=" * 62)
