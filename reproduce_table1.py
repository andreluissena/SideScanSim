#!/usr/bin/env python3
"""
reproduce_table1.py — Reprodução da Tabela 1 do artigo ERBASE/WEIBASE 2026.

Executa o motor com os quatro objetos da cena multiobjeto descrita na Seção 4.4
do artigo e imprime as métricas de ground truth geradas automaticamente,
comparando-as com os valores publicados.

Uso:
    python reproduce_table1.py              # tabela no terminal
    python reproduce_table1.py --figure     # salva também a imagem da cena

Requisitos: numpy, scipy, matplotlib (apenas para --figure).

Referência:
    Sena, A. L. S. e Santos Neto, J. (2026) "SideScanSim: An Open-Source
    Side-Scan Sonar Simulator for Education, Algorithm Development, and
    Machine Learning Training", ERBASE/WEIBASE.
"""
import argparse
import sys

import numpy as np

from sidescan_engine_v4 import (
    SonarParams,
    SeafloorParams,
    SceneObject,
    render_sss_image,
)

# ─── Parâmetros da cena publicada (Seção 4.4) ─────────────────────────
SONAR = SonarParams(
    frequency_kHz=450.0,
    range_m=50.0,
    altitude_m=5.0,
    pixels_per_meter_across=15.0,
    pixels_per_meter_along=15.0,
)

SEAFLOOR = SeafloorParams(sediment_type="medium_sand")

SCENE_LENGTH_M = 80.0
SPECKLE_SEED = 42
TEXTURE_SEED = 123

OBJECTS = [
    SceneObject(shape="sphere", across_m=12.0, along_m=20.0, height_m=0.80,
                width_m=1.6, length_m=1.6, material="steel", name="Sphere"),
    SceneObject(shape="box", across_m=32.0, along_m=40.0, height_m=0.60,
                width_m=1.5, length_m=2.0, material="concrete", name="Block"),
    SceneObject(shape="cylinder", across_m=20.0, along_m=60.0, height_m=0.40,
                width_m=0.6, length_m=6.0, material="steel", name="Pipe"),
    SceneObject(shape="box", across_m=38.0, along_m=70.0, height_m=0.50,
                width_m=1.0, length_m=1.5, material="wood", name="Box"),
]

# Valores publicados na Tabela 1: (L_teórico [m], L_medido [m], erro [%])
PUBLISHED = {
    "Sphere": (2.29, 1.60, 30.0),
    "Block":  (4.36, 4.40, 1.0),
    "Pipe":   (1.74, 1.00, 42.0),
    "Box":    (4.22, 4.20, 1.0),
}

TOL_M = 0.10          # tolerância absoluta em metros (1,5 px a 15 px/m)


def main(salvar_figura: bool = False) -> int:
    image, meta = render_sss_image(
        SONAR, SEAFLOOR, OBJECTS,
        scene_length_m=SCENE_LENGTH_M,
        speckle_seed=SPECKLE_SEED,
        texture_seed=TEXTURE_SEED,
    )

    print()
    print("Tabela 1 — ground truth por objeto (gerado pelo motor)")
    print("=" * 78)
    print(f"{'Objeto':9}{'Material':11}{'x (m)':>7}{'θ (°)':>7}{'H (m)':>7}"
          f"{'L_teo':>8}{'L_med':>8}{'erro %':>8}{'':>4}{'artigo':>8}")
    print("-" * 78)

    tudo_ok = True
    for gt in meta["objects_gt"]:
        nome = gt["name"]
        pub_teo, pub_med, _ = PUBLISHED[nome]
        bate = (abs(gt["shadow_theoretical_m"] - pub_teo) <= TOL_M and
                abs(gt["shadow_measured_m"] - pub_med) <= TOL_M)
        tudo_ok &= bate
        print(f"{nome:9}{gt['material']:11}"
              f"{gt['nadir_distance_m']:7.0f}"
              f"{gt['grazing_angle_deg']:7.0f}"
              f"{gt['height_m']:7.2f}"
              f"{gt['shadow_theoretical_m']:8.2f}"
              f"{gt['shadow_measured_m']:8.2f}"
              f"{gt['shadow_error_pct']:8.1f}"
              f"{'':>4}{'OK' if bate else 'DIVERGE':>8}")

    print("-" * 78)
    print(f"imagem: {image.shape[0]} x {image.shape[1]} px  ·  "
          f"sedimento: {meta['sediment']}  ·  "
          f"BS0 = {meta['BS0_dB']:.0f} dB  ·  "
          f"alpha = {meta['alpha_dB_per_km']:.0f} dB/km")
    print()

    print("Convenção de medida das duas colunas de sombra")
    print("-" * 78)
    print("  L_teo  : H * x / (h - H), medido a partir do CENTRO do objeto.")
    print("           Vem de SceneObject.theoretical_shadow_length().")
    print("  L_med  : maior bloco contíguo da máscara de sombra. Como")
    print("           compute_shadow() exclui os pixels do próprio objeto,")
    print("           esse bloco começa na BORDA AFASTADA, não no centro.")
    print()
    print("  As duas colunas partem de referências diferentes. Em objetos de")
    print("  topo plano a diferença é pequena; em topo curvo o raio do objeto")
    print("  se soma ao deslocamento do ponto de oclusão, e o erro chega a")
    print("  30-42 %. É diferença de referencial, não erro do modelo acústico.")
    print()

    if tudo_ok:
        print(f"RESULTADO: todos os valores reproduzem a Tabela 1 "
              f"dentro de {TOL_M:.2f} m.")
    else:
        print("RESULTADO: há divergência em relação à Tabela 1 publicada.")

    if salvar_figura:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sidescan_engine_v4 import get_sss_colormap, sss_stretch

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.imshow(sss_stretch(image), cmap=get_sss_colormap("amber"),
                  aspect="auto", vmin=0, vmax=1)
        ax.set_title("Tabela 1 — cena multiobjeto (Seção 4.4)")
        ax.set_xlabel("across-track (px)")
        ax.set_ylabel("along-track (px)")
        fig.tight_layout()
        fig.savefig("table1_scene.png", dpi=150)
        print("figura salva em table1_scene.png")

    return 0 if tudo_ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figure", action="store_true",
                   help="salva também a imagem da cena em table1_scene.png")
    args = p.parse_args()
    sys.exit(main(salvar_figura=args.figure))
