"""
SideScanSim — GUI de exploração paramétrica (Streamlit)
=========================================================
Versão mínima, conforme escopo combinado:
  - Sidebar: Sonar / Seafloor / Environment / 1 objeto
  - Botão "Gerar imagem"
  - Preview com colormap selecionável (amber / gray)
  - SEM ground truth visual, SEM presets de domínio, SEM export de dataset
    (isso fica para a Etapa 4, quando Etapa 2/3 estabilizarem)

Uso:
    streamlit run app.py
"""
import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sidescan_engine_v4 import (
    SonarParams,
    SeafloorParams,
    SEDIMENT_TABLE,
    MATERIAL_TABLE,
    create_object,
    render_sss_image,
    get_sss_colormap,
    sss_stretch,
)

st.set_page_config(page_title="SideScanSim — Exploração Paramétrica", layout="wide")
st.title("SideScanSim — Exploração Paramétrica (v4.0)")
st.caption(
    "Ferramenta mínima de exploração. Ground truth, presets de domínio e "
    "export de dataset ficam para a Etapa 4 (interface educacional completa)."
)


# ─────────────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES DE VISUALIZAÇÃO (puramente didáticas/GUI —
# não fazem parte da engine, só leem valores já calculados por ela)
# ─────────────────────────────────────────────────────────────────
def plot_side_geometry(h_alt: float, range_m: float, obj_gt: dict | None):
    """
    Vista lateral (corte transversal) da geometria de aquisição:
    - barra vertical: altitude do sonar acima do fundo
    - barra horizontal: distância ao nadir, com as 3 zonas do SSS
    - se houver objeto: raio acústico, altura do objeto e sombra projetada

    As 3 zonas seguem a estrutura física já usada em calibrated_profile():
      Nadir        (~0–5.5% do range)  — sem retorno lateral útil
      Especular    (~5.5–25% do range) — pico near-nadir (reflexão quase normal)
      Lambertiana  (~25–100% do range) — fundo em plateau, regida por Lambert+TVG
    """
    fig, ax = plt.subplots(figsize=(4.6, 4.2))

    x_max = range_m * 1.05

    # Fundo (seafloor)
    ax.axhline(0, color="#8a6d3b", lw=2, zorder=3)
    ax.fill_between([0, x_max], -0.6, 0, color="#caa472", alpha=0.35, zorder=1)

    # Coluna d'água
    ax.fill_between([0, x_max], 0, h_alt * 1.25, color="#bcdfee", alpha=0.25, zorder=0)

    # 3 zonas (faixas verticais sombreadas + rótulos)
    zone_bounds = [0.0, 0.055 * range_m, 0.25 * range_m, range_m]
    zone_colors = ["#333333", "#f4c542", "#7a9e7e"]
    zone_labels = ["Nadir", "Especular\n(near-range)", "Lambertiana\n(far-range)"]
    for i in range(3):
        ax.axvspan(zone_bounds[i], zone_bounds[i + 1], color=zone_colors[i],
                   alpha=0.12, zorder=0)
        xm = (zone_bounds[i] + zone_bounds[i + 1]) / 2
        ax.text(xm, -0.55, zone_labels[i], ha="center", va="top",
                fontsize=7, color="#444444")

    # Sonar (transdutor) e barra de altitude
    ax.plot(0, h_alt, marker="v", markersize=14, color="black", zorder=5)
    ax.annotate(
        "", xy=(-x_max * 0.035, 0), xytext=(-x_max * 0.035, h_alt),
        arrowprops=dict(arrowstyle="<->", color="dimgray", lw=1.2),
    )
    ax.text(-x_max * 0.07, h_alt / 2, f"h = {h_alt:.1f} m",
            rotation=90, va="center", ha="center", fontsize=8)

    # Objeto + raio acústico + sombra
    if obj_gt is not None:
        x_obj = obj_gt["nadir_distance_m"]
        H = obj_gt["height_m"]
        L_shadow = obj_gt["shadow_theoretical_m"]

        # Ponto onde o raio (transdutor → topo do objeto) toca o fundo de novo
        # = limite teórico da sombra (mesma fórmula da engine/compute_shadow)
        x_far = x_obj + L_shadow

        # Raio acústico rasante (grazing ray) que define a sombra
        ax.plot([0, x_far], [h_alt, 0], color="crimson", lw=1.1, ls="--", zorder=4)

        # Objeto (barra vertical simplificada)
        ax.plot([x_obj, x_obj], [0, H], color="steelblue", lw=5, zorder=4,
                solid_capstyle="round")
        ax.plot(x_obj, H, marker="o", color="steelblue", markersize=5, zorder=5)
        ax.text(x_obj, H + h_alt * 0.06, f"objeto\nH={H:.2f} m",
                ha="center", fontsize=7, color="steelblue")

        # Sombra projetada no fundo
        ax.axvspan(x_obj, x_far, ymin=0.0, ymax=0.05, color="black", alpha=0.65, zorder=2)
        ax.annotate(
            "", xy=(x_obj, -h_alt * 0.18), xytext=(x_far, -h_alt * 0.18),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.0),
        )
        ax.text((x_obj + x_far) / 2, -h_alt * 0.30,
                f"sombra ≈ {L_shadow:.2f} m", ha="center", fontsize=7)

    ax.set_xlim(-x_max * 0.12, x_max)
    ax.set_ylim(-h_alt * 0.45, h_alt * 1.3)
    ax.set_xlabel("Distância ao nadir [m]", fontsize=8)
    ax.set_yticks([])
    ax.set_title("Geometria lateral da cena", fontsize=10)
    fig.tight_layout()
    return fig


def plot_lateral_profile(meta: dict):
    """
    Perfil calibrado de intensidade ao longo do across-track (apenas o lado
    direito da imagem, já calculado internamente pela engine em 'profile').
    Corresponde ao diagnóstico descrito em validation_criteria.md §2.3.
    """
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.plot(meta["x_m"], meta["profile"], color="darkorange", lw=1.5)
    ax.set_xlabel("Distância ao nadir [m]", fontsize=8)
    ax.set_ylabel("Intensidade calibrada", fontsize=8)
    ax.set_title("Perfil lateral de intensidade (sem speckle)", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────
# SIDEBAR — ENTRADA DE PARÂMETROS
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📡 Sonar")
    frequency_kHz = st.slider("Frequência [kHz]", 100, 900, 450, step=10)
    range_m = st.slider("Range (alcance) [m]", 5, 150, 30)
    altitude_m = st.slider("Altitude do veículo [m]", 1.0, 10.0, 4.0, step=0.1)
    pulse_duration_us = st.slider("Duração do pulso [μs]", 10, 200, 50, step=5)
    altitude_jitter_m = st.slider(
        "Altitude jitter (heave) [m]", 0.0, 0.5, 0.05, step=0.01
    )

    st.divider()
    st.header("🌊 Ambiente / Água")
    st.caption(
        "Nota: nesta versão (v4.0, engine congelada do ERBASE) a absorção α "
        "é calculada apenas a partir da frequência (Francois-Garrison). "
        "Salinidade, temperatura e turbidez ainda não modulam α — "
        "ver seção 'Limitações' abaixo."
    )
    water_depth_m = st.number_input("Profundidade da coluna d'água [m]", 1.0, 500.0, 12.0)
    salinity_ppt = st.slider("Salinidade [ppt]", 0, 40, 35)
    temperature_C = st.slider("Temperatura [°C]", 0, 35, 26)
    turbidity = st.selectbox("Turbidez", ["low", "medium", "high"], index=0)

    st.divider()
    st.header("🏖️ Seafloor")
    sediment_type = st.selectbox(
        "Tipo de sedimento",
        list(SEDIMENT_TABLE.keys()),
        format_func=lambda k: f"{k} ({SEDIMENT_TABLE[k]['description']})",
        index=list(SEDIMENT_TABLE.keys()).index("medium_sand"),
    )
    ripple_enabled = st.checkbox("Marcas de corrente (ripples)", value=False)
    ripple_wavelength_m = st.slider(
        "Comprimento de onda do ripple [m]", 0.1, 2.0, 0.5, step=0.05,
        disabled=not ripple_enabled,
    )
    ripple_amplitude = st.slider(
        "Amplitude do ripple", 0.0, 0.3, 0.06, step=0.01,
        disabled=not ripple_enabled,
    )

    st.divider()
    st.header("📦 Objeto")
    add_object = st.checkbox("Adicionar objeto à cena", value=True)
    if add_object:
        shape = st.selectbox(
            "Forma", ["sphere", "box", "cylinder", "mound", "rock", "wreck"]
        )
        material = st.selectbox(
            "Material",
            list(MATERIAL_TABLE.keys()),
            format_func=lambda k: f"{k} ({MATERIAL_TABLE[k]['description']})",
            index=list(MATERIAL_TABLE.keys()).index("steel"),
        )
        nadir_distance_m = st.slider(
            "Distância ao nadir [m]", 0.5, float(range_m) - 0.5, min(12.0, range_m * 0.4)
        )
        along_m_obj = st.slider("Posição along-track [m]", 0.0, 80.0, 40.0)
        height_m_obj = st.slider("Altura do objeto [m]", 0.05, 5.0, 0.4, step=0.05)
        use_custom_dims = st.checkbox("Definir largura/comprimento manualmente", value=False)
        if use_custom_dims:
            width_m_obj = st.slider("Largura (across) [m]", 0.1, 10.0, 0.8, step=0.1)
            length_m_obj = st.slider("Comprimento (along) [m]", 0.1, 20.0, 2.0, step=0.1)
        else:
            width_m_obj = None
            length_m_obj = None

    st.divider()
    st.header("🎨 Visualização")
    colormap_name = st.radio("Colormap", ["amber", "gray"], horizontal=True)
    scene_length_m = st.slider("Extensão along-track da cena [m]", 20, 150, 80)
    speckle_seed = st.number_input("Seed (speckle)", value=42, step=1)
    texture_seed = st.number_input("Seed (textura)", value=123, step=1)

    st.subheader("Brilho / Contraste (só exibição)")
    st.caption(
        "Ajusta apenas como a imagem aparece na tela — não altera os "
        "valores físicos usados no ground truth ou na validação."
    )
    apply_stretch = st.checkbox("Aplicar realce de contraste", value=True)
    low_pct = st.slider(
        "Percentil escuro (sombra/fundo) [%]", 0.0, 10.0, 1.0, step=0.5,
        disabled=not apply_stretch,
        help="Valores abaixo deste percentil viram preto (0).",
    )
    high_pct = st.slider(
        "Percentil claro (highlights) [%]", 90.0, 100.0, 99.0, step=0.5,
        disabled=not apply_stretch,
        help="Valores acima deste percentil viram branco (1).",
    )
    gamma = st.slider(
        "Gama (clarear/escurecer tons médios)", 0.3, 2.0, 0.7, step=0.05,
        disabled=not apply_stretch,
        help="< 1.0 clareia o fundo (recomendado se a imagem está escura); "
             "> 1.0 escurece os tons médios.",
    )

    generate = st.button("🔄 Gerar imagem", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# MONTAGEM DOS PARÂMETROS E CHAMADA DA ENGINE
# ─────────────────────────────────────────────────────────────────
if generate or "last_image" not in st.session_state:

    sonar = SonarParams(
        frequency_kHz=float(frequency_kHz),
        range_m=float(range_m),
        altitude_m=float(altitude_m),
        pulse_duration_us=float(pulse_duration_us),
        altitude_jitter_m=float(altitude_jitter_m),
    )

    seafloor = SeafloorParams(
        sediment_type=sediment_type,
        ripple_wavelength_m=float(ripple_wavelength_m),
        ripple_amplitude=float(ripple_amplitude),
        ripple_enabled=bool(ripple_enabled),
    )

    objects = []
    if add_object:
        obj = create_object(
            shape=shape,
            nadir_distance_m=float(nadir_distance_m),
            along_m=float(along_m_obj),
            height_m=float(height_m_obj),
            material=material,
            width_m=width_m_obj,
            length_m=length_m_obj,
        )
        objects.append(obj)

    image_full, meta = render_sss_image(
        sonar=sonar,
        seafloor=seafloor,
        objects=objects,
        scene_length_m=float(scene_length_m),
        speckle_seed=int(speckle_seed),
        texture_seed=int(texture_seed),
    )

    st.session_state["last_image"] = image_full
    st.session_state["last_meta"] = meta
    st.session_state["last_sonar"] = sonar

# ─────────────────────────────────────────────────────────────────
# ÁREA PRINCIPAL — PREVIEW
# ─────────────────────────────────────────────────────────────────
image_full = st.session_state["last_image"]
meta = st.session_state["last_meta"]
sonar_used = st.session_state["last_sonar"]

col_img, col_info = st.columns([3, 1])

with col_img:
    cmap = get_sss_colormap(colormap_name)

    if apply_stretch:
        image_display = sss_stretch(
            image_full, low_percentile=low_pct, high_percentile=high_pct, gamma=gamma
        )
    else:
        image_display = image_full

    fig, ax = plt.subplots(figsize=(9, 9 * image_display.shape[0] / image_display.shape[1]))
    ax.imshow(image_display, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("Across-track (range) [px]")
    ax.set_ylabel("Along-track [px]")
    ax.set_title(f"SideScanSim v4.0 — {sediment_type}, {frequency_kHz} kHz, alt. {altitude_m} m")
    st.pyplot(fig, use_container_width=True)
    if apply_stretch:
        st.caption(
            "⚠️ Imagem exibida com realce de contraste (apenas visualização). "
            "Os valores físicos usados no ground truth permanecem inalterados."
        )

    # ─── Botão de salvar imagem ───────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="💾 Salvar imagem (PNG)",
        data=buf,
        file_name=f"sidescansim_{sediment_type}_{int(frequency_kHz)}kHz_alt{altitude_m:.1f}m.png",
        mime="image/png",
        use_container_width=True,
    )

with col_info:
    st.subheader("Metadados da cena")
    st.metric("BS₀ do sedimento [dB]", f"{meta['BS0_dB']:.1f}")
    st.metric("α (absorção) [dB/km]", f"{meta['alpha_dB_per_km']:.2f}")
    st.metric("Largura do nadir [px]", meta["nadir_width_px"])
    st.metric("Resolução da imagem", f"{meta['n_along']} × {meta['n_full'] if 'n_full' in meta else image_full.shape[1]} px")

    if meta.get("objects_gt"):
        st.subheader("Objeto(s) na cena")
        for ogt in meta["objects_gt"]:
            with st.expander(f"{ogt['name']} ({ogt['material_description']})"):
                st.write(f"**TS:** {ogt['TS_dB']:.1f} dB")
                st.write(f"**Ângulo de grazing:** {ogt['grazing_angle_deg']:.1f}°")
                st.write(f"**Sombra teórica:** {ogt['shadow_theoretical_m']:.2f} m")
                st.write(f"**Sombra medida:** {ogt['shadow_measured_m']:.2f} m")
                st.write(f"**Erro:** {ogt['shadow_error_pct']:.1f}%")

    st.divider()
    st.subheader("📐 Geometria da cena (vista lateral)")
    obj_gt_first = meta["objects_gt"][0] if meta.get("objects_gt") else None
    fig_geom = plot_side_geometry(
        h_alt=sonar_used.altitude_m,
        range_m=sonar_used.range_m,
        obj_gt=obj_gt_first,
    )
    st.pyplot(fig_geom, use_container_width=True)

    st.subheader("📈 Perfil lateral de intensidade")
    fig_profile = plot_lateral_profile(meta)
    st.pyplot(fig_profile, use_container_width=True)

st.divider()
with st.expander("⚠️ Limitações desta versão mínima"):
    st.markdown(
        """
- **Salinidade / temperatura / turbidez** ainda não alimentam o modelo de
  absorção da engine v4.0 — esses campos estão na interface para fins de
  registro de metadados da cena, mas **não alteram a imagem ainda**.
  A integração física correta (velocidade do som via salinidade/temperatura,
  fator de turbidez sobre α) é um TODO explícito para a Etapa 2.
- **Brilho/contraste**: o slider de realce (`sss_stretch`) altera apenas a
  exibição. Os números do painel "Objeto(s) na cena" e qualquer cálculo de
  validação futuro usam sempre a imagem física original (0-1), não a
  versão com stretch aplicado.
- **Diagrama de geometria lateral**: representa o objeto como uma barra
  vertical simplificada (não a forma real — esfera, caixa, etc.). É um
  esquema didático da geometria de aquisição (altitude, raio acústico,
  sombra), não uma reconstrução 3D do objeto.
- Sem ground truth visual sobreposto na imagem principal (shadow mask, bbox)
  — apenas os números já calculados pela engine aparecem no painel lateral.
- Sem presets de domínio (`domain_scenarios.md`) — cada cena é montada
  manualmente.
- Apenas 1 objeto por cena (lista dinâmica de múltiplos objetos é melhoria
  futura, não bloqueante para o uso exploratório atual).
        """
    )
