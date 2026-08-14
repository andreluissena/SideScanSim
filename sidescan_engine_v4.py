"""
SideScanSim Engine v4.0 — Material-Aware Object Simulation
Extends v3.1 with:
  - MATERIAL_TABLE: acoustic properties by object material (metal, concrete,
    wood, plastic, natural rock, coral)
  - SceneObject.material field with automatic TS and surface speckle
  - create_object() helper for clean scene construction
  - Enriched ground truth metadata per object (theoretical vs measured shadow)
  - Full backward compatibility with v3.1 SceneObject usage

Profile calibrated from 0004_2015.jpg (unchanged from v3.1).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import List, Tuple
from matplotlib.colors import LinearSegmentedColormap


@dataclass
class SonarParams:
    frequency_kHz:           float = 450.0
    range_m:                 float = 30.0
    altitude_m:              float = 4.0
    pulse_duration_us:       float = 50.0
    beam_width_along_rad:    float = 0.017
    sound_speed_mps:         float = 1500.0
    pixels_per_meter_across: float = 15.0
    pixels_per_meter_along:  float = 15.0
    dynamic_range_dB:        float = 40.0
    altitude_jitter_m:       float = 0.05

    @property
    def alpha_dB_per_km(self):
        f = self.frequency_kHz
        return 0.1*f**2/(1+f**2) + 40*f**2/(4100+f**2) + 2.75e-4*f**2


SEDIMENT_TABLE = {
    "rock":           {"BS0_dB": -8.0,  "speckle_var": 0.88, "description": "Rocha"},
    "coarse_gravel":  {"BS0_dB": -15.0, "speckle_var": 0.78, "description": "Cascalho grosso"},
    "coarse_sand":    {"BS0_dB": -22.0, "speckle_var": 0.68, "description": "Areia grossa"},
    "medium_sand":    {"BS0_dB": -25.0, "speckle_var": 0.62, "description": "Areia média"},
    "fine_sand":      {"BS0_dB": -28.0, "speckle_var": 0.55, "description": "Areia fina"},
    "sandy_mud":      {"BS0_dB": -33.0, "speckle_var": 0.45, "description": "Lama arenosa"},
    "soft_mud":       {"BS0_dB": -38.0, "speckle_var": 0.35, "description": "Lama mole"},
    "compacted_sand": {"BS0_dB": -23.0, "speckle_var": 0.60, "description": "Areia compactada"},
    "mixed":          {"BS0_dB": -22.0, "speckle_var": 0.72, "description": "Misto"},
    "gravel":         {"BS0_dB": -18.0, "speckle_var": 0.74, "description": "Cascalho"},
}


# ─── TABELA DE MATERIAIS DE OBJETOS ──────────────────────────────
# Target Strength (TS) baseado em impedância acústica relativa à água.
# Referências: Urick (1983) cap. 7; Lurton (2002) §8.3.
#
# TS = 10·log10(|R|²·A_eff)  onde R é o coeficiente de reflexão de pressão.
# Para simplificação prática: TS aqui representa o boost de intensidade do
# highlight em dB acima do backscatter do fundo — valor positivo = brilhante.
#
# surface_roughness [0–1]: controla a componente difusa do speckle de superfície
#   0.0 = superfície especular lisa (metal polido)  → highlight muito pontual
#   1.0 = superfície muito rugosa (coral, rocha)    → highlight difuso, espalhado
#
# absorption_factor [0–1]: fração da energia absorvida (não refletida)
#   0.0 = reflexão total (metal)
#   1.0 = absorção total (teórico — não ocorre na prática)

MATERIAL_TABLE = {
    "steel": {
        "TS_dB": 22.0,
        "surface_roughness": 0.05,
        "absorption_factor": 0.02,
        "description": "Aço / metal",
        "note": "Reflexão especular dominante. Highlight muito brilhante e nítido.",
    },
    "aluminum": {
        "TS_dB": 20.0,
        "surface_roughness": 0.08,
        "absorption_factor": 0.03,
        "description": "Alumínio",
        "note": "Similar ao aço, ligeiramente menor impedância.",
    },
    "concrete": {
        "TS_dB": 10.0,
        "surface_roughness": 0.55,
        "absorption_factor": 0.15,
        "description": "Concreto / argamassa",
        "note": "Reflexão moderada, difusa. Highlight mais espalhado que metal.",
    },
    "natural_rock": {
        "TS_dB": 8.0,
        "surface_roughness": 0.70,
        "absorption_factor": 0.18,
        "description": "Rocha natural",
        "note": "Backscatter difuso, heterogêneo. Similar ao concreto com mais variância.",
    },
    "coral": {
        "TS_dB": 6.0,
        "surface_roughness": 0.85,
        "absorption_factor": 0.22,
        "description": "Coral / substrato recifal",
        "note": "Estrutura porosa. Backscatter muito difuso, bordas irregulares.",
    },
    "wood": {
        "TS_dB": 2.0,
        "surface_roughness": 0.40,
        "absorption_factor": 0.35,
        "description": "Madeira",
        "note": "Baixa impedância relativa. Highlight fraco, absorção parcial.",
    },
    "plastic": {
        "TS_dB": -2.0,
        "surface_roughness": 0.20,
        "absorption_factor": 0.45,
        "description": "Plástico / polímero",
        "note": "Impedância próxima à água. Reflexão fraca. Difícil de detectar.",
    },
    "fiberglass": {
        "TS_dB": 1.0,
        "surface_roughness": 0.15,
        "absorption_factor": 0.40,
        "description": "Fibra de vidro (FRP)",
        "note": "Comum em cascos de embarcações pequenas. Reflexão fraca.",
    },
    "rubber": {
        "TS_dB": -8.0,
        "surface_roughness": 0.30,
        "absorption_factor": 0.60,
        "description": "Borracha / elastômero",
        "note": "Material anecóico. Absorção alta. Muito difícil de detectar.",
    },
}


@dataclass
class SeafloorParams:
    sediment_type:       str   = "medium_sand"
    ripple_wavelength_m: float = 0.5
    ripple_amplitude:    float = 0.06    # leve — padrão real
    ripple_enabled:      bool  = False   # desligado por padrão


@dataclass
class SceneObject:
    shape:              str
    across_m:           float   # distância horizontal ao nadir [m]
    along_m:            float   # posição ao longo da trajetória [m]
    height_m:           float   # altura do objeto acima do fundo [m]
    width_m:            float = 1.0
    length_m:           float = 2.0
    material:           str   = "steel"       # chave em MATERIAL_TABLE
    target_strength_dB: float = None          # None = usa MATERIAL_TABLE
    name:               str   = "object"

    def effective_TS(self) -> float:
        """
        Retorna o Target Strength efetivo.
        Se target_strength_dB for definido manualmente, usa esse valor (override).
        Caso contrário, usa a tabela de materiais.
        """
        if self.target_strength_dB is not None:
            return self.target_strength_dB
        mat = MATERIAL_TABLE.get(self.material, MATERIAL_TABLE["steel"])
        return mat["TS_dB"]

    def surface_roughness(self) -> float:
        """Rugosidade superficial do material [0–1]."""
        mat = MATERIAL_TABLE.get(self.material, MATERIAL_TABLE["steel"])
        return mat["surface_roughness"]

    def theoretical_shadow_length(self, vehicle_altitude_m: float) -> float:
        """
        Comprimento teórico da sombra acústica [m].
        Fórmula: L = H · x / (h - H)   (válida para H << h)
        Referência: Blondel (2009) §6.2
        """
        H, x, h = self.height_m, self.across_m, vehicle_altitude_m
        if h <= H:
            return float('inf')   # objeto tão alto quanto a altitude — sombra infinita
        return H * x / (h - H)


def create_object(
    shape: str,
    nadir_distance_m: float,
    along_m: float,
    height_m: float,
    material: str = "steel",
    width_m: float = None,
    length_m: float = None,
    name: str = None,
) -> SceneObject:
    """
    Helper para criar SceneObject com parâmetros intuitivos.

    Parâmetros
    ----------
    shape            : 'sphere', 'box', 'cylinder', 'mound', 'rock', 'wreck'
    nadir_distance_m : distância horizontal do nadir ao centro do objeto [m]
                       (equivale a across_m — nomenclatura mais intuitiva)
    along_m          : posição ao longo do eixo de movimento [m]
    height_m         : altura do objeto acima do fundo [m]
    material         : chave em MATERIAL_TABLE (default: 'steel')
    width_m          : largura across-track [m] (default: auto por shape)
    length_m         : comprimento along-track [m] (default: auto por shape)
    name             : identificador textual (default: shape_material)

    Retorna
    -------
    SceneObject pronto para uso em render_sss_image()

    Exemplos
    --------
    # Esfera de aço de 0.8m de diâmetro a 12m do nadir
    obj = create_object('sphere', nadir_distance_m=12, along_m=40,
                        height_m=0.4, material='steel', width_m=0.8)

    # Bloco de concreto 2×1×0.5m a 20m do nadir
    obj = create_object('box', nadir_distance_m=20, along_m=60,
                        height_m=0.5, material='concrete',
                        width_m=2.0, length_m=1.0)
    """
    # Dimensões default razoáveis por shape se não especificadas
    defaults = {
        "sphere":   {"width_m": height_m * 2.0, "length_m": height_m * 2.0},
        "box":      {"width_m": height_m * 3.0, "length_m": height_m * 2.0},
        "cylinder": {"width_m": height_m * 1.2, "length_m": height_m * 4.0},
        "mound":    {"width_m": height_m * 4.0, "length_m": height_m * 4.0},
        "rock":     {"width_m": height_m * 3.5, "length_m": height_m * 3.0},
        "wreck":    {"width_m": height_m * 2.5, "length_m": height_m * 6.0},
    }
    d = defaults.get(shape, {"width_m": height_m * 2.0, "length_m": height_m * 2.0})
    w = width_m  if width_m  is not None else d["width_m"]
    l = length_m if length_m is not None else d["length_m"]
    n = name if name is not None else f"{shape}_{material}"

    return SceneObject(
        shape=shape,
        across_m=nadir_distance_m,
        along_m=along_m,
        height_m=height_m,
        width_m=w,
        length_m=l,
        material=material,
        name=n,
    )

# ─── PERFIL ACROSS-TRACK CALIBRADO ────────────────────────────────
def calibrated_profile(n_across: int, seafloor: 'SeafloorParams',
                        BS0_dB: float = -25.0) -> np.ndarray:
    """
    Perfil across-track calibrado contra imagem real 0004_2015.jpg.

    Forma medida:
    - Nadir (x=0): zero (sombra)
    - Zona de transição (x<8%): subida rápida
    - Pico near-nadir (x≈13-16%): ~0.42  ← especular + Lambert
    - Falloff (16% → 25%): queda rápida para ~0.18
    - Planalto (25% → 70%): relativamente plano ~0.18-0.20
    - Far-range (70%→100%): leve queda para ~0.15

    Dois componentes físicos:
    1. Especular near-nadir: Gaussiana estreita em x≈13%
    2. Lambert+TVG: nível plano ~0.17 (fundo sedimento compensado)
    """
    x = np.linspace(0, 1, n_across)

    # Ajuste pelo tipo de sedimento (BS0 mais alto = fundo mais brilhante)
    bs_adjust = 10.0 ** ((BS0_dB - (-25.0)) / 20.0)  # normalizado para areia média

    # Componente 1: especular near-nadir
    spec_amp   = 0.24 * bs_adjust
    spec_pos   = 0.14
    spec_width = 0.032
    rise       = 1.0 / (1.0 + np.exp(-100.0 * (x - 0.06)))
    specular   = spec_amp * np.exp(-(x - spec_pos)**2 / (2.0*spec_width**2)) * rise

    # Componente 2: Lambert+TVG (fundo plano)
    lambert_base = 0.17 * bs_adjust
    # Leve undulação no mid-range (heterogeneidade macro)
    mid_boost = 0.025 * np.exp(-(x - 0.45)**2 / (2*0.28**2))
    lambert   = lambert_base + mid_boost

    # Máscara de nadir: forçar zero nos primeiros ~5% (nadir shadow)
    nadir_mask = 1.0 / (1.0 + np.exp(-120.0 * (x - 0.055)))
    profile    = np.clip((specular + lambert) * nadir_mask, 0, 1)

    # Ripple marks (opcional, amplitude muito leve)
    if seafloor.ripple_enabled:
        # Comprimento de onda em termos normalizados
        rng = np.random.default_rng(42)
        phase = rng.uniform(0, 2*np.pi)
        ripple = 1.0 + seafloor.ripple_amplitude * np.sin(
            2*np.pi * x / (seafloor.ripple_wavelength_m / 30.0) + phase)
        profile *= ripple

    return profile.astype(np.float32)


# ─── SPECKLE CORRETO ──────────────────────────────────────────────
def speckle_ping_independent(n_along: int, n_across: int,
                              speckle_var: float = 0.62,
                              seed: int = 42) -> np.ndarray:
    """
    Speckle SSS correto: cada linha (ping) é INDEPENDENTE.
    Amplitude de Rayleigh, correlação across ~1.5px, zero along.
    """
    rng  = np.random.default_rng(seed)
    I    = rng.standard_normal((n_along, n_across)).astype(np.float32)
    Q    = rng.standard_normal((n_along, n_across)).astype(np.float32)
    # Correlação APENAS across-track (~1.5px = meia largura de pulso)
    for l in range(n_along):
        I[l] = gaussian_filter1d(I[l], sigma=1.3)
        Q[l] = gaussian_filter1d(Q[l], sigma=1.3)
    amp  = np.sqrt(I**2 + Q**2)
    amp /= amp.mean()
    # speckle multiplicativo: 1 + (amp-1)*var
    return (1.0 + (amp - 1.0) * speckle_var).astype(np.float32)


# ─── TEXTURA MACRO ────────────────────────────────────────────────
def macro_texture(n_along: int, n_across: int, seed: int = 123) -> np.ndarray:
    """
    Heterogeneidade lenta do sedimento.
    Varia APENAS across-track (longa correlação ~30px).
    Amplitude: ±8% do sinal.
    """
    rng    = np.random.default_rng(seed)
    noise  = rng.standard_normal(n_across).astype(np.float32)
    smooth = gaussian_filter1d(noise, sigma=35.0)
    smooth = smooth / (smooth.std() + 1e-8) * 0.08
    tex_1d = 1.0 + smooth
    # Tiny along-track variation (~1%)
    al     = rng.standard_normal(n_along).astype(np.float32)
    al     = gaussian_filter1d(al, sigma=10.0)
    al     = 1.0 + al / (al.std() + 1e-8) * 0.012
    return (np.outer(al, tex_1d)).astype(np.float32)


# ─── ALTURA DOS OBJETOS ───────────────────────────────────────────
def object_height(obj: SceneObject, x_m: np.ndarray, along_m: np.ndarray) -> np.ndarray:
    da = x_m - obj.across_m
    dl = along_m - obj.along_m
    DA, DL = np.meshgrid(da, dl)
    h = np.zeros_like(DA, dtype=np.float32)

    if obj.shape == "box":
        h[(np.abs(DA) <= obj.width_m/2) & (np.abs(DL) <= obj.length_m/2)] = obj.height_m
    elif obj.shape == "sphere":
        R = obj.width_m / 2.0
        r2 = DA**2 + DL**2; m = r2 <= R**2
        h[m] = np.sqrt(np.maximum(R**2 - r2[m], 0))
    elif obj.shape == "cylinder":
        R = obj.width_m / 2.0
        m = (np.abs(DA) <= R) & (np.abs(DL) <= obj.length_m/2)
        h[m] = np.sqrt(np.maximum(R**2 - DA[m]**2, 0))
    elif obj.shape == "mound":
        sa = obj.width_m/4; sl = obj.length_m/4
        h = (obj.height_m * np.exp(-(DA**2/(2*sa**2)+DL**2/(2*sl**2)))).astype(np.float32)
    elif obj.shape == "rock":
        rng = np.random.default_rng(int(abs(obj.across_m)*137+abs(obj.along_m)*31))
        for _ in range(6):
            dx=rng.uniform(-obj.width_m*.3,obj.width_m*.3)
            dy=rng.uniform(-obj.length_m*.3,obj.length_m*.3)
            sa=obj.width_m*rng.uniform(.10,.25); sl=obj.length_m*rng.uniform(.10,.25)
            amp=obj.height_m*rng.uniform(.55,1.0)
            h += amp*np.exp(-((DA-dx)**2/(2*sa**2)+(DL-dy)**2/(2*sl**2)))
        h = np.minimum(h, obj.height_m*1.15).astype(np.float32)
    elif obj.shape == "wreck":
        sa=obj.width_m/3.5; sl=obj.length_m/3.5
        h_hull = (obj.height_m*.55*np.exp(-(DA**2/(2*sa**2)+DL**2/(2*sl**2)))).astype(np.float32)
        hs = np.zeros_like(DA,dtype=np.float32)
        hs[(np.abs(DA)<=obj.width_m*.22)&(np.abs(DL-obj.length_m*.10)<=obj.length_m*.16)]=obj.height_m*.95
        hs[(np.abs(DA)<=obj.width_m*.12)&(np.abs(DL-obj.length_m*.38)<=obj.length_m*.09)]=obj.height_m*.75
        hs[(np.abs(DA)<=obj.width_m*.18)&(np.abs(DL+obj.length_m*.30)<=obj.length_m*.08)]=obj.height_m*.65
        h = np.maximum(h_hull, hs)
    return np.maximum(h, 0.0).astype(np.float32)


# ─── SOMBRA ───────────────────────────────────────────────────────
def compute_shadow(height_grid: np.ndarray, x_m: np.ndarray, h_alt: float) -> np.ndarray:
    """
    Ray casting vetorizado O(N) por linha para sombra acústica.

    Geometria correta:
      Transdutor em (x=0, z=h_alt). Fundo em z=0.
      Raio até o ponto (x_far, 0) tem altura z(x) = h_alt*(1 - x/x_far)
      no ponto x. Um objeto de altura H em x_near bloqueia esse raio se
      H >= z(x_near), ou seja:
          H >= h_alt * (x_far - x_near) / x_far
          h_alt*x_far - h_alt*x_near <= H*x_far
          x_far*(h_alt - H) <= h_alt*x_near
          → x_far <= x_near * h_alt / (h_alt - H)     [limite FINITO da sombra]

    Ou seja, a sombra projetada por um objeto em x_near com altura H cobre
    o intervalo FINITO:
          x_near < x_far <= t(x_near),  onde t(x_near) = x_near*h_alt/(h_alt-H)

    O comprimento dessa sombra é t(x_near) - x_near = H*x_near/(h_alt-H),
    exatamente a fórmula de 01_physics_reference.md §5.1.

    Como vários pixels de objeto podem projetar sombras distintas, o pixel
    x_far está em sombra se estiver dentro do alcance de PELO MENOS UM
    objeto à sua esquerda — ou seja, se x_far <= máximo acumulado de t
    entre todos os x_near < x_far com objeto.

    (Versão anterior usava mínimo acumulado + comparação ">" — isso fazia
    a sombra se estender da posição do objeto até a BORDA da imagem em vez
    de parar no comprimento físico correto. Corrigido aqui.)
    """
    n_along, n_across = height_grid.shape
    shadow = np.zeros((n_along, n_across), dtype=bool)

    for l in range(n_along):
        H = height_grid[l, :]          # alturas do objeto nessa linha

        # Limite (finito) da sombra projetada por cada pixel com objeto
        has_obj = H > 0.005
        t = np.full(n_across, -np.inf, dtype=np.float64)
        mask_valid = has_obj & (H < h_alt * 0.99)
        if not np.any(mask_valid):
            continue
        t[mask_valid] = x_m[mask_valid] * h_alt / (h_alt - H[mask_valid])

        # Máximo acumulado (da esquerda): maior alcance de sombra possível
        # considerando todos os obstáculos vistos até aqui
        max_t = np.maximum.accumulate(t)

        # Pixel j está na sombra se x_m[j] <= max_t[j-1] (dentro do alcance
        # de algum obstáculo anterior) e não é o próprio objeto
        s = np.zeros(n_across, dtype=bool)
        s[1:] = (x_m[1:] <= max_t[:-1]) & (~has_obj[1:])
        shadow[l] = s

    return shadow


# ─── MOTOR PRINCIPAL ──────────────────────────────────────────────
def render_sss_image(
    sonar:          SonarParams,
    seafloor:       SeafloorParams,
    objects:        List[SceneObject],
    scene_length_m: float = 80.0,
    speckle_seed:   int   = 42,
    texture_seed:   int   = 123,
) -> Tuple[np.ndarray, dict]:
    sed    = SEDIMENT_TABLE.get(seafloor.sediment_type, SEDIMENT_TABLE["medium_sand"])
    BS0    = sed["BS0_dB"]
    alpha  = sonar.alpha_dB_per_km
    h_alt  = sonar.altitude_m
    pxa    = sonar.pixels_per_meter_across
    pxl    = sonar.pixels_per_meter_along

    n_across = int(sonar.range_m * pxa)
    n_along  = int(scene_length_m * pxl)

    x_m     = np.linspace(h_alt * 0.01, sonar.range_m, n_across)
    r_m     = np.sqrt(x_m**2 + h_alt**2)
    theta   = np.arctan2(x_m, h_alt)
    along_m = np.linspace(0.0, scene_length_m, n_along)

    # 1. Perfil calibrado
    profile = calibrated_profile(n_across, seafloor, BS0)

    # 2. Textura macro (across only)
    texture = macro_texture(n_along, n_across, texture_seed)

    # 3. Grade 2D base
    image = np.tile(profile, (n_along, 1)) * texture

    # 4. Speckle correto (ping-a-ping independente)
    spk   = speckle_ping_independent(n_along, n_across, sed["speckle_var"], speckle_seed)
    image = (image * spk).astype(np.float32)

    # 5. Alturas
    hgrid = np.zeros((n_along, n_across), dtype=np.float32)
    for obj in objects:
        hg = object_height(obj, x_m, along_m)
        hgrid = np.maximum(hgrid, hg)

    # 6. Sombra
    shadow = compute_shadow(hgrid, x_m, h_alt)
    rng_s  = np.random.default_rng(speckle_seed + 9999)
    shd_lvl = profile.mean() * 0.025
    shd_n   = rng_s.exponential(shd_lvl, (n_along, n_across)).astype(np.float32)
    image[shadow] = shd_n[shadow]

    # 7. Highlights — intensidade e espalhamento dependem do material
    for obj in objects:
        hg   = object_height(obj, x_m, along_m)
        mask = hg > 0.005
        if not np.any(mask): continue

        ts_linear  = 10.0 ** (obj.effective_TS() / 10.0)
        roughness  = obj.surface_roughness()
        absorption = MATERIAL_TABLE.get(obj.material, MATERIAL_TABLE["steel"])["absorption_factor"]

        ax2  = np.tile(x_m, (n_along, 1))

        # Face factor: materiais lisos (metal) concentram energia na face frontal
        # Materiais rugosos (coral, rocha) distribuem o highlight mais amplamente
        face_sharpness = 1.0 - roughness * 0.5   # [0.5, 1.0]
        face = np.clip(
            1.0 - (ax2 - (obj.across_m - obj.width_m/2)) / (obj.width_m + 1e-3) * (0.65 * face_sharpness),
            0.35, 1.0
        )

        hf  = np.clip(hg / (obj.height_m + 1e-6), 0, 1)

        # Speckle de superfície: materiais rugosos têm mais variância no highlight
        rng_mat = np.random.default_rng(int(abs(obj.across_m)*97 + abs(obj.along_m)*53))
        surf_spk = 1.0 + rng_mat.standard_normal((n_along, n_across)).astype(np.float32) * roughness * 0.35
        surf_spk = np.clip(gaussian_filter1d(surf_spk, sigma=1.5, axis=1), 0.1, 2.5)

        # Highlight final: TS × face × altura relativa × absorção × speckle de superfície
        hl = np.clip(
            np.tile(profile, (n_along, 1)) * ts_linear * face * hf * (1.0 - absorption) * surf_spk,
            0, 1.0
        )
        image = np.where(mask & ~shadow, hl, image)

    # 8. Ping jitter (tiny heave)
    rng_j  = np.random.default_rng(speckle_seed + 12345)
    jitter = rng_j.standard_normal(n_along)
    jitter = gaussian_filter1d(jitter, sigma=6.0)
    jitter = 1.0 + jitter / (jitter.std() + 1e-10) * (sonar.altitude_jitter_m / h_alt)
    image  = np.clip(image * jitter[:, np.newaxis], 0, 1)

    # 9. Nadir
    nadir_px = max(8, int(h_alt * pxa * 0.20))
    rng_n    = np.random.default_rng(77777)
    nadir    = np.clip(rng_n.exponential(0.012, (n_along, nadir_px)), 0, 0.04).astype(np.float32)
    ep       = max(1, nadir_px // 8)
    tap      = np.linspace(0.0, 1.0, ep+1)[1:].astype(np.float32)
    nadir[:, :ep] *= tap; nadir[:, -ep:] *= tap[::-1]
    if nadir_px >= 3:
        c = nadir_px // 2
        nadir[:, c] = 0.88          # linha GPS
        if nadir_px >= 5:
            nadir[:, c-1] = 0.30; nadir[:, c+1] = 0.30

    # 10. Composição
    image_full = np.concatenate([np.fliplr(image), nadir, image], axis=1)
    n_full     = image_full.shape[1]

    # Rolloff gradual ao longo de TODA a extensão do range.
    # Fisicamente: Lambert backscatter cai suavemente com ângulo de grazing pequeno
    # (far-range). Transição visível em toda a largura, não apenas nos extremos.
    # Modelo: taper coseno de 1.0 (near-nadir) a 0.65 (far-edge), lento e contínuo.
    # Rolloff: construir exatamente n_full pontos para evitar mismatch de shape
    x_roll = np.linspace(0.0, 1.0, n_full).astype(np.float32)
    # Taper coseno simétrico: 1.0 no centro (nadir), 0.65 nas bordas (far-range)
    # |x_roll - 0.5| vai de 0 (centro) a 0.5 (bordas)
    ro = 1.0 - 0.35 * (1.0 - np.cos(np.pi * np.abs(x_roll - 0.5) * 2.0)) / 2.0
    ro = np.clip(ro, 0.65, 1.0).astype(np.float32)
    image_full = np.clip(image_full * ro[np.newaxis, :], 0, 1)

    # Ground truth por objeto: sombra teórica vs medida
    objects_gt = []
    for obj in objects:
        shadow_theoretical = obj.theoretical_shadow_length(h_alt)
        mat_info = MATERIAL_TABLE.get(obj.material, MATERIAL_TABLE["steel"])

        # Linha along-track mais próxima do centro do objeto
        row_obj = int(np.argmin(np.abs(along_m - obj.along_m)))
        col_obj = int(np.argmin(np.abs(x_m - obj.across_m)))

        # Sombra medida: extensão contígua de pixels em shadow a partir da borda do objeto
        shd_row = shadow[row_obj, :]      # linha de sombra na imagem (lado direito)
        # A sombra começa após o objeto (col >= col_obj)
        shd_after = shd_row[col_obj:]
        if np.any(shd_after):
            # Encontrar blocos contíguos de True
            diff = np.diff(np.concatenate([[0], shd_after.astype(int), [0]]))
            starts = np.where(diff == 1)[0]
            ends   = np.where(diff == -1)[0]
            if len(starts) > 0:
                # Maior bloco de sombra contíguo
                lengths = ends - starts
                shadow_measured = float(lengths.max()) / pxa
            else:
                shadow_measured = 0.0
        else:
            shadow_measured = 0.0

        objects_gt.append({
            "name":                  obj.name,
            "shape":                 obj.shape,
            "material":              obj.material,
            "material_description":  mat_info["description"],
            "TS_dB":                 obj.effective_TS(),
            "height_m":              obj.height_m,
            "width_m":               obj.width_m,
            "length_m":              obj.length_m,
            "nadir_distance_m":      obj.across_m,
            "grazing_angle_deg":     float(np.degrees(np.arctan2(obj.across_m, h_alt))),
            "shadow_theoretical_m":  shadow_theoretical,
            "shadow_measured_m":     shadow_measured,
            "shadow_error_pct":      abs(shadow_measured - shadow_theoretical) / (shadow_theoretical + 1e-6) * 100,
        })

    meta = dict(x_m=x_m, r_m=r_m, theta_rad=theta, along_m=along_m,
                nadir_width_px=nadir_px, n_across=n_across, n_along=n_along,
                shadow_mask=shadow, height_grid=hgrid,
                alpha_dB_per_km=alpha, BS0_dB=BS0,
                sediment=sed.get("description",""), profile=profile,
                objects_gt=objects_gt)
    return image_full.astype(np.float32), meta


# ─── COLORMAPS ────────────────────────────────────────────────────
def get_sss_colormap(name: str = "amber"):
    if name == "amber":
        c = [(0.000,0.000,0.000),(0.045,0.020,0.000),(0.110,0.048,0.000),
             (0.210,0.095,0.000),(0.340,0.175,0.000),(0.490,0.270,0.000),
             (0.630,0.375,0.005),(0.755,0.495,0.025),(0.858,0.635,0.055),
             (0.938,0.790,0.115),(0.978,0.925,0.270),(0.998,0.975,0.680),
             (1.000,1.000,0.940)]
    elif name == "gray":
        c = [(0.000,0.000,0.000),(0.038,0.042,0.050),(0.095,0.098,0.108),
             (0.190,0.192,0.198),(0.305,0.305,0.308),(0.430,0.430,0.428),
             (0.565,0.565,0.562),(0.700,0.700,0.695),(0.835,0.832,0.824),
             (0.940,0.938,0.928),(0.990,0.990,0.982),(1.000,1.000,0.995)]
    else:
        import matplotlib.pyplot as plt; return plt.get_cmap("gray")
    return LinearSegmentedColormap.from_list(f"sss_{name}", c, N=512)


# ─── STRETCH DE EXIBIÇÃO (NÃO altera os dados físicos) ───────────
def sss_stretch(image: np.ndarray,
                 low_percentile:  float = 1.0,
                 high_percentile: float = 99.0,
                 gamma:           float = 1.0) -> np.ndarray:
    """
    Ajuste de brilho/contraste PARA EXIBIÇÃO apenas.

    Importante — separação de responsabilidades:
    -----------------------------------------------
    O array retornado por `render_sss_image()` representa backscatter
    físico relativo (0-1). Esse valor é a base do ground truth (TS_dB,
    shadow_measured_m, métricas de validação do validation_criteria.md).
    NÃO devemos alterar esse array para "melhorar a aparência" — isso
    destruiria a diferenciação física entre materiais e sedimentos
    (ex.: aço TS=22dB vs plástico TS=-2dB ficariam parecidos se forçados
    para a mesma faixa visual).

    Esta função faz o que o equipamento real faz no display do operador:
    aplica um realce de contraste e gama SÓ para visualização, sem alterar
    o dado físico subjacente. Use a imagem original (não-stretched) para
    qualquer cálculo de validação ou ground truth.

    Parâmetros
    ----------
    image           : array 0-1, saída de render_sss_image()
    low_percentile  : percentil mapeado para 0 (sombras ficam mais pretas
                       se este valor for baixo, ex. 0.5-2.0)
    high_percentile : percentil mapeado para 1 (realces ficam mais
                       estourados se este valor for alto, ex. 98-99.5)
    gamma           : correção de gama pós-stretch.
                       gamma < 1.0 → clareia tons médios (fundo)
                       gamma > 1.0 → escurece tons médios
                       gamma = 1.0 → sem correção de gama

    Retorna
    -------
    Array float32, mesma forma, valores 0-1, SOMENTE para exibição.

    Exemplo
    -------
    # Fundo aparecendo muito escuro na tela → realça contraste e clareia
    # um pouco os tons médios:
    img_display = sss_stretch(image_full, low_percentile=1, high_percentile=99, gamma=0.7)
    """
    lo = np.percentile(image, low_percentile)
    hi = np.percentile(image, high_percentile)
    if hi <= lo:
        hi = lo + 1e-6

    stretched = (image - lo) / (hi - lo)
    stretched = np.clip(stretched, 0.0, 1.0)

    if gamma != 1.0:
        stretched = np.power(stretched, gamma)

    return stretched.astype(np.float32)
