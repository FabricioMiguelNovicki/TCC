#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
                                           
   ▓▓▓▓▓▓▓▓  ▓▓▓ ▓▓▓░    ▓▓▓ ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒░   | 
             ▓▓▓  ▓▓▓   ▓▓▓           ▒▒▒         |   Flow Induced Vibration
   ▓▓▓▓▓▓▓▓  ▓▓▓   ▓░  ▓▓▓   ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒    |   on Aeronautical Systems
   ▓▓▓       ▓▓▓      ▒▓▓░                  ▒▒▒   |   and Structures
   ▓▓▓       ▓▓▓     ░▓▓░    ▒▒▒▒▒▒▒▒ ▒▒▒▒▒▒▒▒    |
                                                 
-------------------------------------------------------------------------------
    
Description:
    Blade Element Momentum Theory (BEMT) extended to non-axial uniform inflow

@author: Lorenzo Stabeli Diehl, M.Sc. Student - LVA/UFSC
@date: 25 March 2026

"""

# Busque 'DOC-GUR' ou 'DOC-EXTRA' (Ctrl+F) para localizar as anotacoes.

#%%
# ========================================================================
# EXTERNAL PACKAGES
# ========================================================================

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.special import j0, j1, kv
import re
import os
import glob

#%%
# ========================================================================
# USER-DEFINED SETTINGS
# ========================================================================

# === PROPELLER NAME ===
PROP_NAME = "27X12TH2B"

# === PATH CONFIGURATION ===
PROJECT_FOLDER_NAME = f"{PROP_NAME}_PROJECT"
PROPELLER_DATA_FOLDER_NAME = os.path.join(PROJECT_FOLDER_NAME, f"{PROP_NAME}_DATA")

# [DOC-GUR-01] Cap. 4.1, pp. 41–44 — representacao por elementos de pa.
# TCC 3.3: quantidade de estacoes radiais e azimutais e uma decisao numerica.
# Gur apresenta a integracao radial; a malha azimutal decorre da extensao
# deste solver para escoamento nao axial (nao do caso axial simplificado).
# === DISCRETIZATION ===
N_RADIAL_STATIONS = 200  # Number of radial stations
N_AZIMUTHAL_STATIONS = 72  # Number of azimuthal stations

# === AZIMUTHAL ANGLE WHERE ADVANCING SIDE [DEGREES] BEGINS  ===
ADVANCING_SIDE_START_DEG = 180

# === ROTATION CONFIGURATION ===
ROTATION_DIRECTION = 'CCW'  # Options: 'CCW' (counter-clockwise) or 'CW' (clockwise)

# [DOC-GUR-02] Cap. 4.3–4.4, pp. 50–57 — importancia das correcoes.
# REVISAR: as correcoes aqui selecionadas NAO sao todas as descritas por Gur.
# Kaplan/Karman-Tsien/Korn: compressibilidade; Prandtl: perdas de ponta/raiz.
# Gur 4.3 trata correcoes 3D de angulo de ataque/estol; Gur 4.4 usa outro F_T.
# --- CORRECTION FLAGS ---
USE_TIP_LOSS = True  # Enable/disable Prandtl tip loss factor
USE_HUB_LOSS = True  # Enable/disable Prandtl hub loss factor
USE_LIFT_CORRECTION = True # Enable/disable compressibility correction for lift
LIFT_CORRECTION_METHOD = 'kaplan'  # Options: 'kaplan' or 'karman-tsien'
USE_DRAG_CORRECTION = True  # Enable/disable wave drag correction
CONSIDER_DRAG_DIVERGENCE = True # Enable/disable wave drag correction

# --- SOLVER PARAMETERS ---
BEMT_MAX_ITER = 100000
BEMT_TOLERANCE = 1e-9
BEMT_RELAX = 0.1  # Relaxation factor 
KAPLAN_MAX_ITER = 1000
KAPLAN_TOLERANCE = 1e-6

# [DOC-GUR-03] Cap. 4.1, p. 43 — entradas: condicoes de operacao e ar.
# TCC 3.2.1 / 3.3: V_inf, RPM, incidencia, densidade, viscosidade, som.
# DOC-EXTRA: alpha_P != 0 representa escoamento inclinado (Diehl, 2026).
# NOTA: RUN='Z' usa V_INF=1 m/s, portanto nao e J=0 exatamente.
# === ENVIRONMENTAL AND OPERATING CONDITIONS ===
RUN = 'Z'

if RUN == 'A':
    CASE_NAME = 'TEST_2A'
    V_INF = 4.4502832      # m/s (Freestream velocity)
    OMEGA_RPM = 4751.313107467888   # RPM (Rotation speed)
    ALPHA_P_DEG = 12.8      # degrees (Incidence angle)
    RHO = 1.1934652554227514  # kg/m^3 (Air density at standard conditions)
    MU = 1.8004135069378658e-05  # Ns/m^2 (Dynamic viscosity of air)
    A_SOUND = 345.0174201700278  # m/s (Speed of sound in air at standard conditions)
    GAMMA = 1.4  # Ratio of specific heats for air

elif RUN == 'B':
    CASE_NAME = 'TEST_2B'
    V_INF = 3.832622933333334  # m/s (Freestream velocity)
    OMEGA_RPM = 5511.045879164529  # RPM (Rotation speed)
    ALPHA_P_DEG = 33.5     # degrees (Incidence angle)
    RHO = 1.194891785466579  # kg/m^3 (Air density at standard conditions)
    MU = 1.7991207344472444e-05  # Ns/m^2 (Dynamic viscosity of air)
    A_SOUND = 344.80620541064076  # m/s (Speed of sound in air at standard conditions)
    GAMMA = 1.4  # Ratio of specific heats for air
    
elif RUN == 'C':
    CASE_NAME = 'TEST_2C'
    V_INF = 3.1002224      # m/s (Freestream velocity)
    OMEGA_RPM = 6289.791526489976  # RPM (Rotation speed)
    ALPHA_P_DEG = 12.8     # degrees (Incidence angle)
    RHO = 1.1950541613529115  # kg/m^3 (Air density at standard conditions)
    MU = 1.7986290724286127e-05  # Ns/m^2 (Dynamic viscosity of air)
    A_SOUND = 344.78554624890296  # m/s (Speed of sound in air at standard conditions)
    GAMMA = 1.4  # Ratio of specific heats for air

elif RUN == 'D':
    CASE_NAME = 'TEST_2D'
    V_INF = 3.79984        # m/s (Freestream velocity)
    OMEGA_RPM = 7086.897397130495  # RPM (Rotation speed)
    ALPHA_P_DEG = 35.8     # degrees (Incidence angle)
    RHO = 1.1954962584484736 # kg/m^3 (Air density at standard conditions)
    MU = 1.7982255503123885e-05 # Ns/m^2 (Dynamic viscosity of air)
    A_SOUND = 344.71365122385754  # m/s (Speed of sound in air at standard conditions)
    GAMMA = 1.4  # Ratio of specific heats for air
    
elif RUN == 'Z':
    CASE_NAME = 'TEST_2Z'
    V_INF = 1        # m/s (Freestream velocity)
    OMEGA_RPM = 7200  # RPM (Rotation speed)
    ALPHA_P_DEG = 0     # degrees (Incidence angle)
    RHO = 1.1954962584484736 # kg/m^3 (Air density at standard conditions)
    MU = 1.7982255503123885e-05 # Ns/m^2 (Dynamic viscosity of air)
    A_SOUND = 344.71365122385754  # m/s (Speed of sound in air at standard conditions)
    GAMMA = 1.4  # Ratio of specific heats for air
    
OUTPUT_FOLDER_NAME = os.path.join(PROJECT_FOLDER_NAME, f"BEMT_RESULTS_{CASE_NAME}")

#%%
# ========================================================================
# 1 - LOADING GEOMETRICAL DATA
# ========================================================================

# [DOC-GUR-04] Cap. 4.1, p. 43; geometria detalhada no cap. 3.
# TCC 3.2.1 / 3.2.2: raio externo, raio interno e numero de pas.
# A secao 4.1 pressupoe uma geometria conhecida em cada posicao radial.
# Load propeller constants (Rtip, Rhub, B)
df_prop_constants = pd.read_csv(os.path.join(PROPELLER_DATA_FOLDER_NAME, f'{PROP_NAME}.csv'))
Rtip = float(df_prop_constants[df_prop_constants['property'] == 'Rtip']['file'].values[0])
Rhub = float(df_prop_constants[df_prop_constants['property'] == 'Rhub']['file'].values[0])
B = int(df_prop_constants[df_prop_constants['property'] == 'B']['file'].values[0])
D = 2 * Rtip

# [DOC-GUR-05] Cap. 4.1, pp. 42–43; cap. 3 para definicoes geometricas.
# TCC 3.2.2 / 3.3: c(r), beta(r), t/c e sweep alimentam as secoes.
# O caso simplificado do livro utiliza principalmente corda e pitch local.
# --- Load blade geometry data for chord, twist, t/c and sweep angle ---
data = pd.read_csv(os.path.join(PROPELLER_DATA_FOLDER_NAME, f'{PROP_NAME}_bladespan.csv'))
r_rt = data["r/rt"].values
b_D = data["b/D"].values
t_b = data["t/b"].values
twist_geom = data["twist"].values
MCA = data["circumferential_midpoint"].values
data['chord'] = b_D * D

# Calculate actual radial positions
r_geom = r_rt * (D / 2)

# Sweep angle (degrees) based on circumferential midpoint
sweep = np.zeros_like(r_rt)
sweep[1:] = np.rad2deg(np.arctan2(np.diff(MCA), (np.diff(r_rt) * (D / 2))))

# Create interpolators for chord, twist, t/c and sweep angle
interp_chord_from_geom = interp1d(r_geom, data['chord'], bounds_error=False, fill_value='extrapolate')
interp_twist_from_geom = interp1d(r_geom, np.deg2rad(twist_geom), bounds_error=False, fill_value='extrapolate')
interp_t_over_c = interp1d(r_rt, t_b, bounds_error=False, fill_value='extrapolate')
interp_sweep = interp1d(r_rt, sweep, bounds_error=False, fill_value='extrapolate')

print("Loaded blade geometry data for chord, twist, and compressibility corrections")
print(f"  Chord range: {data['chord'].min():.6f} to {data['chord'].max():.6f} m")
print(f"  Twist range: {twist_geom.min():.2f}° to {twist_geom.max():.2f}°")
print(f"  t/c range: {t_b.min():.4f} to {t_b.max():.4f}")
print(f"  Sweep angle range: {sweep.min():.2f}° to {sweep.max():.2f}°")

print(f"\n{'='*60}")
print(f"USER SETTING: Using {N_RADIAL_STATIONS} radial stations")
print(f"{'='*60}")

# [DOC-GUR-06] Cap. 4.1, Eq. (4.4), p. 43 — integracao ao longo do raio.
# TCC 3.3: malha cossenoidal e largura dr sao escolhas de quadratura numerica.
# REVISAR depois: convergencia dos resultados com refinamento da malha.
# Define BEMT radial stations through interpolation
def cosine_spacing(n_stations, r_hub, r_tip):
    """Generate cosine-spaced radial stations for better tip resolution"""
    theta = np.linspace(0, np.pi, n_stations)
    r_normalized = 0.5 * (1 - np.cos(theta))
    return r_hub + r_normalized * (r_tip - r_hub)

# Generate radial stations
r_stations = cosine_spacing(N_RADIAL_STATIONS, Rhub, Rtip)

# Calculate differential annulus widths
dr_annulus = np.zeros_like(r_stations)
dr_annulus[0] = (r_stations[1] - Rhub) / 2.0
dr_annulus[1:-1] = (r_stations[2:] - r_stations[:-2]) / 2.0
dr_annulus[-1] = (Rtip - r_stations[-2]) / 2.0

# Get chord and twist at r_stations through interpolation
c_stations = interp_chord_from_geom(r_stations)
twist_stations = interp_twist_from_geom(r_stations)

print(f"\nDefined {len(r_stations)} BEMT radial stations (cosine-spaced)")
print(f"  Radial range: r/R = {r_stations[0]/Rtip:.3f} to {r_stations[-1]/Rtip:.3f}")
print(f"  Station spacing: Min Δr = {np.min(np.diff(r_stations)):.6f} m, "
      f"Max Δr = {np.max(np.diff(r_stations)):.6f} m")

#%%
# ========================================================================
# 2 - LOADING AIRFOIL POLAR DATA
# ========================================================================

# [DOC-GUR-07] Cap. 4.2, pp. 45–50 — banco de dados aerodinamicos 2D.
# TCC 2.2.4 / 3.3: leitura de CL(alpha,Re) e CD(alpha,Re) por secao radial.
# O solver importa arquivos ja existentes: nao executa XFOIL.
# A origem, o Mach, a cobertura e a confiabilidade das polares exigem registro.
def load_all_polars(data_folder):
    """Load all polar data from the folder and organize by section (r/R) and Reynolds number."""
    POLARS = {}
    
    # Find all polar files
    pattern = os.path.join(data_folder, "*_polar_section_*_re_*.csv")
    polar_files = glob.glob(pattern)
    
    # Also check for .txt files
    pattern_txt = os.path.join(data_folder, "*_polar_section_*_re_*.txt")
    polar_files.extend(glob.glob(pattern_txt))
    
    print(f"Found {len(polar_files)} polar files")
    
    for filename in polar_files:
        # Extract r/R and Re from filename
        match = re.search(r'section_(\d+)_re_([0-9p]+e[0-9]+)', filename)
        if match:
            r_over_R_val = float(match.group(1)) / 100.0  # '20' -> 0.20
            re_str = match.group(2).replace('p', '.')  # '1p5e5' -> '1.5e5'
            Re_val = float(re_str)
            
            try:
                # Try loading with pandas first
                df_polar = pd.read_csv(filename)
                
                # Handle different possible column names
                if 'Alpha' in df_polar.columns:
                    df_polar.columns = ['Alpha', 'Cl', 'Cd', 'Cm', 'Cpmin_0']
                else:
                    # Assume standard order
                    df_polar.columns = ['Alpha', 'Cl', 'Cd', 'Cm', 'Cpmin_0']
                
                # Store in dictionary
                if r_over_R_val not in POLARS:
                    POLARS[r_over_R_val] = {}
                
                POLARS[r_over_R_val][Re_val] = {
                    'alpha': df_polar['Alpha'].values,
                    'Cl': df_polar['Cl'].values,
                    'Cd': df_polar['Cd'].values
                }
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    # Print summary
    print("\nLoaded polar data summary:")
    for r_over_R in sorted(POLARS.keys()):
        Re_list = sorted(POLARS[r_over_R].keys())
        print(f"  r/R = {r_over_R:.2f}: Re = {Re_list}")
    
    return POLARS

# [DOC-GUR-08] Cap. 4.2, p. 45 — necessidade de dados bem tabulados.
# TCC 2.2.4: interpolacao de coeficientes entre angulos e numeros de Reynolds.
# Se houver somente um Re, o codigo desconsidera a dependencia de Re.
# REVISAR: extrapolacao de alpha/Re para fora da faixa da base.
def create_polar_interpolator(POLARS):
    """Create 2D interpolators (Re, alpha) -> (Cl, Cd) for each radial section."""
    INTERPOLATORS = {}
    
    for r_over_R in POLARS.keys():
        Re_values = sorted(POLARS[r_over_R].keys())
        
        if len(Re_values) == 1:
            # Only one Reynolds number available, create 1D interpolator
            Re_single = Re_values[0]
            alpha_vals = POLARS[r_over_R][Re_single]['alpha']
            Cl_vals = POLARS[r_over_R][Re_single]['Cl']
            Cd_vals = POLARS[r_over_R][Re_single]['Cd']
            
            # Create simple 1D interpolators (Re-independent)
            interp_Cl = interp1d(alpha_vals, Cl_vals, kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
            interp_Cd = interp1d(alpha_vals, Cd_vals, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
            
            # Wrap in a lambda that ignores Re
            INTERPOLATORS[r_over_R] = {
                'Cl': lambda alpha, Re=None, interp=interp_Cl: interp(alpha),
                'Cd': lambda alpha, Re=None, interp=interp_Cd: interp(alpha),
                'Re_range': [Re_single, Re_single]
            }
        else:
            # Multiple Reynolds numbers - create 2D interpolator
            all_alphas = set()
            for Re in Re_values:
                all_alphas.update(POLARS[r_over_R][Re]['alpha'])
            alpha_grid = np.array(sorted(all_alphas))
            
            # Create regular grid for Cl and Cd
            Cl_grid = np.zeros((len(Re_values), len(alpha_grid)))
            Cd_grid = np.zeros((len(Re_values), len(alpha_grid)))
            
            for i, Re in enumerate(Re_values):
                alpha_orig = POLARS[r_over_R][Re]['alpha']
                Cl_orig = POLARS[r_over_R][Re]['Cl']
                Cd_orig = POLARS[r_over_R][Re]['Cd']
                
                # Interpolate to common alpha grid
                Cl_grid[i, :] = np.interp(alpha_grid, alpha_orig, Cl_orig)
                Cd_grid[i, :] = np.interp(alpha_grid, alpha_orig, Cd_orig)
            
            # Create 2D interpolators
            interp_Cl_2D = RegularGridInterpolator(
                (Re_values, alpha_grid), Cl_grid,
                method='linear', bounds_error=False, fill_value=None
            )
            interp_Cd_2D = RegularGridInterpolator(
                (Re_values, alpha_grid), Cd_grid,
                method='linear', bounds_error=False, fill_value=None
            )
            
            INTERPOLATORS[r_over_R] = {
                'Cl': lambda alpha, Re, interp=interp_Cl_2D: interp((Re, alpha)),
                'Cd': lambda alpha, Re, interp=interp_Cd_2D: interp((Re, alpha)),
                'Re_range': [min(Re_values), max(Re_values)]
            }
    
    return INTERPOLATORS

# [DOC-GUR-09] Cap. 4.1, p. 43; cap. 4.2, pp. 45–50.
# TCC 3.3: CL e CD seccionais sao consultados para alpha e Re locais;
# o codigo tambem interpola entre estacoes radiais disponiveis.
# A dependencia de Mach e tratada separadamente por correcoes, nao
# por uma base de polares tabulada diretamente em (alpha, Re, M).
def get_Cl_Cd(Re, alpha_deg, r_over_R, INTERPOLATORS):
    """Get Cl and Cd for given Reynolds number, angle of attack, and radial position."""
    available_sections = sorted(INTERPOLATORS.keys())
    
    # Find closest section or interpolate between sections
    if r_over_R <= available_sections[0]:
        section = available_sections[0]
        Cl = INTERPOLATORS[section]['Cl'](alpha_deg, Re)
        Cd = INTERPOLATORS[section]['Cd'](alpha_deg, Re)
    elif r_over_R >= available_sections[-1]:
        section = available_sections[-1]
        Cl = INTERPOLATORS[section]['Cl'](alpha_deg, Re)
        Cd = INTERPOLATORS[section]['Cd'](alpha_deg, Re)
    else:
        # Interpolate between two closest sections
        idx = np.searchsorted(available_sections, r_over_R)
        r1 = available_sections[idx - 1]
        r2 = available_sections[idx]
        
        # Linear interpolation weight
        weight = (r_over_R - r1) / (r2 - r1)
        
        # Get Cl and Cd from both sections
        Cl1 = INTERPOLATORS[r1]['Cl'](alpha_deg, Re)
        Cd1 = INTERPOLATORS[r1]['Cd'](alpha_deg, Re)
        Cl2 = INTERPOLATORS[r2]['Cl'](alpha_deg, Re)
        Cd2 = INTERPOLATORS[r2]['Cd'](alpha_deg, Re)
        
        # Interpolate
        Cl = Cl1 * (1 - weight) + Cl2 * weight
        Cd = Cd1 * (1 - weight) + Cd2 * weight
    
    return Cl, Cd

# Load all polar data
POLARS = load_all_polars(PROPELLER_DATA_FOLDER_NAME)
INTERPOLATORS = create_polar_interpolator(POLARS)

print(f"\nCreated interpolators for {len(INTERPOLATORS)} radial sections")

#%%
# ========================================================================
# 3 - DEFINING ENVIRONMENTAL AND OPERATING CONDITIONS
# ========================================================================

print("\n" + "="*60)
print("CORRECTION FLAGS:")
print("="*60)
print(f"  Prandtl Tip Loss Factor:      {'ENABLED' if USE_TIP_LOSS else 'DISABLED'}")
print(f"  Prandtl Hub Loss Factor:      {'ENABLED' if USE_HUB_LOSS else 'DISABLED'}")
print(f"  Lift Compressibility:         {'ENABLED' if USE_LIFT_CORRECTION else 'DISABLED'}")
if USE_LIFT_CORRECTION:
    print(f"    Method:                     {LIFT_CORRECTION_METHOD.upper()}")
    if LIFT_CORRECTION_METHOD == 'kaplan':
        print(f"    Max iterations:             {KAPLAN_MAX_ITER}")
        print(f"    Tolerance:                  {KAPLAN_TOLERANCE}")
print(f"  Wave Drag Correction:         {'ENABLED' if USE_DRAG_CORRECTION else 'DISABLED'}")
print("="*60)

# Convert rotation settings to internal conventions
if ROTATION_DIRECTION == 'CCW':
    rotation_sign = 1.0  # Positive angular velocity
elif ROTATION_DIRECTION == 'CW':
    rotation_sign = -1.0  # Negative angular velocity
else:
    raise ValueError("ROTATION_DIRECTION must be 'CCW' or 'CW'")


print("\n" + "="*60)
print("ROTATION CONFIGURATION:")
print("="*60)
print(f"  Direction:        {ROTATION_DIRECTION}")
print("="*60)

Omega = OMEGA_RPM * 2 * np.pi / 60.0  # rad/s
alpha_P_rad = np.deg2rad(ALPHA_P_DEG)

# [DOC-EXTRA-01] Gur, cap. 2, pp. 25–26 — coeficientes adimensionais.
# TCC 2.2 / 3.3: J = V_inf/(n D); normalizacoes para CT e CQ.
# Este trecho nao pertence diretamente as Eqs. (4.1–4.4) do cap. 4.
# J, Advance Ratio: J = V_inf / (n * D)
n = Omega / (2 * np.pi)
J = V_INF / (n * D)

# Non-dimensionalization factors
force_norm = RHO * n**2 * D**4
moment_norm = RHO * n**2 * D**5

# Create output folder
os.makedirs(OUTPUT_FOLDER_NAME, exist_ok=True)
print(f"\nOutput folder created: {OUTPUT_FOLDER_NAME}")

#%%
# ========================================================================
# 4 - COMPRESSIBILITY AND TIP-LOSS CORRECTION FUNCTIONS
# ========================================================================

# [DOC-GUR-10] Cap. 4.4, pp. 56–57 — efeitos de ponta: relacao CONCEITUAL.
# TCC 2.2.3 / 3.3: o codigo usa o fator de PRANDTL.
# NAO confundir com o fator F_T por trechos apresentado por Gur na sec. 4.4.
def prandtl_tip_loss_factor(r, Rtip, B, phi):
    """Calculate Prandtl's tip loss factor."""
    f = (B * (Rtip - r)) / (2 * r * np.sin(np.abs(phi)))
    F = (2.0 / np.pi) * np.arccos(np.exp(-f))
    return F

# [DOC-EXTRA-02] Correcao de raiz do tipo Prandtl.
# Nao foi identificada como implementacao literal da sec. 4.4 de Gur.
# TCC 3.3: documentar separadamente da correcao de ponta.
def prandtl_hub_loss_factor(r, Rhub, B, phi):
    """Calculate Prandtl's hub loss factor."""
    f = (B * (r - Rhub)) / (2 * r * np.sin(np.abs(phi)))
    F = (2.0 / np.pi) * np.arccos(np.exp(-f))
    return F

# [DOC-EXTRA-03] Estimativa do Mach de divergencia de arrasto (Korn).
# Gur, cap. 4.2, discute a dependencia das polares em Mach;
# esta equacao especifica e uma extensao presente no codigo.
# TCC 2.2.4 / 3.3: conferir referencia original e dominio de validade.
def calculate_drag_divergence_mach(Cl_c, t_over_c, sweep_deg):
    """Calculate drag divergence Mach number using Korn equation."""
    sweep_rad = np.deg2rad(sweep_deg)
    cos_sweep = np.cos(sweep_rad)
    
    M_dd = (0.87 - Cl_c / (10 * cos_sweep**2) - t_over_c / cos_sweep) / cos_sweep
        
    return M_dd

# [DOC-EXTRA-04] Correcao de sustentacao por compressibilidade (Kaplan).
# Nao equivale as correcoes 3D de AOA/estol discutidas no cap. 4.3 de Gur.
# TCC 3.3: separar claramente o tratamento de Mach do efeito de rotacao.
def kaplan_compressibility_correction(Cl_i, M, t_over_c, sweep_deg, 
                                     max_iter=KAPLAN_MAX_ITER, 
                                     tolerance=KAPLAN_TOLERANCE,
                                     consider_divergence=True):  # ADD THIS PARAMETER
    """Apply Kaplan compressibility correction to lift coefficient with robust iteration."""
    # Initial guess for M_dd using incompressible Cl
    M_dd = calculate_drag_divergence_mach(Cl_i, t_over_c, sweep_deg)
    Cl_c = Cl_i.copy()
    
    converged = False
    for iter_num in range(max_iter):
        Cl_c_old = Cl_c.copy()
        M_dd_old = M_dd.copy()
        
        # Calculate mu_K for M < M_dd
        term1 = 1.0 / np.sqrt(1 - M**2)
        term2 = (t_over_c / (1 + t_over_c)) * (term1 * (term1 - 1) + 0.25 * (GAMMA + 1) * ((term1**2 - 1)**2))
        mu_K = term1 + term2
        
        # MODIFY THIS SECTION:
        if consider_divergence:
            # Apply correction based on whether M > M_dd
            Cl_c = np.where(
                M <= M_dd,
                Cl_i * mu_K,
                Cl_i * mu_K * (1 - M**2) / (1 - M_dd**2)
            )
            # Update M_dd with new Cl_c
            M_dd = calculate_drag_divergence_mach(Cl_c, t_over_c, sweep_deg)
        else:
            # Simple correction without divergence consideration
            Cl_c = Cl_i * mu_K
            # M_dd doesn't change in this case
        
        # Check convergence
        if isinstance(Cl_c, np.ndarray):
            Cl_error = np.max(np.abs(Cl_c - Cl_c_old))
            M_dd_error = np.max(np.abs(M_dd - M_dd_old))
        else:
            Cl_error = abs(Cl_c - Cl_c_old)
            M_dd_error = abs(M_dd - M_dd_old)
        
        if Cl_error < tolerance and M_dd_error < tolerance:
            converged = True
            break
    
    return Cl_c, M_dd, converged

# [DOC-EXTRA-05] Correcao alternativa de compressibilidade: Karman–Tsien.
# Requer citacao teorica propria (nao atribuir a formulacao a Gur 4.3).
def karman_tsien_correction(Cl_i, M):
    """Apply Karman-Tsien compressibility correction to lift coefficient."""
    sqrt_term = np.sqrt(1 - M**2)
    denominator = sqrt_term + (M**2 / (1 + sqrt_term)) * Cl_i / 2.0
    Cl_c = Cl_i / denominator
    
    return Cl_c

# [DOC-EXTRA-06] Arrasto de onda acima do Mach de divergencia.
# TCC 2.2.4 / 3.3: modelo empirico adicional, alem da sec. 4.3 de Gur.
# REVISAR dominio de validade para as secoes mais externas da pa.
def wave_drag_correction(Cd_i, M, M_dd, consider_divergence=True):  # ADD THIS PARAMETER
    """Apply wave drag correction to drag coefficient."""
    if not consider_divergence:
        # No wave drag correction
        return Cd_i
    
    # Add wave drag term only when M > M_dd
    Cd_wave = np.where(
        M > M_dd,
        1.1 * ((M - M_dd) / (1 - M_dd))**3,
        0.0
    )
    
    Cd_c = Cd_i + Cd_wave
    
    return Cd_c

#%%
# ========================================================================
# 5 - UNSTEADY THIN-AIRFOIL AERODYNAMIC FUNCTIONS
# ========================================================================

# [DOC-EXTRA-07] Aerodinamica nao estacionaria: funcao de Theodorsen.
# Fora da formulacao principal do cap. 4 do Gur.
# TCC 3.3 / 3.5: fundamentar por referencias complementares (ex.: Diehl).
def theodorsen_function(sigma_val):
    """Calculate Theodorsen's function C(σ) for incompressible flow."""
    K0 = kv(0, 1j * sigma_val)
    K1 = kv(1, 1j * sigma_val)
    
    C_theo = K1 / (K0 + K1)
    
    return C_theo

# [DOC-EXTRA-08] Funcao de Sears incompressivel para resposta harmonica.
# Nao se trata da integracao estacionaria das Eqs. (4.2–4.4) de Gur.
def incompressible_sears_function(sigma_val):
    """Calculate incompressible Sears function S_M=0(σ)."""
    # Bessel functions of first kind
    J0_val = j0(sigma_val)
    J1_val = j1(sigma_val)
    
    # Modified Bessel functions of second kind
    C = theodorsen_function(sigma_val)
    
    # Sears function using equation 21
    S_incomp = (J0_val - 1j * J1_val) * C + 1j*J1_val
    
    return S_incomp

# [DOC-EXTRA-09] Extensao compressivel da funcao de Sears.
# TCC 3.3 / 3.5: representa resposta nao estacionaria relevante
# para os termos-fonte do modelo aeroacustico, nao descrita no cap. 4.
def compressible_sears_function(sigma_val, M_eff, zeta):
    """Calculate compressible Sears function S(σ,M)."""
    
    # Prandtl-Glauert factor
    zeta_sq = zeta**2
    
    # Scaled reduced frequency
    sigma_scaled = sigma_val / zeta_sq
    
    # Get incompressible Sears function at scaled frequency
    S_incomp = incompressible_sears_function(sigma_scaled)
    
    # Bessel term: J₀(M²σ/ζ²) + iJ₁(M²σ/ζ²)
    M_sq_sigma = M_eff**2 * sigma_scaled
    bessel_term = j0(M_sq_sigma) + 1j * j1(M_sq_sigma)
    
    # Correction factor f(M) from equation 23
    f_M = (1 - zeta) * np.log(M_eff) + zeta * np.log(1 + zeta) - np.log(2)
    # Phase correction term
    phase_correction = np.exp(-1j * sigma_val * f_M / zeta_sq)
    
    # Complete compressible Sears function
    S_comp = S_incomp * bessel_term * phase_correction
    
    # Check condition and print warning
    check_value = sigma_val * M_eff / zeta_sq
    if check_value >= 1.0:
        print(f"    WARNING: σM/ζ² = {check_value:.4f} >= 1.0")

    return S_comp

#%%
# ========================================================================
# 6 - BEMT ENGINE
# ========================================================================

# [DOC-GUR-11] Cap. 4.1, pp. 41–45 — nucleo de elemento de pa.
# TCC 3.3: este procedimento combina cinemática local, CL/CD, dT/dr e dQ/dr.
# As velocidades induzidas sao fechadas por balanco de momentum
# (aprofundar nos caps. 5–6 de Gur); o caso nao axial e extensao do livro.
def bemt_solve(r, c, twist_rad, V_inf, Omega, alpha_P_rad, rho, mu, B, r_over_R, 
               INTERPOLATORS, t_over_c, sweep_deg, a_sound, 
               use_tip_loss, use_hub_loss, use_lift_corr, use_drag_corr, lift_method,
               rotation_sign):  # NEW PARAMETERS
    """Solves for the induced velocities and sectional force/torque for a single radial station."""
    
    # Azimuthal angles with new convention
    # Base angles from 0 to 2π
    psi = np.linspace(0, 2 * np.pi, N_AZIMUTHAL_STATIONS, endpoint=False)
    
    # Sort to maintain monotonic progression for output
    sort_idx = np.argsort(psi)   
    
    # Apply rotation sign to angular velocity
    Omega_signed = Omega
    
    # [DOC-EXTRA-10] Escoamento nao axial: V_inf decomposta em axial e no plano.
    # TCC 3.3: incidencia alpha_P produz velocidade dependente do azimute.
    # Referencia complementar: Diehl (2026), sec. 2.4 e cap. 4.
    # Parameters derived from non-axial inflow
    V_A = V_inf * np.cos(alpha_P_rad)  # Axial component of freestream
    V_Z = rotation_sign * V_inf * np.sin(alpha_P_rad)  # In-plane component of freestream

    # [DOC-GUR-12] Cap. 4.1, Fig. 4.4, p. 45 — iteracao acoplada.
    # Chutes iniciais das velocidades induzidas; o modelo de inducoes
    # propriamente dito e assunto dos capitulos posteriores do livro.
    # Initial guess for induced velocities
    V_iA_old = 0.01 * V_A * np.ones(N_AZIMUTHAL_STATIONS)
    V_iT_old = np.zeros(N_AZIMUTHAL_STATIONS)
    
    max_iter = BEMT_MAX_ITER
    tolerance = BEMT_TOLERANCE
    a_relax = BEMT_RELAX

    for iter_count in range(max_iter):
        V_iA = V_iA_old.copy()
        V_iT = V_iT_old.copy()

        # [DOC-GUR-13] Cap. 4.1, Figs. 4.2–4.3, pp. 41–43.
        # Componentes locais da velocidade incluindo inducoes e rotacao.
        # V_t tambem varia com psi devido ao escoamento nao axial.
        # Axial velocity component
        V_a = V_A + V_iA

        # Tangential velocity component (uses signed Omega)
        V_t = r * Omega_signed + V_Z * np.sin(psi) - V_iT
        
        # [DOC-GUR-14] Cap. 4.1, Fig. 4.3, pp. 42–43.
        # Triangulo de velocidades: phi = atan2(V_a, V_t);
        # W = sqrt(V_a^2+V_t^2); alpha = beta_local - phi.
        # TCC 3.3: passo inicial para consultar CL e CD.
        # Inflow angle
        phi = np.arctan2(V_a, V_t)

        # Local relative wind speed
        W = np.sqrt(V_a**2 + V_t**2)
        
        # Angle of attack
        alpha_rad = twist_rad - phi
        alpha_deg = np.rad2deg(alpha_rad)

        # [DOC-GUR-15] Cap. 4.1, pp. 42–44; base 2D no cap. 4.2.
        # Re local = rho*W*c/mu; Mach local = W/a.
        # Parametros afetam CL e CD; conferir faixa suportada pelas polares.
        # Reynolds number
        Re = rho * W * c / mu

        # Get incompressible Cl and Cd
        Cl_i, Cd_i = get_Cl_Cd(Re, alpha_deg, r_over_R, INTERPOLATORS)

        # Calculate local Mach number
        M = W / a_sound

        # Apply lift compressibility correction
        # Apply lift compressibility correction
        # [DOC-EXTRA-11] Este bloco aplica correcao de Mach sobre CL/CD.
        # NAO e implementacao das tres correcoes de AOA da secao 4.3 do Gur.
        # A correção inclui M_dd e t/c, sweep; documentar citacoes proprias.
        if use_lift_corr:
            if lift_method == 'kaplan':
                Cl_c, M_dd, converged = kaplan_compressibility_correction(
                    Cl_i, M, t_over_c, sweep_deg,
                    consider_divergence=CONSIDER_DRAG_DIVERGENCE  # ADD THIS
                )
                if not converged and iter_count == 0:
                    print(f"      Warning: Kaplan correction did not converge at r/R = {r_over_R:.3f}")
            elif lift_method == 'karman-tsien':
                Cl_c = karman_tsien_correction(Cl_i, M)
                M_dd = calculate_drag_divergence_mach(Cl_c, t_over_c, sweep_deg)
            else:
                raise ValueError(f"Unknown lift correction method: {lift_method}")
        else:
            Cl_c = Cl_i
            M_dd = calculate_drag_divergence_mach(Cl_c, t_over_c, sweep_deg)

        # Apply wave drag correction
        if use_drag_corr:
            Cd_c = wave_drag_correction(Cd_i, M, M_dd, 
                                       consider_divergence=CONSIDER_DRAG_DIVERGENCE)  # ADD THIS
        else:
            Cd_c = Cd_i

        # [DOC-GUR-16] Cap. 4.4, pp. 56–57 — efeito de ponta (nao mesma equacao).
        # Aqui Prandtl multiplica CL, enquanto CD permanece sem esse fator.
        # Gur, Eq. (4.2), ilustra F_T sobre ambos os termos seccionais.
        # REVISAR a escolha fisica antes de comparar modelos/alterar formulas.
        # Apply Prandtl tip loss factor to lift coefficient
        if use_tip_loss:
            Pr_tip = prandtl_tip_loss_factor(r, Rtip, B, phi)
            Cl_tip = Cl_c * Pr_tip
            Cd_tip = Cd_c
        else:
            Cl_tip = Cl_c
            Cd_tip = Cd_c
        
        if use_hub_loss:
            Pr_hub = prandtl_hub_loss_factor(r, Rhub, B, phi)
            CL = Cl_tip * Pr_hub
            CD = Cd_tip
        else:
            CL = Cl_tip   
            CD = Cd_tip
        
        # [DOC-GUR-17] Cap. 4.1, Eqs. (4.1–4.3), pp. 42–43.
        # dT/dr = 0.5*rho*W^2*c*(CL*cos(phi)-CD*sin(phi)) por pa.
        # dQ/dr = 0.5*rho*W^2*c*r*(CL*sin(phi)+CD*cos(phi)) por pa.
        # A multiplicacao por B abaixo produz as contribuicoes de todas as pas.
        # TCC 3.3: equacoes centrais do calculo de empuxo e torque.
        # Sectional force/torque per unit radius, PER BLADE
        Delta_T_per_blade = 0.5 * rho * W**2 * c * (CL * np.cos(phi) - CD * np.sin(phi))
        Delta_Q_per_blade = 0.5 * rho * W**2 * c * (CL * np.sin(phi) + CD * np.cos(phi)) * r

        # Total sectional force/torque per unit radius for all blades
        Delta_T = Delta_T_per_blade * B
        Delta_Q = Delta_Q_per_blade * B

        # [DOC-GUR-18] Cap. 4.1, Fig. 4.4, p. 45: fechamento BEM + inducoes.
        # As formulas numericas de V_iA e V_iT vem da TEORIA DO MOMENTUM;
        # estudar Gur, caps. 5 e 6, antes de afirmar equivalencia exata.
        # REVISAR tambem as aproximacoes sob escoamento inclinado.
        # BEMT Momentum Balance to solve for induced velocities
        V_a_safe = np.where(np.abs(V_a) < 1e-6, 1e-6, V_a)
        
        V_iA_new = Delta_T / (4.0 * np.pi * r * rho * V_a_safe)
        V_iT_new = Delta_Q / (4.0 * np.pi * r**2 * rho * V_a_safe)

        # Relaxation
        if V_A != 0.0:
            a_new = V_iA_new / V_A
            a_old = V_iA / V_A
            a_new_relaxed = a_old * (1 - a_relax) + a_new * a_relax
            V_iA_new = a_new_relaxed * V_A
        
        # [DOC-GUR-19] Cap. 4.1, Fig. 4.4, p. 45 — criterio de parada.
        # Codigo usa relaxacao para a indução axial e compara incrementos
        # de velocidades induzidas; tolerancias sao parametros numericos locais.
        # Check for convergence
        if np.all(np.abs(V_iA_new - V_iA_old) < tolerance * max(abs(V_A), 0.1)) and \
           np.all(np.abs(V_iT_new - V_iT_old) < tolerance * max(abs(r * Omega_signed), 0.1)):
            if iter_count > 5:
                break

        # Update induced velocities
        V_iA_old = V_iA_new
        V_iT_old = V_iT_new
    
    if iter_count == max_iter - 1:
        print(f"  Warning: Max iterations reached at r/R = {r_over_R:.3f}")
        
    # Apply sorting to all output arrays to maintain consistency
    Delta_T = Delta_T[sort_idx]
    Delta_Q = Delta_Q[sort_idx]
    M = M[sort_idx]
    M_dd = M_dd[sort_idx]
    W = W[sort_idx]
    phi = phi[sort_idx]
    
    return Delta_T, Delta_Q, psi, M, M_dd, W, V_a, V_t, phi, alpha_rad, Re

#%%
# ========================================================================
# 7 - UNSTEADINESS CORRECTION
# ========================================================================
# [DOC-EXTRA-12] Correcao harmonica (Sears), alem do cap. 4 do Gur.
# TCC 3.3 / 3.5: FT das distribuicoes quase-estacionarias, correcao
# de harmonicos e reconstrucao no azimute para entradas aeroacusticas.
# REVISAR: comentario original fala em '2 harmonics', mas o codigo
# guarda N_HARMONICS_TO_KEEP=16 (k=0 a 15).
def apply_sears_correction(Delta_CT_disk_qs, Delta_CQ_disk_qs, psi, 
                           r_stations, c_stations, Omega, V_inf, alpha_P_rad, 
                           r_disk, M_disk, W_disk, V_a_disk, V_t_disk):
    """
    Apply Sears function correction to CT' and CQ' distributions.
    
    Only the first 2 harmonics (k=0 and k=1) are corrected and retained.
    
    Returns corrected distributions in the azimuthal domain.
    """
    
    N_HARMONICS_TO_KEEP = 16
    
    # Storage for corrected coefficients
    Delta_CT_disk_uns = np.zeros_like(Delta_CT_disk_qs)
    Delta_CQ_disk_uns = np.zeros_like(Delta_CQ_disk_qs)
    
    # Storage for azimuthal harmonics (only 2 harmonics: k=0 and k=1)
    Delta_CT_disk_uns_FFT = np.zeros((N_RADIAL_STATIONS, N_HARMONICS_TO_KEEP), dtype=complex)
    Delta_CQ_over_r_disk_uns_FFT = np.zeros((N_RADIAL_STATIONS, N_HARMONICS_TO_KEEP), dtype=complex)
    
    V_a_disk_FFT = np.zeros((N_RADIAL_STATIONS, N_HARMONICS_TO_KEEP), dtype=complex)
    V_t_disk_FFT = np.zeros((N_RADIAL_STATIONS, N_HARMONICS_TO_KEEP), dtype=complex)
    
    print("\n" + "="*60)
    print("Applying Compressible Sears Function Correction")
    print("="*60)
    
    for i in range(N_RADIAL_STATIONS):
        r = r_stations[i]
        c = c_stations[i]
        M_avg = np.mean(M_disk[i, :])
        W_avg = np.mean(W_disk[i, :])
        
        # Get sectional coefficients at this radius (azimuthal variation)
        Delta_CT_qs = Delta_CT_disk_qs[i, :]
        Delta_CQ_over_r_qs = Delta_CQ_disk_qs[i, :] / r
        V_a_qs = V_a_disk[i, :]
        V_t_qs = V_t_disk[i, :]
        
        # Prandtl-Glauert factor
        zeta = np.sqrt(1 - M_avg**2)
        
        # Compute FFT (quasi-steady harmonics)
        # Normalize by N_AZIMUTHAL_STATIONS for consistent amplitude scaling
        Delta_CT_qs_FFT = np.fft.rfft(Delta_CT_qs) / N_AZIMUTHAL_STATIONS
        Delta_CQ_over_r_qs_FFT = np.fft.rfft(Delta_CQ_over_r_qs) / N_AZIMUTHAL_STATIONS
        V_a_qs_FFT = np.fft.rfft(V_a_qs) / N_AZIMUTHAL_STATIONS
        V_t_qs_FFT = np.fft.rfft(V_t_qs) / N_AZIMUTHAL_STATIONS

        # Allocating arrays for only 2 harmonics
        Delta_CT_uns_FFT = np.zeros(N_HARMONICS_TO_KEEP, dtype=complex)
        Delta_CQ_over_r_uns_FFT = np.zeros(N_HARMONICS_TO_KEEP, dtype=complex)
        
        # Process only k=0 and k=1
        for k in range(N_HARMONICS_TO_KEEP):
            if k == 0:
                # Mean value - no correction needed
                Delta_CT_uns_FFT[k] = Delta_CT_qs_FFT[k]
                Delta_CQ_over_r_uns_FFT[k] = Delta_CQ_over_r_qs_FFT[k]
            else:
                # k=1: Compute reduced frequency: σ = kΩc/(2V)
                sigma_k = (k * Omega * c) / (2 * W_avg)
                
                # Get compressible Sears function
                S_k = compressible_sears_function(sigma_k, M_avg, zeta)
                
                # Apply transfer function correction
                Delta_CT_uns_FFT[k] = Delta_CT_qs_FFT[k] * S_k
                Delta_CQ_over_r_uns_FFT[k] = Delta_CQ_over_r_qs_FFT[k] * S_k
        
        # Store the 2 harmonics for output
        Delta_CT_disk_uns_FFT[i, :] = Delta_CT_uns_FFT
        Delta_CQ_over_r_disk_uns_FFT[i, :] = Delta_CQ_over_r_uns_FFT
        
        V_a_disk_FFT[i, :] = V_a_qs_FFT[:N_HARMONICS_TO_KEEP]
        V_t_disk_FFT[i, :] = V_t_qs_FFT[:N_HARMONICS_TO_KEEP]

        # Reconstruct time-domain signal from only 2 harmonics
        # Multiply by N_AZIMUTHAL_STATIONS to compensate for irfft normalization
        Delta_CT_disk_uns[i, :] = np.fft.irfft(Delta_CT_uns_FFT * N_AZIMUTHAL_STATIONS, n=N_AZIMUTHAL_STATIONS)
        Delta_CQ_disk_uns[i, :] = np.fft.irfft(Delta_CQ_over_r_uns_FFT * N_AZIMUTHAL_STATIONS, n=N_AZIMUTHAL_STATIONS) * r
        
        if (i + 1) % 10 == 0 or i == N_RADIAL_STATIONS - 1:
            print(f"  Corrected station {i+1}/{N_RADIAL_STATIONS}: r/R = {r_stations[i]/Rtip:.3f}")
    
    print("\nSears correction complete!")
    
    return Delta_CT_disk_uns, Delta_CQ_disk_uns, Delta_CT_disk_uns_FFT, Delta_CQ_over_r_disk_uns_FFT, V_a_disk_FFT, V_t_disk_FFT

#%%
# ========================================================================
# 8 - BEMT ROUTINE (QUASI-STEADY)
# ========================================================================

print("\nSolving BEMT...")

# === ARTIFICIAL AZIMUTHAL SHIFT CONFIGURATION ===
advancing_shift_rad = np.deg2rad(ADVANCING_SIDE_START_DEG)
d_psi_shift = 2 * np.pi / N_AZIMUTHAL_STATIONS
shift_indices = int(np.round(advancing_shift_rad / d_psi_shift))

# Storage for quasi-steady results
total_T_qs = 0.0
total_Q_qs = 0.0
total_Y_qs = 0.0
total_Z_qs = 0.0
total_n_qs = 0.0
total_m_qs = 0.0

# Allocate arrays for disk distributions of local quasi-steady quantities (radial × azimuth)
r_disk = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
psi_disk = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
phi_disk = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
alpha_disk = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
M_disk = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
M_dd_disk = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
W_disk = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
V_a_disk = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
V_t_disk = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
Re_disk = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))

Delta_T_disk_qs = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
Delta_Y_disk_qs = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
Delta_Z_disk_qs = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
Delta_Q_disk_qs = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
Delta_n_disk_qs = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
Delta_m_disk_qs = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))

for i, r in enumerate(r_stations):
    r_over_R = r / Rtip
    c = c_stations[i]
    twist_rad = twist_stations[i]
    
    # Get local t/c and sweep angle
    t_over_c_local = interp_t_over_c(r_over_R)
    sweep_local = interp_sweep(r_over_R)

    # Solve BEMT for this radial station
    Delta_T_qs, Delta_Q_qs, psi, M, M_dd, W, V_a, V_t, phi, alpha_rad, Re = bemt_solve(
        r, c, twist_rad, V_INF, Omega, alpha_P_rad, RHO, MU, B, r_over_R, 
        INTERPOLATORS, t_over_c_local, sweep_local, A_SOUND,
        USE_TIP_LOSS, USE_HUB_LOSS, USE_LIFT_CORRECTION, USE_DRAG_CORRECTION, 
        LIFT_CORRECTION_METHOD, rotation_sign
    )
    
    # ========================================================================
    # APPLY ARTIFICIAL AZIMUTHAL SHIFT
    # ========================================================================
    Delta_T_qs = np.roll(Delta_T_qs, shift_indices)
    Delta_Q_qs = np.roll(Delta_Q_qs, shift_indices)
    M = np.roll(M, shift_indices)
    M_dd = np.roll(M_dd, shift_indices)
    W = np.roll(W, shift_indices)
    V_a = np.roll(V_a, shift_indices)
    V_t = np.roll(V_t, shift_indices)
    phi = np.roll(phi, shift_indices)
    alpha_rad = np.roll(alpha_rad, shift_indices)
    Re = np.roll(Re, shift_indices)
    
    # Secondary forces and moments (quasi-steady) - computed AFTER shift
    Delta_Y_qs = - (Delta_Q_qs / r) * np.cos(psi)
    Delta_Z_qs = - (Delta_Q_qs / r) * np.sin(psi)
    Delta_n_qs = - Delta_T_qs * r * np.sin(psi)
    Delta_m_qs = - Delta_T_qs * r * np.cos(psi)
    
    # ========================================================================
    # STORE DATA (QUASI-STEADY)
    # ========================================================================
    r_disk[i, :]    = np.ones_like(psi) * r
    psi_disk[i, :]  = psi
    phi_disk[i, :]  = phi
    alpha_disk[i, :] = alpha_rad
    M_disk[i, :]    = M
    M_dd_disk[i, :] = M_dd
    W_disk[i, :]    = W
    V_a_disk[i, :]    = V_a
    V_t_disk[i, :]    = V_t
    Re_disk[i, :]    = Re

    Delta_T_disk_qs[i, :] = Delta_T_qs
    Delta_Y_disk_qs[i, :] = Delta_Y_qs
    Delta_Z_disk_qs[i, :] = Delta_Z_qs
    Delta_Q_disk_qs[i, :] = Delta_Q_qs
    Delta_n_disk_qs[i, :] = Delta_n_qs
    Delta_m_disk_qs[i, :] = Delta_m_qs
    
    # ========================================================================
    # [DOC-GUR-20] Cap. 4.1, Eq. (4.4), p. 43 — relacao conceitual.
    # Gur integra ao longo do raio para carregamentos axisimetricos.
    # Aqui calcula-se PRIMEIRO a media em psi pela extensao nao axial.
    # AZIMUTHAL INTEGRATION (QUASI-STEADY)
    # ========================================================================

    d_psi = psi[1] - psi[0] if len(psi) > 1 else 2 * np.pi / len(psi)

    # Integral Propeller Forces
    T_prime_avg_qs = np.sum((Delta_T_qs / (2.0 * np.pi)) * d_psi)
    Y_prime_avg_qs = np.sum((Delta_Y_qs / (2.0 * np.pi)) * d_psi)
    Z_prime_avg_qs = np.sum((Delta_Z_qs / (2.0 * np.pi)) * d_psi)
    Q_prime_avg_qs = np.sum((Delta_Q_qs / (2.0 * np.pi)) * d_psi)
    n_prime_avg_qs = np.sum((Delta_n_qs / (2.0 * np.pi)) * d_psi)
    m_prime_avg_qs = np.sum((Delta_m_qs / (2.0 * np.pi)) * d_psi)
    
    # ========================================================================
    # [DOC-GUR-21] Cap. 4.1, Eq. (4.4), p. 43 — integracao de dT/dr e dQ/dr.
    # TCC 3.3: soma com pesos dr_annulus aproxima a integral radial.
    # As contribuições seccionais ja contem multiplicacao pelo numero B.
    # RADIAL INTEGRATION (QUASI-STEADY)
    # ========================================================================
    dr = dr_annulus[i]

    total_T_qs += T_prime_avg_qs * dr
    total_Q_qs += Q_prime_avg_qs * dr
    total_Y_qs += Y_prime_avg_qs * dr
    total_Z_qs += Z_prime_avg_qs * dr
    total_n_qs += n_prime_avg_qs * dr
    total_m_qs += m_prime_avg_qs * dr

# ========================================================================
# [DOC-EXTRA-13] Gur, cap. 2, pp. 25–26; nao e a Eq. (4.4).
# TCC 2.2 / 3.8: CT=T/(rho*n^2*D^4), CQ=Q/(rho*n^2*D^5).
# Saidas úteis na comparacao numérico-experimental entre propellers.
# NON-DIMENSIONALIZATION (QUASI-STEADY)
# ========================================================================

Delta_CT_disk_qs = Delta_T_disk_qs / force_norm
Delta_CY_disk_qs = Delta_Y_disk_qs / force_norm
Delta_CZ_disk_qs = Delta_Z_disk_qs / force_norm
Delta_CQ_disk_qs = Delta_Q_disk_qs / moment_norm
Delta_Cn_disk_qs = Delta_n_disk_qs / moment_norm
Delta_Cm_disk_qs = Delta_m_disk_qs / moment_norm

total_CT_qs = total_T_qs / force_norm
total_CY_qs = total_Y_qs / force_norm
total_CZ_qs = total_Z_qs / force_norm
total_CQ_qs = total_Q_qs / moment_norm
total_Cn_qs = total_n_qs / moment_norm
total_Cm_qs = total_m_qs / moment_norm

#%%
# ========================================================================
# [DOC-EXTRA-14] Saida nao estacionaria apos tratamento Sears.
# Nao faz parte do modelo BEM axial elementar do cap. 4 de Gur.
# TCC 3.5: revisar como essas distribuicoes alimentam Hanson.
# 9 - BEMT UNSTEADY CORRECTION
# ========================================================================

Delta_CT_disk_uns, Delta_CQ_disk_uns, \
Delta_CT_uns_FFT, Delta_CQ_over_r_uns_FFT, V_a_disk_FFT, V_t_disk_FFT = apply_sears_correction(
    Delta_CT_disk_qs,
    Delta_CQ_disk_qs,
    psi,
    r_stations,
    c_stations,
    Omega,
    V_INF,
    alpha_P_rad,
    r_disk,
    M_disk,
    W_disk,
    V_a_disk,
    V_t_disk
)

total_T_uns = 0.0
total_Q_uns = 0.0
total_Y_uns = 0.0
total_Z_uns = 0.0
total_n_uns = 0.0
total_m_uns = 0.0

# Allocate arrays for disk distributions of local unsteady quantities (radial × azimuth)
Delta_T_disk_uns = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
Delta_Y_disk_uns = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
Delta_Z_disk_uns = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
Delta_Q_disk_uns = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
Delta_n_disk_uns = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))
Delta_m_disk_uns = np.zeros((N_RADIAL_STATIONS, N_AZIMUTHAL_STATIONS))

for i, r in enumerate(r_stations):

    # Get azimuthal variation of thrust and torque at current radial station
    Delta_CT_uns = Delta_CT_disk_uns[i, :]
    Delta_CQ_uns = Delta_CQ_disk_uns[i, :]
    
    # Convert to dimensional quantities
    Delta_T_uns = Delta_CT_uns * force_norm
    Delta_Q_uns = Delta_CQ_uns * moment_norm
    
    # Secondary forces and moments (unsteady)
    Delta_Y_uns = - (Delta_Q_uns / r) * np.cos(psi)
    Delta_Z_uns = - (Delta_Q_uns / r) * np.sin(psi)
    Delta_n_uns = - Delta_T_uns * r * np.sin(psi)
    Delta_m_uns = - Delta_T_uns * r * np.cos(psi)
    
    # ========================================================================
    # STORE DATA (UNSTEADY)
    # ========================================================================
    
    Delta_T_disk_uns[i, :] = Delta_T_uns
    Delta_Y_disk_uns[i, :] = Delta_Y_uns
    Delta_Z_disk_uns[i, :] = Delta_Z_uns
    Delta_Q_disk_uns[i, :] = Delta_Q_uns
    Delta_n_disk_uns[i, :] = Delta_n_uns
    Delta_m_disk_uns[i, :] = Delta_m_uns
    
    # ========================================================================
    # AZIMUTHAL INTEGRATION (UNSTEADY)
    # ========================================================================

    d_psi = psi[1] - psi[0] if len(psi) > 1 else 2 * np.pi / len(psi)

    # Integral Propeller Forces
    T_prime_avg_uns = np.sum((Delta_T_uns / (2.0 * np.pi)) * d_psi)
    Y_prime_avg_uns = np.sum((Delta_Y_uns / (2.0 * np.pi)) * d_psi)
    Z_prime_avg_uns = np.sum((Delta_Z_uns / (2.0 * np.pi)) * d_psi)
    Q_prime_avg_uns = np.sum((Delta_Q_uns / (2.0 * np.pi)) * d_psi)
    n_prime_avg_uns = np.sum((Delta_n_uns / (2.0 * np.pi)) * d_psi)
    m_prime_avg_uns = np.sum((Delta_m_uns / (2.0 * np.pi)) * d_psi)
    
    # ========================================================================
    # RADIAL INTEGRATION (UNSTEADY)
    # ========================================================================
    dr = dr_annulus[i]  # ← ADD THIS LINE - you need the differential element!

    total_T_uns += T_prime_avg_uns * dr
    total_Q_uns += Q_prime_avg_uns * dr
    total_Y_uns += Y_prime_avg_uns * dr
    total_Z_uns += Z_prime_avg_uns * dr
    total_n_uns += n_prime_avg_uns * dr
    total_m_uns += m_prime_avg_uns * dr

# ========================================================================
# NON-DIMENSIONALIZATION (UNSTEADY)
# ========================================================================

Delta_CY_disk_uns = Delta_Y_disk_uns / force_norm
Delta_CZ_disk_uns = Delta_Z_disk_uns / force_norm
Delta_Cn_disk_uns = Delta_n_disk_uns / moment_norm
Delta_Cm_disk_uns = Delta_m_disk_uns / moment_norm

total_CT_uns = total_T_uns / force_norm
total_CY_uns = total_Y_uns / force_norm
total_CZ_uns = total_Z_uns / force_norm
total_CQ_uns = total_Q_uns / moment_norm
total_Cn_uns = total_n_uns / moment_norm
total_Cm_uns = total_m_uns / moment_norm

#%%
# ========================================================================
# [DOC-GUR-22] Saidas globais derivadas da integracao radial (cap. 4.1).
# TCC 3.3 e 3.8: T, Q, forcas/momentos e coeficientes adimensionais;
# estes arquivos apoiam a validacao numérico-experimental.
# 10 - SAVING INTEGRAL RESULTS TO CSV
# ========================================================================

print("\n" + "="*60)
print("Saving Results to CSV Files")
print("="*60)

# Create dictionary for integral results
integral_results = {
    'Parameter': ['V_inf', 'OMEGA_RPM', 'alpha_P_deg', 'J', 'n', 'D', 'rho', 'a_sound',
                  'C_T', 'C_Y', 'C_Z', 'C_Q', 'C_n', 'C_m',
                  'total_T', 'total_Y', 'total_Z', 'total_Q', 'total_n', 'total_m'],
    'Value': [V_INF, OMEGA_RPM, ALPHA_P_DEG, J, n, D, RHO, A_SOUND,
              total_CT_uns, total_CY_uns, total_CZ_uns, total_CQ_uns, total_Cn_uns, total_Cm_uns,
              total_T_uns, total_Y_uns, total_Z_uns, total_Q_uns, total_n_uns, total_m_uns],
    'Unit': ['m/s', 'RPM', 'deg', '-', 'Hz', 'm', 'kg/m³', 'm/s',
             '-', '-', '-', '-', '-', '-',
             'N', 'N', 'N', 'Nm', 'Nm', 'Nm'],
    'Description': ['Freestream velocity', 'Rotational speed', 'Propeller incidence angle',
                   'Advance ratio', 'Rotation frequency', 'Propeller diameter',
                   'Air density', 'Speed of sound',
                   'Thrust coefficient', 'Side force coefficient', 'Normal force coefficient',
                   'Torque coefficient', 'Rolling moment coefficient', 'Pitching moment coefficient',
                   'Total thrust', 'Total side force', 'Total normal force',
                   'Total torque', 'Total rolling moment', 'Total pitching moment']
}

df_integral = pd.DataFrame(integral_results)
integral_csv_path = os.path.join(OUTPUT_FOLDER_NAME, 'integral_results.csv')
df_integral.to_csv(integral_csv_path, index=False)

#%%
# ========================================================================
# [DOC-GUR-23] Saidas locais relacionadas a formulacao da sec. 4.1.
# TCC 3.3 / 3.5: campos por raio e azimute, inclusive dT/dr e dQ/dr,
# que permitem observar distribuicoes e construir fontes para Hanson.
# 11 - SAVING DISK DISTRIBUTIONS TO CSV
# ========================================================================

print("\n" + "="*60)
print("Saving Disk Distributions to CSV Files")
print("="*60)

# Create a meshgrid for easier data organization
r_mesh, psi_mesh = np.meshgrid(r_stations, np.linspace(0, 2*np.pi, N_AZIMUTHAL_STATIONS, endpoint=False), indexing='ij')

# Flatten all arrays for CSV storage
n_points = N_RADIAL_STATIONS * N_AZIMUTHAL_STATIONS

# Create DataFrame with geometric and flow field data
df_disk_data = pd.DataFrame({
    'r': r_mesh.flatten(),
    'r/R': (r_mesh / Rtip).flatten(),
    'psi_rad': psi_mesh.flatten(),
    'psi_deg': np.rad2deg(psi_mesh.flatten()),
    'phi_rad': phi_disk.flatten(),
    'phi_deg': np.rad2deg(phi_disk.flatten()),
    'alpha_rad': alpha_disk.flatten(),
    'alpha_deg': np.rad2deg(alpha_disk.flatten()),
    'M': M_disk.flatten(),
    'M_dd': M_dd_disk.flatten(),
    'W': W_disk.flatten(),
    'V_a': V_a_disk.flatten(),
    'V_t': V_t_disk.flatten(),
    'Re': Re_disk.flatten()
})

# Save geometric and flow field data
disk_data_csv_path = os.path.join(OUTPUT_FOLDER_NAME, 'disk_flowfield_data.csv')
df_disk_data.to_csv(disk_data_csv_path, index=False)

# Create DataFrame for quasi-steady dimensional distributions
df_disk_qs_dimensional = pd.DataFrame({
    'r': r_mesh.flatten(),
    'r/R': (r_mesh / Rtip).flatten(),
    'psi_rad': psi_mesh.flatten(),
    'psi_deg': np.rad2deg(psi_mesh.flatten()),
    'Delta_T': Delta_T_disk_qs.flatten(),
    'Delta_Y': Delta_Y_disk_qs.flatten(),
    'Delta_Z': Delta_Z_disk_qs.flatten(),
    'Delta_Q': Delta_Q_disk_qs.flatten(),
    'Delta_n': Delta_n_disk_qs.flatten(),
    'Delta_m': Delta_m_disk_qs.flatten()
})

# Save quasi-steady dimensional distributions
qs_dim_csv_path = os.path.join(OUTPUT_FOLDER_NAME, 'disk_distributions_qs_dimensional.csv')
df_disk_qs_dimensional.to_csv(qs_dim_csv_path, index=False)

# Create DataFrame for quasi-steady non-dimensional distributions
df_disk_qs_nondim = pd.DataFrame({
    'r': r_mesh.flatten(),
    'r/R': (r_mesh / Rtip).flatten(),
    'psi_rad': psi_mesh.flatten(),
    'psi_deg': np.rad2deg(psi_mesh.flatten()),
    'Delta_CT': Delta_CT_disk_qs.flatten(),
    'Delta_CY': Delta_CY_disk_qs.flatten(),
    'Delta_CZ': Delta_CZ_disk_qs.flatten(),
    'Delta_CQ': Delta_CQ_disk_qs.flatten(),
    'Delta_Cn': Delta_Cn_disk_qs.flatten(),
    'Delta_Cm': Delta_Cm_disk_qs.flatten()
})

# Save quasi-steady non-dimensional distributions
qs_nondim_csv_path = os.path.join(OUTPUT_FOLDER_NAME, 'disk_distributions_qs_nondimensional.csv')
df_disk_qs_nondim.to_csv(qs_nondim_csv_path, index=False)

# Create DataFrame for unsteady dimensional distributions
df_disk_uns_dimensional = pd.DataFrame({
    'r': r_mesh.flatten(),
    'r/R': (r_mesh / Rtip).flatten(),
    'psi_rad': psi_mesh.flatten(),
    'psi_deg': np.rad2deg(psi_mesh.flatten()),
    'Delta_T': Delta_T_disk_uns.flatten(),
    'Delta_Y': Delta_Y_disk_uns.flatten(),
    'Delta_Z': Delta_Z_disk_uns.flatten(),
    'Delta_Q': Delta_Q_disk_uns.flatten(),
    'Delta_n': Delta_n_disk_uns.flatten(),
    'Delta_m': Delta_m_disk_uns.flatten()
})

# Save unsteady dimensional distributions
uns_dim_csv_path = os.path.join(OUTPUT_FOLDER_NAME, 'disk_distributions_uns_dimensional.csv')
df_disk_uns_dimensional.to_csv(uns_dim_csv_path, index=False)

# Create DataFrame for unsteady non-dimensional distributions
df_disk_uns_nondim = pd.DataFrame({
    'r': r_mesh.flatten(),
    'r/R': (r_mesh / Rtip).flatten(),
    'psi_rad': psi_mesh.flatten(),
    'psi_deg': np.rad2deg(psi_mesh.flatten()),
    'Delta_CT': Delta_CT_disk_uns.flatten(),
    'Delta_CY': Delta_CY_disk_uns.flatten(),
    'Delta_CZ': Delta_CZ_disk_uns.flatten(),
    'Delta_CQ': Delta_CQ_disk_uns.flatten(),
    'Delta_Cn': Delta_Cn_disk_uns.flatten(),
    'Delta_Cm': Delta_Cm_disk_uns.flatten()
})

# Save unsteady non-dimensional distributions
uns_nondim_csv_path = os.path.join(OUTPUT_FOLDER_NAME, 'disk_distributions_uns_nondimensional.csv')
df_disk_uns_nondim.to_csv(uns_nondim_csv_path, index=False)

# Create a summary DataFrame comparing quasi-steady vs unsteady at each point
df_comparison = pd.DataFrame({
    'r': r_mesh.flatten(),
    'r/R': (r_mesh / Rtip).flatten(),
    'psi_rad': psi_mesh.flatten(),
    'psi_deg': np.rad2deg(psi_mesh.flatten()),
    'Delta_CT_qs': Delta_CT_disk_qs.flatten(),
    'Delta_CT_uns': Delta_CT_disk_uns.flatten(),
    'Delta_CT_diff_pct': 100 * (Delta_CT_disk_uns.flatten() - Delta_CT_disk_qs.flatten()) / (Delta_CT_disk_qs.flatten() + 1e-10),
    'Delta_CQ_qs': Delta_CQ_disk_qs.flatten(),
    'Delta_CQ_uns': Delta_CQ_disk_uns.flatten(),
    'Delta_CQ_diff_pct': 100 * (Delta_CQ_disk_uns.flatten() - Delta_CQ_disk_qs.flatten()) / (Delta_CQ_disk_qs.flatten() + 1e-10)
})

# Save comparison data
comparison_csv_path = os.path.join(OUTPUT_FOLDER_NAME, 'disk_distributions_comparison.csv')
df_comparison.to_csv(comparison_csv_path, index=False)

# ========================================================================
# [DOC-EXTRA-15] Componentes harmonicas de carregamentos e velocidades.
# TCC 3.5 / 3.6: possivel interface de dados com o modulo Hanson.
# Verificar convencoes, unidades e indices usados no consumidor.
# SAVE AZIMUTHAL HARMONICS DATA
# ========================================================================

# Create harmonic index array
N_AZIMUTHAL_HARMONICS = Delta_CT_uns_FFT.shape[1]
harmonic_indices = np.arange(N_AZIMUTHAL_HARMONICS)

# Create DataFrame for Delta_CT FFT data
fft_ct_data = {
    'harmonic_k': harmonic_indices
}

for i, r_val in enumerate(r_stations):
    r_over_R = r_val / Rtip
    fft_ct_data[f'r/R_{r_over_R:.3f}_real'] = Delta_CT_uns_FFT[i, :].real
    fft_ct_data[f'r/R_{r_over_R:.3f}_imag'] = Delta_CT_uns_FFT[i, :].imag

df_fft_ct = pd.DataFrame(fft_ct_data)
fft_ct_csv_path = os.path.join(OUTPUT_FOLDER_NAME, 'fft_Delta_CT_uns.csv')
df_fft_ct.to_csv(fft_ct_csv_path, index=False)

# Create DataFrame for Delta_CQ_over_r FFT data
fft_cq_data = {
    'harmonic_k': harmonic_indices
}

for i, r_val in enumerate(r_stations):
    r_over_R = r_val / Rtip
    fft_cq_data[f'r/R_{r_over_R:.3f}_real'] = Delta_CQ_over_r_uns_FFT[i, :].real
    fft_cq_data[f'r/R_{r_over_R:.3f}_imag'] = Delta_CQ_over_r_uns_FFT[i, :].imag

df_fft_cq = pd.DataFrame(fft_cq_data)
fft_cq_csv_path = os.path.join(OUTPUT_FOLDER_NAME, 'fft_Delta_CQ_over_r_uns.csv')
df_fft_cq.to_csv(fft_cq_csv_path, index=False)

# Create DataFrame for V_a FFT data
fft_V_a_data = {
    'harmonic_k': harmonic_indices
}

for i, r_val in enumerate(r_stations):
    r_over_R = r_val / Rtip
    fft_V_a_data[f'r/R_{r_over_R:.3f}_real'] = V_a_disk_FFT[i, :].real
    fft_V_a_data[f'r/R_{r_over_R:.3f}_imag'] = V_a_disk_FFT[i, :].imag

df_fft_V_a = pd.DataFrame(fft_V_a_data)
fft_V_a_csv_path = os.path.join(OUTPUT_FOLDER_NAME, 'fft_V_a.csv')
df_fft_V_a.to_csv(fft_V_a_csv_path, index=False)

# Create DataFrame for V_t FFT data
fft_V_t_data = {
    'harmonic_k': harmonic_indices
}

for i, r_val in enumerate(r_stations):
    r_over_R = r_val / Rtip
    fft_V_t_data[f'r/R_{r_over_R:.3f}_real'] = V_t_disk_FFT[i, :].real
    fft_V_t_data[f'r/R_{r_over_R:.3f}_imag'] = V_t_disk_FFT[i, :].imag

df_fft_V_t = pd.DataFrame(fft_V_t_data)
fft_V_t_csv_path = os.path.join(OUTPUT_FOLDER_NAME, 'fft_V_t.csv')
df_fft_V_t.to_csv(fft_V_t_csv_path, index=False)

print("\n" + "="*60)
print("All disk distributions saved successfully!")
print("="*60)

#%%
# --- 13. Print Results ---

print("\n" + "="*60)
print("BEMT Non-Axial Inflow Analysis Results")
corrections_active = []
if USE_TIP_LOSS:
    corrections_active.append("Prandtl Tip Loss")
if USE_HUB_LOSS:
    corrections_active.append("Prandtl Hub Loss")
if USE_LIFT_CORRECTION:
    corrections_active.append(f"{LIFT_CORRECTION_METHOD.capitalize()} Lift Correction")
if USE_DRAG_CORRECTION:
    corrections_active.append("Wave Drag")
if corrections_active:
    print(f"(Active corrections: {', '.join(corrections_active)})")
else:
    print("(No corrections applied)")
    
print("="*60)
print("\nOperating Conditions:")
print(f"  Freestream velocity: {V_INF} m/s")
print(f"  Propeller incidence: {ALPHA_P_DEG}°")
print(f"  Rotational speed: {OMEGA_RPM} RPM")
print(f"  Advance ratio J: {J:.4f}")
print("\nIntegral Forces and Moments (Quasi-Steady):")
print("-"*60)
print(f"  Thrust:           {total_T_qs:.6f} N")
print(f"  Side force:       {total_Y_qs:.6f} N")
print(f"  Normal force:     {total_Z_qs:.6f} N")
print(f"  Torque:           {total_Q_qs:.6f} Nm")
print(f"  Rolling moment:   {total_n_qs:.6f} Nm")
print(f"  Pitching moment:  {total_m_qs:.6f} Nm")
print("="*60)
print("\nIntegral Forces and Moments (Unsteady):")
print("-"*60)
print(f"  Thrust:           {total_T_uns:.6f} N")
print(f"  Side force:       {total_Y_uns:.6f} N")
print(f"  Normal force:     {total_Z_uns:.6f} N")
print(f"  Torque:           {total_Q_uns:.6f} Nm")
print(f"  Rolling moment:   {total_n_uns:.6f} Nm")
print(f"  Pitching moment:  {total_m_uns:.6f} Nm")
print("="*60)

# Print Mach number statistics
print("\nMach Number Statistics:")
print("-"*60)
print(f"  Local Mach number range: {M_disk.min():.4f} to {M_disk.max():.4f}")
print(f"  M_dd range: {M_dd_disk.min():.4f} to {M_dd_disk.max():.4f}")
print("="*60)