import numpy as np
import matplotlib.pyplot as plt
from galpy.potential.McMillan17 import McMillan17
from galpy.potential import evaluateRforces, evaluatezforces
from galpy.potential import evaluatePotentials
from numba import njit, objmode
from numba import prange
from scipy.interpolate import CubicSpline
import pandas as pd
from datetime import datetime

# =======================================================
# CALCULATING GALACTIC ACCELERATION WITH GALPY
# =======================================================
def calculate_galactic_acc(x, y, z):
    R = np.sqrt(x ** 2 + y ** 2)

    # For dividing zero error
    R = np.where(R == 0, 1e-10, R)

    # McMillan unit translation
    ro = 8.21
    vo = 233.1
    R_int = R / ro
    z_int = z / ro


    F_R_galaxy_int = evaluateRforces(McMillan17, R_int, z_int, use_physical=False)
    F_z_galaxy_int = evaluatezforces(McMillan17, R_int, z_int, use_physical=False)

    conversion_factor = (vo ** 2) / ro
    F_R_galaxy = F_R_galaxy_int * conversion_factor
    F_z_galaxy = F_z_galaxy_int * conversion_factor

    ax_galaxy = F_R_galaxy * (x / R)
    ay_galaxy = F_R_galaxy * (y / R)
    az_galaxy = F_z_galaxy

    return ax_galaxy, ay_galaxy, az_galaxy


# =======================================================
# CALCULATING N-BODY ACCELERATION
# =======================================================
@njit(parallel=True)
def calculate_nbody_acc(x, y, z, G, masses, eps):
    N = len(x)
    ax_nbody = np.zeros(N)
    ay_nbody = np.zeros(N)
    az_nbody = np.zeros(N)


    for i in prange(N):
        dx = x - x[i]
        dy = y - y[i]
        dz = z - z[i]
        r_sqr = dx ** 2 + dy ** 2 + dz ** 2 + eps ** 2
        ivme = (G * masses) / (r_sqr ** 1.5)

        ax_nbody[i] = np.sum(ivme * dx)
        ay_nbody[i] = np.sum(ivme * dy)
        az_nbody[i] = np.sum(ivme * dz)



    return ax_nbody, ay_nbody, az_nbody


# =======================================================
# LEAPFROG ENGINE
# =======================================================
@njit(parallel=False)
def leapfrog_engine(step_num, dt, G, masses, eps, x, y, z, vx, vy, vz):
    N_star = len(x)
    x_past = np.zeros((step_num, N_star))
    y_past = np.zeros((step_num, N_star))
    z_past = np.zeros((step_num, N_star))

    vx_past = np.zeros((step_num, N_star))
    vy_past = np.zeros((step_num, N_star))
    vz_past = np.zeros((step_num, N_star))

    ax_n, ay_n, az_n = calculate_nbody_acc(x, y, z, G, masses, eps)

    # Safely calling galpy function with objmode from Numba
    with objmode(ax_g='float64[:]', ay_g='float64[:]', az_g='float64[:]'):
        ax_g, ay_g, az_g = calculate_galactic_acc(x, y, z)

    ax = ax_n + ax_g
    ay = ay_n + ay_g
    az = az_n + az_g

    print_interval = step_num // 10

    for step in prange(step_num):

        # KICK 1
        vx += ax * (dt / 2.0)
        vy += ay * (dt / 2.0)
        vz += az * (dt / 2.0)

        # DRIFT
        x += vx * dt
        y += vy * dt
        z += vz * dt

        # New Acceleration: N-Body
        ax_n, ay_n, az_n = calculate_nbody_acc(x, y, z, G, masses, eps)

        # New Acceleration: Galactic
        with objmode(ax_g='float64[:]', ay_g='float64[:]', az_g='float64[:]'):
            ax_g, ay_g, az_g = calculate_galactic_acc(x, y, z)

        ax = ax_n + ax_g
        ay = ay_n + ay_g
        az = az_n + az_g

        # KICK 2
        vx += ax * (dt / 2.0)
        vy += ay * (dt / 2.0)
        vz += az * (dt / 2.0)


        x_past[step] = x
        y_past[step] = y
        z_past[step] = z


        vx_past[step] = vx
        vy_past[step] = vy
        vz_past[step] = vz

        # Shows the progress
        if step % print_interval == 0:
            progress = int((step / step_num) * 100)
            print("Progress: %", progress)

    print("Simulation Completed")
    return x_past, y_past, z_past, vx_past, vy_past, vz_past




# ==========================================
# INITIAL CONDITIONS AND RUNNING
# ==========================================
G = 4.3009e-6
eps = 0.2

masses = np.array([1.0, 5.0, 3.0, 1.5], dtype=np.float64)
x = np.array([8.0, 4.0, 12.0, 6.1], dtype=np.float64)
y = np.array([0.0, 0.0, 0.0, 0.8], dtype=np.float64)
z = np.array([0.1, 0.5, 0.05, 1.2], dtype=np.float64)
vx = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
vy = np.array([220.0, 240.0, 190.0, 231.3], dtype=np.float64)
vz = np.array([10.0, 20.0, 5.0, 8.0], dtype=np.float64)

N_star = len(x)

t_target = 5.0
dt = 0.005
step_num = int(t_target / dt)

print("Leapfrog Simulation Starts... Total step:", step_num)


x_past, y_past, z_past, vx_past, vy_past, vz_past = leapfrog_engine(
    step_num, dt, G, masses, eps, x, y, z, vx, vy, vz
)

print("Calculating energy...")

# =======================================================
# CALCULATING ENERGY
# =======================================================
# 1. Kinetic Energy
v_sqr = vx_past**2 + vy_past**2 + vz_past**2
K = np.sum(0.5 * masses * v_sqr, axis=1)

# 2. Galactic Potential Energy
ro = 8.21
vo = 233.1

R_past = np.sqrt(x_past**2 + y_past**2)
R_past = np.where(R_past == 0, 1e-10, R_past)

R_int = R_past / ro
z_int = z_past / ro

Phi_int = evaluatePotentials(McMillan17, R_int.flatten(), z_int.flatten(), use_physical=False)
Phi_physical = Phi_int.reshape(R_int.shape) * (vo**2)
U_gal = np.sum(masses * Phi_physical, axis=1)

# 3. N-Body Potential Energy
U_nbody = np.zeros(step_num)
for i in range(N_star):
    for j in range(i + 1, N_star):
        dx = x_past[:, i] - x_past[:, j]
        dy = y_past[:, i] - y_past[:, j]
        dz = z_past[:, i] - z_past[:, j]
        r = np.sqrt(dx**2 + dy**2 + dz**2 + eps**2)
        U_nbody -= G * masses[i] * masses[j] / r

# 4. Total Energy and Relative Error (Delta E / E_0)
E_total = K + U_gal + U_nbody
E_0 = E_total[0] # initial energy
energy_error = (E_total - E_0) / np.abs(E_0)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# ==========================================
# SAVING
# ==========================================
# Time series (Gyr)
time_unit = 0.97779222
time_gyr = np.linspace(0, t_target, step_num) * time_unit

data_dic = {"Time_Gyr": time_gyr}

for i in range(N_star):
    data_dic[f"E_sum"] = E_total
    data_dic[f"R_{i + 1}"] = R_past[:, i]
    data_dic[f"x_{i+1}"] = x_past[:, i]
    data_dic[f"y_{i+1}"] = y_past[:, i]
    data_dic[f"z_{i+1}"] = z_past[:, i]
    data_dic[f"vx_{i + 1}"] = vx_past[:, i]
    data_dic[f"vy_{i + 1}"] = vy_past[:, i]
    data_dic[f"vz_{i + 1}"] = vz_past[:, i]

df = pd.DataFrame(data_dic)
df.to_csv(f"nbody_results_{current_time}.csv", index=False)


# ==========================================
# PLOTTING
# ==========================================
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))

t_raw = np.linspace(0, t_target, step_num)
t_soft = np.linspace(0, t_target, step_num * 5)

for i in range(N_star):
    cs_x = CubicSpline(t_raw, x_past[:, i])
    cs_y = CubicSpline(t_raw, y_past[:, i])
    ax1.plot(cs_x(t_soft), cs_y(t_soft), label=f'star {i + 1}')

ax1.plot(0, 0, marker='*', color='black', markersize=15)
ax1.set_title('X - Y Plane', fontsize=14)
ax1.set_xlabel('X (kpc)')
ax1.set_ylabel('Y (kpc)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)


for i in range(N_star):
    cs_R = CubicSpline(t_raw, R_past[:, i])
    cs_z = CubicSpline(t_raw, z_past[:, i])
    ax2.plot(cs_R(t_soft), cs_z(t_soft))

ax2.set_title('R - Z Plane', fontsize=14)
ax2.set_xlabel('R (kpc)')
ax2.set_ylabel('Z (kpc)')
ax2.grid(True, linestyle='--', alpha=0.6)

time_axis = t_raw * 0.97779222

ax3.plot(time_axis, E_total, color='red', linewidth=1.5)
ax3.set_title('Total Energy', fontsize=14)
ax3.set_xlabel(r'$\tau\ (Gyr)$')
ax3.set_ylabel(r'$E\ (km/s)^2$')
ax3.grid(True, linestyle='--', alpha=0.6)

ax4.plot(time_axis, energy_error, color='purple', linewidth=1.5)
ax4.axhline(0, color='black', linestyle='--', linewidth=1)
ax4.set_title('Conservation of Energy', fontsize=14)
ax4.set_xlabel(r'$\tau\ (Gyr)$')
ax4.set_ylabel(r'$\Delta E / |E_0|$')
ax4.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(f"nbody_plot_{current_time}.png", dpi=300)
plt.show()