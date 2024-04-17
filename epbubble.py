import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Constants
M = 2.01568e-3  # Hydrogen molar mass (kg/mol)
sig = 0.024  # Surface tension (N/m)
B = 8.314  # Gas constant (J/(mol*K))
T = 293.15  # Temperature (Kelvin)
k = 1.291e-7  # Diffusion coefficient of hydrogen in water (m^2/s)
d = 0.0692  # Henry's law constant for solubility
dt = 1e-12  # Time step (s)
initr = 0.92e-8  # Initial radius of the bubble (m)
Nnh3 = 387439  # Initial number of NH3 molecules
mih = 2 * 1.67356e-27  # Mass of H2 molecule (kg)

# Function to calculate the change in radius over time
def f3(x, tau, f, R0, y):
    vdel = tau / (R0 * 0.28077)
    term1 = 1 - (x / R0) ** 2 - 2 * vdel * (1 / (1 - f) - 2 / 3) * (1 - x / R0)
    term2 = 2 * vdel ** 2 / (1 - f) * (1 / (1 - f) - 2 / 3)
    term3 = vdel + 1 - f
    term5d = vdel + (1 - f) * x / R0
    term5e = term3 / term5d
    term5 = np.log(term5e) if term5e > 0 else -100  # Handle potential log domain error
    omg = 2 * k * d / R0 ** 2
    term4 = omg * y * (1 - f)
    return term1 + term2 * term5 - term4

# Initialize arrays and variables
R = [initr]
t = [0]
fr = []
tau = 2 * M * sig / (B * T)  # Calculate tau
dm = abs(4 * np.pi * initr ** 2 * 10 * (0.28077 + 2 * tau / (3 * initr)) * dt)
Nh2 = dm / mih
molfrac = Nh2 / (Nh2 + Nnh3)
f = molfrac * 12489.70 / 3.3555
fr.append(f)

# Simulation loop
for i in range(14200):
    ti = (i + 1) * dt
    res = fsolve(f3, initr, args=(tau, f, initr, dt))
    R.append(res[0])
    dRdt = (res[0] - initr) / dt
    dm -= 4 * np.pi * initr ** 2 * dRdt * (0.28077 + 2 * tau / (3 * initr)) * dt
    Nh2 = dm / mih
    molfrac = Nh2 / (Nh2 + Nnh3)
    f = molfrac * 12489.70 / 3.3555 # molfrac * henry constant / partial press.
    fr.append(f)
    initr = res[0]
    t.append(ti)

# Plotting
fig = plt.figure(figsize=(10, 5))
ax = fig.add_axes([0.15, 0.15, 0.85, 0.85])
timea = np.array(t) * 1e9  # Convert time to ns
pressa = np.array(R) * 1e9 / (initr * 1e9)  # Convert radius to nm
ax.plot(timea, pressa)
plt.xlabel('Time (ns)')
plt.ylabel('Radius (nm)')
plt.title('Bubble Radius vs Time')
plt.show()
