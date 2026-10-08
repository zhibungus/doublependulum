import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
from matplotlib.colors import hsv_to_rgb

#parameters
l1, l2 = 1.0, 1.0
m1, m2 = 1.0, 1.0
g = 9.81

N = 300 #Number of pendulums in each axis
totaltime = 5
fps = 30
stepsperframe = 20

#all regular arithmetic and functions are elementwise unless specified (eg. np.cross)
def derivs(row):
    t1, t2, omega1, omega2 = row
    a1 = (l2 * m2)/(l1 * (m1 + m2)) * np.cos(t1 - t2)
    a2 = l1 / l2 * np.cos(t1 - t2)
    f1 = (-l2 * m2 * omega2**2) / (l1 * (m1 + m2)) * np.sin(t1 - t2) - g / l1 * np.sin(t1)
    f2 = (l1 * omega1**2) / l2 * np.sin(t1 - t2) - g / l2 * np.sin(t2)
    det = (1 - a1 * a2)
    domega1 = (f1 - a1 * f2) / det
    domega2 = (f2 - a2 * f1) / det
    return np.array([omega1, omega2, domega1, domega2]) #returns a (4, N, N array)

#RK4 
def rk4(state, step):
    a = derivs(state)
    b = derivs((state + (step * a) / 2))
    c = derivs((state + (step * b) / 2))
    d = derivs((state + step * c))
    return (state + (step / 6) * (a + 2*b + 2*c + d))

#colour map
def colour(t1, t2):
    hue = (t1 % (2*np.pi)) / (2*np.pi) #varies from 0 to 1
    val = 0.3 + 0.7 * (0.5 + 0.5 * np.cos(t2)) #varies from .3 to 1.0
    sat = np.full_like(hue, 0.9)
    return hsv_to_rgb(np.stack([hue, sat, val], axis=-1)) #channel number is last

step = 1 / (fps * stepsperframe)
grid = np.linspace(-np.pi , np.pi, N, endpoint=False)
th1, th2 = np.meshgrid(grid, grid)
state = np.array([th1, th2, np.zeros_like(th1), np.zeros_like(th1)])

def run(state):
    frames = np.empty((fps*totaltime + 1, N, N, 3), dtype=np.uint8) #smaller data type
    frames[0] = colour(state[0], state[1]) * 255
    for i in range(fps * totaltime):
        for _ in range(stepsperframe): 
            state = rk4(state, step)
        frames[i+1] = colour(state[0], state[1]) * 255
    return frames
frames = run(state)

#Plotting the figure
fig , ax = plt.subplots(figsize = (6,6)) #figsize is in inches. pixels = figsize * dpi
im = ax.imshow(
    frames[0],
    origin="lower",
    extent=[-180, 180, -180, 180],
    interpolation="nearest"
)

ax.set_xlabel("initial θ₁ (deg)")
ax.set_ylabel("initial θ₂ (deg)")
label = ax.text(0.05, 0.95, "", transform=ax.transAxes, va="top", color="w", family="monospace", bbox={'fc': 'k', 'alpha': 0.5, 'ec': 'none'}) #i could've used dict() instead

def update(i):
    label.set_text(f"t = {i/fps:.2f}s")
    im.set_data(frames[i])
    return im, label    # blit needs the changed artists returned

ani = FuncAnimation(
    fig, update,
    frames= totaltime * fps + 1,
    interval=1/fps * 1000,   # interval is automatically processed in ms, *1000 is the conversion
    blit=True             # redraw only the artists above, not the whole axes
)

plt.show()
