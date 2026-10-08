import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 

l1, l2 = 1.0, 1.0
theta1, theta2 = 90 * (np.pi/180), 90 * (np.pi/180)
m1, m2 = 1.0, 1.0
g = 9.81
fps = 60
stepsperframe = 10
totaltime = 5
dt = 1 / fps

angles = np.empty((totaltime*fps + 1 , 4))

#using the functions defined in the writeup
def derivs(t1, t2, omega1, omega2):
    a1 = (l2 * m2)/(l1 * (m1 + m2)) * np.cos(t1 - t2)
    a2 = l1 / l2 * np.cos(t1 - t2)
    f1 = (-l2 * m2 * omega2**2) / (l1 * (m1 + m2)) * np.sin(t1 - t2) - g / l1 * np.sin(t1)
    f2 = (l1 * omega1**2) / l2 * np.sin(t1 - t2) - g / l2 * np.sin(t2)
    domega1 = (f1 - a1 * f2) / (1 - a1 * a2)
    domega2 = (f2 - a2 * f1) / (1 - a1 * a2)
    return np.array([omega1, omega2, domega1, domega2])

#RK4 -> note that the actual formula is given in terms of f(t, x) but this derivative does not vary with t in this case so the term was dropped.
def nextstep(row, step):
    a = derivs(*row)
    b = derivs(*(row + (step * a) / 2))
    c = derivs(*(row + (step * b) / 2))
    d = derivs(*(row + step * c))
    return (row + (step / 6) * (a + 2*b + 2*c + d))

def nextframe(row):
    stepsize = 1/(fps * stepsperframe)
    for _ in range(stepsperframe): #loops stepsperframe times
        row = nextstep(row, stepsize)
    return row

#Array of angles over time
angles[0, :] = [theta1, theta2, 0, 0]
for framenum in range(fps * totaltime):
    angles[framenum + 1] = nextframe(angles[framenum])


def energy(t1, t2, omega1, omega2):
    return ( (0.5 * m1 * l1**2 * omega1**2) + (0.5 * m2) * (l1**2 * omega1**2 + l2**2 * omega2**2 + 2 * l1 * omega1 * l2 * omega2 * np.cos(t1 - t2)) - (m1 + m2) * g * l1 * np.cos(t1) - m2 * g * l2 * np.cos(t2) )

#Plotting the figure
x1 = l1 * np.sin(angles[:, 0] + np.pi)
x2 = x1 + l2 * np.sin(angles[:, 1] + np.pi)
y1 = l1 * np.cos(angles[:, 0] + np.pi)
y2 = y1 + l2 * np.cos(angles[:, 1] + np.pi)
E = [energy(*row) for row in angles]

fig , ax = plt.subplots()
ax.set_aspect("equal")
lim = 1.05 * (l1 + l2)
ax.set_ylim(-lim, lim)
ax.set_xlim(-lim, lim)

line, = ax.plot([],[], "o-", lw=2) 
trail, = ax.plot([], [], '-', lw=1, alpha=0.4)
timelabel = ax.text(0.05, 0.95, '', transform=ax.transAxes)
energylabel = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha="right")

trailframes = 120
def update(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]]) #(0,0) is the pivot, (x1,y1) is bob 1 and (x2,y2) is bob 2
    pastline = max(0, i - trailframes)
    trail.set_data(x2[pastline : i], y2[pastline : i])
    timelabel.set_text(f't = {i*dt:.2f}s')
    energylabel.set_text(f"{E[i]:.3e}")
    
    return line, trail, timelabel, energylabel           # blit needs the changed artists returned

ani = FuncAnimation(
    fig, update,
    frames=np.shape(angles)[0],
    interval=dt * 1000,   # interval is automatically processed in ms, *1000 is the conversion
    blit=True             # redraw only the artists above, not the whole axes
)

plt.show()
