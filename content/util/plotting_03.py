import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import math
import numpy as np

def car_transform(position, x, h, reverse=False):
    return (transforms.Affine2D()
        .scale(-1 if reverse else 1, 1)
        .rotate(-1*math.atan2(x[position-2]-x[position+2], 4*h))
        .scale(0.1)
        .translate(h*position, x[position] + 0.03)
    )

def init_animation(x0, rand, N, h):
    ax = plt.gca()
    # Hänge
    left_x = np.array([-0.1, 0.0, 0.5, -0.1])
    left_y = np.array([rand[0], rand[0], -1.0, -1.0])
    right_x = np.array([2.1, 2.0, 1.5, 2.1])
    right_y = np.array([rand[1], rand[1], -1.0, -1.0])
    ax.fill(left_x,  left_y,  "green", linewidth=2, edgecolor="green")
    ax.fill(right_x, right_y, "green", linewidth=2, edgecolor="green")
    # unbelastete Brücke
    ax.plot(np.linspace(0.0, 2.0, N+1), x0, "r:")
    # belastete Brücke
    ax.plot(np.linspace(0.0, 2.0, N+1), x0, "b-")
    # Räder und Auto
    ax.add_patch(
        plt.Circle(
            (h, x0[0] + 0.015),
            0.015,
            color='black',
        )
    )
    ax.add_patch(
        plt.Circle(
            (4*h, x0[4] + 0.016),
            0.015,
            color='black',
        )
    )
    car = patches.Polygon(
        [
            (-1, 0.0), (1, 0.0), (1, 0.15), 
            (0.5, 0.25), (0, 0.5), (-0.5, 0.5), 
            (-1, 0.25)
        ],
        color='black',
    )
    car.set_transform(car_transform(2, x0, h) + ax.transData)
    ax.add_patch(car)

def draw_frame(
    frame,
    rand,
    N,
    h,
    gewicht_b,
    gewicht_a,
    alpha,
    beta,
    gamma,
    vorwaerts_rueckwaerts_substitution,
):
    if frame % 10 == 0:
        print(f"Animation  wird generiert ({frame}/{2*N - 8})")
    if frame < N-4:
        reverse = False
        position = frame + 2
    else:
        reverse = True
        position = N + N - 7 - frame
    # Berechnung
    z = np.full(N-1, -gewicht_b * h**2)
    z[0] += rand[0]
    z[position] -= gewicht_a
    z[-1] += rand[1]
    x = np.hstack(
        [rand[0], vorwaerts_rueckwaerts_substitution(alpha, beta, gamma, z), rand[1]],
    )
    # Plotting
    ax = plt.gca()
    ax.lines[1].set_data(np.linspace(0.0, 2.0, N+1), x)
    ax.patches[2].set_center((h * (position - 2), x[position-2] + 0.015))
    ax.patches[3].set_center((h * (position + 2), x[position+2] + 0.016))
    ax.patches[4].set_transform(car_transform(position, x, h, reverse) + ax.transData)
