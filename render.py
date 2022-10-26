import math
import re

import pygame as pg
import numpy as np


def get_proj_mat(fov, ar):
    mat = np.array([
        [-1 / math.tan(fov / 2), 0, 0],
        [0, ar / math.tan(fov / 2), 0]
    ])
    return mat


def get_rot_mat(azi, alt):
    azi_rot = np.array([
        [math.cos(azi), 0, math.sin(azi)],
        [0, 1, 0],
        [-math.sin(azi), 0, math.cos(azi)]
    ])
    alt_rot = np.array([[1, 0, 0],
        [0, math.cos(-alt), -math.sin(-alt)],
        [0, math.sin(-alt), math.cos(-alt)]
    ])
    mat = np.dot(alt_rot, azi_rot)
    return mat


# Reading file
with open("models/sphere.obj", "r") as f:
    file = f.read()
    n_verts = len(re.findall(r"^v ", file, flags=re.MULTILINE))
    n_faces = len(re.findall(r"^f ", file, flags=re.MULTILINE))
    verts = np.empty((n_verts, 3))  # Nx3 (vertices, xyz)
    faces = np.empty((n_faces, 3), dtype="int16")  # Nx3 (faces, 3 vertices)

    f.seek(0, 0)  # Go back to beginning of file
    n_v = 0
    n_f = 0
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        if re.findall(r"v ", line):  # Line of vertex
            vert = line.split(" ")
            verts[n_v] = np.array([float(x) for x in vert[1:]])
            n_v += 1
        elif re.findall(r"f ", line):  # Line of vertex
            face = re.findall(r" .*?/", line)
            faces[n_f] = np.array([int(x[1:-1]) for x in face[:]], dtype="int16")
            n_f += 1


# Init display
win = pg.display.set_mode((0, 0))
pg.mouse.set_visible(False)
pg.event.set_grab(True)
w, h = pg.display.get_window_size()
clock = pg.time.Clock()

# Player coordinates
x, y, z = 0, 0, 0
azi, alt = 0, 0  # Player orientation angles
speed = 0.01

proj = get_proj_mat(70*math.pi/180, w/h)

while True:
    dt = clock.tick(60)
    # Handle events
    for event in pg.event.get():
        # Handle mouse scroll
        if event.type == pg.MOUSEWHEEL:
            speed *= 1.1**event.y
    # Handle key presses
    keys = pg.key.get_pressed()
    if keys[pg.K_d]:
        x += dt * speed * math.cos(azi)
        z += dt * speed * math.sin(azi)
    if keys[pg.K_a]:
        x -= dt * speed * math.cos(azi)
        z -= dt * speed * math.sin(azi)
    if keys[pg.K_w]:
        x += dt * speed * math.sin(azi)
        z -= dt * speed * math.cos(azi)
    if keys[pg.K_s]:
        x -= dt * speed * math.sin(azi)
        z += dt * speed * math.cos(azi)
    if keys[pg.K_SPACE]:
        y += dt * speed
    if keys[pg.K_LSHIFT]:
        y -= dt * speed

    # Handle mouse move
    mouse_move = pg.mouse.get_rel()
    azi += 0.002 * mouse_move[0]
    alt += 0.002 * -mouse_move[1]
    alt = max(-math.pi/2, min(math.pi/2, alt))

    # Translate objects
    verts_rot = verts - np.array([[x, y, z]])
    # Rotate objects
    rot_mat = get_rot_mat(azi, alt)
    verts_rot = np.dot(rot_mat, verts_rot.T).T

    # Project objects
    projected = np.dot(proj, verts_rot.T)  # 2x3 @ 3xN -> 2xN
    # Divide each point by its z value
    projected = projected.T / -np.absolute(verts_rot[:, 2].reshape(-1, 1))  # Nx2 / Nx1 broadcast
    # Rescale to fit screen
    rescale = np.array([
        [w/2, h/2]
    ])
    projected = projected*rescale + rescale  # Nx2 and 1x2

    # Draw objects
    # Clear display
    win.fill((255, 255, 255))
    tri = pg.Surface((w, h))
    tri.fill((255, 255, 255))
    tri.set_colorkey((255, 255, 255))
    for i in faces:
        a, b, c = i  # 3 verts of triangle
        # If object is behind screen
        if verts_rot[a-1, 2] > -0.01 or verts_rot[b-1, 2] > -0.01 or verts_rot[c-1, 2] > -0.01:
            continue

        # Draw triangles
        pg.draw.polygon(tri, (0, 0, 0), (
            (projected[a - 1][0], projected[a - 1][1]),
            (projected[b - 1][0], projected[b - 1][1]),
            (projected[c - 1][0], projected[c - 1][1]),
        ), width=1)
    win.blit(tri, (0, 0))
    pg.display.update()
