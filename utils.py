import numpy as np
import re
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D


def gather_point_cloud(file):
    # takes a .off file as input and returns ordered lists of the x, y and z coordinates
    # corresponding to each point
    x = []
    input_file = open(file, 'r')
    for line in input_file:
        coord = re.search(r'([\w-]+\.\w+)\s([\w-]+\.\w+)\s([\w-]+\.\w+)', line)
        if coord:
            if len(x) < 1:
                x = np.array([float(coord.group(1))])
                y = np.array([float(coord.group(2))])
                z = np.array([float(coord.group(3))])
            else:
                x = np.concatenate((x, [float(coord.group(1))]))
                y = np.concatenate((y, [float(coord.group(2))]))
                z = np.concatenate((z, [float(coord.group(3))]))

    return x, y, z


def create_2d_grid(x, y, pixel_radius):
    # takes 2d point cloud coordinates and returns a structured grid of unfilled pixels,
    # with size determined by coordinate bounds and spacing determined by the pixel_radius,
    # which is the width in mm

    # determine grid bounds
    x_max = np.amax(x)
    x_min = np.amin(x)

    y_max = np.amax(y)
    y_min = np.amin(y)

    pixels = np.zeros((int((x_max - x_min) / (2 * pixel_radius)), int((y_max - y_min) / (2 * pixel_radius))))

    print("Grid size: {}".format(pixels.shape))

    return pixels


def create_3d_grid(x, y, z, voxel_radius):
    # performs the same function in 3d

    # determine grid bounds
    x_max = np.amax(x)
    x_min = np.amin(x)

    y_max = np.amax(y)
    y_min = np.amin(y)

    z_max = np.amax(z)
    z_min = np.amin(z)

    voxels = np.zeros((int((x_max - x_min) / (2 * voxel_radius)),
                       int((y_max - y_min) / (2 * voxel_radius)),
                       int((z_max - z_min) / (2 * voxel_radius))))

    print("Grid size: {}".format(voxels.shape))

    return voxels


def characterize_faces_2d(file, view='xy'):
    # obtains the point cloud for a CAD file followed by the corner vertices of each triangular face
    # and returns lists of their coordinates p, q and r
    x, y, z = gather_point_cloud(file)
    p_tot = []
    input_file = open(file, 'r')
    for line in input_file:
        face = re.search(r'\w\s(\w+)\s(\w+)\s(\w+)', line)

        if face:
            # determine vertices of face
            one = int(face.group(1))
            two = int(face.group(2))
            thr = int(face.group(3))

            # discard either x, y or z coordinates depending on the view
            if view == 'xy':
                p = np.array([x[one], y[one]])
                q = np.array([x[two], y[two]])
                r = np.array([x[thr], y[thr]])
            elif view == 'xz':
                p = np.array([x[one], z[one]])
                q = np.array([x[two], z[two]])
                r = np.array([x[thr], z[thr]])
            elif view == 'yz':
                p = np.array([y[one], z[one]])
                q = np.array([y[two], z[two]])
                r = np.array([y[thr], z[thr]])

            if len(p_tot) < 1:
                p_tot = [p]
                q_tot = [q]
                r_tot = [r]
            else:
                p_tot += [p]
                q_tot += [q]
                r_tot += [r]

    return p_tot, q_tot, r_tot


def characterize_faces_3d(file):
    # performs the same function in 3d, while additionally
    # calculating the 2d infinite plane equation of each face
    x, y, z = gather_point_cloud(file)
    A_tot = []
    input_file = open(file, 'r')
    for line in input_file:
        face = re.search(r'\w\s(\w+)\s(\w+)\s(\w+)', line)

        if face:
            # determine vertices of face
            one = int(face.group(1))
            two = int(face.group(2))
            thr = int(face.group(3))

            p = np.array([x[one], y[one], z[one]])
            q = np.array([x[two], y[two], z[two]])
            r = np.array([x[thr], y[thr], z[thr]])
            # calculate equation of infinite plane in form A1*x + A2*y + A3*z = d, for the face
            A = np.cross(np.subtract(q, p), np.subtract(r, p))
            d = np.dot(A, p)

            if len(A_tot) < 1:
                p_tot = [p]
                q_tot = [q]
                r_tot = [r]
                A_tot = [A]
                d_tot = [d]
            else:
                p_tot += [p]
                q_tot += [q]
                r_tot += [r]
                A_tot += [A]
                d_tot += [d]

    return p_tot, q_tot, r_tot, A_tot, d_tot


def barycentric_vectors(p, q, r, s):
    # defines whether or not a point lying on a plane is within a particular triangle on that plane
    # refer to: blackpawn.com/texts/pointinpoly/
    # p, q and r are triangle vertices, s is a point to be tested
    v0 = np.subtract(q, p)
    v1 = np.subtract(r, p)
    v2 = np.subtract(s, p)

    scalar00 = np.dot(v0, v0)
    scalar01 = np.dot(v0, v1)
    scalar02 = np.dot(v0, v2)
    scalar11 = np.dot(v1, v1)
    scalar12 = np.dot(v1, v2)

    denom = 1 / (scalar00 * scalar11 - scalar01 * scalar01)
    u = (scalar11 * scalar02 - scalar01 * scalar12) * denom
    v = (scalar00 * scalar12 - scalar01 * scalar02) * denom

    return u, v


def voxel_fill(voxels):
    # takes a hollow shell of voxels and fills the interior
    print("Filling voxels...")
    # Cycle through voxels by iterating through 3 dimensions
    for g in range(voxels.shape[0]):
        for h in range(voxels.shape[1]):
            for i in range(voxels.shape[2]):
                # Ignore all filled voxels
                if voxels[g, h, i] == 1:
                    continue
                # If it hasn't yet, check whether it is within shape or not
                # Do this by iterating one by one through voxels forward and backward in 3 dimensions
                # to test whether a filled voxel is met in all directions
                else:
                    # assume the voxel is to be filled
                    filled = 1
                    direction = 0
                    # iterate through 6 directions
                    # stop search if any direction is found to be clear of voxels
                    while filled and direction < 6:
                        ctr = 0
                        if direction == 0:
                            while g + ctr < voxels.shape[0]:
                                check = voxels[g + ctr, h, i]
                                ctr += 1
                                if check == 1:
                                    filled = 1
                                    break
                                else:
                                    filled = 0

                        elif direction == 1:
                            while g - ctr >= 0:
                                check = voxels[g - ctr, h, i]
                                ctr += 1
                                if check == 1:
                                    filled = 1
                                    break
                                else:
                                    filled = 0

                        elif direction == 2:
                            while h + ctr < voxels.shape[1]:
                                check = voxels[g, h + ctr, i]
                                ctr += 1
                                if check == 1:
                                    filled = 1
                                    break
                                else:
                                    filled = 0

                        elif direction == 3:
                            while h - ctr >= 0:
                                check = voxels[g, h - ctr, i]
                                ctr += 1
                                if check == 1:
                                    filled = 1
                                    break
                                else:
                                    filled = 0

                        elif direction == 4:
                            while i + ctr < voxels.shape[2]:
                                check = voxels[g, h, i + ctr]
                                ctr += 1
                                if check == 1:
                                    filled = 1
                                    break
                                else:
                                    filled = 0

                        elif direction == 5:
                            while i - ctr >= 0:
                                check = voxels[g, h, i - ctr]
                                ctr += 1
                                if check == 1:
                                    filled = 1
                                    break
                                else:
                                    filled = 0

                        direction += 1

                    if filled:
                        voxels[g, h, i] = 1

    return voxels


def pixelize(file, pixel_radius, view='xy'):
    # takes a CAD mesh file and turns it into a set of 2d pixels, of width pixel_radius,
    # that are filled if their centre lies within a triangular face,
    # viewed from directly above the axes specified

    x, y, z = gather_point_cloud(file)

    # Reorientate axes as x-y depending on view
    if view == 'xy':
        pass
    elif view == 'xz':
        y = z
    elif view == 'yz':
        x = y
        y = z
    else:
        sys.exit("Error: invalid view. Please specify view 'xy', 'xz' or 'yz'")

    # determine coordinates for starting corner of grid
    x_min = np.amin(x)
    y_min = np.amin(y)
    pixels = create_2d_grid(x, y, pixel_radius)

    p, q, r = characterize_faces_2d(file, view=view)

    for g in range(pixels.shape[0]):
        for h in range(pixels.shape[1]):
            print("Point: ({}, {})".format(g, h))
            # centre point of each pixel being checked is scrolled across
            point = np.array(((2*g + 1) * pixel_radius + x_min, (2*h + 1) * pixel_radius + y_min))

            # for each point, look through all faces to until one is found that it lies within
            for face in range(len(p)):
                # using a Barycentric coordinates method,
                # determine whether s is within the bounds of the triangular face
                if p[face][0] == q[face][0] == r[face][0]:
                    continue
                elif p[face][1] == q[face][1] == r[face][1]:
                    continue

                u, v = barycentric_vectors(p[face], q[face], r[face], point)
                # if the closest point is within the triangle, fill the voxel
                if u >= 0 and v >= 0 and u + v < 1:
                    pixels[g, h] = 1
                    break

    # plot the pixels on a grid
    fig = plt.figure(figsize=(int(np.amax(x)), int(np.amax(y))))
    plt.imshow(np.flip(pixels.T, axis=0), cmap='Blues')
    plt.show()

    return pixels


def voxelize(file, voxel_radius):
    # performs the same function in 3d, creating a grid of voxels rather than pixels

    x, y, z = gather_point_cloud(file)
    x_min = np.amin(x)
    y_min = np.amin(y)
    z_min = np.amin(z)

    voxels = create_3d_grid(x, y, z, voxel_radius)
    p, q, r, A, d = characterize_faces_3d(file)

    for g in range(voxels.shape[0]):
        for h in range(voxels.shape[1]):
            for i in range(voxels.shape[2]):
                print("Point: {}, {}, {}".format(g, h, i))
                point = np.array(((2*g + 1) * voxel_radius + x_min, (2*h + 1) * voxel_radius + y_min,
                                  (2*i + 1) * voxel_radius + z_min))

                for face in range(len(A)):
                    # determine the closest point, s, on the infinite plane of the face
                    # to the point being checked
                    # if p, q and r lie along a straight line then A = 0 and a plane cannot be calculated
                    # subsequent analysis of the plane must be skipped
                    if A[face][0] != 0 or A[face][1] != 0 or A[face][2] != 0:
                        k = (np.dot(point, A[face]) - d[face]) / np.dot(A[face], A[face])
                        s = np.array([point[0] - k * A[face][0], point[1] - k * A[face][1], point[2] - k * A[face][2]])
                    else:
                        continue

                    # if the closest point on the plane is beyond the corners of the point's voxel,
                    # (corner being 1.41 times the width away from the centre)
                    # the voxel will not be filled
                    if np.abs(np.linalg.norm(point - s)) > 1.41 * voxel_radius:
                        pass
                    else:
                        # using a Barycentric coordinates method,
                        # determine whether s is within the bounds of the triangular face
                        u, v = barycentric_vectors(p[face], q[face], r[face], s)

                        # if the closest point is within the triangular face, fill the voxel.
                        if u >= 0 and v >= 0 and u + v < 1:
                            voxels[g, h, i] = 1
                            break

    voxels = voxel_fill(voxels)
    print("Voxels filled")

    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.voxels(voxels, alpha=0.5, edgecolor="k")
    plt.show()

    return voxels


def plot_cloud(file, dimensions, view='xy', size=np.pi * 5, colour='b'):
    # plots a point cloud of vertices from a CAD file
    # dimensions can be 2 or 3, view can be 'xy', 'xz' or 'yz'
    x, y, z = gather_point_cloud(file)

    if dimensions == 2:
        if view == 'xy':
            pass
        elif view == 'xz':
            y = z
        elif view == 'yz':
            x = y
            y = z
        else:
            sys.exit("Error: invalid view. Please specify view 'xy', 'xz' or 'yz'")

        fig = plt.figure(figsize=(int(np.amax(x)), int(np.amax(y))))

    if dimensions == 3:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, s=size, color=colour)
    elif dimensions == 2:
        plt.scatter(x, y, s=size)

    plt.show()


def plot_mesh(file, dimensions, view='xy', colour='b'):
    # plots a mesh of a CAD file
    x, y, z = gather_point_cloud(file)

    if view == 'xy':
        fig = plt.figure(figsize=(int(np.amax(x)), int(np.amax(y))))
    elif view == 'xz':
        fig = plt.figure(figsize=(int(np.amax(x)), int(np.amax(z))))
    elif view == 'yz':
        fig = plt.figure(figsize=(int(np.amax(y)), int(np.amax(z))))

    if dimensions == 3:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')

    input_file = open(file, 'r')
    for line in input_file:

        match = re.search(r'\w\s(\w+)\s(\w+)\s(\w+)', line)
        if match:
            point_1 = int(match.group(1))
            point_2 = int(match.group(2))
            point_3 = int(match.group(3))

            a = np.array([x[point_1], x[point_2], x[point_3], x[point_1]])
            b = np.array([y[point_1], y[point_2], y[point_3], y[point_1]])
            c = np.array([z[point_1], z[point_2], z[point_3], z[point_1]])

            if dimensions == 3:
                ax.plot(a, b, c, color=colour)

            elif dimensions == 2:
                if view == 'xy':
                    plt.plot(a, b, color=colour)
                elif view == 'xz':
                    plt.plot(a, c, color=colour)
                elif view == 'yz':
                    plt.plot(b, c, color=colour)
                else:
                    sys.exit("Error: invalid view. Please specify view 'xy', 'xz' or 'yz'")

    plt.show()
