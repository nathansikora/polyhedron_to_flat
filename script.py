import numpy as np
from stl import mesh
from itertools import product
import networkx as nx
from networkx.algorithms import tree


#m = mesh.Mesh.from_file(r"D:\Personal\3D\TTES.stl")
m = mesh.Mesh.from_file(r"D:\Personal\3D\blabla.stl")
f2d = []

for v in m.vectors:
    angs = np.zeros(3)
    dists = np.zeros(3)
    new_v = np.zeros(3, dtype=np.complex)
    new_v[1] = np.linalg.norm(v[1, :] - v[0, :])

    cur_a = np.arccos(np.sum((v[1, :] - v[0, :]) * (v[2, :] - v[1, :])) / (
                np.linalg.norm(v[1, :] - v[0, :]) * np.linalg.norm(v[2, :] - v[1, :])))
    cur_d = np.linalg.norm(v[2, :] - v[1, :])
    new_v[2] = new_v[1] + cur_d * np.exp(1j * cur_a)

    new_v -= np.mean(new_v)
    f2d.append(new_v)


bool_face_mat = np.zeros((m.vectors.shape[0], m.vectors.shape[0]))
match_face_mat = -np.ones((m.vectors.shape[0], m.vectors.shape[0]), dtype=list)
for (fi, f), (gi, g) in product(enumerate(m.vectors), enumerate(m.vectors)):
    if fi == gi:
        continue
    match = []
    for k in range(3):
        for j in range(3):
            if all(f[k, :] == g[j, :]):
                match.append((k, j))                

    if len(match) == 2:
        bool_face_mat[fi, gi] = 1
        match_face_mat[fi, gi] = match


def create_net(sptG, cur_edge=0, prv_edge=None, data=None, is_first=True):
    if data is None:
        data = {}
        data['treemat'] = nx.to_numpy_array(sptG)
        data['shift_faces'] = {}
        data['visited'] = []
        data['all_lines'] = []

    cur_face = f2d[cur_edge]
    lines_to_add = [{0, 1}, {1, 2}, {2, 0}]
    if prv_edge is not None:
        prv_face = data['shift_faces'][prv_edge]
        vert_match = match_face_mat[prv_edge, cur_edge]
        prv_v1, prv_v2 = prv_face[vert_match[0][0]], prv_face[vert_match[1][0]]
        cur_v1, cur_v2 = cur_face[vert_match[0][1]], cur_face[vert_match[1][1]]

        lines_to_add.remove({vert_match[0][1], vert_match[1][1]})

        prv_orient = np.imag(np.log(prv_v2 - prv_v1))
        cur_orient = np.imag(np.log(cur_v2 - cur_v1))
        cur_face = cur_face * np.exp(1j * (prv_orient +0 - cur_orient))
        cur_face += prv_v1 - cur_face[vert_match[0][1]]


        #plot_tri(prv_face)
        #plot_tri(cur_face)

    data['all_lines'] += [tuple(cur_face[e] for e in list(l)) for l in lines_to_add]
    data['shift_faces'][cur_edge] = cur_face

    data['visited'].append(cur_edge)
    for next_edge in np.nonzero(data['treemat'][cur_edge, :])[0]:
        if next_edge in data['visited']:
            continue
        create_net(sptG, next_edge, cur_edge, data, is_first=False)
    return data if is_first else None

import matplotlib.pyplot as plt
def plot_tri(tri):
    plt.plot([tri[k].real for k in [0,1,2,0]], [tri[k].imag for k in [0,1,2,0]])

def plot_line(l):
    plt.plot([l[0].real, l[1].real], [l[0].imag, l[1].imag])

G = nx.convert_matrix.from_numpy_matrix(bool_face_mat)
sptG = tree.maximum_spanning_tree(G)

def is_intersecting(p11, p12, p21, p22):
    if p11.real > p12.real:
        p11, p12 = p12, p11
    if p21.real > p22.real:
        p21, p22 = p22, p21
    m1 = (p12.imag - p11.imag)/(p12.real - p11.real)
    b1 = p11.imag - p11.real * m1

    m2 = (p22.imag - p21.imag) / (p22.real - p21.real)
    b2 = p21.imag - p21.real * m2

    xzero = (b2-b1)/(m1-m2)

    if xzero > p11.real + 0.0001 and xzero < p12.real - 0.0001 and xzero > p21.real + 0.0001 and xzero < p22.real - 0.0001:
        return True
        plot_line((p11, p12))
        plot_line((p21, p22))
    return False


intersect_count = 1
while intersect_count > 0:
    unused_edges = list(g for g in G.edges if g not in sptG.edges)
    new_edge = unused_edges[np.random.randint(len(unused_edges))]
    sptG.add_edge(*new_edge)
    cycle = nx.find_cycle(sptG)
    if new_edge in cycle:
        cycle.remove(new_edge)
    elif (new_edge[1], new_edge[0]) in cycle:
        cycle.remove((new_edge[1], new_edge[0]))
    else:
        raise Exception

    sptG.remove_edge(*cycle[np.random.randint(len(cycle))])
    data = create_net(sptG)
    intersect_count = 0
    for (k1, l1), (k2, l2) in product(enumerate(data['all_lines']), enumerate(data['all_lines'])):
        if k1 <= k2:
            continue
        if is_intersecting(l1[0], l1[1], l2[0], l2[1]):
            intersect_count += 1
    print(intersect_count)
    if False:
        minl, maxl = np.inf, -np.inf
        for tri in data['shift_faces'].values():
            for e in tri:
                minl = e.real if e.real < minl else minl
                minl = e.imag if e.imag < minl else minl
                maxl = e.real if e.real > maxl else maxl
                maxl = e.imag if e.imag > maxl else maxl
            plot_tri(tri)
        plt.xlim((minl - 0.1*np.abs(minl), maxl + 0.1*np.abs(maxl)))
        plt.ylim((minl - 0.1*np.abs(minl), maxl + 0.1*np.abs(maxl)))


a = 5


