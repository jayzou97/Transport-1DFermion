import numpy as np
from scipy.linalg import expm
import time
import multiprocessing as mp
import random


inv_density = 6     # inverse of density
int_range = 3
Ms = 16800
process = 28
Ms1 = np.int(Ms/process)

step1 = 60  # the number of measurements of stage 1
t_step10 = 0.02  # time interval
step2 = 240  # the number of measurements of stage 2
t_step20 = 0.1  # time interval
step = step1 + step2

l_min = 1
l_max = 9
l_step = 2
l0 = np.arange(l_min, l_max, l_step)
num_result = len(l0)

interval = int_range
num_period = 4

Q0 = np.zeros(num_result)
tau0 = np.zeros(num_result)
c0 = np.zeros((num_result, step))

st1 = time.time()

filename1 = 'P{}R{}Re{}_1.txt'.format(inv_density, int_range, num_period)    # basic information
filename5 = 'P{}R{}Re{}_5.txt'.format(inv_density, int_range, num_period)    # contrast
filename6 = 'P{}R{}Re{}_6.txt'.format(inv_density, int_range, num_period)    # Q0
filename7 = 'P{}R{}Re{}_7.txt'.format(inv_density, int_range, num_period)    # tau0
filename2 = 'P{}R{}Re{}_2.txt'.format(inv_density, int_range, num_period)    # density
filename3 = 'P{}R{}Re{}_3.txt'.format(inv_density, int_range, num_period)    # density
filename4 = 'P{}R{}Re{}_4.txt'.format(inv_density, int_range, num_period)    # density
filename8 = 'P{}R{}Re{}_8.txt'.format(inv_density, int_range, num_period)    # density
filename_dense = [filename2, filename3, filename4, filename8]

with open(filename1, 'a+') as f:
    f.write('Ms={}, T_precise={}, step1={}, t_step1={}, step2={}, t_step2={}, T_total={}'.format(Ms,
                    step1*t_step10, step1, t_step10, step2, t_step20, step1*t_step10+step2*t_step20))
    f.write('\n')


def linear(x, A, B):
    return A*x+B


def albreminder(x): 
    n = np.size(x, 0)
    y = []
    for i in range(n + 1):
        c = list(range(n + 1))
        c.remove(i)
        subi = x[:, c]
        s = np.linalg.det(subi)
        if i % 2 == 0:
            y.append(s)
        else:
            y.append(-s)
    y = np.array(y)
    return y


def sample(x, L2, N):
    a = np.random.permutation(N)
    r = x[:, a[0]]
    prob = r * np.conjugate(r)
    prob = prob.real
    r0 = np.random.choice(L2, 1, p=prob)
    row = list(r0)
    m = np.zeros(L2)
    m[r0] = 1
    if N ==1:
        return m
    else:
        for i in range(1, N):
            column = a[0:i + 1]
            submat = x[np.ix_(row, column)]
            sub = albreminder(submat)
            prob = []
            div = np.prod(range(1, i + 2))  
            for j in range(L2):
                det = np.dot(x[[j], column], sub)
                p = np.square(np.abs(det)) / div
                prob.append(p.real)
            probs = sum(prob)
            prob = prob / probs
            ri = np.random.choice(L2, 1, p=prob)
            m[ri] = 1
            row.append(ri[0])
        return m


def relax(_):
    random.seed(_)
    dense_part = np.zeros((step, L1))
    dense_part_error = np.zeros((step, L1))

    H = np.zeros((L2, L2))

    for i in range(L2 - 1):
        H[i, i + 1] = -1
        H[i + 1, i] = -1

    for s in range(step):
        if s < step1:
            t = s * t_step1
        else:
            t = step1 * t_step1 + (s - step1) * t_step2

        h = H * t * 1j
        U = expm(h)
        U2 = U.real - 1j * U.imag
        U = U2[:, initial_logical]

        num = np.zeros(L1)
        for ms in range(Ms1):
            m = sample(U, L2, N)  # logical basis, L2
            
            A1 = np.zeros((N, N), dtype=complex)
            a1 = 0
            for i in range(L2):
                if m[i] == 1:
                    A1[a1] = U[i]
                    a1 = a1 + 1
                else:
                    continue
            
            d1 = np.linalg.det(A1)  # determinant
            if d1 == 0:
                print(0)
                continue  
            else:
                n = m.copy()
                i = 0
                while i < len(n):
                    if n[i] == 1:
                        for _ in range(int_range):
                            n = np.insert(n, i, 0)
                        i = i + 1 + int_range
                    else:
                        i = i + 1
                for _ in range(int_range):
                    n = np.delete(n, 0)

                num = num + n

        dense_part[s] = num / Ms1
        dense_part_error[s] = np.sqrt(dense_part[s] - dense_part[s] ** 2) / np.sqrt(Ms1)

    pack = [dense_part, dense_part_error]
    return pack


def multicore():
    pool = mp.Pool(processes=process)
    res_pack = pool.map(relax, range(process))
    dense = sum([x[0] for x in res_pack]) / process
    dense_error = sum([x[1] for x in res_pack]) / (process * np.sqrt(process))
    pack = [dense, dense_error]
    return pack


for l in l0:
    random.seed()

    particle_period = l + 1
    l1 = np.int((l - l_min)/l_step)
    N = particle_period * num_period
    period = inv_density * particle_period
    residue = period - particle_period * (int_range + 1)
    start_period = np.int(residue / 2) + 1
    center_period = start_period + (particle_period - 1) * (interval + 1) / 2
    L1 = num_period * period
    Q = 2 * np.pi / period
    Q0[l1] = Q

    t_step1 = t_step10 / Q
    t_step2 = t_step20 / Q

    initial = np.arange(start_period, start_period + particle_period * (interval + 1), interval + 1)
    a = initial.copy()
    for i in range(1, num_period):
        b = a + i * period
        initial = np.append(initial, b)
    L2 = L1 + int_range - N * int_range

    initial_logical = [initial[i] - int_range * i for i in range(N)]

    results_pack = multicore()
    density = results_pack[0]
    density_error = results_pack[1]

    fourier = np.cos(Q * (np.arange(L1) - center_period)).reshape(L1, 1)
    c = np.dot(density, fourier).reshape(step)

    c2 = c / c[0]

    c0[l1] = c2

    c1 = np.abs(c2 - 0.6)
    tau1 = np.argmin(c1) * t_step1
    tau = tau1 / np.log(1 / 0.6)
    tau0[l1] = tau

    # write in
    with open(filename1, 'a+') as f:
        f.write('#{}: L={}, N={} \n'.format(l, L1, N))
        f.write('initial: physical basis:{}  ,  logical:{} \n'.format(initial, initial_logical))
        f.write('density:{}    error:{} \n'.format(np.max(density[10]), np.max(density_error[10])))
        f.write('Q={}, tau={}  \n'.format(Q0[l1], tau0[l1]))
        f.write('\n')

    with open(filename5, 'a+') as f:
        for ind in range(step):
            f.write('%3.8f' % c[ind])
            f.write('\t')
        f.write('\n')

    with open(filename_dense[l1], 'a+') as f:
        for ind in range(step):
            for indc in range(L1):
                f.write('%3.8f' % density[ind, indc])
                f.write('\t')
            f.write('\n')
        f.write('\n')


with open(filename6, 'a+') as f:
    for ind in range(num_result):
        f.write('%3.8f' % Q0[ind])
        f.write('\t')
    f.write('\n')

with open(filename7, 'a+') as f:
    for ind in range(num_result):
        f.write('%3.8f' % tau0[ind])
        f.write('\t')
    f.write('\n')

st2 = time.time()
st = st2 - st1

with open(filename1, 'a+') as f:
    f.write('time: {} (s), {} (h) \n'.format(st, st/3600))
    f.write('\n')


