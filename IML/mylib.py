import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pylab
import time
from sklearn.cluster import KMeans
import pandas as pd
from mpl_toolkits import mplot3d

strats = ("plusplus", "uniform", "random", "choice")
msg = ("KMeans++\n", "Равномерное распределение\n", "Случайные числа\n", "Выбор из данных точек\n")


def uniform(ratio, k, eps=0.1):
    a = int(np.sqrt(k))
    b = k // a + (k % a != 0)
    while abs(a / b - ratio) >= eps:
        if a == 1:
            break
        a -= 1
        b = k // a + (k % a != 0)
    return a, b


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def strat_of_init(matr, strat="uniform", k=2, show_work=-1, eps=0.0001):
    if k < 2:
        k = 2
    x = np.asarray(matr[:, 0])
    y = np.asarray(matr[:, 1])
    length = len(x)
    xc = np.zeros(k)
    yc = np.zeros(k)
    
    if strat == "random":
        xc = np.random.rand(k) * (x.max() - x.min()) + x.min()
        yc = np.random.rand(k) * (y.max() - y.min()) + y.min()
        if show_work >= 0:
            pylab.figure(show_work)
            plt.scatter(xc, yc, marker = "*", c = "r")
        return xc, yc
    if strat == "choice":
        r = np.random.choice(length, k, replace = False)
        xc = x[r]
        yc = y[r]
        if show_work >= 0:
            pylab.figure(show_work)
            plt.scatter(xc, yc, marker = "*", c = "r")
        return xc, yc
    if strat == "plusplus":
        r = np.random.choice(k, replace = False)
        xc[0] = x[r]
        yc[0] = y[r]
        for ready in range(k - 1):
            dx2 = np.zeros(length)
            sumdx2 = 0
            for i in range(length):
                dx2[i] = dist(xc[ready], yc[ready], x[i], y[i])
                for j in range(ready):
                    dx2[i] = min(dx2[i], dist(xc[j], yc[j], x[i], y[i]))
                sumdx2 += dx2[i]
            sumdx2 *= np.random.rand()
            sumdx = 0
            for i in range(length):
                sumdx += dx2[i]
                if sumdx > sumdx2:
                    xc[ready + 1] = x[i]
                    yc[ready + 1] = y[i]
                    break
        return xc, yc
    x_len = x.max()-x.min()
    y_len = y.max()-y.min()
    uni_x, uni_y = uniform(x_len/y_len, k, eps=eps)
    units = np.random.choice(uni_x * uni_y, k, replace=False)
    xc = np.zeros(k)
    yc = np.zeros(k)
    for i in range(k):
        xc[i] = x.min() + x_len / (2 * uni_x) + (units[i] % uni_x) * x_len / uni_x
        yc[i] = y.min() + y_len / (2 * uni_y) + (units[i] % uni_y) * y_len / uni_y
    xline, yline = 0, 0
    for i in range(uni_x + 1):
        xline = x.min() + i * x_len / uni_x
    for i in range(uni_y + 1):
        yline = y.min() + i * y_len / uni_y
    if show_work >= 0:
        pylab.figure(show_work)
        plt.scatter(xc, yc, marker="*", c="r")
        plt.plot([xline, xline], [y.min(), y.max()])
        plt.plot([x.min(), x.max()], [yline, yline])
    return xc, yc


def filter_by_clust(a, i, length):
    return list(filter(lambda q: a[q] == i, range(length)))


def KMeans4(matr, k=2, seed=0, eps=0.0001, limit_of_rec=500, show_work=-1, strat = "plusplus"):
    np.random.seed(seed)
    x = np.asarray(matr[:, 0])
    y = np.asarray(matr[:, 1])
    length = len(x)
    
    xc, yc = strat_of_init(matr, strat = strat, k = k, show_work=show_work)
    matr = np.concatenate([x, y, np.random.randint(0, k, length, dtype = int)])
    matr = matr.reshape(3, length).T
    limit = limit_of_rec
    while True:
        limit -= 1
        if limit == 0:
            if show_work >= 0:
                print("Calculated with the achievement of the limit number of iterations: ", limit_of_rec)
            break
        for i in range(length):
            clast = k - 1
            dist_to_clast = dist(matr[i][0], matr[i][1], xc[k - 1], yc[k - 1])
            for j in range(k - 1):
                cur_dist = dist(matr[i][0], matr[i][1], xc[j], yc[j])
                if cur_dist < dist_to_clast:
                    dist_to_clast = cur_dist
                    clast = j
            matr[i][2] = clast
        sums = 0
        for i in range(k):
            fil = filter_by_clust(matr[:, 2], i, length)
            if len(fil) == 0:
                sums = eps + 1
                r = np.random.randint(length)
                xc[i] = x[r]
                yc[i] = y[r]
            else:
                xcc = xc[i]
                ycc = yc[i]
                xc[i] = np.mean(x[fil])
                yc[i] = np.mean(y[fil])
                sums += dist(xc[i], yc[i], xcc, ycc)
        if sums < eps:
            if show_work >= 0:
                print("Calculated with given accuracy, number of iterations =", limit_of_rec - limit)
            break
    if show_work >= 0:
        pylab.figure(show_work)
        plt.scatter(xc, yc, marker = "v", c = "black")
    return matr[:, 2], xc, yc


def best_K(matr, k_max=0, show_work=-1, eps=0.25, d_eps=0.0001, strat_mean = True, strat = "plusplus"):
    x = np.asarray(matr[:, 0])
    y = np.asarray(matr[:, 1])
    length = len(x)
    if k_max == 0:
        k_max = int(pow(length, 1/3)) + 2
    if k_max < 2:
        k_max = 3
    arr = np.array(range(k_max - 2)) + 2
    inter_max = np.zeros(k_max - 2)
    exter_min = np.zeros(k_max - 2)
    inter_mean = np.zeros(k_max - 2)
    
    for e in range(k_max - 2):
        a, xc, yc = KMeans4(matr, k = arr[e], strat = strat)
        flag = True
        inter_num = 0
        for i in range(arr[e]):
            j = i + 1
            while j < arr[e]:
                d = dist(xc[i], yc[i], xc[j], yc[j])
                if flag:
                    exter_min[e] = d
                    flag = False
                exter_min[e] = min(exter_min[e], d)
                j += 1
            fil = filter_by_clust(a, i, length)
            for j in fil:
                inter_num += 1
                d = dist(xc[i], yc[i], x[j], y[j])
                inter_mean[e] += d
                inter_max[e] = max(inter_max[e], d)
        inter_mean[e] /= inter_num
    out = np.diff(exter_min/inter_max)
    arr = arr[:-1]
    if strat_mean:
        out = np.diff(inter_mean)
    f = list(filter(lambda i: abs(out[i]) < eps, range(len(out))))
    while len(f)==0:
        eps += d_eps
        f = list(filter(lambda i: abs(out[i]) < eps, range(len(out))))
    if show_work >= 0:
        pylab.figure(0, figsize = (15, 10))
        plt.plot(arr, out)
    if strat_mean:
        return arr[f[0]], eps
    return arr[f[0] - 1], eps


def clusters_sorted(clust, xc, yc, x_s = 0, y_s = 0):
    k = xc.shape[0]
    length = clust.shape[0]
    cen_clust_sorted = sorted(list(range(k)), key = lambda i: dist(xc[i], yc[i], x_s, y_s))
    clust += k
    for i in range(k):
        f = filter_by_clust(clust, cen_clust_sorted[i] + k, length)
        clust[f] = i
    return clust, xc[cen_clust_sorted], yc[cen_clust_sorted]


def sort_by_clust(matr, clust, xc, yc, corner_left_lower = [True, True]):
    clust = np.asarray(clust)
    length = len(clust)
    k = len(xc)
    dot = np.zeros(2)
    for i in range(2):
        if corner_left_lower[i]:
            dot[i] = matr[:, i].min()
        else:
            dot[i] = matr[:, i].max()
    clust, xc, yc = clusters_sorted(clust, xc, yc, dot[0], dot[1])
    matr = np.concatenate([matr.T.ravel(), clust]).reshape(3, length).T
    matr = np.array(sorted(matr, key = lambda x: x[2]))
    return matr[:, :2], clust, xc, yc


def clusters_centers(matr, clust):
    k = clust.max() + 1
    length = matr.shape[0]
    x = np.asarray(matr[:, 0])
    y = np.asarray(matr[:, 1])
    xc = np.zeros(k)
    yc = np.zeros(k)
    for i in range(k):
        fil = filter_by_clust(clust, i, length)
        xc[i] = np.mean(x[fil])
        yc[i] = np.mean(y[fil])
    return xc, yc


def init_t_and_a_arr(n_tests):
    mytype =  [("plusplus", 'float32'), ("uniform", 'float32'), ("random", 'float32'), ("choice", 'float32')]
    t = np.array(np.zeros(n_tests), dtype = mytype)
    a = np.array(np.zeros(n_tests), dtype = mytype)
    return t, a


def strats_compar(k = 2, n = 400, n_tests = 10, gen_tests = True, tests = ()):
    if gen_tests:
        tests = tuple([(n, k)] * n_tests)
    else:
        n_tests = tests.shape[0]
    times, accurs = init_t_and_a_arr(n_tests)
    for i in range(len(tests)):
        x, _ = make_blobs(n_samples = tests[i][0], n_features = 2, centers = tests[i][1], random_state = i)
        true_time = time.time()
        true_cl = KMeans(n_clusters = tests[i][1], random_state = 4).fit_predict(x)
        true_time = time.time() - true_time
        xc, yc = clusters_centers(x, true_cl)
        true_cl, _, _ = clusters_sorted(true_cl, xc, yc)
        for strat in strats:
            t = time.time()
            cl, xc, yc = KMeans4(x, k = tests[i][1], strat = strat)
            t = time.time() - t
            cl, _, _ = clusters_sorted(cl, xc, yc)
            times[strat][i] = t / true_time
            overl = list(true_cl == cl).count(True)
            accurs[strat][i] = overl / tests[i][0]
    
    if gen_tests:
        pylab.figure(0, figsize = (15, 10))
        pylab.subplot(211)
        for i in range(len(strats)):
            plt.plot(range(n_tests), times[strats[i]], label = msg[i])
        plt.legend(loc=(1, 0.4))
        plt.grid(lw=2)
        plt.xlabel("Номера тестов")
        plt.ylabel("Затраченное время")
        pylab.subplot(212)
        for i in range(len(strats)):
            plt.plot(range(n_tests), accurs[strats[i]], label = msg[i])
        plt.legend(loc=(1, 0.4))
        plt.grid(lw=2)
        plt.xlabel("Номера тестов")
        plt.ylabel("Точность")
        plt.show()
    
    return times, accurs
    
    
def show_best_K(k, n, n_tests, strat_mean = False):
    for i in range(n_tests):
        x, _ = make_blobs(n_samples = n, n_features = 2, centers = k, random_state = i)
        best_K(x, show_work = 0, strat_mean = strat_mean)
    plt.xlabel("Количество кластеров")
    if strat_mean:
        plt.ylabel("Среднее внутрикластерное расстояние")
    else:
        plt.ylabel("Отношение минимального внешнекластерного и\nмаксимального внутрикластерного расстояний")
    plt.axvline(k)
 
                   
def show_strats(n, k_for_means, k_for_blobs, after_dot = 7):
    x, _ = make_blobs(n_samples = n, n_features = 2, centers = k_for_blobs, random_state = 4)
    xtext = x[:, 0].min() + 1
    ytext = x[:, 1].max() - 1
    pylab.figure(0, figsize = (15, 10))
    for i in range(len(strats)):
        t = time.time()
        st, xc, yc = KMeans4(x, k = k_for_means, strat = strats[i])
        t = time.time() - t
        st, _, _ = clusters_sorted(st, xc, yc)
        pylab.subplot(2, 2, i + 1)
        plt.scatter(x[:, 0], x[:, 1], c = st)
        pylab.text(xtext, ytext, msg[i] + "Время: " + str(t)[:after_dot])
    plt.show()
    

def test_eps(tests, epss, strat = "plusplus", strat_mean = True):
    epss = np.asarray(epss)
    num_tests = tests.shape[0]
    num_eps = epss.shape[0]
    accurs = np.zeros(num_eps)
    for i in range(num_eps):
        accurs[i] = 0
        for j in range(num_tests):
            x, _ = make_blobs(n_samples = tests[j][0], n_features = 2, centers = tests[j][1], random_state = 4)
            k, _ = best_K(x, eps = epss[i], strat = strat, strat_mean = strat_mean)
            if k == tests[j][1]:
                accurs[i] += 1
    accurs /= num_tests
    plt.plot(epss, accurs, label = "Среднее n = " + "%.0f"%np.mean(tests[:, 0]))
    plt.scatter(0., 1., c = "w")
    plt.xlabel("eps")
    plt.ylabel("Точность предположения")
    plt.legend(loc=(0.1, 0.2))
    return epss[filter_by_clust(accurs, accurs.max(), num_eps)]


def test_time_once(n, k):
    x, _ = make_blobs(n_samples = n, n_features = 2, centers = k, random_state = 4)
    t = time.time()
    KMeans4(x, k = k, strat = "uniform")
    t = time.time() - t
    return t


def test_time_mean(n, k, times):
    s = 0
    for i in range(times):
        s += test_time_once(n, k)
    return s / times


def test_time(n_start = 100, n_end = 1000, n_tests = 10, k_start = 2, k_end = 4, times = 10):
    k_tests = k_end - k_start + 1
    n = np.linspace(n_start, n_end, n_tests, dtype = np.int32)
    k = np.linspace(k_start, k_end, k_tests, dtype = np.int32)
    tests = np.zeros(n_tests * k_tests).reshape(n_tests, k_tests)
    for i in range(n_tests):
        for j in range(k_tests):
            tests[i][j] = test_time_mean(n[i], k[j], times)
    k, n = np.meshgrid(k, n)
    
    ax = plt.axes(projection='3d')
    pylab.xlabel = ("Количество точек")
    pylab.ylabel = ("Количество кластеров")
    ax.plot_surface(k, n, tests, cmap = "gist_rainbow")
    
    
def test_time_best_K(n_start = 100, n_end = 1000, n_tests = 10, times = 10):
    n = np.linspace(n_start, n_end, n_tests, dtype = np.int32)
    tests = np.zeros(n_tests)
    for i in range(n_tests):
        for j in range(times):
            x, _ = make_blobs(n_samples = n[i], n_features = 2, centers = 4, random_state = j)
            t = time.time()
            best_K(x, strat_mean = False, strat = "uniform")
            tests[i] += time.time() - t
    tests /= times
    plt.plot(n, tests, c="r")
    plt.xlabel("Количество точек")
    plt.ylabel("Продолжительность работы функции")
    

def make_matr_dist(matr):
    length = matr.shape[0]
    matr_dist = np.zeros(length * length).reshape(length, length)
    for i in range(length):
        matr_dist[i] = dist(matr[i][0], matr[i][1], matr[:, 0], matr[:, 1])
    return matr_dist