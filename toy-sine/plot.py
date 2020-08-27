import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_data_and_label(ax, data_all, label_all, domain=False):
    # data_all = (data_all - data_all.mean(0, keepdims=True)) / data_all.std(0, keepdims=True)

    cmap = matplotlib.cm.get_cmap('rainbow')
    num = int(np.max(label_all)) + 1
    l_color = [cmap(ele)[:3] for ele in np.linspace(0, 1, num)]

    for i in range(num):
        data_sub = data_all[label_all == i, :]
        if domain:
            ax.plot(data_sub[:, 0], data_sub[:, 1], '.', color=l_color[i % len(l_color)])
        else:
            ax.plot(data_sub[:, 0], data_sub[:, 1], ['x', '.'][i], color=['b', 'r'][i], alpha=0.5)


def plot_dataset(info):
    asp = 0.75
    fig, ax = plt.subplots(1, 2, figsize=(6 * 2, 6 * asp))
    data = info['data']
    label = info['label']
    domain = info['domain']
    plot_data_and_label(ax[0], data, domain, True)
    plot_data_and_label(ax[1], data, label, False)
    ax[0].set_title('Color encodes domain', fontsize=15)
    ax[1].set_title('Color encodes label', fontsize=15)
    plt.tight_layout(pad=0)
    plt.show()


def plot_data_with_boundry(data, label, opt):
    assert label.min() == 0 and label.max() == 1
    from sklearn import svm

    data = (data - data.mean(0, keepdims=True)) / data.std(0, keepdims=True)

    if opt.data == "circle":
        xlim = [-1.8, 1.8]
        ylim = [-2.5, 1.8]
    elif opt.data == "sine":
        xlim = [-1.77, 1.77]
        ylim = [-2.4, 2.4]

    asp = 0.75
    fig, ax = plt.subplots(1, 1, figsize=(6, 6 * asp))

    N = 300
    x = np.linspace(xlim[0], xlim[1], N)
    y = np.linspace(ylim[0], ylim[1], N)

    if opt.data == "sine":
        gamma = 3
        if opt.model in ['cua']:
            gamma = 30
        if opt.model in ['dann', 'zhao']:
            gamma = 15

        if opt.model in ['adda', 'mdd']:
            aug_data, aug_label = data, label

        else:
            num = 500
            xx = np.random.rand(num) * 4 - 2.0
            yy = np.random.rand(num) * 1.5 - 3.0

            dy = np.sin((xx + 1.71) / 1.72 * np.pi * 2 * 2)
            yy += dy

            aug_data_2 = np.concatenate([xx[:, None], yy[:, None]], 1)
            aug_label_2 = np.zeros(len(aug_data_2))

            xx = np.random.rand(num) * 4 - 2.0
            yy = np.random.rand(num) * 1.5 + 1.5
            yy += dy

            aug_data_3 = np.concatenate([xx[:, None], yy[:, None]], 1)
            aug_label_3 = np.ones(len(aug_data_3))

            aug_data = np.concatenate([data, aug_data_2, aug_data_3], 0)
            aug_label = np.concatenate([label, aug_label_2, aug_label_3], 0)

        svc = svm.SVC(kernel='rbf', gamma=gamma, C=1).fit(aug_data, aug_label)
        # data, label = aug_data, aug_label

    elif opt.data == "circle":

        if opt.model in ['cua', 'cida']:
            num = 100

            ang = np.random.rand(num) * np.pi
            mag = np.linspace(0, 1, num)
            xx = np.cos(ang) * mag
            yy = np.sin(ang) * mag
            yy = yy * 3 - 2.5
            aug_data_1 = np.concatenate([xx[:, None], yy[:, None]], 1)
            aug_label_1 = np.zeros(len(aug_data_1))

            xx = np.random.rand(num) * 0.5 - 2.0
            yy = np.random.rand(num) * 1 + 0.5
            aug_data_2 = np.concatenate([xx[:, None], yy[:, None]], 1)
            aug_label_2 = np.ones(len(aug_data_2))

            xx = np.random.rand(num) * 0.5 + 1.5
            yy = np.random.rand(num) * 1 + 0.5
            aug_data_3 = np.concatenate([xx[:, None], yy[:, None]], 1)
            aug_label_3 = np.ones(len(aug_data_3))

            aug_data = np.concatenate([data, aug_data_1, aug_data_2, aug_data_3], 0)
            aug_label = np.concatenate([label, aug_label_1, aug_label_2, aug_label_3], 0)

        else:
            aug_data, aug_label = data, label

        svc = svm.SVC(kernel='rbf', gamma=0.7, C=1).fit(aug_data, aug_label)
        if opt.model == 'cua':
            svc = svm.SVC(kernel='rbf', gamma=7, C=1).fit(aug_data, aug_label)

        # data, label = aug_data, aug_label


    X, Y = np.meshgrid(x, y)
    Z = svc.predict(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)

    if opt.gt == 0:
        plt.contour(X, Y, Z, levels=[0.5], colors=['k'], linewidths=4)

    for i in range(2):
        plt.plot(data[label == i, 0], data[label == i, 1], ['x', '.'][i], color=['b', 'r'][i], alpha=0.5)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout(pad=0)
