import torch
import numpy as np


class Perceptron:
    def __init__(self, d):
        self.d = d
        self.w = torch.randn(d, 1) / np.sqrt(d)
        self.b = torch.zeros(1)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        pre = torch.matmul(self.w.T, x) + self.b
        pred = torch.where(pre >= 0, torch.ones_like(pre), torch.zeros_like(pre))
        return pred.squeeze()

    def backward(self, x, yp, yt, lr):
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        delta = yp - yt
        self.w -= lr * delta * x
        self.b -= lr * delta

    def get_angle(self):
        if self.d != 2:
            raise ValueError("2D only")
        return (torch.atan(self.w[1, 0] / self.w[0, 0]) + np.pi / 2).item()

    def get_line(self, xr):
        if self.d != 2:
            raise ValueError("2D only")
        w1, w2 = self.w[0, 0].item(), self.w[1, 0].item()
        b = self.b.item()
        x1 = torch.linspace(xr[0], xr[1], 100)
        x2 = -(w1 * x1 + b) / w2
        return x1.numpy(), x2.numpy()

    def acc(self, X, y):
        c = 0
        for i in range(len(X)):
            if self.forward(X[i]).item() == y[i].item():
                c += 1
        return c / len(X)


if __name__ == "__main__":
    import os
    from datasets import create_linear_dataset, create_xor_dataset
    from training_utils import train_perceptron
    from visualization import plot_dataset, plot_training_curve, plot_decision_boundary_perceptron

    os.makedirs('results', exist_ok=True)

    Xl, yl = create_linear_dataset()
    Xx, yx = create_xor_dataset()

    plot_dataset(Xl, yl, "Linear Dataset", save_path='results/part1_linear_dataset.png')
    plot_dataset(Xx, yx, "XOR Dataset", save_path='results/part1_xor_dataset.png')

    pl = Perceptron(d=2)
    al, angl = train_perceptron(pl, Xl, yl, epochs=100, lr=0.001, verbose=False)

    px = Perceptron(d=2)
    ax, angx = train_perceptron(px, Xx, yx, epochs=100, lr=0.1, verbose=False)

    plot_training_curve(al, angl, "Perceptron - Linear", save_path='results/part1_linear_training.png')
    plot_training_curve(ax, angx, "Perceptron - XOR", save_path='results/part1_xor_training.png')

    plot_decision_boundary_perceptron(Xl, yl, pl, "Perceptron Boundary - Linear", save_path='results/part1_linear_boundary.png')
    plot_decision_boundary_perceptron(Xx, yx, px, "Perceptron Boundary - XOR", save_path='results/part1_xor_boundary.png')

    print(f"Linear: {al[-1]:.2%} | XOR: {ax[-1]:.2%}")
