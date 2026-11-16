import torch
import numpy as np


class TwoLayerNN:
    def __init__(self, input_dim, hidden_dim):
        self.d = input_dim
        self.h = hidden_dim
        self.w1 = torch.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
        self.b1 = torch.zeros(hidden_dim, 1)
        self.w2 = torch.randn(hidden_dim, 1) / np.sqrt(hidden_dim)
        self.b2 = torch.zeros(1)
        self.x = None
        self.p1 = None
        self.o1 = None
        self.p2 = None
        self.yp = None

    def sig(self, x):
        return 1 / (1 + torch.exp(-torch.clamp(x, -100, 100)))

    def dsig(self, x):
        s = self.sig(x)
        return s * (1 - s)

    def forward(self, x):
        self.x = x.view(-1, 1)
        self.p1 = torch.matmul(self.w1, self.x) + self.b1
        self.o1 = self.sig(self.p1)
        self.p2 = torch.matmul(self.w2.T, self.o1) + self.b2
        self.yp = self.sig(self.p2)
        return self.yp.squeeze()

    def backward(self, yt, lr):
        if not isinstance(yt, torch.Tensor):
            yt = torch.tensor([[yt]], dtype=torch.float32)
        elif len(yt.shape) == 0:
            yt = yt.view(1, 1)
        elif len(yt.shape) == 1:
            yt = yt.view(1, 1)

        dy = self.yp - yt.item()
        dp2 = dy * self.dsig(self.p2)
        db2 = dp2
        dw2 = dp2 * self.o1
        do1 = dp2 * self.w2
        dp1 = do1 * self.dsig(self.p1)
        db1 = dp1
        dw1 = torch.matmul(dp1, self.x.T)

        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2.squeeze()

    def predict(self, x):
        return 1.0 if self.forward(x) >= 0.5 else 0.0

    def acc(self, X, y):
        c = 0
        for i in range(len(X)):
            yp = self.predict(X[i])
            yt = y[i].item() if isinstance(y[i], torch.Tensor) else y[i]
            if yp == yt:
                c += 1
        return c / len(X)

    def grid(self, r1, r2, res=20):
        if self.d != 2:
            raise ValueError("2D only")
        x1 = torch.linspace(r1[0], r1[1], res)
        x2 = torch.linspace(r2[0], r2[1], res)
        g1, g2 = torch.meshgrid(x1, x2, indexing='xy')
        preds = torch.zeros_like(g1)
        for i in range(res):
            for j in range(res):
                x = torch.tensor([g1[i, j], g2[i, j]])
                preds[i, j] = self.predict(x)
        return g1.numpy(), g2.numpy(), preds.numpy()


if __name__ == "__main__":
    import os
    from datasets import create_linear_dataset, create_xor_dataset
    from training_utils import train_multiple_instances, find_best_model
    from visualization import plot_training_curve, plot_decision_boundary_nn

    os.makedirs('results', exist_ok=True)

    Xl, yl = create_linear_dataset()
    Xx, yx = create_xor_dataset()

    ml, al, fl = train_multiple_instances(TwoLayerNN, Xl, yl, num_instances=10, epochs=1000, lr=0.1, input_dim=2, hidden_dim=2, verbose=False)
    mx, ax, fx = train_multiple_instances(TwoLayerNN, Xx, yx, num_instances=10, epochs=1000, lr=0.1, input_dim=2, hidden_dim=2, verbose=False)

    bml, bal, _ = find_best_model(ml, al)
    bmx, bax, _ = find_best_model(mx, ax)

    plot_training_curve(bal, title="Two-Layer NN - Linear", save_path='results/part2_linear_training.png')
    plot_training_curve(bax, title="Two-Layer NN - XOR", save_path='results/part2_xor_training.png')

    plot_decision_boundary_nn(Xl, yl, bml, "NN Boundary - Linear", save_path='results/part2_linear_boundary.png')
    plot_decision_boundary_nn(Xx, yx, bmx, "NN Boundary - XOR", save_path='results/part2_xor_boundary.png')

    pl = sum(a == 1.0 for a in fl)
    px = sum(a == 1.0 for a in fx)
    print(f"Linear: {pl}/10 | XOR: {px}/10")
