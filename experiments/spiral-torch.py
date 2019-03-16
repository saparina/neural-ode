import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from neuralode.adjoint import *
import numpy as np


def to_np(x):
    return x.detach().cpu().numpy()


class LinearOdeTrained(OdeWithGrad):
    def __init__(self):
        super(LinearOdeTrained, self).__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.fc2 = nn.Linear(2, 2, bias=True)

    def forward(self, x, t):
        return self.fc2(self.fc1(x))


class LinearODE(OdeWithGrad):
    def __init__(self, W):
        super(LinearODE, self).__init__()
        self.fc1 = nn.Linear(2, 2, bias=False)
        self.fc1.weight = nn.Parameter(W)

    def forward(self, x, t):
        return self.fc1(x)


def plot_traj(obs=None, times=None, trajs=None, figsize=(16, 8), true_traj=None):
    plt.figure(figsize=figsize)
    if obs is not None:
        if times is None:
            times = [None] * len(obs)
        for o, t in zip(obs, times):
            o, t = to_np(o), to_np(t).squeeze()
            for b_i in range(o.shape[1]):
                plt.scatter(o[:, b_i, 0], o[:, b_i, 1], c=t)
    if trajs is not None:
        for z in trajs:
            z = to_np(z)
            plt.plot(z[:, 0, 0], z[:, 0, 1], lw=1.5)
    if true_traj is not None:
        plt.plot(true_traj[:,0, 0], true_traj[:,0, 1])

    plt.show()

W = torch.tensor([[-0.1, -1.], [1., -0.1]])
ode_trained = NeuralODE(LinearOdeTrained())
ode_true = NeuralODE(LinearODE(W))


z0 = torch.tensor([[0.6, 0.3]], requires_grad=True, dtype=torch.float32)
plot_freq = 50
t_max = 6.29*3
n_points = 200
n_steps = 2000
index_np = np.arange(0, n_points, 1, dtype=np.int)
index_np = np.hstack([index_np[:, None]])
times_np = np.linspace(0, t_max, num=n_points)
times_np = np.hstack([times_np[:, None]])

times = torch.from_numpy(times_np[:, :, None]).to(z0)
obs = ode_true(z0, times, sequence=True).detach()
obs = obs + torch.randn_like(obs) * 0.01

min_delta_time = 1.0
max_delta_time = 5.0
max_points_num = 32


def create_batch():
    t0 = np.random.uniform(0, t_max - max_delta_time)
    t1 = t0 + np.random.uniform(min_delta_time, max_delta_time)

    idx = sorted(np.random.permutation(index_np[(times_np > t0) & (times_np < t1)])[:max_points_num])

    obs_ = obs[idx]
    ts_ = times[idx]
    return obs_, ts_


# Train Neural ODE
optimizer = torch.optim.Adam(ode_trained.parameters(), lr=0.01)
for i in range(n_steps):
    obs_, ts_ = create_batch()

    z_ = ode_trained(obs_[0], ts_, sequence=True)
    loss = F.mse_loss(z_, obs_.detach())
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    if i % plot_freq == 0:
        z_p = ode_trained(z0, times, sequence=True)

z_p = ode_trained(z0, times, sequence=True)
plot_traj(obs=[obs], times=[times], trajs=[z_p])
# exit()