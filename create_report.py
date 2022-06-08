import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.interpolate import CubicSpline

all_runs = sorted(glob("runs/*/", recursive = True))

epochs = 20
x = np.arange(0, 20, step = 1)

def read_run(run):
    event_acc = EventAccumulator(run)
    event_acc.Reload()
    w_times, steps, train_loss = zip(*event_acc.Scalars("Loss/Train"))
    w_times, steps, val_loss = zip(*event_acc.Scalars("Loss/Valid"))
    w_times, steps, train_score = zip(*event_acc.Scalars("Score/Train"))
    w_times, steps, val_score = zip(*event_acc.Scalars("Score/Valid"))

    return {"loss": [train_loss, val_loss], "score": [train_score, val_score]}

def plot_scenario(scenario, plot_name):
    fig1, ax1 = plt.subplots(figsize = (6, 4))
    fig2, ax2 = plt.subplots(figsize = (6, 4))
    
    ax1.set_title("Training loss")
    ax2.set_title("Validation loss")
    
    ax1.set_xticks(range(0, 20, 2))
    ax2.set_xticks(range(0, 20, 2))
    
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    for run in scenario:
        data = read_run(run)
        train_loss = CubicSpline(x, data["loss"][0])
        val_loss = CubicSpline(x, data["loss"][1])
        ax1.plot(x, train_loss(x), label = run)
        ax2.plot(x, val_loss(x), label = run)
    
    ax1.legend(loc = "upper right", bbox_to_anchor = (2.5, 1))
    ax2.legend(loc = "upper right", bbox_to_anchor = (2.5, 1))
    
    fig1.savefig("figures/" + plot_name + "_train_loss.png", bbox_inches = "tight")
    fig2.savefig("figures/" + plot_name + "_val_loss.png", bbox_inches = "tight")

optimizer_runs = [[run for run in all_runs if "train=0.7" in run and "test=0.1" in run and "lr=0.001" in run],
                  [run for run in all_runs if "train=0.7" in run and "test=0.1" in run and "lr=0.0001" in run],
                  [run for run in all_runs if "train=0.7" in run and "test=0.1" in run and "lr=1e-05" in run]]

for i in range(len(optimizer_runs)):
    plot_scenario(optimizer_runs[i], "optimizer_comparison_" + str(i))
    
    
    
    