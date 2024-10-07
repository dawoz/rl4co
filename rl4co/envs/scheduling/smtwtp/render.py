import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm, colormaps
from tensordict.tensordict import TensorDict

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td, actions=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(min(20, len(actions) // 2), min(15, len(actions)//3)))
    
    current_time = td['job_release_time'][actions[0]].item()
    times = [current_time]
    
    job_weight_range = [torch.min(td['job_weight']), torch.max(td['job_weight'])]
    
    for a in actions:        
        ax.barh(y=a, width=td['job_process_time'][a], left=current_time, height=1, color=plt.cm.rainbow(a/len(actions)), edgecolor='k')
        
        # normalized_weight = (td['job_weight'][a] - job_weight_range[0]) / (job_weight_range[1] - job_weight_range[0]) if tardy else 0
        # todo print weight        
        
        ax.plot(td['job_due_time'][a], a, color='r', marker='s', alpha=1, linewidth=3)
        # ax.plot([td['job_due_time'][a], current_time + td['job_process_time'][a].item()], [a, a], color='r', alpha=0.5, linestyle='dashed', linewidth=1)
        times.append(td['job_due_time'][a].item())
        
        if not (td['job_release_time'] == 0).all():
            ax.plot(td['job_release_time'][a], a, color='g', marker='>', alpha=1, linewidth=3)
            # ax.plot([td['job_release_time'][a], current_time], [a, a], color='g', alpha=0.5, linestyle='dashed', linewidth=1)
            times.append(td['job_release_time'][a].item())
        
        current_time += td['job_process_time'][a].item()
        times.append(current_time)

    for a in actions:
        ax.plot([0, max(times)], [a, a], color='gray', alpha=0.5, linewidth=0.5)
      
    ax.set_yticks(list(range(1,len(actions)+1)), labels=[f'Job {j+1}' for j in range(len(td['job_process_time'])-1)])
    # ax.set_xticks(times, labels=[f'{t:.1f}' for t in times], rotation=90)
    ax.set_xlim(0, max(times))
    ax.set_xlabel("Time")
    
    # shift y labels to the right side
    ax.yaxis.tick_right()
    
    ax.set_ylim(0.5, len(td['job_process_time'])-0.5)
    
    xticks = np.linspace(0, np.floor(max(times)), 10).tolist() + [max(times)]
    
    # show the last xticklabel as well. Make it visible but dont change the xticks
    ax.set_xticks(xticks, labels=[f'{t:.1f}' for t in xticks])
    ax.grid(axis='x', alpha=0.5, linestyle='dotted')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    
    return ax