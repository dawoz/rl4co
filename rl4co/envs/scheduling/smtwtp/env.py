from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger

from .generator import SMTWTPGenerator
from .render import render

log = get_pylogger(__name__)


class SMTWTPEnv(RL4COEnvBase):
    """
    Single Machine Total Weighted Tardiness Problem environment as described in DeepACO (https://arxiv.org/pdf/2309.14032.pdf)
    SMTWTP is a scheduling problem in which a set of jobs must be processed on a single machine.
    Each job i has a processing time, a weight, and a due date. The objective is to minimize the sum of the weighted tardiness of all jobs,
    where the weighted tardiness of a job is defined as the product of its weight and the duration by which its completion time exceeds its due date.
    At each step, the agent chooses a job to process. The reward is 0 unless the agent processes all the jobs.
    In that case, the reward is (-)objective value of the processing order: maximizing the reward is equivalent to minimizing the objective.

    Observation:
        - job_due_time: the due time of each job
        - job_release_time: the release time of each job
        - job_weight: the weight of each job
        - job_process_time: the process time of each job
        - current_node: the current node
        - action_mask: a mask of available actions
        - current_time: the current time

    Constants:
        - num_job: number of jobs
        - min_time_span: lower bound of jobs' due time. By default, jobs' due time is uniformly sampled from (min_time_span, max_time_span)
        - max_time_span: upper bound of jobs' due time. By default, it will be set to num_job / 2
        - min_job_weight: lower bound of jobs' weights. By default, jobs' weights are uniformly sampled from (min_job_weight, max_job_weight)
        - max_job_weight: upper bound of jobs' weights
        - min_process_time: lower bound of jobs' process time. By default, jobs' process time is uniformly sampled from (min_process_time, max_process_time)
        - max_process_time: upper bound of jobs' process time
        - release_times: whether to include release times

    Finishing condition:
        - All jobs are processed
    
    Reward:
        - The reward is 0 unless the agent processes all the jobs.
        - In that case, the reward is (-)objective value of the processing order: maximizing the reward is equivalent to minimizing the objective.

    Args:
        generator: SMTWTPGenerator as insstance generator
        generator_params: parameters for the generator
    """

    name = "smtwtp"

    def __init__(
        self,
        generator: SMTWTPGenerator = None,
        generator_params: dict = {},
        feasibility_penalty: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = SMTWTPGenerator(**generator_params)
        self.generator = generator
        self.feasibility_penalty = feasibility_penalty
        self._make_spec(self.generator)

    @staticmethod
    def _step(td: TensorDict) -> TensorDict:
        current_job = td["action"]

        # Set not visited to 0 (i.e., we visited the node)
        available = td["action_mask"].scatter(
            -1, current_job.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )

        # Increase used time
        selected_process_time = td["job_process_time"][
            torch.arange(current_job.size(0)), current_job
        ]
        current_time = td["current_time"] + selected_process_time.unsqueeze(-1)

        # We are done there are no unvisited locations
        done = torch.count_nonzero(available, dim=-1) <= 0

        # The reward is calculated outside via get_reward for efficiency, so we set it to 0 here
        reward = torch.zeros_like(done)

        td.update(
            {
                "current_job": current_job,
                "current_time": current_time,
                "action_mask": available,
                "reward": reward,
                "done": done,
            }
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        device = td.device

        init_job_due_time = td["job_due_time"]
        init_job_release_time = td["job_release_time"]
        init_job_process_time = td["job_process_time"]
        init_job_weight = td["job_weight"]

        # Other variables
        current_job = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        current_time = torch.zeros((*batch_size, 1), dtype=torch.int64, device=device)
        available = torch.ones(
            (*batch_size, self.generator.num_job + 1), dtype=torch.bool, device=device
        )
        available[:, 0] = 0  # mask the starting dummy node

        return TensorDict(
            {
                "job_due_time": init_job_due_time,
                "job_release_time": init_job_release_time,
                "job_weight": init_job_weight,
                "job_process_time": init_job_process_time,
                "current_job": current_job,
                "current_time": current_time,
                "action_mask": available,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: SMTWTPGenerator) -> None:
        self.observation_spec = CompositeSpec(
            job_due_time=BoundedTensorSpec(
                low=generator.min_time_span,
                high=generator.max_time_span,
                shape=(generator.num_job + 1,),
                dtype=torch.float32,
            ),
            job_release_time=BoundedTensorSpec(
                low=generator.min_time_span,
                high=generator.max_time_span,
                shape=(generator.num_job + 1,),
                dtype=torch.float32,
            ),
            job_weight=BoundedTensorSpec(
                low=generator.min_job_weight,
                high=generator.max_job_weight,
                shape=(generator.num_job + 1,),
                dtype=torch.float32,
            ),
            job_process_time=BoundedTensorSpec(
                low=generator.min_process_time,
                high=generator.max_process_time,
                shape=(generator.num_job + 1,),
                dtype=torch.float32,
            ),
            current_node=UnboundedDiscreteTensorSpec(
                shape=(1,),
                dtype=torch.int64,
            ),
            action_mask=UnboundedDiscreteTensorSpec(
                shape=(generator.num_job + 1,),
                dtype=torch.bool,
            ),
            current_time=UnboundedContinuousTensorSpec(
                shape=(1,),
                dtype=torch.float32,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_job + 1,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = UnboundedDiscreteTensorSpec(shape=(1,), dtype=torch.bool)

    def _get_reward(self, td, actions) -> TensorDict:
        out = self.get_extra_metrics(td, actions)
        return out.get("reward")

    def check_solution_validity(self, td, actions):
        log.warning("Checking solution validity is not implemented for SMTWTP")
        pass

    @staticmethod
    def render(td, actions=None, ax=None):
        return render(td, actions, ax)
    
    def get_extra_metrics(self, td, actions):
        job_due_time = td["job_due_time"]
        job_release_time = td["job_release_time"]
        job_weight = td["job_weight"]
        job_process_time = td["job_process_time"]

        batch_idx = torch.arange(
            job_process_time.shape[0], device=job_process_time.device
        ).unsqueeze(1)

        ordered_process_time = job_process_time[batch_idx, actions]
        ordered_due_time = job_due_time[batch_idx, actions]
        ordered_release_time = job_release_time[batch_idx, actions]
        ordered_job_weight = job_weight[batch_idx, actions]
        presum_process_time = torch.cumsum(
            ordered_process_time, dim=1
        )  # ending time of each job
        job_tardiness = presum_process_time - ordered_due_time
        job_tardiness[job_tardiness < 0] = 0
        job_weighted_tardiness = ordered_job_weight * job_tardiness
        
        # penalize jobs that start before their release time
        ordered_start_time = presum_process_time - ordered_process_time
        ordered_start_time += ordered_release_time[:, 0, None]  # add release time of the first scheduled job
        job_too_early_time = (ordered_release_time - ordered_start_time).clamp(min=0)

        weghted_tardiness = job_weighted_tardiness.sum(-1)
        too_early_time = job_too_early_time.sum(-1)
        feasibility = (too_early_time == 0).float()
        reward = - weghted_tardiness - self.feasibility_penalty * too_early_time
        
        return {
            "reward": reward,
            "total_weighted_tardiness": weghted_tardiness,
            "too_early_time": too_early_time,
            "feasibility": feasibility
        }
        
