import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import wandb
import torch
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from tqdm import trange

from agilerl.algorithms.core.base import RLAlgorithm
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.utils.utils import (
    init_wandb,
    save_population_checkpoint,
    tournament_selection_and_mutation,
)

InitDictType = Optional[Dict[str, Any]]
PopulationType = List[RLAlgorithm]


def train_on_policy(
    env: gym.Env,
    env_name: str,
    algo: str,
    pop: PopulationType,
    INIT_HP: InitDictType = None,
    MUT_P: InitDictType = None,
    swap_channels: bool = False,
    max_steps: int = 1000000,
    evo_steps: int = 10000,
    eval_steps: Optional[int] = None,
    eval_loop: int = 1,
    target: Optional[float] = None,
    tournament: Optional[TournamentSelection] = None,
    mutation: Optional[Mutations] = None,
    checkpoint: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    overwrite_checkpoints: bool = False,
    save_elite: bool = False,
    elite_path: Optional[str] = None,
    wb: bool = False,
    verbose: bool = True,
    accelerator: Optional[Accelerator] = None,
    wandb_api_key: Optional[str] = None,
) -> Tuple[PopulationType, List[List[float]]]:
    """The general on-policy RL training function. Returns trained population of agents
    and their fitnesses.

    :param env: The environment to train in. Can be vectorized.
    :type env: Gym-style environment
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: list[RLAlgorithm]
    :param INIT_HP: Dictionary containing initial hyperparameters, defaults to None
    :type INIT_HP: dict, optional
    :param MUT_P: Dictionary containing mutation parameters, defaults to None
    :type MUT_P: dict, optional
    :param swap_channels: Swap image channels dimension from last to first
        [H, W, C] -> [C, H, W], defaults to False
    :type swap_channels: bool, optional
    :param max_steps: Maximum number of steps in environment, defaults to 1000000
    :type max_steps: int, optional
    :param evo_steps: Evolution frequency (steps), defaults to 10000
    :type evo_steps: int, optional
    :param eval_steps: Number of evaluation steps per episode. If None, will evaluate until
        environment terminates or truncates. Defaults to None
    :type eval_steps: int, optional
    :param eval_loop: Number of evaluation episodes, defaults to 1
    :type eval_loop: int, optional
    :param target: Target score for early stopping, defaults to None
    :type target: float, optional
    :param tournament: Tournament selection object, defaults to None
    :type tournament: object, optional
    :param mutation: Mutation object, defaults to None
    :type mutation: object, optional
    :param checkpoint: Checkpoint frequency (steps), defaults to None
    :type checkpoint: int, optional
    :param checkpoint_path: Location to save checkpoint, defaults to None
    :type checkpoint_path: str, optional
    :param overwrite_checkpoints: Overwrite previous checkpoints during training, defaults to False
    :type overwrite_checkpoints: bool, optional
    :param save_elite: Boolean flag indicating whether to save elite member at the end
        of training, defaults to False
    :type save_elite: bool, optional
    :param elite_path: Location to save elite agent, defaults to None
    :type elite_path: str, optional
    :param wb: Weights & Biases tracking, defaults to False
    :type wb: bool, optional
    :param verbose: Display training stats, defaults to True
    :type verbose: bool, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wandb_api_key: API key for Weights & Biases, defaults to None
    :type wandb_api_key: str, optional

    :return: Trained population of agents and their fitnesses
    :rtype: list[RLAlgorithm], list[list[float]]
    """
    assert isinstance(
        algo, str
    ), "'algo' must be the name of the algorithm as a string."
    assert isinstance(max_steps, int), "Number of steps must be an integer."
    assert isinstance(evo_steps, int), "Evolution frequency must be an integer."
    if target is not None:
        assert isinstance(
            target, (float, int)
        ), "Target score must be a float or an integer."
    if checkpoint is not None:
        assert isinstance(checkpoint, int), "Checkpoint must be an integer."
    assert isinstance(
        wb, bool
    ), "'wb' must be a boolean flag, indicating whether to record run with W&B"
    assert isinstance(verbose, bool), "Verbose must be a boolean."
    if save_elite is False and elite_path is not None:
        warnings.warn(
            "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
        )
    if checkpoint is None and checkpoint_path is not None:
        warnings.warn(
            "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
        )

    if wb:
        init_wandb(
            algo=algo,
            env_name=env_name,
            init_hyperparams=INIT_HP,
            mutation_hyperparams=MUT_P,
            wandb_api_key=wandb_api_key,
            accelerator=accelerator,
        )

    # Detect if environment is vectorised
    if hasattr(env, "num_envs"):
        is_vectorised = True
        num_envs = env.num_envs
    else:
        is_vectorised = False
        num_envs = 1

    save_path = (
        checkpoint_path.split(".pt")[0]
        if checkpoint_path is not None
        else "{}-EvoHPO-{}-{}".format(
            env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
        )
    )

    if accelerator is not None:
        print(f"\nDistributed training on {accelerator.device}...")
    else:
        print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    if accelerator is not None:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
            disable=not accelerator.is_local_main_process,
        )
    else:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
        )

    pop_loss = [[] for _ in pop]
    pop_fitnesses = []
    total_steps = 0
    loss = None
    checkpoint_count = 0

    # Pre-training mutation
    if accelerator is None and mutation is not None:
        pop = mutation.mutation(pop, pre_training_mut=True)

    # RL training loop
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        if accelerator is not None:
            accelerator.wait_for_everyone()
        pop_episode_scores = []
        pop_fps = []
        for agent_idx, agent in enumerate(pop):  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0
            start_time = time.time()
            for _ in range(-(evo_steps // -agent.learn_step)):

                states = []
                actions = []
                log_probs = []
                rewards = []
                dones = []
                values = []
                truncs = []

                learn_steps = 0
                for idx_step in range(-(agent.learn_step // -num_envs)):

                    if swap_channels:
                        state = obs_channels_to_first(state)

                    # Get next action from agent
                    action_mask = info.get("action_mask", None)
                    action, log_prob, _, value = agent.get_action(
                        state, action_mask=action_mask
                    )

                    if not is_vectorised:
                        action = action[0]
                        log_prob = log_prob[0]
                        value = value[0]

                    next_state, reward, done, trunc, info = env.step(
                        action
                    )  # Act in environment

                    total_steps += num_envs
                    steps += num_envs
                    learn_steps += num_envs

                    states.append(state)
                    actions.append(action)
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    dones.append(done)
                    values.append(value)
                    truncs.append(trunc)

                    state = next_state
                    scores += np.array(reward)

                    if not is_vectorised:
                        done = [done]
                        trunc = [trunc]

                    for idx, (d, t) in enumerate(zip(done, trunc)):
                        if d or t:
                            completed_episode_scores.append(scores[idx])
                            agent.scores.append(scores[idx])
                            scores[idx] = 0

                    pbar.update(num_envs)

                # pbar.update(learn_steps // len(pop))

                if swap_channels:
                    next_state = obs_channels_to_first(next_state)

                experiences = (
                    states,
                    actions,
                    log_probs,
                    rewards,
                    dones,
                    values,
                    next_state,
                )
                # Learn according to agent's RL algorithm
                loss = agent.learn(experiences)
                pop_loss[agent_idx].append(loss)

            agent.steps[-1] += steps
            fps = steps / (time.time() - start_time)
            pop_fps.append(fps)
            pop_episode_scores.append(completed_episode_scores)

        # Evaluate population
        fitnesses = [
            agent.test(
                env, swap_channels=swap_channels, max_steps=eval_steps, loop=eval_loop
            )
            for agent in pop
        ]
        pop_fitnesses.append(fitnesses)
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else "0 completed episodes"
            )
            for episode_scores in pop_episode_scores
        ]

        if wb:
            wandb_dict = {
                "global_step": (
                    total_steps * accelerator.state.num_processes
                    if accelerator is not None and accelerator.is_main_process
                    else total_steps
                ),
                "fps": np.mean(pop_fps),
                "train/mean_score": np.mean(
                    [
                        mean_score
                        for mean_score in mean_scores
                        if not isinstance(mean_score, str)
                    ]
                ),
                "eval/mean_fitness": np.mean(fitnesses),
                "eval/best_fitness": np.max(fitnesses),
            }

            agent_loss_dict = {
                f"train/agent_{index}_loss": np.mean(loss_[-10:])
                for index, loss_ in enumerate(pop_loss)
            }
            wandb_dict.update(agent_loss_dict)

            if accelerator is not None:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    wandb.log(wandb_dict)
                accelerator.wait_for_everyone()
            else:
                wandb.log(wandb_dict)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

        # Early stop if consistently reaches target
        if target is not None:
            if (
                np.all(
                    np.greater([np.mean(agent.fitness[-10:]) for agent in pop], target)
                )
                and len(pop[0].steps) >= 100
            ):
                if wb:
                    wandb.finish()
                return pop, pop_fitnesses

        # Tournament selection and population mutation
        if tournament and mutation is not None:
            pop = tournament_selection_and_mutation(
                population=pop,
                tournament=tournament,
                mutation=mutation,
                env_name=env_name,
                algo=algo,
                elite_path=elite_path,
                save_elite=save_elite,
                accelerator=accelerator,
            )

        if verbose:
            fitness = ["%.2f" % fitness for fitness in fitnesses]
            avg_fitness = ["%.2f" % np.mean(agent.fitness[-5:]) for agent in pop]
            avg_score = ["%.2f" % np.mean(agent.scores[-10:]) for agent in pop]
            agents = [agent.index for agent in pop]
            num_steps = [agent.steps[-1] for agent in pop]
            muts = [agent.mut for agent in pop]
            pbar.update(0)

            print(
                f"""
                --- Global Steps {total_steps} ---
                Fitness:\t\t{fitness}
                Score:\t\t{mean_scores}
                5 fitness avgs:\t{avg_fitness}
                10 score avgs:\t{avg_score}
                Agents:\t\t{agents}
                Steps:\t\t{num_steps}
                Mutations:\t\t{muts}
                """,
                end="\r",
            )

        # Save model checkpoint
        if checkpoint is not None:
            if pop[0].steps[-1] // checkpoint > checkpoint_count:
                save_population_checkpoint(
                    population=pop,
                    save_path=save_path,
                    overwrite_checkpoints=overwrite_checkpoints,
                    accelerator=accelerator,
                )
                checkpoint_count += 1

    if wb:
        if accelerator is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.finish()
            accelerator.wait_for_everyone()
        else:
            wandb.finish()

    pbar.close()
    return pop, pop_fitnesses


def train_on_policy_with_tensorboard(
    env: Any,
    env_name: str,
    algo: str,
    pop: List,
    INIT_HP: Dict = None,
    MUT_P: Dict = None,
    swap_channels: bool = False,
    max_steps: int = 1000000,
    evo_steps: int = 10000,
    eval_steps: Optional[int] = None,
    eval_loop: int = 1,
    target: Optional[float] = None,
    tournament: Optional[Any] = None,
    mutation: Optional[Any] = None,
    checkpoint: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    overwrite_checkpoints: bool = False,
    save_elite: bool = False,
    elite_path: Optional[str] = None,
    wb: bool = False,
    tensorboard: bool = False,  # New parameter to enable TensorBoard
    tensorboard_dir: Optional[str] = None,  # New parameter for TensorBoard log directory
    verbose: bool = True,
    accelerator: Optional[Accelerator] = None,
    wandb_api_key: Optional[str] = None,
) -> Tuple[List, List[List[float]]]:
    """The general on-policy RL training function. Returns trained population of agents
    and their fitnesses.

    :param env: The environment to train in. Can be vectorized.
    :type env: Gym-style environment
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: list[RLAlgorithm]
    :param INIT_HP: Dictionary containing initial hyperparameters, defaults to None
    :type INIT_HP: dict, optional
    :param MUT_P: Dictionary containing mutation parameters, defaults to None
    :type MUT_P: dict, optional
    :param swap_channels: Swap image channels dimension from last to first
        [H, W, C] -> [C, H, W], defaults to False
    :type swap_channels: bool, optional
    :param max_steps: Maximum number of steps in environment, defaults to 1000000
    :type max_steps: int, optional
    :param evo_steps: Evolution frequency (steps), defaults to 10000
    :type evo_steps: int, optional
    :param eval_steps: Number of evaluation steps per episode. If None, will evaluate until
        environment terminates or truncates. Defaults to None
    :type eval_steps: int, optional
    :param eval_loop: Number of evaluation episodes, defaults to 1
    :type eval_loop: int, optional
    :param target: Target score for early stopping, defaults to None
    :type target: float, optional
    :param tournament: Tournament selection object, defaults to None
    :type tournament: object, optional
    :param mutation: Mutation object, defaults to None
    :type mutation: object, optional
    :param checkpoint: Checkpoint frequency (steps), defaults to None
    :type checkpoint: int, optional
    :param checkpoint_path: Location to save checkpoint, defaults to None
    :type checkpoint_path: str, optional
    :param overwrite_checkpoints: Overwrite previous checkpoints during training, defaults to False
    :type overwrite_checkpoints: bool, optional
    :param save_elite: Boolean flag indicating whether to save elite member at the end
        of training, defaults to False
    :type save_elite: bool, optional
    :param elite_path: Location to save elite agent, defaults to None
    :type elite_path: str, optional
    :param wb: Weights & Biases tracking, defaults to False
    :type wb: bool, optional
    :param tensorboard: TensorBoard tracking, defaults to False
    :type tensorboard: bool, optional
    :param tensorboard_dir: Directory for TensorBoard logs, defaults to None
    :type tensorboard_dir: str, optional
    :param verbose: Display training stats, defaults to True
    :type verbose: bool, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    :param wandb_api_key: API key for Weights & Biases, defaults to None
    :type wandb_api_key: str, optional

    :return: Trained population of agents and their fitnesses
    :rtype: list[RLAlgorithm], list[list[float]]
    """
    assert isinstance(
        algo, str
    ), "'algo' must be the name of the algorithm as a string."
    assert isinstance(max_steps, int), "Number of steps must be an integer."
    assert isinstance(evo_steps, int), "Evolution frequency must be an integer."
    if target is not None:
        assert isinstance(
            target, (float, int)
        ), "Target score must be a float or an integer."
    if checkpoint is not None:
        assert isinstance(checkpoint, int), "Checkpoint must be an integer."
    assert isinstance(
        wb, bool
    ), "'wb' must be a boolean flag, indicating whether to record run with W&B"
    assert isinstance(
        tensorboard, bool
    ), "'tensorboard' must be a boolean flag, indicating whether to record run with TensorBoard"
    assert isinstance(verbose, bool), "Verbose must be a boolean."
    if save_elite is False and elite_path is not None:
        warnings.warn(
            "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True."
        )
    if checkpoint is None and checkpoint_path is not None:
        warnings.warn(
            "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined."
        )

    # Initialize W&B if enabled
    if wb:
        init_wandb(
            algo=algo,
            env_name=env_name,
            init_hyperparams=INIT_HP,
            mutation_hyperparams=MUT_P,
            wandb_api_key=wandb_api_key,
            accelerator=accelerator,
        )
    
    # Initialize TensorBoard if enabled
    tb_writer = None
    if tensorboard:
        if tensorboard_dir is None:
            tensorboard_dir = f"runs/{env_name}-{algo}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Only create writer on main process if using accelerator
        if accelerator is None or accelerator.is_main_process:
            tb_writer = SummaryWriter(log_dir=tensorboard_dir)
            
            # Log hyperparameters
            if INIT_HP is not None:
                for agent_idx, agent in enumerate(pop):
                    hparams = {}
                    # Add agent hyperparameters
                    for key, val in INIT_HP.items():
                        if isinstance(val, (int, float, str, bool)):
                            hparams[key] = val
                    
                    # Add agent-specific hyperparameters
                    for attr in dir(agent):
                        if not attr.startswith('_') and not callable(getattr(agent, attr)) and attr not in ['actor', 'critic', 'device', 'algo', 'index', 'scores', 'steps', 'fitness', 'mut']:
                            try:
                                value = getattr(agent, attr)
                                if isinstance(value, (int, float, str, bool)):
                                    hparams[attr] = value
                            except:
                                pass
                    
                    # Add mutation parameters if available
                    if MUT_P is not None:
                        for key, val in MUT_P.items():
                            if isinstance(val, (int, float, str, bool)):
                                hparams[f"mut_{key}"] = val
                    
                    # Log hyperparameters with a dummy metric
                    tb_writer.add_hparams(
                        hparams, 
                        {'hparam/agent_id': agent_idx}
                    )

    # Detect if environment is vectorised
    if hasattr(env, "num_envs"):
        is_vectorised = True
        num_envs = env.num_envs
    else:
        is_vectorised = False
        num_envs = 1

    save_path = (
        checkpoint_path.split(".pt")[0]
        if checkpoint_path is not None
        else "{}-EvoHPO-{}-{}".format(
            env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
        )
    )

    if accelerator is not None:
        print(f"\nDistributed training on {accelerator.device}...")
    else:
        print("\nTraining...")
        if tensorboard and tb_writer is not None:
            print(f"TensorBoard logs at: {tensorboard_dir}")
            print(f"To view: tensorboard --logdir={tensorboard_dir}")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    if accelerator is not None:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
            disable=not accelerator.is_local_main_process,
        )
    else:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
        )

    pop_loss = [[] for _ in pop]
    pop_fitnesses = []
    total_steps = 0
    loss = None
    checkpoint_count = 0

    # Pre-training mutation
    if accelerator is None and mutation is not None:
        pop = mutation.mutation(pop, pre_training_mut=True)

    # RL training loop
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        if accelerator is not None:
            accelerator.wait_for_everyone()
        pop_episode_scores = []
        pop_fps = []
        for agent_idx, agent in enumerate(pop):  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0
            start_time = time.time()
            for _ in range(-(evo_steps // -agent.learn_step)):

                states = []
                actions = []
                log_probs = []
                rewards = []
                dones = []
                values = []
                truncs = []

                learn_steps = 0
                for idx_step in range(-(agent.learn_step // -num_envs)):

                    if swap_channels:
                        state = obs_channels_to_first(state)

                    # Get next action from agent
                    action_mask = info.get("action_mask", None)
                    action, log_prob, _, value = agent.get_action(
                        state, action_mask=action_mask
                    )

                    if not is_vectorised:
                        action = action[0]
                        log_prob = log_prob[0]
                        value = value[0]

                    next_state, reward, done, trunc, info = env.step(
                        action
                    )  # Act in environment

                    total_steps += num_envs
                    steps += num_envs
                    learn_steps += num_envs

                    states.append(state)
                    actions.append(action)
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    dones.append(done)
                    values.append(value)
                    truncs.append(trunc)

                    state = next_state
                    scores += np.array(reward)

                    if not is_vectorised:
                        done = [done]
                        trunc = [trunc]

                    for idx, (d, t) in enumerate(zip(done, trunc)):
                        if d or t:
                            completed_episode_scores.append(scores[idx])
                            agent.scores.append(scores[idx])
                            scores[idx] = 0

                    pbar.update(num_envs)

                # pbar.update(learn_steps // len(pop))

                if swap_channels:
                    next_state = obs_channels_to_first(next_state)

                experiences = (
                    states,
                    actions,
                    log_probs,
                    rewards,
                    dones,
                    values,
                    next_state,
                )
                # Learn according to agent's RL algorithm
                loss = agent.learn(experiences)
                pop_loss[agent_idx].append(loss)
                
                # Log loss to TensorBoard
                if tensorboard and tb_writer is not None and len(pop_loss[agent_idx]) > 0:
                    if accelerator is None or accelerator.is_main_process:
                        tb_writer.add_scalar(
                            f'train/agent_{agent_idx}_loss', 
                            np.mean(pop_loss[agent_idx][-10:]), 
                            total_steps
                        )

            agent.steps[-1] += steps
            fps = steps / (time.time() - start_time)
            pop_fps.append(fps)
            pop_episode_scores.append(completed_episode_scores)
            
            # Log agent metrics to TensorBoard
            if tensorboard and tb_writer is not None:
                if accelerator is None or accelerator.is_main_process:
                    if len(completed_episode_scores) > 0:
                        tb_writer.add_scalar(
                            f'train/agent_{agent_idx}_score', 
                            np.mean(completed_episode_scores), 
                            total_steps
                        )
                    tb_writer.add_scalar(
                        f'train/agent_{agent_idx}_fps', 
                        fps, 
                        total_steps
                    )

        # Evaluate population
        fitnesses = [
            agent.test(
                env, swap_channels=swap_channels, max_steps=eval_steps, loop=eval_loop
            )
            for agent in pop
        ]
        pop_fitnesses.append(fitnesses)
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else "0 completed episodes"
            )
            for episode_scores in pop_episode_scores
        ]

        # Prepare metrics dict for logging
        metrics_dict = {
            "global_step": (
                total_steps * accelerator.state.num_processes
                if accelerator is not None and accelerator.is_main_process
                else total_steps
            ),
            "fps": np.mean(pop_fps),
            "train/mean_score": np.mean(
                [
                    mean_score
                    for mean_score in mean_scores
                    if not isinstance(mean_score, str)
                ]
            ),
            "eval/mean_fitness": np.mean(fitnesses),
            "eval/best_fitness": np.max(fitnesses),
        }

        agent_loss_dict = {
            f"train/agent_{index}_loss": np.mean(loss_[-10:])
            for index, loss_ in enumerate(pop_loss)
        }
        metrics_dict.update(agent_loss_dict)

        # Log to W&B if enabled
        if wb:
            if accelerator is not None:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    wandb.log(metrics_dict)
                accelerator.wait_for_everyone()
            else:
                wandb.log(metrics_dict)
        
        # Log to TensorBoard if enabled
        if tensorboard and tb_writer is not None:
            if accelerator is None or accelerator.is_main_process:
                for metric_name, metric_value in metrics_dict.items():
                    tb_writer.add_scalar(metric_name, metric_value, total_steps)
                
                # Additional TensorBoard specific metrics
                tb_writer.add_scalar('eval/min_fitness', np.min(fitnesses), total_steps)
                tb_writer.add_scalar('eval/fitness_std', np.std(fitnesses), total_steps)
                
                # Log histograms for actions and values if available
                if len(actions) > 0:
                    try:
                        actions_tensor = torch.cat([a.flatten() for a in actions]) if isinstance(actions[0], torch.Tensor) else torch.tensor(actions).flatten()
                        tb_writer.add_histogram('train/action_distribution', actions_tensor, total_steps)
                    except:
                        pass  # Skip if tensor operations fail
                
                if len(values) > 0:
                    try:
                        values_tensor = torch.cat([v.flatten() for v in values]) if isinstance(values[0], torch.Tensor) else torch.tensor(values).flatten()
                        tb_writer.add_histogram('train/value_distribution', values_tensor, total_steps)
                    except:
                        pass  # Skip if tensor operations fail
                
                # Log individual agent fitness
                for agent_idx, fitness in enumerate(fitnesses):
                    tb_writer.add_scalar(f'eval/agent_{agent_idx}_fitness', fitness, total_steps)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

        # Early stop if consistently reaches target
        if target is not None:
            if (
                np.all(
                    np.greater([np.mean(agent.fitness[-10:]) for agent in pop], target)
                )
                and len(pop[0].steps) >= 100
            ):
                if wb:
                    wandb.finish()
                if tensorboard and tb_writer is not None:
                    tb_writer.close()
                return pop, pop_fitnesses

        # Tournament selection and population mutation
        if tournament and mutation is not None:
            pop = tournament_selection_and_mutation(
                population=pop,
                tournament=tournament,
                mutation=mutation,
                env_name=env_name,
                algo=algo,
                elite_path=elite_path,
                save_elite=save_elite,
                accelerator=accelerator,
            )
            
            # Log mutation information to TensorBoard
            if tensorboard and tb_writer is not None:
                if accelerator is None or accelerator.is_main_process:
                    mutation_counts = {}
                    for agent in pop:
                        if hasattr(agent, 'mut'):
                            if agent.mut not in mutation_counts:
                                mutation_counts[agent.mut] = 0
                            mutation_counts[agent.mut] += 1
                    
                    for mut_name, count in mutation_counts.items():
                        tb_writer.add_scalar(f'mutation/{mut_name}', count, total_steps)

        if verbose:
            fitness = ["%.2f" % fitness for fitness in fitnesses]
            avg_fitness = ["%.2f" % np.mean(agent.fitness[-5:]) for agent in pop]
            avg_score = ["%.2f" % np.mean(agent.scores[-10:]) for agent in pop]
            agents = [agent.index for agent in pop]
            num_steps = [agent.steps[-1] for agent in pop]
            muts = [agent.mut for agent in pop]
            pbar.update(0)

            print(
                f"""
                --- Global Steps {total_steps} ---
                Fitness:\t\t{fitness}
                Score:\t\t{mean_scores}
                5 fitness avgs:\t{avg_fitness}
                10 score avgs:\t{avg_score}
                Agents:\t\t{agents}
                Steps:\t\t{num_steps}
                Mutations:\t\t{muts}
                """,
                end="\r",
            )

        # Save model checkpoint
        if checkpoint is not None:
            if pop[0].steps[-1] // checkpoint > checkpoint_count:
                save_population_checkpoint(
                    population=pop,
                    save_path=save_path,
                    overwrite_checkpoints=overwrite_checkpoints,
                    accelerator=accelerator,
                )
                checkpoint_count += 1

    if wb:
        if accelerator is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.finish()
            accelerator.wait_for_everyone()
        else:
            wandb.finish()
    
    # Close TensorBoard writer
    if tensorboard and tb_writer is not None:
        if accelerator is None or accelerator.is_main_process:
            tb_writer.close()

    pbar.close()
    return pop, pop_fitnesses