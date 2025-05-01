# general imports
import argparse
import gc
import coloredlogs
import logging
from pathlib import Path
import time
import json
import gymnasium
import torch
import numpy as np
import random

# benchmark imports
from gym_gz_ws.gym_gz.envs.gym_gz_env import GymGzEnv, Timestamp
from stable_baselines3 import PPO
from gym_gz_ws.gym_gz.envs.ur5e_env import UR5eEnv
from stable_baselines3.common.callbacks import (
    CallbackList,
    StopTrainingOnMaxEpisodes,
    BaseCallback,
)


logger = logging.getLogger(__name__)
coloredlogs.install(
    fmt="%(asctime)s %(levelname)-6s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# CLI arguments
parser = argparse.ArgumentParser(
    description="Run the benchmark",
)
parser.add_argument(
    "--result-dir",
    help="Path to the target directory for the results",
    default="benchmark-results",
)
parser.add_argument(
    "--repetitions", help="How often the benchmark is repeated", type=int, required=True
)
parser.add_argument(
    "--seed", help="What seed to use for everything", type=int, required=True
)
parser.add_argument("--gui", help="Run benchmark with GUI", action="store_true")
parser.add_argument("--verbose", help="Print verbose output", action="store_true")

args = parser.parse_args()


# BENCHMARK
class TimestampModelUpdateCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_rollout_end(self) -> None:
        # runs before policy is updated
        self.training_env.env_method("measure", Timestamp.MODEL_UPDATE)


def benchmark_routine(
    episodes: int,
    max_steps_per_episode: int,
    perform_measurements: bool,
    seed: int,
    gui: bool,
    run_result_dir: Path,
    verbose: bool = False,
):
    ### start (setup): training run
    start_setup_timestamp = time.time_ns()  # epoch time

    env = GymGzEnv(
        max_steps_per_episode=max_steps_per_episode,
        result_dir=run_result_dir,
        perform_measurements=perform_measurements,
        seed=seed,
        gui=gui,
    )
    # limits episode length
    env = gymnasium.wrappers.TimeLimit(env, max_steps_per_episode)

    # limits amount of episodes
    callbacks = CallbackList(
        [
            TimestampModelUpdateCallback(
                verbose=int(verbose)
            ),  # sets timestamp when model update starts
            StopTrainingOnMaxEpisodes(
                episodes, verbose=int(verbose)
            ),  # limits amount of episodes
        ]
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=int(verbose),
        seed=seed,
        device="cpu",
        batch_size=256,
        n_steps=256,
    )
    model.set_random_seed(seed)
    # (also done before benchmark routine is called, but doesn't
    #  hurt if it's set multiple times)

    ### start (training): training run
    start_training_timestamp_sim = env.unwrapped.get_sim_timestamp(
        timeout_sec=5.0
    )  # always call this first, might take a bit
    start_training_timestamp_real = time.time_ns()

    model.learn(
        total_timesteps=int(
            1e12
        ),  # effectively infinite steps, training is stopped by env wrapper and callback
        callback=callbacks,
    )

    stop_training_timestamp_sim = env.get_sim_timestamp(
        timeout_sec=5.0
    )  # always call this first, might take a bit

    stop_training_timestamp_real = time.time_ns()
    ### end: training run

    # save data
    logger.debug("Training run finished, closing env.")
    env.close()  # saves measurements

    with open(
        run_result_dir.joinpath("start_stop_timestamps.json"), "w"
    ) as file:  # save start / stop timestamps
        json.dump(
            {
                "start_setup": start_setup_timestamp,
                "start_training_real": start_training_timestamp_real,
                "start_training_sim": start_training_timestamp_sim,
                "stop_training_real": stop_training_timestamp_real,
                "stop_training_sim": stop_training_timestamp_sim,
            },
            file,
        )


# settings
EPISODES = 15
MAX_STEPS_PER_EPISODE = 500

SLEEP_AFTER_RUN_SEC = 120

SEED = args.seed
GUI = args.gui
REPETITIONS = args.repetitions

VERBOSE = args.verbose
if VERBOSE:
    coloredlogs.set_level(logging.DEBUG)

RESULT_DIR = Path(args.result_dir).resolve()
RESULT_DIR.mkdir(parents=True, exist_ok=True)


with open(RESULT_DIR.joinpath("benchmark-settings.json"), "w") as file:
    json.dump(
        {
            "episodes": EPISODES,
            "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
            "seed": SEED,
            "gui": GUI,
            "repetitions": REPETITIONS,
        },
        file,
    )

# reproducibility
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(mode=True)
np.random.seed(SEED)
random.seed(SEED)
# also, seed for PPO is set

# first iterations WITH measurement
logger.info(
    f"Starting benchmark with {REPETITIONS} repetitions and WITH simulation measurements..."
)

for run in range(1, REPETITIONS + 1):
    run_result_dir = RESULT_DIR.joinpath(f"m-true_run-{run:02}")
    run_result_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Start run {run}")
    benchmark_routine(
        episodes=EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        perform_measurements=True,
        seed=SEED,
        gui=GUI,
        run_result_dir=run_result_dir,
    )

    logger.info(f"Finished run {run}.")

    logger.info("Running garbage collection...")
    gc.collect()
    logger.info("Garbage collection finished.")

    logger.info(f"Sleeping for {SLEEP_AFTER_RUN_SEC} seconds..")
    time.sleep(SLEEP_AFTER_RUN_SEC)
    logger.info("Finished sleeping")

logger.info("Finished benchmark with WITH simulation measurements.")

# second, iterations WITHOUT measurement
logger.info(
    f"Starting benchmark with {REPETITIONS} repetitions and WITHOUT simulation measurements..."
)

for run in range(1, REPETITIONS + 1):
    run_result_dir = RESULT_DIR.joinpath(f"m-false_run-{run:02}")
    run_result_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Start run {run}")
    benchmark_routine(
        episodes=EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        perform_measurements=False,
        seed=SEED,
        gui=GUI,
        run_result_dir=run_result_dir,
    )

    logger.info(f"Finished run {run}.")

    if run == REPETITIONS:
        break # don't need to sleep, benchmark is completely finished

    logger.info("Running garbage collection...")
    gc.collect()
    logger.info("Garbage collection finished.")

    logger.info(f"Sleeping for {SLEEP_AFTER_RUN_SEC} seconds..")
    time.sleep(SLEEP_AFTER_RUN_SEC)
    logger.info("Finished sleeping")


logger.info("Finished benchmark with WITHOUT simulation measurements.")
logger.info("Benchmark finished successfully!")
