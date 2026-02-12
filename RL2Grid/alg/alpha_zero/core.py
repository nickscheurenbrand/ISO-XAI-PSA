from collections import deque
from time import time
import random

from .agent import AlphaZeroNetwork
from .config import get_alg_args
from common.checkpoint import AlphaZeroCheckpoint, CheckpointSaver
from common.imports import *
from common.logger import Logger
from env.eval import Evaluator


class AlphaZero:
    """AlphaZero-style policy/value training with self-play."""

    def __init__(self, env: gym.Env, run_name: str, start_time: float,
                 args: Dict[str, Any], ckpt: CheckpointSaver) -> None:
        if args.action_type != "topology":
            raise ValueError("AlphaZero currently supports only topology (discrete) environments.")
        if args.n_envs != 1:
            print("[AlphaZero] Ignoring n_envs>1 and using a single self-play environment.")
            args.n_envs = 1

        if not ckpt.resumed:
            args = ap.Namespace(**vars(args), **vars(get_alg_args()))

        device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")
        self.args = args
        self.device = device
        self.env = env
        self.run_name = run_name
        self.start_time = start_time

        self.network = AlphaZeroNetwork(env.observation_space, env.action_space)
        self.network.to(device)
        self.optimizer = optim.Adam(self.network.parameters(),
                                    lr=args.az_learning_rate,
                                    weight_decay=args.az_weight_decay)

        self.replay_buffer: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=args.az_replay_size)
        self.global_step = 0
        self.init_iteration = 1

        if ckpt.resumed:
            args = ckpt.loaded_run['args']
            self.args = args
            self.network.load_state_dict(ckpt.loaded_run['network'])
            self.optimizer.load_state_dict(ckpt.loaded_run['optimizer'])
            self.global_step = ckpt.loaded_run['global_step']
            self.init_iteration = ckpt.loaded_run.get('last_iteration', 1)
            stored_buffer = ckpt.loaded_run.get('replay_buffer', [])
            for sample in stored_buffer:
                self.replay_buffer.append((sample['state'], sample['policy'], sample['value']))

        logger = Logger(run_name, args) if args.track else None
        evaluator = Evaluator(args, logger, device)

        try:
            for iteration in range(self.init_iteration, args.az_total_iterations + 1):
                batch_data, steps = self._self_play()
                self.global_step += steps
                for sample in batch_data:
                    self.replay_buffer.append(sample)

                if len(self.replay_buffer) >= args.az_batch_size:
                    for _ in range(args.az_train_epochs):
                        self._train_step()

                if iteration % args.az_eval_freq == 0:
                    evaluator.evaluate(self.global_step, self.network)

                if args.verbose:
                    print(
                        f"[AlphaZero] Iteration {iteration} | Buffer {len(self.replay_buffer)} | "
                        f"Global step {self.global_step}"
                    )

                if (time() - start_time) / 60 >= args.time_limit:
                    break
        finally:
            ckpt.set_record(
                args,
                self.network,
                self.optimizer,
                self.global_step,
                self._buffer_snapshot(),
                "" if not logger else logger.wb_path,
                iteration
            )
            ckpt.save()
            if logger:
                logger.close()
            self.env.close()

    def _buffer_snapshot(self) -> List[Dict[str, Any]]:
        snapshot: List[Dict[str, Any]] = []
        for state, policy, value in list(self.replay_buffer):
            snapshot.append(
                {
                    'state': state.astype(np.float32),
                    'policy': policy.astype(np.float32),
                    'value': float(value)
                }
            )
        return snapshot

    def _self_play(self) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], int]:
        dataset: List[Tuple[np.ndarray, np.ndarray, float]] = []
        total_steps = 0
        for _ in range(self.args.az_self_play_games):
            obs, _ = self.env.reset()
            trajectory: List[Tuple[np.ndarray, np.ndarray]] = []
            rewards: List[float] = []
            done = False
            steps = 0
            while not done and steps < self.args.az_max_episode_steps:
                obs_tensor = th.tensor(obs, dtype=th.float32, device=self.device)
                with th.no_grad():
                    logits, _ = self.network.predict(obs_tensor)
                    policy = F.softmax(logits / self.args.az_temperature, dim=-1).cpu().numpy()[0]
                if self.args.az_dirichlet_epsilon > 0:
                    dirichlet = np.random.dirichlet([self.args.az_dirichlet_alpha] * len(policy))
                    policy = (1 - self.args.az_dirichlet_epsilon) * policy + self.args.az_dirichlet_epsilon * dirichlet
                policy = policy / np.sum(policy)
                action = np.random.choice(len(policy), p=policy)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                trajectory.append((obs.astype(np.float32), policy.astype(np.float32)))
                rewards.append(float(reward))
                obs = next_obs
                done = bool(terminated or truncated)
                steps += 1
            returns = self._discount_returns(rewards)
            for (state, policy), ret in zip(trajectory, returns):
                dataset.append((state, policy, ret))
            total_steps += steps
        return dataset, total_steps

    def _discount_returns(self, rewards: List[float]) -> List[float]:
        discounted = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.args.az_gamma * running
            discounted.append(running)
        discounted.reverse()
        return discounted

    def _train_step(self) -> None:
        batch = random.sample(self.replay_buffer, self.args.az_batch_size)
        states = th.tensor(np.stack([b[0] for b in batch]), dtype=th.float32, device=self.device)
        policy_targets = th.tensor(np.stack([b[1] for b in batch]), dtype=th.float32, device=self.device)
        value_targets = th.tensor([b[2] for b in batch], dtype=th.float32, device=self.device)

        logits, values = self.network(states)
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(values, value_targets)
        loss = self.args.az_policy_loss_weight * policy_loss + self.args.az_value_loss_weight * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.args.az_max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.args.az_max_grad_norm)
        self.optimizer.step()
