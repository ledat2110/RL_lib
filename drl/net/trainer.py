import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import numpy as np
import time
import gym
import drl
import collections

from tensorboardX import SummaryWriter

class Trainer:
    def __init__ (self):
        self.rewards = []
        self.steps = []
        self.looses = []
        self.epslions = []
        self.fpss = []
        self.trackers = []
        self.iteration = 0 
        self.ts_eps = time.time()
        self.ts = time.time()
        self.best_m_reward = None
        self.stop = False
        self.parallel = False

    def add_tracker (self, tracker: drl.tracker.Tracker):
        assert isinstance(tracker, drl.tracker.Tracker)
        self.trackers.append(tracker)

    def add_exp_source (self, exp_source: drl.experience.ExperienceSource, initial_size: int):
        assert isinstance(exp_source, drl.experience.ExperienceSource)
        self.exp_source = exp_source
        self.initial_size = initial_size

    def add_net (self, net: nn.Module):
        assert isinstance(net, nn.Module)
        self.net = net

    def add_agent (self, agent: drl.agent.BaseAgent):
        assert isinstance(agent, drl.experience.BaseAgent)
        self.agent = agent

    def add_target_agent (self, tgt_agent: drl.agent.TargetNet):
        assert isinstance(tgt_agent, drl.agent.TargetNet)
        self.target_agent = tgt_agent

    def add_tensorboard_writer (self, writer: SummaryWriter):
        assert isinstance(writer, SummaryWriter)
        self.writer = writer

    def _update_tracker (self, idx: int):
        for tracker in self.trackers:
            tracker.update(idx)

    def _end_of_iteration (self):
        reward, step = self.exp_source.reward_step()

        if reward is not None:
            self.rewards.append(reward)
            self.steps.append(step)
            self._fps(step)
            self._cal_metric()

            self._print_out()

            self._update_best_m_reward()
            self._check_stop_reward()

    def _fps (self, step):
        speed = step / (time.time() - self.ts_eps)
        self.ts_eps = time.time()
        self.fpss.append(speed)

    def _cal_metric (self):
        self.m_reward = np.mean(self.rewards[-100:])
        self.m_fps = np.mean(self.fpss[-100:])
        self.m_step = np.mean(self.steps[-100:])

    def _print_out (self):
        time_elapsed = time.time() - self.ts
        episode = len(self.rewards)

        print("Iter: %d, Episoded %d: reward=%.3f, steps=%.3f, speed=%.3f fps, elapsed: %.3f"%(self.iteration, episode, self.m_reward, self.m_step, self.m_fps, time_elapsed))

    def _update_tensorboard (self):
        m_loss = np.mean(self.looses[-100:])
        self.writer.add_scalar("m_reward", self.m_reward, self.iteration)
        self.writer.add_scalar("loss", m_loss, self.iteration)
        self.writer.add_scalar("avg_fps", self.m_fps, self.iteration)

    def _update_best_m_reward (self):
        if self.best_m_reward is None or self.best_m_reward < self.m_reward:
            torch.save(self.net, "best_%.0f.dat"%self.m_reward)
            if self.best_m_reward is not None:
                print("Best mean reward update %.3f -> %.3f"%(self.best_m_reward, self.m_reward))
            self.best_m_reward = self.m_reward

    def _check_stop_reward (self):
        m_reward = np.mean(self.rewards[-100:])
        if m_reward > self.stop_reward:
            self.stop = True

    def run (self, optimizer, loss, batch_size: int, stop_reward: float, tb_iteration: int, sync_iteration: int):
        self.stop_reward = stop_reward
        while True:
            self.iteration += 1
            self._update_tracker(self.iteration)
            self.exp_source.play_steps()

            self._end_of_iteration()
            if self.stop:
                break

            if len(self.exp_source.buffer) < self.initial_size:
                continue

            if self.iteration % sync_iteration == 0:
                self.target_agent.sync()

            optimizer.zero_grad()
            batch = self.exp_source.buffer.sample(batch_size)
            loss_t = loss(batch)
            self.looses.append(loss_t.item())
            loss_t.backward()
            optimizer.step()

            if self.iteration % tb_iteration == 0:
                self._update_tensorboard()

        self.writer.close()

class ParallelTrainer (Trainer):
    def __init__ (self):
        super(ParallelTrainer, self).__init__()

    def add_buffer (self, rp_buffer: drl.experience.ReplayBuffer, initial_size: int):
        assert isinstance(rp_buffer, drl.experience.ReplayBuffer)
        self.rp_buffer = rp_buffer
        self.initial_size = initial_size

    def run_parallel (self, exp_queue: mp.Queue, optimizer, loss, batch_size: int, stop_reward: float, tb_iteration: int, sync_iteration: int):
        self.stop_reward = stop_reward

        while True:
            reward, step = None, None

            while exp_queue.qsize() > 0:
                exp = exp_queue.get()
                if isinstance(exp, drl.experience.EpisodeEnded):
                    reward, step = exp.reward, exp.step
                else:
                    self.rp_buffer.append(exp)

            if reward is not None:
                self.rewards.append(reward)
                self.steps.append(step)
                self._fps(step)
                self._cal_metric()

                self._print_out()

                self._update_best_m_reward()
                self._check_stop_reward()

                self.iteration += step

            if self.stop:
                break

            if len(self.rp_buffer) < self.initial_size:
                continue

            if self.iteration % sync_iteration == 0:
                self.target_agent.sync()

            optimizer.zero_grad()
            batch = self.rp_buffer.sample(batch_size)
            loss_t = loss(batch)
            self.looses.append(loss_t.item())
            loss_t.backward()
            optimizer.step()

            if self.iteration % tb_iteration == 0:
                self._update_tensorboard()

        self.writer.close()

