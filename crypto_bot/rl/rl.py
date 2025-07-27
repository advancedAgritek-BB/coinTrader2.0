# rl.py - Reinforcement Learning Strategy Selector for CoinTrader2.0
# This module implements an RL-based strategy selector using Proximal Policy Optimization (PPO) from Stable-Baselines3.
# It trains on historical PnL data from trades.csv (integrated with CoinTrader_Trainer logs) to learn optimal strategy 
# selection per market regime. For 2025 profitability, it uses PPO for stable, sample-efficient learning on discrete 
# actions (strategy indices), with LSTM policy for sequential OHLCV data. Rewards are based on simulated PnL to maximize 
# BTC growth (e.g., favoring sniper_bot in volatiles for 5-10x pumps, grid_bot in sideways for 5-8% yields).

# Installation note: pip install stable-baselines3 gymnasium numpy pandas torch
# Ensure CoinTrader_Trainer has generated trades.csv with columns: regime, strategy, pnl, timestamp, adx, rsi, vol_z, etc.

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from typing import Dict, Tuple, List, Callable, Optional
import os
from datetime import datetime
import logging
import torch
from ta.trend import ADXIndicator

from crypto_bot.utils.regime_pnl_tracker import get_recent_win_rate

# Strategy map - list of available strategies (from meta_selector or strategy_router)
STRATEGIES = [
    "trend_bot",       # 0: EMA/ADX trends for bull runs (2025 BTC $100k+)
    "grid_bot",        # 1: Dynamic grids for sideways (delta-neutral yields 5-8%)
    "sniper_bot",      # 2: Pump detection for Solana memes (5-10x volatiles)
    "dex_scalper",     # 3: EMA divergence scalps on DEX (fast edges)
    "dip_hunter",      # 4: RSI/vol dips for mean-reversion (quick 2-3x)
    "mean_bot",        # 5: Keltner/BB mean-reversion (stable in ranges)
    "breakout_bot",    # 6: BB/KC squeeze breakouts (high-momentum 3-5x)
    "momentum_bot",    # 7: Donchian/volume momentum strategy
    "bounce_scalper",  # 8: Patterns + RSI/vol for bounces (80% win scalps)
    "micro_scalp_bot", # 9: EMA cross + wicks for micro-scalps (high-freq)
    "dca_bot",         # 10: Averaging for longs (safety net)
    "solana_scalping"  # 11: MACD/RSI for Solana scalps (mempool-integrated)
]
NUM_STRATEGIES = len(STRATEGIES)

# Features for state (regime indicators from regime_classifier + vol/normalized)
FEATURES = ['adx', 'rsi', 'rsi_z', 'vol_z', 'bb_z', 'atr_pct', 'ema_cross', 'regime_prob_trending', 'regime_prob_volatile', 'regime_prob_sideways']  # 10 features

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('crypto_bot/logs/rl_selector.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class StrategySelectionEnv(gym.Env):
    """Custom Gym environment for RL strategy selection in crypto trading."""
    def __init__(self, historical_data: pd.DataFrame):
        super(StrategySelectionEnv, self).__init__()
        self.historical_data = historical_data
        self.action_space = spaces.Discrete(NUM_STRATEGIES)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(FEATURES),), dtype=np.float32)
        self.current_step = 0
        self.episode_length = 100
        self.max_steps = len(historical_data) - self.episode_length - 1
        self.reward_scaling = 1e-4

        from crypto_bot.strategy_router import get_strategy_by_name
        self.strategy_fns: List[Callable] = [get_strategy_by_name(s) for s in STRATEGIES]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        self.current_step = np.random.randint(0, self.max_steps)
        state = self._get_state(self.current_step)
        self.done = False
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        df_slice = self.historical_data.iloc[self.current_step:self.current_step + self.episode_length]
        strategy_fn = self.strategy_fns[action]
        score, direction = strategy_fn(df_slice)

        entry_price = df_slice['close'].iloc[-1]
        future_pnl_period = 5
        if self.current_step + self.episode_length + future_pnl_period < len(self.historical_data):
            exit_price = self.historical_data['close'].iloc[self.current_step + self.episode_length + future_pnl_period]
            pnl = (exit_price - entry_price) / entry_price if direction == "long" else (entry_price - exit_price) / entry_price
            reward = pnl * self.reward_scaling * score
        else:
            reward = 0.0

        win_rate = get_recent_win_rate(strategy=STRATEGIES[action])
        if win_rate < 0.5:
            reward -= 0.1

        self.current_step += 1
        self.done = self.current_step >= self.max_steps
        state = self._get_state(self.current_step)
        return state, reward, self.done, False, {}

    def _get_state(self, step: int) -> np.ndarray:
        row = self.historical_data.iloc[step]
        state = np.array([row[f] for f in FEATURES], dtype=np.float32)
        return state

    def render(self, mode='human'):
        pass

class RLStrategySelector:
    def __init__(self, model_path: str = "crypto_bot/models/rl_selector.zip"):
        self.model_path = model_path
        self.model = None
        self.strategies = STRATEGIES
        self.load_model()

    def train(self, historical_data: pd.DataFrame, timesteps: int = 100000, eval_freq: int = 10000):
        logger.info("Training RL selector...")
        env = DummyVecEnv([lambda: StrategySelectionEnv(historical_data)])
        eval_env = DummyVecEnv([lambda: StrategySelectionEnv(historical_data)])

        policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])], activation_fn=torch.nn.ReLU)
        self.model = PPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            learning_rate=0.00025,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log="crypto_bot/logs/tb_rl_selector",
            device="auto"
        )

        eval_callback = EvalCallback(eval_env, eval_freq=eval_freq, deterministic=True, render=False)
        self.model.learn(total_timesteps=timesteps, callback=eval_callback)
        self.model.save(self.model_path)
        logger.info(f"RL model trained and saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = PPO.load(self.model_path)
            logger.info(f"Loaded RL model from {self.model_path}")
        else:
            logger.warning("No RL model found; train first.")

    def select(self, state: np.ndarray) -> str:
        if self.model is None:
            logger.warning("No trained model; using default.")
            return self.strategies[0]

        action, _ = self.model.predict(state, deterministic=True)
        strategy = self.strategies[action]
        logger.info(f"RL selected: {strategy} for state {state}")
        return strategy


def train_rl_selector():
    historical_data = pd.read_csv("crypto_bot/logs/trades.csv")
    historical_data['adx'] = ADXIndicator(historical_data['high'], historical_data['low'], historical_data['close'], window=14).adx()
    selector = RLStrategySelector()
    selector.train(historical_data)


def get_rl_strategy(regime_state: np.ndarray) -> str:
    selector = RLStrategySelector()
    return selector.select(regime_state)


if __name__ == "__main__":
    train_rl_selector()
