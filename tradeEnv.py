import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    IMPROVED Trading Environment with:
    - Fractional position sizing (continuous actions)
    - Better reward shaping (log returns + smoothing)
    - Random starting points (prevents memorization)
    - Proper normalization (mean/std, no leakage)
    - Proportional transaction costs
    - Better observation scaling
    """
    
    def __init__(self, data: pd.DataFrame, initial_cash=100000.0, 
                 transaction_cost=0.001, lookback_window=10, 
                 random_start=True):
        super(TradingEnv, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.random_start = random_start
        
        # For reward normalization
        self.reward_history = []
        self.max_reward_history = 100
        
        # State variables
        self.cash = initial_cash
        self.shares = 0.0
        self.current_step = lookback_window
        self.start_step = lookback_window
        self.total_trades = 0
        self.portfolio_history = []
        
        # CONTINUOUS action space: position target in [-1, 1]
        # -1 = 100% cash (sell all), 0 = 50% stocks/50% cash, +1 = 100% stocks
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation: [normalized_price_data, portfolio_state]
        num_price_features = len(self.data.columns)
        num_portfolio_features = 3  # cash_ratio, position_ratio, unrealized_pnl
        obs_size = (lookback_window * num_price_features) + num_portfolio_features
        
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(obs_size,), dtype=np.float32
        )
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # RANDOM STARTING POINT to prevent memorization
        if self.random_start and len(self.data) > self.lookback_window + 100:
            max_start = len(self.data) - 100  # Leave room for episode
            self.start_step = np.random.randint(self.lookback_window, max_start)
        else:
            self.start_step = self.lookback_window
        
        self.current_step = self.start_step
        self.cash = self.initial_cash
        self.shares = 0.0
        self.total_trades = 0
        self.portfolio_history = [self.initial_cash]
        self.reward_history = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        # Clip action to valid range
        action = np.clip(action[0], -1.0, 1.0)
        
        current_price = self.data.iloc[self.current_step]['Close']
        portfolio_before = self._get_portfolio_value()
        
        # FRACTIONAL POSITION SIZING
        # Map action [-1, 1] to target position [0, 1]
        # -1 = 0% in stocks (all cash)
        # 0 = 50% in stocks
        # +1 = 100% in stocks
        target_position_ratio = (action + 1.0) / 2.0
        
        # Calculate target shares based on portfolio value
        target_value_in_stocks = target_position_ratio * portfolio_before
        target_shares = target_value_in_stocks / current_price
        
        # Execute trade to reach target position
        shares_delta = target_shares - self.shares
        
        # HOLD ZONE: Only trade if change is significant (>5% of portfolio)
        position_change_pct = abs(shares_delta * current_price) / portfolio_before if portfolio_before > 0 else 0
        
        if position_change_pct > 0.05:  # Only trade if change > 5%
            if shares_delta > 0:  # Buying
                self._buy(current_price, shares_delta)
            elif shares_delta < 0:  # Selling
                self._sell(current_price, -shares_delta)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Calculate portfolio value change
        portfolio_after = self._get_portfolio_value()
        self.portfolio_history.append(portfolio_after)
        
        # IMPROVED REWARD: Log returns (more stable than percentage)
        if portfolio_before > 0:
            log_return = np.log(portfolio_after / portfolio_before)
        else:
            log_return = 0.0
        
        # PROPORTIONAL transaction cost penalty (INCREASED to prevent overtrading)
        position_change = abs(shares_delta * current_price) / portfolio_before if portfolio_before > 0 else 0
        transaction_penalty = self.transaction_cost * position_change * 200  # Much stronger penalty!
        
        reward = log_return - transaction_penalty
        
        # NORMALIZE REWARD using recent history
        self.reward_history.append(reward)
        if len(self.reward_history) > self.max_reward_history:
            self.reward_history.pop(0)
        
        if len(self.reward_history) > 10:
            reward_std = np.std(self.reward_history) + 1e-8
            reward = reward / reward_std
        
        # Small bonus for positive cumulative returns
        cumulative_return = (portfolio_after - self.initial_cash) / self.initial_cash
        if cumulative_return > 0:
            reward += 0.001
        
        info = {
            'portfolio_value': portfolio_after,
            'cash': self.cash,
            'shares': self.shares,
            'total_trades': self.total_trades,
            'step': self.current_step,
            'position_ratio': self.shares * current_price / portfolio_after if portfolio_after > 0 else 0
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _buy(self, price, shares_to_buy):
        """Buy specified number of shares"""
        if shares_to_buy <= 0:
            return
        
        cost_per_share = price * (1 + self.transaction_cost)
        total_cost = shares_to_buy * cost_per_share
        
        # Can only buy what we can afford
        affordable_shares = self.cash / cost_per_share
        actual_shares = min(shares_to_buy, affordable_shares)
        
        if actual_shares > 0.01:
            actual_cost = actual_shares * cost_per_share
            self.shares += actual_shares
            self.cash -= actual_cost
            self.total_trades += 1
    
    def _sell(self, price, shares_to_sell):
        """Sell specified number of shares"""
        if shares_to_sell <= 0:
            return
        
        # Can only sell what we have
        actual_shares = min(shares_to_sell, self.shares)
        
        if actual_shares > 0.01:
            proceeds = actual_shares * price * (1 - self.transaction_cost)
            self.cash += proceeds
            self.shares -= actual_shares
            self.total_trades += 1
    
    def _get_portfolio_value(self):
        """Calculate total portfolio value"""
        current_price = self.data.iloc[self.current_step]['Close']
        return self.cash + (self.shares * current_price)
    
    def _get_observation(self):
        """
        Get observation with PROPER NORMALIZATION (no information leakage)
        """
        # Get lookback window
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        window_data = self.data.iloc[start_idx:end_idx].copy()
        
        # NORMALIZE BY MEAN AND STD (not by first value!)
        normalized_data = pd.DataFrame()
        for col in window_data.columns:
            mean = window_data[col].mean()
            std = window_data[col].std() + 1e-8
            normalized_data[col] = (window_data[col] - mean) / std
        
        # Flatten price data
        price_obs = normalized_data.values.flatten()
        
        # Portfolio state (already normalized ratios)
        current_price = self.data.iloc[self.current_step]['Close']
        portfolio_value = self._get_portfolio_value()
        
        if portfolio_value > 0:
            cash_ratio = self.cash / portfolio_value
            position_ratio = (self.shares * current_price) / portfolio_value
            
            # Unrealized PnL (normalized)
            pnl = (portfolio_value - self.initial_cash) / self.initial_cash
            pnl = np.clip(pnl, -1, 1)  # Clip to reasonable range
        else:
            cash_ratio = 1.0
            position_ratio = 0.0
            pnl = -1.0
        
        portfolio_obs = np.array([cash_ratio, position_ratio, pnl], dtype=np.float32)
        
        # Combine and clip to observation space
        obs = np.concatenate([price_obs, portfolio_obs]).astype(np.float32)
        obs = np.clip(obs, -10, 10)  # Prevent extreme values
        
        return obs
    
    def get_metrics(self):
        """Calculate performance metrics"""
        if len(self.portfolio_history) < 2:
            return {}
        
        portfolio_values = np.array(self.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        total_return = (portfolio_values[-1] - self.initial_cash) / self.initial_cash
        
        # Sharpe ratio
        if len(returns) > 1 and np.std(returns) != 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': portfolio_values[-1],
            'total_trades': self.total_trades
        }