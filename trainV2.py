"""
FIXED TRADING BOT - Applying Learnings from Performance Analysis

Key fixes:
1. SIMPLIFIED reward: log returns + transaction penalty only
2. DISCRETE action space: HOLD (0), BUY (1), SELL (2)
3. NO random_start: preserve sequential context for trend learning
4. MULTI-STOCK validation: validate on 3-5 stocks instead of 1
5. NO reward normalization: only normalize observations
6. REDUCED observation complexity: simpler, focused features
7. Better early stopping logic
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.utils import set_random_seed
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import os
import glob
import torch

os.makedirs('trading_bot_fixed', exist_ok=True)
os.makedirs('trading_bot_fixed/models', exist_ok=True)
os.makedirs('trading_bot_fixed/logs', exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")


class FixedTradingEnv(gym.Env):
    """
    FIXED Trading Environment with:
    - Discrete actions (HOLD/BUY/SELL)
    - Simple reward (log returns + transaction cost)
    - No random start
    - Simplified observations
    """
    
    def __init__(self, df, initial_cash=100000, transaction_cost=0.001, lookback_window=10):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # FIX 1: DISCRETE ACTION SPACE
        # 0 = HOLD, 1 = BUY (go all-in stocks), 2 = SELL (go all-in cash)
        self.action_space = spaces.Discrete(3)
        
        # FIX 2: SIMPLIFIED OBSERVATION SPACE
        # Only essential features: Close, Volume, RSI, MACD, SMA ratios
        n_features = 5  # Close ratio, Volume, RSI, MACD, SMA trend
        n_lookback = n_features * lookback_window
        n_position = 3  # cash_ratio, stock_ratio, unrealized_pnl
        
        obs_size = n_lookback + n_position
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(obs_size,), dtype=np.float32
        )
        
        # Precompute normalizations
        self.price_mean = self.df['Close'].mean()
        self.price_std = self.df['Close'].std() + 1e-8
        self.volume_mean = self.df['Volume'].mean()
        self.volume_std = self.df['Volume'].std() + 1e-8
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # FIX 3: NO RANDOM START - always start from beginning
        self.current_step = self.lookback_window
        
        self.cash = self.initial_cash
        self.shares = 0
        self.portfolio_value = self.initial_cash
        self.prev_portfolio_value = self.initial_cash
        
        self.trades_count = 0
        self.entry_price = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """SIMPLIFIED observation with only essential features"""
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Lookback features: only 5 essential indicators
        lookback_features = []
        for i in range(start_idx, end_idx):
            row = self.df.iloc[i]
            
            # 1. Price ratio (relative to current)
            close_ratio = (row['Close'] - current_price) / (current_price + 1e-8)
            
            # 2. Volume (normalized)
            volume_norm = np.tanh((row['Volume'] - self.volume_mean) / self.volume_std)
            
            # 3. RSI (centered and scaled)
            rsi_norm = (row['RSI'] - 50) / 50
            
            # 4. MACD (bounded)
            macd_norm = np.tanh(row['MACD'] / 10)
            
            # 5. SMA trend (20 vs 50)
            sma_trend = np.tanh((row['SMA_20'] - row['SMA_50']) / (row['SMA_50'] + 1e-8) * 10)
            
            lookback_features.extend([
                close_ratio, volume_norm, rsi_norm, macd_norm, sma_trend
            ])
        
        lookback_array = np.array(lookback_features, dtype=np.float32)
        
        # Position features: only 3 essential values
        stock_value = self.shares * current_price
        total_value = self.cash + stock_value
        cash_ratio = self.cash / (total_value + 1e-8)
        stock_ratio = stock_value / (total_value + 1e-8)
        
        # Unrealized P&L
        unrealised_pnl = 0
        if self.shares > 0 and self.entry_price > 0:
            unrealised_pnl = np.tanh((current_price - self.entry_price) / (self.entry_price + 1e-8))
        
        position_features = np.array([
            cash_ratio,
            stock_ratio,
            unrealised_pnl
        ], dtype=np.float32)
        
        observation = np.concatenate([lookback_array, position_features])
        observation = np.clip(observation, -10, 10)
        
        return observation
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Store previous values
        stock_value = self.shares * current_price
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_value = self.cash + stock_value
        
        trade_occurred = False
        
        # FIX 4: DISCRETE ACTIONS - clear buy/sell/hold decisions
        if action == 1:  # BUY - go all-in on stock
            if self.cash > 0:
                shares_to_buy = self.cash / (current_price * (1 + self.transaction_cost))
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                
                if cost <= self.cash:
                    self.cash -= cost
                    self.shares += shares_to_buy
                    self.trades_count += 1
                    trade_occurred = True
                    if self.entry_price == 0:
                        self.entry_price = current_price
        
        elif action == 2:  # SELL - go all-in on cash
            if self.shares > 0:
                proceeds = self.shares * current_price * (1 - self.transaction_cost)
                self.cash += proceeds
                self.shares = 0
                self.trades_count += 1
                trade_occurred = True
                self.entry_price = 0
        
        # action == 0 is HOLD - do nothing
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Recalculate portfolio value
        stock_value = self.shares * current_price
        self.portfolio_value = self.cash + stock_value
        
        # FIX 5: SIMPLIFIED REWARD - just log returns + transaction penalty
        step_return = (self.portfolio_value - self.prev_portfolio_value) / (self.prev_portfolio_value + 1e-8)
        
        # Base reward: log return (handles multiplicative returns well)
        reward = np.log1p(step_return) * 100
        
        # Small penalty for trading (discourages overtrading)
        if trade_occurred:
            reward -= 0.05
        
        # That's it! No complex bonuses, penalties, or Sharpe adjustments
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position_ratio': stock_value / (self.portfolio_value + 1e-8),
            'trades': self.trades_count,
            'action': action
        }
        
        truncated = False
        return self._get_observation(), reward, done, truncated, info
    
    def render(self):
        pass


def load_and_process_data(data_folder, ticker_list=None, min_days=200):
    """Load and process stock data"""
    print(f"Loading data from: {data_folder}")
    print("="*60)
    
    if ticker_list:
        csv_files = [os.path.join(data_folder, f"{ticker}.csv") for ticker in ticker_list]
        csv_files = [f for f in csv_files if os.path.exists(f)]
    else:
        csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    print(f"Found {len(csv_files)} CSV files\n")
    
    processed_data = {}
    
    for csv_file in csv_files:
        ticker = os.path.basename(csv_file).replace('.csv', '')
        
        try:
            df = pd.read_csv(csv_file)
            
            # Handle date column
            date_col = None
            for col in ['Date', 'date', 'DATE', 'Unnamed: 0']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
            
            required = ['Close', 'High', 'Low', 'Open', 'Volume']
            if not all(col in df.columns for col in required):
                print(f"✗ Skipping {ticker}: Missing columns")
                continue
            
            if len(df) < min_days:
                print(f"✗ Skipping {ticker}: Only {len(df)} days")
                continue
            
            print(f"Processing {ticker}... ", end='')
            
            # Compute only essential indicators
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            
            df = df.dropna()
            
            if len(df) < 100:
                print(f"too many NaN")
                continue
            
            features = ['Close', 'High', 'Low', 'Open', 'Volume', 'RSI', 'MACD', 'SMA_20', 'SMA_50']
            df = df[features]
            
            processed_data[ticker] = df
            print(f"✓ {len(df)} days")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Loaded {len(processed_data)} stocks")
    print(f"{'='*60}\n")
    
    return processed_data


def split_data(df, train_ratio=0.7, val_ratio=0.15):
    """Split data chronologically"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def make_env(df, initial_cash=100000, rank=0):
    """Create trading environment - NO random start"""
    def _init():
        env = FixedTradingEnv(df, initial_cash=initial_cash, 
                             transaction_cost=0.001, 
                             lookback_window=10)
        env.reset(seed=rank)
        return env
    set_random_seed(rank)
    return _init


def train_model_fixed(train_data, val_data, total_timesteps=500000, n_envs=4):
    """FIXED training with discrete actions and simplified reward"""
    print("\n" + "="*60)
    print("TRAINING FIXED MODEL")
    print("="*60 + "\n")
    
    # Create training environments
    train_envs = []
    env_counter = 0
    for ticker, df in train_data.items():
        for i in range(max(1, n_envs // len(train_data))):
            train_envs.append(make_env(df, rank=env_counter))
            env_counter += 1
        print(f"  Training env: {ticker} ({len(df)} days)")
    
    print(f"\nUsing {len(train_envs)} parallel environments")
    
    env = SubprocVecEnv(train_envs)
    
    # FIX 6: ONLY NORMALIZE OBSERVATIONS, NOT REWARDS
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # FIX 7: MULTI-STOCK VALIDATION (use 3-5 stocks, not just 1)
    val_envs = []
    val_tickers = list(val_data.keys())[:min(3, len(val_data))]  # Use up to 3 stocks for validation
    
    print(f"\nValidation stocks: {val_tickers}")
    
    for idx, ticker in enumerate(val_tickers):
        val_envs.append(make_env(val_data[ticker], rank=9000 + idx))
    
    val_env = DummyVecEnv(val_envs)
    val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    
    # Early stopping - more patience
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=6,  # Increased patience
        min_evals=10,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='trading_bot_fixed/models/',
        log_path='trading_bot_fixed/logs/',
        eval_freq=max(20000 // len(train_envs), 2000),
        deterministic=True,
        n_eval_episodes=len(val_envs) * 3,  # Multiple episodes per validation stock
        callback_after_eval=stop_callback,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // len(train_envs), 5000),
        save_path='trading_bot_fixed/models/',
        name_prefix='trading_model_fixed'
    )
    
    print(f"\nTraining configuration:")
    print(f"  Parallel environments: {len(train_envs)}")
    print(f"  Validation stocks: {len(val_envs)}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Device: {DEVICE}")
    print(f"  Action space: Discrete(3) - HOLD/BUY/SELL")
    print(f"  Reward: Simplified (log returns + transaction penalty)\n")
    
    # FIXED PPO hyperparameters for discrete actions
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048 // len(train_envs),
        batch_size=128,  # Smaller batch for discrete actions
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Slightly more exploration for discrete actions
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Simpler network for discrete actions
            activation_fn=torch.nn.Tanh
        ),
        verbose=1,
        device=DEVICE,
        tensorboard_log="trading_bot_fixed/logs/"
    )
    
    print("Starting training...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    model.save("trading_bot_fixed/models/final_model_fixed")
    env.save("trading_bot_fixed/models/vec_normalize_fixed.pkl")
    
    print("\n✓ Training complete!")
    return model, env


def evaluate_model(model, test_data, env_normalise):
    """Evaluate on test data"""
    print("\n" + "="*60)
    print("EVALUATING FIXED MODEL")
    print("="*60 + "\n")
    
    results = {}
    
    for ticker, df in test_data.items():
        print(f"Testing {ticker}...")
        
        test_env = DummyVecEnv([make_env(df, rank=10000)])
        test_env = VecNormalize(test_env, training=False, norm_obs=True, 
                                norm_reward=False, clip_obs=10.0)
        test_env.obs_rms = env_normalise.obs_rms
        
        obs = test_env.reset()
        done = False
        actions = []
        portfolio_values = []
        positions = []
        
        action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
            if 'portfolio_value' in info[0]:
                portfolio_values.append(info[0]['portfolio_value'])
                positions.append(info[0].get('position_ratio', 0))
                actions.append(int(action[0]))
                action_counts[int(action[0])] += 1
        
        if len(portfolio_values) < 2:
            print(f"  ✗ Not enough data")
            continue
        
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        initial_cash = 100000
        total_return = (portfolio_values[-1] - initial_cash) / initial_cash
        
        if len(returns) > 1 and np.std(returns) != 0:
            sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0
        
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        max_drawdown = np.min(drawdown)
        
        buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[10]) / df['Close'].iloc[10]
        
        # Count actual trades (action changes)
        trade_count = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
        
        results[ticker] = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'buy_hold_return': buy_hold_return,
            'actions': actions,
            'action_counts': action_counts,
            'portfolio_values': list(portfolio_values),
            'positions': positions,
            'trades': trade_count
        }
        
        print(f"  RL Return:    {total_return*100:+7.2f}%")
        print(f"  Buy&Hold:     {buy_hold_return*100:+7.2f}%")
        print(f"  Outperform:   {(total_return-buy_hold_return)*100:+7.2f}%")
        print(f"  Sharpe:       {sharpe_ratio:7.2f}")
        print(f"  Max DD:       {max_drawdown*100:7.2f}%")
        print(f"  Trades:       {trade_count}")
        print(f"  Actions:      HOLD={action_counts[0]} BUY={action_counts[1]} SELL={action_counts[2]}\n")
    
    return results


def plot_results(results, save_path='trading_bot_fixed/results_fixed.png'):
    """Plot results with discrete action indicators"""
    n_stocks = len(results)
    fig, axes = plt.subplots(n_stocks, 3, figsize=(18, 5*n_stocks))
    if n_stocks == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (ticker, data) in enumerate(results.items()):
        # Portfolio value
        axes[idx, 0].plot(data['portfolio_values'], linewidth=2, color='#2ecc71', label='RL Portfolio')
        axes[idx, 0].axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial')
        axes[idx, 0].set_title(f'{ticker} - Portfolio Value', fontweight='bold', fontsize=12)
        axes[idx, 0].set_ylabel('Value ($)', fontsize=10)
        axes[idx, 0].grid(True, alpha=0.3)
        axes[idx, 0].legend()
        
        # Position size with buy/sell markers
        axes[idx, 1].plot(data['positions'], linewidth=2, color='#3498db', label='Stock Position')
        
        # Mark BUY and SELL actions
        actions = data['actions']
        for i in range(len(actions)):
            if actions[i] == 1:  # BUY
                axes[idx, 1].scatter(i, data['positions'][i], color='green', s=30, alpha=0.6, marker='^')
            elif actions[i] == 2:  # SELL
                axes[idx, 1].scatter(i, data['positions'][i], color='red', s=30, alpha=0.6, marker='v')
        
        axes[idx, 1].set_title(f'{ticker} - Position Size (▲=BUY ▼=SELL)', fontweight='bold', fontsize=12)
        axes[idx, 1].set_ylabel('Position Ratio', fontsize=10)
        axes[idx, 1].set_ylim(-0.1, 1.1)
        axes[idx, 1].grid(True, alpha=0.3)
        axes[idx, 1].legend()
        
        # Action distribution
        action_counts = data['action_counts']
        actions_labels = ['HOLD', 'BUY', 'SELL']
        actions_values = [action_counts[0], action_counts[1], action_counts[2]]
        colors = ['#95a5a6', '#2ecc71', '#e74c3c']
        
        bars = axes[idx, 2].bar(actions_labels, actions_values, color=colors, alpha=0.7, edgecolor='black')
        axes[idx, 2].set_title(f'{ticker} - Action Distribution', fontweight='bold', fontsize=12)
        axes[idx, 2].set_ylabel('Count', fontsize=10)
        axes[idx, 2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx, 2].text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Results saved: {save_path}")


def main():
    """Main training pipeline with all fixes applied"""
    print("\n" + "="*60)
    print("FIXED TRADING BOT")
    print(f"Device: {DEVICE}")
    print("="*60)
    print("\nKey fixes applied:")
    print("  ✓ Discrete actions (HOLD/BUY/SELL)")
    print("  ✓ Simplified reward (log returns + transaction penalty)")
    print("  ✓ No random start (preserves trends)")
    print("  ✓ Multi-stock validation")
    print("  ✓ No reward normalization")
    print("  ✓ Reduced observation complexity")
    print("="*60 + "\n")
    
    DATA_FOLDER = "stockData"
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    TOTAL_TIMESTEPS = 500000
    N_ENVS = 4
    
    if not os.path.exists(DATA_FOLDER):
        print(f"✗ Error: {DATA_FOLDER} not found")
        return
    
    # Load data
    data_dict = load_and_process_data(DATA_FOLDER, TICKERS)
    
    if len(data_dict) == 0:
        print("✗ No data loaded")
        return
    
    # Split data
    train_data, val_data, test_data = {}, {}, {}
    
    print("\nSplitting data:")
    print("=" * 60)
    for ticker, df in data_dict.items():
        tr, va, te = split_data(df)
        train_data[ticker] = tr
        val_data[ticker] = va
        test_data[ticker] = te
        print(f"{ticker:6} | Train: {len(tr):4} | Val: {len(va):4} | Test: {len(te):4}")
    
    # Train model
    model, env_normalise = train_model_fixed(train_data, val_data, TOTAL_TIMESTEPS, N_ENVS)
    
    # Evaluate model
    results = evaluate_model(model, test_data, env_normalise)
    
    # Plot results
    plot_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS (FIXED MODEL)")
    print("="*60)
    
    avg_return = np.mean([r['total_return'] for r in results.values()])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in results.values()])
    avg_buy_hold = np.mean([r['buy_hold_return'] for r in results.values()])
    avg_trades = np.mean([r['trades'] for r in results.values()])
    avg_dd = np.mean([r['max_drawdown'] for r in results.values()])
    
    print(f"\nAverage RL Return:     {avg_return*100:+7.2f}%")
    print(f"Average Buy&Hold:      {avg_buy_hold*100:+7.2f}%")
    print(f"Outperformance:        {(avg_return-avg_buy_hold)*100:+7.2f}%")
    print(f"Average Sharpe:        {avg_sharpe:7.2f}")
    print(f"Average Max Drawdown:  {avg_dd*100:7.2f}%")
    print(f"Average Trades:        {avg_trades:.1f}")
    
    # Win rate
    win_count = sum(1 for r in results.values() if r['total_return'] > r['buy_hold_return'])
    win_rate = win_count / len(results) * 100
    print(f"Win Rate vs Buy&Hold:  {win_rate:.1f}%")
    
    # Action distribution across all stocks
    total_holds = sum(r['action_counts'][0] for r in results.values())
    total_buys = sum(r['action_counts'][1] for r in results.values())
    total_sells = sum(r['action_counts'][2] for r in results.values())
    total_actions = total_holds + total_buys + total_sells
    
    print(f"\nOverall Action Distribution:")
    print(f"  HOLD:  {total_holds:5d} ({total_holds/total_actions*100:.1f}%)")
    print(f"  BUY:   {total_buys:5d} ({total_buys/total_actions*100:.1f}%)")
    print(f"  SELL:  {total_sells:5d} ({total_sells/total_actions*100:.1f}%)")
    
    if avg_return > avg_buy_hold:
        print("\n✓ Fixed bot OUTPERFORMED buy and hold!")
    else:
        print("\n⚠ Performance below buy and hold")
        print("\nNote: In efficient markets, beating buy&hold is extremely difficult.")
        print("The simplified approach should at least reduce overtrading and losses.")
    
    print(f"\n✓ Model saved: trading_bot_fixed/models/final_model_fixed.zip")
    print(f"✓ Normalizer saved: trading_bot_fixed/models/vec_normalize_fixed.pkl")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()