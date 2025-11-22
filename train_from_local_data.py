"""
IMPROVED TRAINING SCRIPT
Uses the enhanced TradingEnv with better hyperparameters
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from tradeEnv import TradingEnv
import matplotlib.pyplot as plt
import os
import glob

os.makedirs('trading_bot_v2', exist_ok=True)
os.makedirs('trading_bot_v2/models', exist_ok=True)
os.makedirs('trading_bot_v2/logs', exist_ok=True)


def load_and_process_local_data(data_folder, ticker_list=None, min_days=200):
    """Load stock data from local CSV files"""
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
            
            # Technical indicators
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
            df['Signal'] = ta.macd(df['Close'])['MACDs_12_26_9']
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['EMA_12'] = ta.ema(df['Close'], length=12)
            df['BB_upper'] = ta.bbands(df['Close'])['BBU_5_2.0']
            df['BB_lower'] = ta.bbands(df['Close'])['BBL_5_2.0']
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            
            df = df.dropna()
            
            if len(df) < 100:
                print(f"too many NaN")
                continue
            
            features = ['Close', 'RSI', 'MACD', 'Signal', 'SMA_20', 'SMA_50', 
                       'EMA_12', 'BB_upper', 'BB_lower', 'ATR', 'Volume']
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


def make_env(df, initial_cash=100000, random_start=True):
    """Create trading environment"""
    def _init():
        return TradingEnv(df, initial_cash=initial_cash, 
                         transaction_cost=0.001, 
                         lookback_window=10,
                         random_start=random_start)
    return _init


def train_model(train_data, val_data, total_timesteps=500000):
    """Train PPO with IMPROVED HYPERPARAMETERS"""
    print("\n" + "="*60)
    print("TRAINING MODEL V2 (IMPROVED)")
    print("="*60 + "\n")
    
    # Create training environments
    train_envs = []
    for ticker, df in train_data.items():
        train_envs.append(make_env(df, random_start=True))
        print(f"  Training env: {ticker} ({len(df)} days)")
    
    # Vectorize and normalize
    env = DummyVecEnv(train_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Validation environment (no random start for consistent eval)
    val_ticker = list(val_data.keys())[0]
    val_env = DummyVecEnv([make_env(val_data[val_ticker], random_start=False)])
    val_env = VecNormalize(val_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Callbacks
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='trading_bot_v2/models/',
        log_path='trading_bot_v2/logs/',
        eval_freq=10000,
        deterministic=True,
        n_eval_episodes=3
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='trading_bot_v2/models/',
        name_prefix='trading_model_v2'
    )
    
    print(f"\nTraining on {len(train_envs)} stocks for {total_timesteps:,} timesteps")
    print("Using IMPROVED hyperparameters...\n")
    
    # IMPROVED PPO HYPERPARAMETERS
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Lower exploration to reduce overtrading (was 0.05)
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),  # Bigger network
        verbose=1,
        tensorboard_log="trading_bot_v2/logs/"
    )
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    model.save("trading_bot_v2/models/final_model_v2")
    env.save("trading_bot_v2/models/vec_normalize_v2.pkl")
    
    print("\n✓ Training complete!")
    return model, env


def evaluate_model(model, test_data, env_normalize):
    """Evaluate on test data"""
    print("\n" + "="*60)
    print("EVALUATING V2")
    print("="*60 + "\n")
    
    results = {}
    
    for ticker, df in test_data.items():
        print(f"Testing {ticker}...")
        
        # Test env with NO random start
        test_env = DummyVecEnv([make_env(df, random_start=False)])
        test_env = VecNormalize(test_env, training=False, norm_obs=True, 
                                norm_reward=False, clip_obs=10.0)
        test_env.obs_rms = env_normalize.obs_rms
        test_env.ret_rms = env_normalize.ret_rms
        
        obs = test_env.reset()
        done = False
        actions = []
        portfolio_values = []
        positions = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
            if 'portfolio_value' in info[0]:
                portfolio_values.append(info[0]['portfolio_value'])
                positions.append(info[0].get('position_ratio', 0))
                actions.append(action[0][0])  # Continuous action value
        
        # Calculate metrics
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
        
        # Count actual trades (position changes > 5%)
        position_changes = np.abs(np.diff(positions))
        significant_trades = np.sum(position_changes > 0.05)
        
        results[ticker] = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'buy_hold_return': buy_hold_return,
            'actions': actions,
            'portfolio_values': list(portfolio_values),
            'positions': positions,
            'trades': significant_trades
        }
        
        print(f"  RL Return:    {total_return*100:+7.2f}%")
        print(f"  Buy&Hold:     {buy_hold_return*100:+7.2f}%")
        print(f"  Outperform:   {(total_return-buy_hold_return)*100:+7.2f}%")
        print(f"  Sharpe:       {sharpe_ratio:7.2f}")
        print(f"  Max DD:       {max_drawdown*100:7.2f}%")
        print(f"  Trades:       {significant_trades}\n")
    
    return results


def plot_results(results, save_path='trading_bot_v2/results_v2.png'):
    """Plot results with position sizes"""
    fig, axes = plt.subplots(len(results), 3, figsize=(18, 5*len(results)))
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (ticker, data) in enumerate(results.items()):
        # Portfolio value
        axes[idx, 0].plot(data['portfolio_values'], linewidth=2, color='green')
        axes[idx, 0].axhline(y=100000, color='gray', linestyle='--', alpha=0.5)
        axes[idx, 0].set_title(f'{ticker} - Portfolio Value', fontweight='bold')
        axes[idx, 0].set_ylabel('Value ($)')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Position over time (continuous)
        axes[idx, 1].plot(data['positions'], linewidth=2, color='blue')
        axes[idx, 1].set_title(f'{ticker} - Position Size', fontweight='bold')
        axes[idx, 1].set_ylabel('Position Ratio (0=cash, 1=stocks)')
        axes[idx, 1].set_ylim(-0.1, 1.1)
        axes[idx, 1].grid(True, alpha=0.3)
        
        # Action distribution
        actions = np.array(data['actions'])
        axes[idx, 2].hist(actions, bins=50, color='purple', alpha=0.7)
        axes[idx, 2].set_title(f'{ticker} - Action Distribution', fontweight='bold')
        axes[idx, 2].set_xlabel('Action (-1=sell, 0=hold, +1=buy)')
        axes[idx, 2].set_ylabel('Frequency')
        axes[idx, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Results saved: {save_path}")


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("IMPROVED TRADING BOT V2")
    print("="*60)
    
    # Configuration
    DATA_FOLDER = "stockData"
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    TOTAL_TIMESTEPS = 100000
    
    
    if not os.path.exists(DATA_FOLDER):
        print(f"✗ Error: {DATA_FOLDER} not found")
        return
    
    # Load data
    data_dict = load_and_process_local_data(DATA_FOLDER, TICKERS)
    
    if len(data_dict) == 0:
        print("✗ No data loaded")
        return
    
    # Split data
    train_data, val_data, test_data = {}, {}, {}
    
    print("Splitting data:")
    print("-" * 60)
    for ticker, df in data_dict.items():
        tr, va, te = split_data(df)
        train_data[ticker] = tr
        val_data[ticker] = va
        test_data[ticker] = te
        print(f"{ticker:6} | Train: {len(tr):4} | Val: {len(va):4} | Test: {len(te):4}")
    
    # Train
    model, env_normalize = train_model(train_data, val_data, TOTAL_TIMESTEPS)
    
    # Evaluate
    results = evaluate_model(model, test_data, env_normalize)
    
    # Plot
    plot_results(results)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS V2")
    print("="*60)
    
    avg_return = np.mean([r['total_return'] for r in results.values()])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in results.values()])
    avg_buy_hold = np.mean([r['buy_hold_return'] for r in results.values()])
    avg_trades = np.mean([r['trades'] for r in results.values()])
    
    print(f"\nAverage RL Return:     {avg_return*100:+7.2f}%")
    print(f"Average Buy&Hold:      {avg_buy_hold*100:+7.2f}%")
    print(f"Outperformance:        {(avg_return-avg_buy_hold)*100:+7.2f}%")
    print(f"Average Sharpe:        {avg_sharpe:7.2f}")
    print(f"Average Trades:        {avg_trades:.1f}")
    
    if avg_return > avg_buy_hold:
        print("\n✓ V2 Bot outperformed buy-and-hold!")
    else:
        print("\n⚠ V2 Bot underperformed - may need more training")
    
    print(f"\n✓ Model saved: trading_bot_v2/models/final_model_v2.zip")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()