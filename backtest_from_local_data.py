"""
BACKTEST TRADING BOT - LOCAL DATA VERSION
Test your trained model on local CSV files
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tradeEnv import TradingEnv
import matplotlib.pyplot as plt
import os


def load_and_process_csv(csv_path):
    """
    Load and process a single CSV file
    """
    print(f"Loading: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Handle date column
    date_col = None
    for col in ['Date', 'date', 'DATE', 'Unnamed: 0']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    
    # Check columns
    required = ['Close', 'High', 'Low', 'Open', 'Volume']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Missing required columns. Has: {list(df.columns)}")
    
    print(f"  Original data: {len(df)} days")
    
    # Calculate indicators
    print("  Calculating indicators...")
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MACD'] = ta.macd(df['Close'])['MACD_12_26_9']
    df['Signal'] = ta.macd(df['Close'])['MACDs_12_26_9']
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['EMA_12'] = ta.ema(df['Close'], length=12)
    df['BB_upper'] = ta.bbands(df['Close'])['BBU_5_2.0']
    df['BB_lower'] = ta.bbands(df['Close'])['BBL_5_2.0']
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Drop NaN and select features
    df = df.dropna()
    features = ['Close', 'RSI', 'MACD', 'Signal', 'SMA_20', 'SMA_50', 
               'EMA_12', 'BB_upper', 'BB_lower', 'ATR', 'Volume']
    df = df[features]
    
    print(f"  After processing: {len(df)} days")
    
    if len(df) < 50:
        raise ValueError("Not enough data after processing")
    
    return df


def backtest_strategy(model, data, ticker_name, initial_cash=100000):
    """Backtest the trained model"""
    print("\n" + "="*60)
    print(f"BACKTESTING: {ticker_name}")
    print("="*60 + "\n")
    
    # Create environment
    env = DummyVecEnv([lambda: TradingEnv(data, initial_cash=initial_cash)])
    
    # Load normalization
    if os.path.exists("trading_bot/models/vec_normalize.pkl"):
        env = VecNormalize.load("trading_bot/models/vec_normalize.pkl", env)
        env.training = False
        env.norm_reward = False
    
    # Run backtest
    obs = env.reset()
    done = False
    
    portfolio_values = []
    actions_taken = []
    prices = []
    timestamps = []
    cash_history = []
    shares_history = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if 'portfolio_value' in info[0]:
            portfolio_values.append(info[0]['portfolio_value'])
            cash_history.append(info[0]['cash'])
            shares_history.append(info[0]['shares'])
            actions_taken.append(action[0])
            
            step_idx = info[0]['step']
            if step_idx < len(data):
                prices.append(data.iloc[step_idx]['Close'])
                if hasattr(data.index, 'to_pydatetime'):
                    timestamps.append(data.index[step_idx])
                else:
                    timestamps.append(step_idx)
    
    # Get metrics
    base_env = env.envs[0]
    metrics = base_env.get_metrics()
    
    return {
        'portfolio_values': portfolio_values,
        'actions': actions_taken,
        'prices': prices,
        'timestamps': timestamps,
        'cash': cash_history,
        'shares': shares_history,
        'metrics': metrics
    }


def plot_backtest(results, ticker, save_path='trading_bot/backtest_local.png'):
    """Create detailed backtest plots"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    timestamps = results['timestamps']
    
    # 1. Portfolio Value
    axes[0].plot(timestamps, results['portfolio_values'], linewidth=2, color='green')
    axes[0].axhline(y=100000, color='red', linestyle='--', alpha=0.5, label='Initial ($100k)')
    axes[0].set_title(f'{ticker} - Portfolio Value', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Value ($)', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Stock Price + Signals
    axes[1].plot(timestamps, results['prices'], label='Stock Price', alpha=0.7)
    
    actions = np.array(results['actions'])
    buy_mask = actions == 1
    sell_mask = actions == 2
    
    if np.any(buy_mask):
        buy_times = [timestamps[i] for i in range(len(actions)) if buy_mask[i]]
        buy_prices = [results['prices'][i] for i in range(len(actions)) if buy_mask[i]]
        axes[1].scatter(buy_times, buy_prices, color='green', marker='^', 
                       s=100, label='Buy', zorder=5)
    
    if np.any(sell_mask):
        sell_times = [timestamps[i] for i in range(len(actions)) if sell_mask[i]]
        sell_prices = [results['prices'][i] for i in range(len(actions)) if sell_mask[i]]
        axes[1].scatter(sell_times, sell_prices, color='red', marker='v', 
                       s=100, label='Sell', zorder=5)
    
    axes[1].set_title('Price with Trading Signals', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Price ($)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. Cash vs Shares Value
    shares_value = [s * p for s, p in zip(results['shares'], results['prices'])]
    axes[2].plot(timestamps, results['cash'], label='Cash', linewidth=2)
    axes[2].plot(timestamps, shares_value, label='Shares Value', linewidth=2)
    axes[2].set_title('Portfolio Composition', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Value ($)', fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].fill_between(timestamps, 0, results['cash'], alpha=0.3)
    axes[2].fill_between(timestamps, 0, shares_value, alpha=0.3)
    
    # 4. Action Distribution
    actions_count = [np.sum(actions == i) for i in range(3)]
    axes[3].bar(['Hold', 'Buy', 'Sell'], actions_count, color=['gray', 'green', 'red'])
    axes[3].set_title('Action Distribution', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('Count', fontsize=12)
    axes[3].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(actions_count):
        axes[3].text(i, v + max(actions_count)*0.02, str(v), 
                    ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Plot saved: {save_path}")
    plt.close()


def compare_strategies(results, data, initial_cash=100000):
    """Compare RL vs Buy & Hold"""
    start_price = data.iloc[10]['Close']
    end_price = data.iloc[-1]['Close']
    
    shares_bought = initial_cash / start_price
    final_bh = shares_bought * end_price
    bh_return = (final_bh - initial_cash) / initial_cash
    
    rl_return = results['metrics']['total_return']
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"\nInitial Capital: ${initial_cash:,.2f}")
    print(f"\nBuy & Hold:")
    print(f"  Final Value:  ${final_bh:,.2f}")
    print(f"  Return:       {bh_return*100:.2f}%")
    print(f"\nRL Trading Bot:")
    print(f"  Final Value:  ${results['portfolio_values'][-1]:,.2f}")
    print(f"  Return:       {rl_return*100:.2f}%")
    print(f"  Sharpe:       {results['metrics']['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {results['metrics']['max_drawdown']*100:.2f}%")
    print(f"  Trades:       {results['metrics']['total_trades']}")
    print(f"\nOutperformance: {(rl_return - bh_return)*100:+.2f}%")
    
    if rl_return > bh_return:
        print("\n✓ RL strategy outperformed buy-and-hold!")
    else:
        print("\n✗ RL strategy underperformed buy-and-hold")
    print("="*60 + "\n")


def main():
    """Main backtesting script"""
    print("\n" + "="*60)
    print("BACKTEST TRADING BOT - LOCAL DATA")
    print("="*60)
    
    # Check for trained model
    if not os.path.exists("trading_bot/models/final_model.zip"):
        print("\n✗ No trained model found!")
        print("Please run train_from_local_data.py first.")
        return
    
    print("\n✓ Loading trained model...")
    model = PPO.load("trading_bot/models/final_model")
    
    # ========== CONFIGURATION ==========
    # Enter the path to your CSV file
    CSV_FILE = input("\nEnter path to CSV file (e.g., ppolstm/stockData/AAPL.csv): ").strip()
    
    # Or hardcode it:
    # CSV_FILE = "ppolstm/stockData/TSLA.csv"
    # ===================================
    
    if not os.path.exists(CSV_FILE):
        print(f"\n✗ File not found: {CSV_FILE}")
        return
    
    # Extract ticker name from filename
    ticker = os.path.basename(CSV_FILE).replace('.csv', '')
    
    # Load and process data
    try:
        data = load_and_process_csv(CSV_FILE)
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        return
    
    # Run backtest
    results = backtest_strategy(model, data, ticker)
    
    # Compare with buy & hold
    compare_strategies(results, data)
    
    # Plot results
    plot_backtest(results, ticker)
    
    print("✓ Backtesting complete!\n")


if __name__ == "__main__":
    main()