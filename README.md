# Improved Trading Bot with Reinforcement Learning

A significantly improved stock trading bot using PPO (Proximal Policy Optimization) that addresses all major issues in the original implementation.

## 🔧 Major Improvements

### 1. **Fixed Reward Function**
**Before:** Rewarded based on price direction (if price went up/down), not actual profit
```python
# OLD - TERRIBLE REWARD
if price > old_price:
    reward = 1  # This doesn't reflect actual profit!
```

**After:** Rewards based on actual portfolio value changes
```python
# NEW - PROPER REWARD
reward = (portfolio_after - portfolio_before) / portfolio_before
```

### 2. **Better Action Space**
**Before:** Only Buy (0) or Sell (1) - no way to hold!
**After:** Hold (0), Buy (1), Sell (2) - agent can now wait for better opportunities

### 3. **Fixed Variable Names**
**Before:** Inconsistent use of `self.shares` vs `self.shares_held`
**After:** Consistent variable naming throughout

### 4. **Portfolio State in Observations**
**Before:** Agent couldn't see what it owned
**After:** Observations include:
- Lookback window of price data
- Cash ratio
- Shares value ratio
- Number of shares held

### 5. **Transaction Costs**
**Before:** No transaction costs (unrealistic)
**After:** 0.1% transaction cost per trade (realistic)

### 6. **Proper Train/Val/Test Split**
**Before:** No data splitting
**After:** 70% train, 15% validation, 15% test

### 7. **Multiple Stocks Training**
**Before:** Trained only on TSLA (overfitting)
**After:** Trains on multiple stocks for better generalization

### 8. **Better Metrics**
**Before:** No performance metrics
**After:** Tracks:
- Total return
- Sharpe ratio
- Maximum drawdown
- Comparison with buy-and-hold

## 📦 Installation

### Step 1: Install Required Packages

```bash
pip install gymnasium numpy pandas yfinance pandas-ta scikit-learn stable-baselines3 sb3-contrib torch matplotlib
```

**Package versions (recommended):**
- gymnasium >= 0.29.0
- stable-baselines3 >= 2.0.0
- torch >= 2.0.0
- pandas >= 2.0.0
- yfinance >= 0.2.0
- pandas-ta >= 0.3.14b

### Step 2: Download the Files

Save these three files in the same directory:
1. `tradeEnv.py` - The improved trading environment
2. `train_trading_bot.py` - Training script
3. `backtest_trading_bot.py` - Backtesting script

## 🚀 Usage

### Training the Model

```bash
python train_trading_bot.py
```

**What it does:**
1. Downloads stock data for multiple tickers (AAPL, MSFT, GOOGL, TSLA, AMZN)
2. Calculates technical indicators (RSI, MACD, SMA, etc.)
3. Splits data into train/validation/test sets
4. Trains a PPO model for 500,000 timesteps
5. Evaluates on test data
6. Saves model to `trading_bot/models/final_model.zip`

**Training output:**
- Models saved to: `trading_bot/models/`
- Logs saved to: `trading_bot/logs/`
- Results plot: `trading_bot/results.png`

**Expected training time:** 30-60 minutes on CPU, 10-20 minutes on GPU

### Backtesting on New Data

After training, test the model on any stock:

```bash
python backtest_trading_bot.py
```

You'll be prompted to enter:
- Ticker symbol (e.g., AAPL, NVDA, META)
- Start date (default: 2024-01-01)
- End date (default: 2024-11-01)

**Output:**
- Performance comparison with buy-and-hold
- Detailed metrics (return, Sharpe ratio, max drawdown)
- Visualization saved to `trading_bot/backtest_results.png`

## 📊 Understanding the Results

### Key Metrics

1. **Total Return**: Percentage change in portfolio value
   - Example: 15.3% means you made 15.3% profit

2. **Sharpe Ratio**: Risk-adjusted returns (higher is better)
   - < 1: Poor
   - 1-2: Good
   - \> 2: Excellent

3. **Maximum Drawdown**: Largest peak-to-trough decline
   - Example: -12% means portfolio dropped 12% at worst point

4. **Outperformance**: RL return minus buy-and-hold return
   - Positive = beat the market
   - Negative = underperformed

### Interpreting Actions

- **Hold (0)**: Wait and observe
- **Buy (1)**: Purchase shares with available cash
- **Sell (2)**: Sell all shares to cash

## 🎯 Customization

### Training on Different Stocks

Edit `train_trading_bot.py`:

```python
# Change this line (around line 200)
TICKERS = ['YOUR', 'STOCKS', 'HERE']
```

### Adjusting Training Duration

```python
# More timesteps = better training but longer time
TOTAL_TIMESTEPS = 500000  # Increase for better results
```

### Changing Initial Capital

```python
# In train_trading_bot.py or backtest_trading_bot.py
INITIAL_CASH = 50000  # Default is 100000
```

### Modifying Transaction Costs

Edit `tradeEnv.py`:

```python
def __init__(self, data, initial_cash=100000.0, 
             transaction_cost=0.001):  # Change this (0.001 = 0.1%)
```

## 🐛 Troubleshooting

### Issue: "No module named 'gymnasium'"
**Solution:** Install gymnasium: `pip install gymnasium`

### Issue: "No data available"
**Solution:** Check your internet connection, yfinance may be rate-limited. Wait a few minutes and try again.

### Issue: Model performs poorly
**Solutions:**
- Train longer (increase TOTAL_TIMESTEPS to 1,000,000+)
- Add more diverse stocks to training
- Adjust hyperparameters in PPO model
- Ensure you have enough historical data (3+ years recommended)

### Issue: "KeyError: 'Close'"
**Solution:** Some stocks have insufficient data. Remove problematic tickers from the list.

## 📈 Performance Tips

1. **Train on multiple stocks** for better generalization
2. **Use 3+ years of data** for training
3. **Increase training timesteps** to 1M+ for production use
4. **Validate on out-of-sample data** before real trading
5. **Monitor transaction costs** - more trades = lower profit

## ⚠️ Important Notes

**This is for educational purposes only!**

- Past performance does not guarantee future results
- Do NOT use this for real trading without extensive testing
- Markets are unpredictable and models can fail
- Always do your own research and risk assessment
- Consider paper trading first

## 🔍 Why This is Better

| Aspect | Original Code | Improved Version |
|--------|---------------|------------------|
| Reward Function | Price direction | Actual portfolio returns |
| Action Space | Buy/Sell only | Hold/Buy/Sell |
| Observations | Price data only | Price + portfolio state |
| Training Data | Single stock | Multiple stocks |
| Transaction Costs | None | 0.1% per trade |
| Data Split | None | Train/Val/Test |
| Metrics | None | Comprehensive |
| Evaluation | None | Backtest + visualizations |

## 📚 Next Steps

1. **Improve features**: Add more technical indicators
2. **Try different algorithms**: A2C, SAC, TD3
3. **Hyperparameter tuning**: Optimize learning rate, batch size, etc.
4. **Ensemble models**: Combine multiple models
5. **Risk management**: Add stop-loss, position sizing
6. **Multi-asset**: Trade multiple stocks simultaneously

## 🤝 Contributing

Feel free to improve this code by:
- Adding more technical indicators
- Implementing better risk management
- Adding support for options/futures
- Creating a web interface
- Implementing real-time trading capabilities

## 📄 License

This code is provided as-is for educational purposes. Use at your own risk.

---

**Good luck with your trading bot! 🚀📈**
