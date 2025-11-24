# Reinforcement Learning with RNN-based Market State Representation for Algorithmic Trading

An end-to-end project that trains, evaluates, and visualizes RL trading agents which use RNNs (LSTM / GRU) or MLPs as market-state encoders. Includes an ablation study (LSTM vs GRU vs MLP), a production-grade LSTM agent trained with PPO, and a Streamlit GUI to load models, backtest and generate live signals.

---

## 🚀 Project Summary (one-liner)

Train RL agents (BUY / SELL / HOLD) on historical stock data using RNN-based feature extractors and compare them against rule baselines (MA Crossover, Buy & Hold). Provide an interactive Streamlit app for backtesting and signal generation.

---

## Why this project?

* Financial markets are time-series — RNNs (LSTM/GRU) can capture temporal dependencies.
* Reinforcement learning lets an agent learn an objective (risk-adjusted outperformance).
* Ablation study helps evaluate whether temporal memory (RNN) adds value over MLP or classical rules.

---

## Repository layout (what each file/folder does)

```
dl_project/
├── trading_rl_rnn.py            # Train the enhanced LSTM PPO agent (long training, saved model)
├── ablation_study.py            # Train LSTM/GRU/MLP agents, evaluate, compare vs MA Crossover & B&H, save plot
├── app.py                       # Streamlit GUI for loading a saved model, backtest, visualizations & current signal
├── models/                      # Saved trained models (zips)
│   ├── ppo_lstm_agent_300k.zip
│   ├── ppo_gru_agent_300k.zip
│   ├── ppo_mlp_agent_(no_rnn)_300k.zip
│   └── ppo_trading_lstm_enhanced.zip
├── ablation_study_results.png   # Ablation visualization
├── README.md                    # <-- you are reading it
└── requirements.txt             # Python packages used
```

---

## Key components explained (high level)

### 1. Data & features

* Uses `yfinance` to download historical stock data (Open, High, Low, Close, Volume).
* Computes technical indicators per timestep:

  * `daily_return`, `SMA_50`, `SMA_200`, `EMA_20`, `EMA_50`, `RSI`, `ATR`, `MACD`, `MACD_signal`, `MACD_hist`, `volatility`
* Sliding window: each observation = last `WINDOW` days × `F` features (default WINDOW=60).

### 2. Custom Gym Environment (`TradingEnv`)

* Observation: `(WINDOW, F)` array (sequence of features).
* Action space: `Discrete(3)` → {0: Hold, 1: Buy, 2: Sell}
* Execution logic: buy/sell 1 share only (long-only), pay transaction cost.
* Portfolio tracking: `balance`, `shares`, `portfolio_value`.
* Reward shaping: `reward = agent_ret - market_weight * market_ret` (with optional hold penalty and scaling). This encourages *outperforming the market* rather than absolute profit only.

### 3. Feature extractors / Policies

* `LSTMExtractor` / `GRUExtractor` / `MLPExtractor` inherit from SB3 `BaseFeaturesExtractor`.

  * LSTM/GRU transform `(batch, WINDOW, F)` → `(batch, hidden_dim)` using the final hidden state `h[-1]`.
  * MLP flattens window and passes through an MLP to create a latent.
* Policies (`LSTMPolicy`, `GRUPolicy`, `MLPPolicy`) instruct SB3 to use the chosen extractor.

### 4. RL algorithm

* Uses `stable-baselines3` PPO for on-policy optimization.
* Trains each agent for a fixed number of timesteps (e.g., 300k for ablation; 1M for enhanced LSTM).
* Saves models to `models/`.

### 5. Baselines

* **MA Crossover**: simple rule-based system:

  * If `SMA_50 > SMA_200` → buy (if not holding)
  * If `SMA_50 < SMA_200` → sell (if holding)
  * Trades 1 share at a time, includes transaction cost in calculation.
* **Buy & Hold (B&H)**:

  * Buy as many shares as initial cash allows at the first test price and hold to the end.
  * Used as a primary baseline (long-only buy-and-hold return & drawdown).

### 6. Ablation study (ablation_study.py)

* Trains LSTM, GRU, MLP with identical PPO hyperparameters.
* Evaluates each on the test period (2020–2024).
* Computes metrics: Sharpe, Sortino, Max Drawdown, CAGR, Final portfolio.
* Plots comparisons and saves `ablation_study_results.png`.

### 7. Streamlit app (app.py)

* Interactive GUI to:

  * Fetch ticker data for chosen date ranges,
  * Load a saved PPO `.zip` model (upload or local path),
  * Run deterministic backtest of the model on test data and compare vs Buy & Hold,
  * Show metrics (Final, Sharpe, MaxDD, CAGR, Sortino, Total Trades),
  * Show equity curve and action scatter,
  * Produce current signal by feeding the latest window to model (`HOLD / BUY / SELL`) and display latest price.

---

## How the full workflow runs (example: select a stock `SPY`)

1. **Choose time ranges & window** in the Streamlit GUI or in script defaults.
2. **Data fetch** (yfinance) → features added → `MinMaxScaler` applied.
3. **Make windows**: sliding windows of shape `(n_windows, WINDOW, F)`.
4. **Load model** (e.g., `models/ppo_trading_lstm_enhanced.zip`) with SB3 `PPO.load()`.
5. **Backtest**:

   * Reset `TradingEnv` on test windows.
   * Loop: `action = model.predict(obs, deterministic=True)` → `env.step(action)` → record `portfolio_value`.
   * Build agent equity curve and buy & hold curve for comparison.
6. **Compute metrics**:

   * Daily returns → Sharpe (annualized), Sortino, Max Drawdown, CAGR.
7. **Visualize**: equity curves, actions, bar plots of metrics.
8. **Signal**: last window fed to `model.predict(last_obs)` → map to `HOLD/BUY/SELL`. Latest price displayed.

---

## Quick start (run locally)

> Recommended Python: 3.9–3.11 (matches venv used). Use a dedicated venv.

1. Clone repository:

```bash
git clone <your-repo-url>
cd dl_project
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

2. Train (example: ablation study)

```bash
python ablation_study.py
# This will train 3 agents (LSTM/GRU/MLP) for default timesteps and save models into ./models/
```

3. Train enhanced LSTM (long run)

```bash
python trading_rl_rnn.py
# Trains the enhanced LSTM PPO for more timesteps and saves ppo_trading_lstm_enhanced.zip
```

4. Run Streamlit UI

```bash
streamlit run app.py
# Open http://localhost:8501 in your browser
```

---

## Typical commands / notes & troubleshooting

* **Gym vs Gymnasium warning**: If you see `Gym has been unmaintained... upgrade to Gymnasium`, replace imports or install gymnasium:

  ```bash
  pip install gymnasium
  # or ensure your code says `import gymnasium as gym`
  ```
* **NumPy scalar/shape warnings**: Ensure `test_prices` and `train_prices` are 1-D arrays (`.reshape(-1)` or `np.asarray(...).reshape(-1)`) and that observations passed to `model.predict` have the same shape expected by the model:

  * For single-window prediction: `last_obs = X_test[-1:].astype(np.float32)` gives shape `(1, WINDOW, F)` which SB3 expects for vectorized Box observations.
  * Convert model output action to scalar safely:

    ```python
    action, _ = model.predict(last_obs, deterministic=True)
    action_int = int(np.asarray(action).reshape(()))
    ```
* **Streamlit TypeError when formatting arrays**: `test_prices[-1]` must be a scalar float; if it’s a 1-element `np.ndarray`, cast: `latest_price = float(np.asarray(test_prices).reshape(-1)[-1])`.

---

## Metrics definitions (brief)

* **Sharpe ratio**: mean(daily_returns) / std(daily_returns) × sqrt(252)
* **Sortino ratio**: mean(daily_returns) / std(downside_returns) × sqrt(252)
* **Max Drawdown (MDD)**: max decline from peak to trough experienced by portfolio
* **CAGR**: annualized growth rate between first and last portfolio values

---

## What is MA Crossover?

A simple rule-based strategy using moving averages:

* Compute short-term SMA (50-day) and long-term SMA (200-day).
* If SMA_short crosses above SMA_long → bullish → **Buy**.
* If SMA_short crosses below SMA_long → bearish → **Sell**.
  This strategy attempts to follow medium/long-term trends.

---

## Results 
<img width="1512" height="982" alt="Screenshot 2025-11-06 at 8 42 00 AM" src="https://github.com/user-attachments/assets/ea6702a3-dcb3-4dbc-9d89-7e29b18a42a0" />
<img width="1512" height="982" alt="Screenshot 2025-11-06 at 8 42 31 AM" src="https://github.com/user-attachments/assets/22e899dd-e665-4115-8ea8-0b97b4907439" />
<img width="1512" height="982" alt="Screenshot 2025-11-06 at 8 42 55 AM" src="https://github.com/user-attachments/assets/cb422070-83f9-40eb-bb45-00b6caf0d086" />
<img width="1512" height="982" alt="Screenshot 2025-11-06 at 8 43 03 AM" src="https://github.com/user-attachments/assets/940572fc-ba5d-49fa-b7ef-5763b648d934" />
<img width="3024" height="1964" alt="Screenshot 2025-11-06 at 8 43 24 AM" src="https://github.com/user-attachments/assets/9ccc58de-defe-4036-a3b7-cffc24aa58d4" />

---
## License

MIT License — feel free to reuse and adapt for research or portfolio demos.

---

## Contact

Sruthi Vihitha Potluri — `sruthivihitha_potluri@srmap.edu.in`
Repo owner: [SruthiVihitha](https://github.com/SruthiVihitha) 

---

## Acknowledgements

Built with: `yfinance`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `torch`, `stable-baselines3`, `gymnasium` / `gym`, and `streamlit`.

---
