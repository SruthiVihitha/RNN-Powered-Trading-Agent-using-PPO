# ==== Ablation Study: PPO Trading Agents ====
# Compares: LSTM vs GRU vs MLP vs MA Crossover vs Buy&Hold
import os, warnings, numpy as np, pandas as pd, yfinance as yf, matplotlib.pyplot as plt
import gymnasium as gym
import torch as th
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
th.manual_seed(SEED)

# ==== Load & Prepare Data ====
df = yf.download("SPY", start="2010-01-01", end="2024-12-31", auto_adjust=True)
df = df.rename(columns={"Adj Close":"AdjClose"})[["Open","High","Low","Close","Volume"]].copy()

# Features
df["daily_return"] = df["Close"].pct_change()
df["SMA_50"] = df["Close"].rolling(50).mean()
df["SMA_200"] = df["Close"].rolling(200).mean()
df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

def rsi(close, n=14):
    diff = close.diff()
    up, down = diff.clip(lower=0), -diff.clip(upper=0)
    gain = up.ewm(alpha=1/n, adjust=False).mean()
    loss = down.ewm(alpha=1/n, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100/(1+rs))

df["RSI"] = rsi(df["Close"], 14)
df["TR"] = np.maximum(df["High"] - df["Low"], np.maximum(
    np.abs(df["High"] - df["Close"].shift(1)),
    np.abs(df["Low"] - df["Close"].shift(1))))
df["ATR"] = df["TR"].rolling(14).mean()

ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
df["volatility"] = df["daily_return"].rolling(20).std()
df.dropna(inplace=True)

train_df = df.loc["2010-01-01":"2019-12-31"].copy()
test_df  = df.loc["2020-01-01":"2024-12-31"].copy()

features = ["Open","High","Low","Close","Volume","daily_return","SMA_50","SMA_200",
            "RSI","EMA_20","EMA_50","ATR","MACD","MACD_signal","MACD_hist","volatility"]
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_df[features])
scaled_test  = scaler.transform(test_df[features])

WINDOW = 60
def make_windows(arr, w):
    return np.asarray([arr[i:i+w, :] for i in range(len(arr)-w+1)], dtype=np.float32)

X_train = make_windows(scaled_train, WINDOW)
X_test  = make_windows(scaled_test, WINDOW)
train_prices = train_df["Close"].iloc[WINDOW-1:].to_numpy(dtype=np.float64)
test_prices  = test_df["Close"].iloc[WINDOW-1:].to_numpy(dtype=np.float64)

# ==== Trading Environment ====
class TradingEnv(gym.Env):
    def __init__(self, X_windows, end_prices, initial_balance=10_000.0, transaction_cost=0.001):
        super().__init__()
        self.X, self.price = X_windows, end_prices
        self.N, self.W, self.F = X_windows.shape
        self.initial_balance, self.tc = float(initial_balance), float(transaction_cost)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.W, self.F), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None: np.random.seed(seed)
        self.i, self.balance, self.shares = 0, float(self.initial_balance), 0
        self.portfolio = self.balance
        return self.X[self.i].astype(np.float32), {}

    def step(self, action):
        epsilon, price_now = 1e-9, float(self.price[self.i])
        price_prev = float(self.price[self.i-1]) if self.i > 0 else price_now
        prev_portfolio = self.portfolio

        if action == 1 and self.shares == 0:
            cost = price_now * (1 + self.tc)
            if self.balance >= cost:
                self.shares, self.balance = 1, max(self.balance - cost, 0.0)
        elif action == 2 and self.shares == 1:
            self.balance += price_now * (1 - self.tc)
            self.shares = 0

        self.portfolio = self.balance + self.shares * price_now
        agent_ret = (self.portfolio - prev_portfolio) / max(prev_portfolio, epsilon)
        mkt_ret = (price_now - price_prev) / max(price_prev, epsilon)
        reward = (agent_ret - 0.5 * mkt_ret) * 100.0
        if action == 0: reward -= 0.02

        self.i += 1
        done = self.i >= self.N
        obs = np.zeros((self.W, self.F), dtype=np.float32) if done else self.X[self.i].astype(np.float32)
        return obs, float(reward), done, False, {"portfolio_value": float(self.portfolio)}

# ==== Feature Extractors ====
class LSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.lstm = nn.LSTM(observation_space.shape[1], features_dim, batch_first=True)
        self._features_dim = features_dim
    def forward(self, obs):
        _, (h, _) = self.lstm(obs)
        return h[-1]

class GRUExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        self.gru = nn.GRU(observation_space.shape[1], features_dim, batch_first=True)
        self._features_dim = features_dim
    def forward(self, obs):
        _, h = self.gru(obs)
        return h[-1]

class MLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        W, F = observation_space.shape
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(W * F, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
        self._features_dim = features_dim
    def forward(self, obs):
        return self.net(obs)

# ==== Policy Classes ====
class LSTMPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, features_extractor_class=LSTMExtractor, 
                        features_extractor_kwargs={"features_dim":64}, **kwargs)

class GRUPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, features_extractor_class=GRUExtractor,
                        features_extractor_kwargs={"features_dim":64}, **kwargs)

class MLPPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, features_extractor_class=MLPExtractor,
                        features_extractor_kwargs={"features_dim":64}, **kwargs)

# ==== Training Function ====
def train_agent(policy_class, name, timesteps=300_000):
    print(f"\n{'='*60}\nTraining {name}...\n{'='*60}")
    vec_env = DummyVecEnv([lambda: Monitor(TradingEnv(X_train, train_prices))])
    vec_env.seed(SEED)

    model = PPO(
        policy_class, vec_env,
        verbose=1,                     # show logs
        learning_rate=1e-5,
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99, gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.02, vf_coef=0.5,
        max_grad_norm=0.5,
        seed=SEED
    )
    model.learn(total_timesteps=timesteps)

    # Save each model for later use in the app
    os.makedirs("models", exist_ok=True)
    safe_name = name.lower().replace(" ", "_")
    path = f"models/ppo_{safe_name}_{timesteps//1000}k.zip"
    model.save(path)
    print(f"[saved] {path}")

    return model


# ==== Evaluation Function ====
def evaluate_agent(model, name):
    env = TradingEnv(X_test, test_prices)
    obs, _ = env.reset()
    values, actions = [env.portfolio], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, _, done, _, info = env.step(int(action))
        values.append(info["portfolio_value"])
    return np.array(values), actions

# ==== MA Crossover Baseline ====
def ma_crossover_baseline():
    print(f"\n{'='*60}\nEvaluating MA Crossover Baseline...\n{'='*60}")
    balance, shares, values = 10_000.0, 0, []

    for i in range(len(test_prices)):
        price = float(test_prices[i])
        idx = WINDOW - 1 + i
        sma50 = float(test_df["SMA_50"].iloc[idx])
        sma200 = float(test_df["SMA_200"].iloc[idx])

        # Buy signal: SMA50 crosses above SMA200
        if sma50 > sma200 and shares == 0 and balance >= price * 1.001:
            shares = 1
            balance -= price * 1.001  # include fee
        # Sell signal: SMA50 crosses below SMA200
        elif sma50 < sma200 and shares == 1:
            balance += price * 0.999  # include fee
            shares = 0

        values.append(balance + shares * price)

    # Ensure 1-D array and prepend initial cash for alignment with other curves
    vals = np.array(values, dtype=np.float64).reshape(-1)
    vals = np.insert(vals, 0, 10_000.0)
    return vals, []


# ==== Metrics ====
def compute_metrics(values, prices=None):
    values = np.asarray(values, dtype=np.float64).reshape(-1)  # <— flatten to 1D
    epsilon = 1e-9
    returns = np.diff(values) / (values[:-1] + epsilon)
    
    
    sharpe = np.mean(returns) / (np.std(returns) + epsilon) * np.sqrt(252)
    downside = returns[returns < 0]
    sortino = np.mean(returns) / (np.std(downside) + epsilon) * np.sqrt(252)
    
    peak = np.maximum.accumulate(values)
    mdd = np.max((peak - values) / (peak + epsilon))
    
    years = len(values) / 252
    cagr = (values[-1] / values[0]) ** (1/years) - 1 if years > 0 else 0
    
    return {"sharpe": sharpe, "sortino": sortino, "mdd": mdd, "cagr": cagr, "final": values[-1]}

# ==== Run Ablation Study ====
print("\n" + "="*60)
print("ABLATION STUDY: RNN-based RL Trading Agents")
print("="*60)

results = {}

# 1. LSTM Agent
model_lstm = train_agent(LSTMPolicy, "LSTM Agent", timesteps=300_000)
values_lstm, actions_lstm = evaluate_agent(model_lstm, "LSTM")
results["LSTM"] = compute_metrics(values_lstm, test_prices)

# 2. GRU Agent
model_gru = train_agent(GRUPolicy, "GRU Agent", timesteps=300_000)
values_gru, actions_gru = evaluate_agent(model_gru, "GRU")
results["GRU"] = compute_metrics(values_gru, test_prices)

# 3. MLP Agent (no recurrence)
model_mlp = train_agent(MLPPolicy, "MLP Agent (No RNN)", timesteps=300_000)
values_mlp, actions_mlp = evaluate_agent(model_mlp, "MLP")
results["MLP"] = compute_metrics(values_mlp, test_prices)

# 4. MA Crossover
values_ma, _ = ma_crossover_baseline()
results["MA Cross"] = compute_metrics(values_ma, test_prices)

# 5. Buy & Hold
initial_cash = 10_000.0
shares_bh = initial_cash / test_prices[0]
values_bh = np.insert(shares_bh * test_prices, 0, initial_cash)
results["Buy&Hold"] = compute_metrics(values_bh, test_prices)

# ==== Print Results ====
print("\n" + "="*80)
print("ABLATION STUDY RESULTS (Test Period: 2020-2024)")
print("="*80)
print(f"\n{'Strategy':<15} {'Final ($)':<12} {'Sharpe':<10} {'Sortino':<10} {'MaxDD':<10} {'CAGR':<10}")
print("-"*80)

for name, metrics in results.items():
    print(f"{name:<15} ${metrics['final']:>10,.2f} {metrics['sharpe']:>9.3f} {metrics['sortino']:>9.3f} "
          f"{metrics['mdd']:>9.3f} {metrics['cagr']:>8.2%}")

print("="*80)

# ==== Visualizations ====
plt.figure(figsize=(16, 10))

# Equity curves
plt.subplot(2, 2, 1)
plt.plot(values_lstm, label='LSTM', linewidth=2)
plt.plot(values_gru, label='GRU', linewidth=2)
plt.plot(values_mlp, label='MLP', linewidth=2)
plt.plot(values_ma, label='MA Crossover', linewidth=2)
plt.plot(values_bh, label='Buy & Hold', linewidth=2, linestyle='--')
plt.title('Equity Curves Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Time Steps')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Sharpe comparison
plt.subplot(2, 2, 2)
names = list(results.keys())
sharpes = [results[n]['sharpe'] for n in names]
colors = ['green' if s > results['Buy&Hold']['sharpe'] else 'red' for s in sharpes]
plt.bar(names, sharpes, color=colors, alpha=0.7)
plt.axhline(y=results['Buy&Hold']['sharpe'], color='black', linestyle='--', label='Buy&Hold Baseline')
plt.title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.grid(True, alpha=0.3)

# MaxDD comparison
plt.subplot(2, 2, 3)
mdds = [results[n]['mdd'] for n in names]
colors = ['green' if m < results['Buy&Hold']['mdd'] else 'red' for m in mdds]
plt.bar(names, mdds, color=colors, alpha=0.7)
plt.axhline(y=results['Buy&Hold']['mdd'], color='black', linestyle='--', label='Buy&Hold Baseline')
plt.title('Maximum Drawdown Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Max Drawdown')
plt.legend()
plt.grid(True, alpha=0.3)

# CAGR comparison
plt.subplot(2, 2, 4)
cagrs = [results[n]['cagr'] * 100 for n in names]
colors = ['green' if c > results['Buy&Hold']['cagr']*100 else 'red' for c in cagrs]
plt.bar(names, cagrs, color=colors, alpha=0.7)
plt.axhline(y=results['Buy&Hold']['cagr']*100, color='black', linestyle='--', label='Buy&Hold Baseline')
plt.title('CAGR Comparison', fontsize=14, fontweight='bold')
plt.ylabel('CAGR (%)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ablation_study_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Ablation study complete! Results saved to 'ablation_study_results.png'")