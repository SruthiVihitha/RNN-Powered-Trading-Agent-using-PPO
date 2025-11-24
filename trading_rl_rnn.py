# ==== Imports ====
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

# ==== Reproducibility ====
SEED = 42
np.random.seed(SEED)
th.manual_seed(SEED)

# ==== Data (SPY) ====
df = yf.download("SPY", start="2010-01-01", end="2024-12-31", auto_adjust=True)
df = df.rename(columns={"Adj Close":"AdjClose"})[["Open","High","Low","Close","Volume"]].copy()

# ==== Enhanced Features ====
df["daily_return"] = df["Close"].pct_change()
df["SMA_50"] = df["Close"].rolling(50).mean()
df["SMA_200"] = df["Close"].rolling(200).mean()
df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

# RSI
def rsi(close, n=14):
    diff = close.diff()
    up, down = diff.clip(lower=0), -diff.clip(upper=0)
    gain = up.ewm(alpha=1/n, adjust=False).mean()
    loss = down.ewm(alpha=1/n, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100/(1+rs))
df["RSI"] = rsi(df["Close"], 14)

# ATR (14)
df["TR"] = np.maximum(
    df["High"] - df["Low"],
    np.maximum(
        np.abs(df["High"] - df["Close"].shift(1)),
        np.abs(df["Low"] - df["Close"].shift(1))
    )
)
df["ATR"] = df["TR"].rolling(14).mean()

# MACD (12, 26, 9)
ema12 = df["Close"].ewm(span=12, adjust=False).mean()
ema26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

# Volatility (rolling std 20d)
df["volatility"] = df["daily_return"].rolling(20).std()

df.dropna(inplace=True)

# ==== Split ====
train_df = df.loc["2010-01-01":"2019-12-31"].copy()
test_df  = df.loc["2020-01-01":"2024-12-31"].copy()

features = ["Open","High","Low","Close","Volume","daily_return","SMA_50","SMA_200",
            "RSI","EMA_20","EMA_50","ATR","MACD","MACD_signal","MACD_hist","volatility"]
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_df[features])
scaled_test  = scaler.transform(test_df[features])

# ==== Windows ====
WINDOW = 60
def make_windows(arr, w):
    X = [arr[i:i+w, :] for i in range(len(arr)-w+1)]
    return np.asarray(X, dtype=np.float32)
X_train = make_windows(scaled_train, WINDOW)
X_test  = make_windows(scaled_test, WINDOW)
train_prices = train_df["Close"].iloc[WINDOW-1:].to_numpy(dtype=np.float64)
test_prices  = test_df["Close"].iloc[WINDOW-1:].to_numpy(dtype=np.float64)
N_train, W, F = X_train.shape

# ==== Enhanced Trading Env with Improved Reward Shaping ====
class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(self, X_windows, end_prices, initial_balance=10_000.0, transaction_cost=0.001):
        super().__init__()
        self.X = X_windows
        self.price = end_prices
        self.N, self.W, self.F = X_windows.shape
        self.initial_balance = float(initial_balance)
        self.tc = float(transaction_cost)

        self.action_space = gym.spaces.Discrete(3)  # 0 Hold, 1 Buy, 2 Sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.W, self.F), dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.i = 0
        self.balance = float(self.initial_balance)
        self.shares = 0
        self.portfolio = self.balance
        self.history = []
        return self.X[self.i].astype(np.float32), {}

    def step(self, action):
        epsilon = 1e-9
        price_now = float(self.price[self.i])
        price_prev = float(self.price[self.i-1]) if self.i > 0 else price_now
        
        prev_portfolio = self.portfolio

        # Execute action (1 share only, long-only with balance guards)
        if action == 1 and self.shares == 0:
            cost = price_now * (1 + self.tc)
            if self.balance >= cost:
                self.shares = 1
                self.balance = max(self.balance - cost, 0.0)
                self.history.append(("Buy", self.i, price_now))
        elif action == 2 and self.shares == 1:
            proceeds = price_now * (1 - self.tc)
            self.balance += proceeds
            self.shares = 0
            self.history.append(("Sell", self.i, price_now))

        # Update portfolio value
        self.portfolio = self.balance + self.shares * price_now

        # Risk-adjusted outperformance reward
        agent_ret = (self.portfolio - prev_portfolio) / max(prev_portfolio, epsilon)
        mkt_ret = (price_now - price_prev) / max(price_prev, epsilon)
        reward = agent_ret - 0.5 * mkt_ret
        
        # HOLD penalty to discourage idling
        if action == 0:
            reward -= 2e-4

        # Scale for better gradient signal
        reward *= 100.0

        self.i += 1
        done = self.i >= self.N
        terminated = done
        truncated = False
        obs = np.zeros((self.W, self.F), dtype=np.float32) if done else self.X[self.i].astype(np.float32)
        info = {
            "portfolio_value": float(self.portfolio), 
            "balance": float(self.balance), 
            "shares_held": int(self.shares)
        }
        return obs, float(reward), terminated, truncated, info

# ==== LSTM Feature Extractor ====
class LSTMExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        W, F = observation_space.shape
        self.lstm = nn.LSTM(input_size=F, hidden_size=features_dim, batch_first=True)
        self._features_dim = features_dim

    def forward(self, obs: th.Tensor) -> th.Tensor:
        out, (h, c) = self.lstm(obs)
        return h[-1]

class LSTMPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=LSTMExtractor,
            features_extractor_kwargs={"features_dim":64},
            **kwargs
        )

# ==== Vectorized env for training ====
def make_train_env():
    env = TradingEnv(X_train, train_prices, initial_balance=10_000.0, transaction_cost=0.001)
    return Monitor(env)
vec_env = DummyVecEnv([make_train_env])
vec_env.seed(SEED)

# ==== PPO Training with Enhanced Hyperparameters ====
device = "cuda" if th.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = PPO(
    LSTMPolicy,
    vec_env,
    verbose=1,
    learning_rate=1e-5,
    n_steps=2048,
    batch_size=1024,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.15,
    ent_coef=0.02,        # Increased exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    seed=SEED,
    device=device
)

print("Training...")
model.learn(total_timesteps=1_000_000)  # Increased training time
os.makedirs("./models", exist_ok=True)
model.save("./models/ppo_trading_lstm_enhanced")

# ==== Deterministic Backtest (Test split) ====
test_env = TradingEnv(X_test, test_prices, initial_balance=10_000.0, transaction_cost=0.001)
obs, _ = test_env.reset()
agent_values, actions = [test_env.portfolio], []
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    actions.append(int(action))
    obs, reward, terminated, truncated, info = test_env.step(int(action))
    done = terminated or truncated
    agent_values.append(info["portfolio_value"])
agent_values = np.asarray(agent_values, dtype=np.float64)

# Buy & Hold aligned curve
epsilon = 1e-9
bh_prices = test_prices
initial_cash = 10_000.0
shares_bh = 0.0 if bh_prices[0] <= 0 else initial_cash / bh_prices[0]
bh_curve = np.insert(shares_bh * bh_prices, 0, initial_cash).astype(np.float64)

# Align lengths
m = min(len(agent_values), len(bh_curve))
agent_values, bh_curve = agent_values[:m], bh_curve[:m]

# ==== Metrics ====
def daily_returns(v):
    return np.diff(v) / (v[:-1] + epsilon) if len(v) > 1 else np.array([0.0])

def sharpe(v):
    r = daily_returns(v)
    return float(np.mean(r) / (np.std(r) + epsilon) * np.sqrt(252))

def max_drawdown(v):
    peak = np.maximum.accumulate(v)
    dd = (peak - v) / (peak + epsilon)
    return float(np.max(dd))

agent_sr, bh_sr = sharpe(agent_values), sharpe(bh_curve)
agent_mdd, bh_mdd = max_drawdown(agent_values), max_drawdown(bh_curve)
num_trades = sum(a in (1,2) for a in actions)

print("\n=== Performance (Test) ===")
print(f"Trades: {num_trades}")
print(f"Sharpe  Agent: {agent_sr:.3f} | B&H: {bh_sr:.3f} | Diff: {agent_sr - bh_sr:+.3f}")
print(f"MaxDD   Agent: {agent_mdd:.3f} | B&H: {bh_mdd:.3f} | Diff: {agent_mdd - bh_mdd:+.3f}")
print(f"Final   Agent: ${agent_values[-1]:.2f} | B&H: ${bh_curve[-1]:.2f} | Diff: ${agent_values[-1] - bh_curve[-1]:+.2f}")
print(f"\nAgent vs B&H: {'✓ BETTER' if agent_sr >= bh_sr and agent_mdd <= bh_mdd else '✗ WORSE'} (Sharpe & DrawDown)")

# ==== Plots ====
plt.figure(figsize=(14,6))
plt.plot(agent_values, label=f"Agent (Sharpe={agent_sr:.3f}, MDD={agent_mdd:.3f})", linewidth=2)
plt.plot(bh_curve, label=f"Buy & Hold (Sharpe={bh_sr:.3f}, MDD={bh_mdd:.3f})", linewidth=2, alpha=0.7)
plt.title("Equity Curves (Test) - Enhanced PPO Agent", fontsize=14, fontweight='bold')
plt.xlabel("Steps")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,4))
colors = ['gray' if a == 0 else 'green' if a == 1 else 'red' for a in actions]
plt.scatter(range(len(actions)), actions, s=12, c=colors, alpha=0.6)
plt.yticks([0,1,2], ["Hold","Buy","Sell"])
plt.title(f"Actions over Time (Test) - Total Trades: {num_trades}", fontsize=12, fontweight='bold')
plt.xlabel("Steps")
plt.ylabel("Action")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()