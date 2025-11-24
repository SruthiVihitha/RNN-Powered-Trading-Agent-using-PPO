import io, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import gymnasium as gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO

st.set_page_config(page_title="RL Trading GUI", layout="wide")
window = 60
st.sidebar.info("Window size fixed to 60 to match the trained models.")

# =============================
# Feature engineering helpers
# =============================
def rsi(close, n=14):
    diff = close.diff()
    up, down = diff.clip(lower=0), -diff.clip(upper=0)
    gain = up.ewm(alpha=1/n, adjust=False).mean()
    loss = down.ewm(alpha=1/n, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100/(1+rs))

def add_features(df):
    df = df.copy()
    df["daily_return"] = df["Close"].pct_change()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    df["RSI"] = rsi(df["Close"], 14)
    tr = np.maximum(df["High"] - df["Low"], np.maximum(
        (df["High"] - df["Close"].shift(1)).abs(), (df["Low"] - df["Close"].shift(1)).abs()))
    df["ATR"] = tr.rolling(14).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    df["volatility"] = df["daily_return"].rolling(20).std()
    return df

FEATURES = ["Open","High","Low","Close","Volume","daily_return","SMA_50","SMA_200",
            "RSI","EMA_20","EMA_50","ATR","MACD","MACD_signal","MACD_hist","volatility"]

def make_windows(arr, w):
    return np.asarray([arr[i:i+w, :] for i in range(len(arr)-w+1)], dtype=np.float32)

# =============================
# Environment
# =============================
class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, X_windows, end_prices, initial_balance=10_000.0, transaction_cost=0.001,
                 hold_penalty=0.0002, market_weight=0.5):
        super().__init__()
        self.X = X_windows
        # 🔧 ensure prices are a 1-D float array (fixes scalar warnings)
        self.price = np.asarray(end_prices, dtype=np.float64).reshape(-1)

        self.N, self.W, self.F = X_windows.shape
        self.initial_balance = float(initial_balance)
        self.tc = float(transaction_cost)
        self.hold_penalty = float(hold_penalty)
        self.market_weight = float(market_weight)

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.W, self.F), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.i = 0
        self.balance = float(self.initial_balance)
        self.shares = 0
        self.portfolio = self.balance
        return self.X[self.i].astype(np.float32), {}

    def step(self, action):
        eps = 1e-9
        # already scalar values now
        price_now = self.price[self.i]
        price_prev = self.price[self.i-1] if self.i > 0 else price_now
        prev_value = self.portfolio

        # Execute actions
        if action == 1 and self.shares == 0:  # BUY
            cost = price_now * (1 + self.tc)
            if self.balance >= cost:
                self.balance -= cost
                self.shares = 1
        elif action == 2 and self.shares == 1:  # SELL
            self.balance += price_now * (1 - self.tc)
            self.shares = 0

        self.portfolio = self.balance + self.shares * price_now

        # Returns
        agent_ret = (self.portfolio - prev_value) / max(prev_value, eps)
        mkt_ret = (price_now - price_prev) / max(price_prev, eps)

        reward = agent_ret - self.market_weight * mkt_ret
        if action == 0:
            reward -= self.hold_penalty
        reward *= 100.0

        self.i += 1
        done = self.i >= self.N
        obs = (np.zeros((self.W, self.F), dtype=np.float32)
               if done else self.X[self.i].astype(np.float32))
        info = {"portfolio_value": float(self.portfolio),
                "price": float(price_now), "shares": int(self.shares)}
        return obs, float(reward), done, False, info


# =============================
# Feature extractors & policies
# =============================
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
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(W*F, 128), nn.ReLU(),
                                 nn.Linear(128, features_dim), nn.ReLU())
        self._features_dim = features_dim
    def forward(self, obs): return self.net(obs)

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

# =============================
# Metrics
# =============================
def compute_metrics(values):
    eps = 1e-9
    r = np.diff(values) / (values[:-1] + eps)
    sharpe = (r.mean() / (r.std() + eps)) * np.sqrt(252) if len(r)>1 else 0.0
    downside = r[r < 0]
    sortino = (r.mean() / (downside.std() + eps)) * np.sqrt(252) if len(downside)>1 else 0.0
    peak = np.maximum.accumulate(values)
    mdd = np.max((peak - values) / (peak + eps))
    years = len(values) / 252
    cagr = (values[-1]/values[0])**(1/years)-1 if years>0 else 0.0
    return dict(sharpe=sharpe, sortino=sortino, mdd=mdd, cagr=cagr, final=values[-1])

def evaluate_agent(model, X_test, test_prices, window):
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

# =============================
# UI Header
# =============================
st.title("🤖 RL Trading Agent Analyzer")
st.markdown("**Interactive backtesting and signal generation for LSTM/GRU/MLP trading agents**")

# =============================
# Sidebar controls
# =============================
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Ticker", value="SPY")
train_start = st.sidebar.date_input("Train start", pd.to_datetime("2010-01-01"))
train_end   = st.sidebar.date_input("Train end",   pd.to_datetime("2019-12-31"))
test_start  = st.sidebar.date_input("Test start",  pd.to_datetime("2020-01-01"))
test_end    = st.sidebar.date_input("Test end",    pd.to_datetime("2024-12-31"))

st.sidebar.markdown("---")
st.sidebar.subheader("Model Settings")
window = st.sidebar.number_input("LSTM Window", min_value=20, max_value=200, value=60, step=5)
tcost = st.sidebar.number_input("Transaction Cost (%)", value=0.10, step=0.05) / 100.0
hold_pen = st.sidebar.number_input("Hold Penalty (×1e-4)", value=2) * 1e-4
mkt_w = st.sidebar.slider("Market Outperformance Weight", 0.0, 1.0, 0.5, 0.05)

model_type = st.sidebar.selectbox("Model Type", ["LSTM","GRU","MLP"])

st.sidebar.markdown("---")
st.sidebar.subheader("Load Trained Model")
uploaded_zip = st.sidebar.file_uploader("Upload PPO .zip", type=["zip"])
model_path = st.sidebar.text_input("Or local path (.zip)", value="./models/ppo_trading_lstm_enhanced.zip")

st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
run_bt = col1.button("📊 Backtest", use_container_width=True)
signal_btn = col2.button("🎯 Signal", use_container_width=True)

# =============================
# Data fetch + prepare
# =============================
@st.cache_data(show_spinner=False)
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df = df.rename(columns={"Adj Close":"AdjClose"})[["Open","High","Low","Close","Volume"]]
    df = add_features(df).dropna()
    return df

if run_bt or signal_btn:
    with st.spinner(f"Fetching {ticker} data..."):
        df = fetch_data(ticker, train_start, test_end)

    train_df = df.loc[str(train_start):str(train_end)].copy()
    test_df  = df.loc[str(test_start):str(test_end)].copy()

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_df[FEATURES])
    scaled_test  = scaler.transform(test_df[FEATURES])

    X_train = make_windows(scaled_train, window)
    X_test  = make_windows(scaled_test, window)
    train_prices = np.asarray(train_df["Close"].iloc[window-1:].to_numpy(), dtype=np.float64).reshape(-1)
    test_prices  = np.asarray(test_df["Close"].iloc[window-1:].to_numpy(),  dtype=np.float64).reshape(-1)


    # Load model
    policy_class = {"LSTM": LSTMPolicy, "GRU": GRUPolicy, "MLP": MLPPolicy}[model_type]

    model = None
    if uploaded_zip is not None:
        with st.spinner("Loading uploaded model..."):
            tmp = "uploaded_model.zip"
            with open(tmp, "wb") as f: 
                f.write(uploaded_zip.read())
            model = PPO.load(tmp, device="cpu")
    elif model_path and os.path.exists(model_path):
        with st.spinner("Loading model from path..."):
            model = PPO.load(model_path, device="cpu")
    else:
        st.error("❌ Please upload a trained PPO .zip or provide a valid local path.")
        st.stop()

    # =============================
    # Backtest
    # =============================
    if run_bt:
        with st.spinner("Running backtest..."):
            values, actions = evaluate_agent(model, X_test, test_prices, window)

            # Buy & Hold baseline
            initial_cash = 10_000.0
            shares_bh = initial_cash / test_prices[0]
            values_bh = np.insert(shares_bh * test_prices, 0, initial_cash)

            m_agent = compute_metrics(values)
            m_bh = compute_metrics(values_bh)

        st.success("✅ Backtest Complete!")
        
        # Metrics comparison
        st.subheader("📈 Performance Comparison (Test Period)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delta_final = m_agent['final'] - m_bh['final']
            st.metric("Final Value", f"${m_agent['final']:,.2f}", 
                     f"${delta_final:+,.2f} vs B&H", delta_color="normal")
        with col2:
            delta_sharpe = m_agent['sharpe'] - m_bh['sharpe']
            st.metric("Sharpe Ratio", f"{m_agent['sharpe']:.3f}", 
                     f"{delta_sharpe:+.3f} vs B&H", delta_color="normal")
        with col3:
            delta_mdd = m_agent['mdd'] - m_bh['mdd']
            st.metric("Max Drawdown", f"{m_agent['mdd']:.3f}", 
                     f"{delta_mdd:+.3f} vs B&H", delta_color="inverse")
        with col4:
            delta_cagr = m_agent['cagr'] - m_bh['cagr']
            st.metric("CAGR", f"{m_agent['cagr']:.2%}", 
                     f"{delta_cagr:+.2%} vs B&H", delta_color="normal")

        # Additional metrics
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Sortino (Agent)", f"{m_agent['sortino']:.3f}")
        with col6:
            num_trades = sum(a in (1,2) for a in actions)
            st.metric("Total Trades", f"{num_trades:,}")

        # Visualizations
        st.subheader("📊 Equity Curves & Actions")
        
        fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        
        # Equity curves
        ax[0].plot(values, label=f"{model_type} Agent", lw=2.5, color='#1f77b4')
        ax[0].plot(values_bh, label="Buy & Hold", lw=2.5, ls="--", color='#ff7f0e', alpha=0.8)
        ax[0].set_title(f"{ticker} Equity Curves (Test Period)", fontsize=14, fontweight='bold')
        ax[0].set_ylabel("Portfolio Value ($)", fontsize=12)
        ax[0].grid(alpha=0.3, linestyle='--')
        ax[0].legend(fontsize=11, loc='upper left')
        ax[0].set_facecolor('#f8f9fa')

        # Actions
        colors = ['gray' if a == 0 else 'green' if a == 1 else 'red' for a in actions]
        ax[1].scatter(range(len(actions)), actions, c=colors, s=15, alpha=0.6)
        ax[1].set_title("Trading Actions Over Time", fontsize=14, fontweight='bold')
        ax[1].set_xlabel("Time Steps", fontsize=12)
        ax[1].set_ylabel("Action", fontsize=12)
        ax[1].set_yticks([0, 1, 2])
        ax[1].set_yticklabels(['HOLD', 'BUY', 'SELL'])
        ax[1].grid(alpha=0.3, linestyle='--', axis='y')
        ax[1].set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        st.pyplot(fig)

        # Summary assessment
        risk_adj_better = m_agent['sharpe'] >= m_bh['sharpe'] and m_agent['mdd'] <= m_bh['mdd']
        if risk_adj_better:
            st.success("✅ **Agent outperforms Buy & Hold** on risk-adjusted basis (Sharpe ≥ B&H, MaxDD ≤ B&H)")
        else:
            st.warning("⚠️ **Agent underperforms Buy & Hold** on risk-adjusted basis")

    # =============================
    # Current Signal
    # =============================
    if signal_btn:
        with st.spinner("Generating signal..."):
            last_obs = X_test[-1:,:,:]

            # Ensure scalar action
            action, _ = model.predict(last_obs, deterministic=True)
            action_int = int(np.asarray(action).reshape(()))

            mapping = {0:"HOLD 📊", 1:"BUY 🟢", 2:"SELL 🔴"}
            signal_text = mapping[action_int]

            # GUI color mapping
            signal_color = {"HOLD 📊": "blue", "BUY 🟢": "green", "SELL 🔴": "red"}[signal_text]

            # Ensure latest price is scalar
            latest_price = float(np.asarray(test_prices).reshape(-1)[-1])

        st.subheader("🎯 Current Signal")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"## :{signal_color}[{signal_text}]")
            st.markdown(f"**Ticker:** {ticker} | **Model:** {model_type}")
            st.markdown(f"**Latest Price:** **${latest_price:.2f}**")


st.sidebar.markdown("---")
st.sidebar.markdown("**Tips:**")
st.sidebar.markdown("- Train model with `trading_rl_rnn.py`")
st.sidebar.markdown("- Upload the saved `.zip` file")
st.sidebar.markdown("- Or point to local model path")
st.sidebar.markdown("- Adjust transaction costs to match training")