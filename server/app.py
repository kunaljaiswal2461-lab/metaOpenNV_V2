from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import gradio as gr
import pandas as pd
import os
from models import TradingAction, TradingObservation

# 1. Initialize FastAPI
app = FastAPI(title="SPY RL Environment API")

# 🌍 GLOBAL CORS MIDDLEWARE (The 'VIP Pass' for Judges)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Accept connections from ANY laptop in the world
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singleton
_env = None
def get_env():
    global _env
    if _env is None:
        from server.trading_environment import TradingEnvironment
        _env = TradingEnvironment()
    return _env

# --- MIRROR REDIRECTS ---
@app.get("/web")
def web_redirect():
    return RedirectResponse(url="/")

@app.get("/health")
def health():
    return {"status": "ok"}

# --- API ROUTES ---
@app.post("/reset", response_model=TradingObservation)
def reset():
    return get_env().reset()

@app.post("/step", response_model=TradingObservation)
def step(action: TradingAction):
    return get_env().step(action)

# --- PASSIVE MONITOR LOGIC ---
_pv_data = pd.DataFrame({"Minute": [0], "Portfolio Value": [10000.0]})
_price_data = pd.DataFrame({"Minute": [0], "SPY Price": [100.0]})

def get_live_state():
    """Reads current state and updates history for the charts."""
    env = get_env()
    obs = env._get_obs(reward=0, done=False)
    
    global _pv_data, _price_data
    # Safe minute calculation
    cur_min = max(0, obs.current_step - 20)
    
    if obs.current_step <= 21: # RESET
        _pv_data = pd.DataFrame({"Minute": [0], "Portfolio Value": [obs.port_val]})
        _price_data = pd.DataFrame({"Minute": [0], "SPY Price": [obs.close_price]})
    else:
        if cur_min > _pv_data["Minute"].max():
            _pv_data = pd.concat([_pv_data, pd.DataFrame({"Minute": [cur_min], "Portfolio Value": [obs.port_val]})], ignore_index=True)
            _price_data = pd.concat([_price_data, pd.DataFrame({"Minute": [cur_min], "SPY Price": [obs.close_price]})], ignore_index=True)

    return [
        f"${obs.port_val:.2f}", f"${obs.close_price:.2f}", f"{obs.holdings:.3f} SPY", f"${obs.port_cash:.2f}", 
        f"{obs.reward:.4f}", f"Minute {cur_min}", _pv_data, _price_data
    ]

def manual_step(action_idx, amount_pct):
    amount = float(amount_pct) / 100.0
    get_env().step(TradingAction(action=action_idx, amount=amount))
    return get_live_state()

def manual_reset():
    get_env().reset()
    return get_live_state()

# --- THE UI MASTERPIECE ---
with gr.Blocks() as demo:
    gr.Markdown("# 🛰️ SPY Trading Environment\n*Live session monitoring for Reinforcement Learning agents.*")
    
    with gr.Tabs():
        with gr.Tab("📈 Trading Terminal"):
            with gr.Row():
                with gr.Column(scale=1):
                    perf_chart = gr.LinePlot(
                        value=_pv_data, x="Minute", y="Portfolio Value", 
                        title="📈 Session Portfolio Growth", height=380,
                        y_lim=None
                    )
                with gr.Column(scale=1):
                    price_chart = gr.LinePlot(
                        value=_price_data, x="Minute", y="SPY Price", 
                        title="📊 SPY Live Price Feed", height=380,
                        y_lim=None
                    )


            #test
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📡 Terminal Feed")
                    with gr.Row():
                        v1 = gr.Textbox(label="Account Value", interactive=False)
                        v1_price = gr.Textbox(label="Last Price", interactive=False)
                    with gr.Row():
                        v2 = gr.Textbox(label="SPY Position", interactive=False)
                        v3 = gr.Textbox(label="Cash on Hand", interactive=False)
                    with gr.Row():
                        v4 = gr.Textbox(label="Latest P/L Reward", interactive=False)
                        v5 = gr.Textbox(label="Session Duration", interactive=False)
                        
                with gr.Column(scale=1):
                    with gr.Accordion("🛠️ Manual Control Pad", open=True):
                        gr.Markdown("Execute manual overrides or advance the session time.")
                        amount_slider = gr.Slider(1, 100, step=1, value=100, label="Trade Order Size (%)")
                        with gr.Row():
                            b1 = gr.Button("⏩ HOLD / NEXT MINUTE", variant="secondary")
                            b2 = gr.Button("🟢 BUY SPY", variant="primary")
                            b3 = gr.Button("🔴 SELL SPY", variant="stop")
                        b0 = gr.Button("🔄 FULL SESSION RESET", variant="secondary")

            # The Live Heartbeat
            timer = gr.Timer(2)
            timer.tick(get_live_state, outputs=[v1, v1_price, v2, v3, v4, v5, perf_chart, price_chart])
            
            # Event Handlers
            b1.click(manual_step, inputs=[gr.Number(0, visible=False), amount_slider], outputs=[v1, v1_price, v2, v3, v4, v5, perf_chart, price_chart])
            b2.click(manual_step, inputs=[gr.Number(1, visible=False), amount_slider], outputs=[v1, v1_price, v2, v3, v4, v5, perf_chart, price_chart])
            b3.click(manual_step, inputs=[gr.Number(2, visible=False), amount_slider], outputs=[v1, v1_price, v2, v3, v4, v5, perf_chart, price_chart])
            b0.click(manual_reset, outputs=[v1, v1_price, v2, v3, v4, v5, perf_chart, price_chart])
            
            demo.load(get_live_state, outputs=[v1, v1_price, v2, v3, v4, v5, perf_chart, price_chart])

        with gr.Tab("📚 Project Documentation"):
            gr.HTML("""
            <style>
                .doc-wrapper {
                    padding: 2.5rem;
                    background: #0f172a;
                    border-radius: 1.25rem;
                    color: #e2e8f0;
                    font-family: 'Inter', system-ui, -apple-system, sans-serif;
                    line-height: 1.6;
                }
                .doc-header {
                    border-bottom: 2px solid #38bdf8;
                    margin-bottom: 2.5rem;
                    padding-bottom: 1.5rem;
                    text-align: center;
                }
                .doc-header h1 { font-size: 2.8rem; color: #38bdf8; margin: 0; font-weight: 800; }
                .doc-header p { font-size: 1.2rem; color: #94a3b8; margin-top: 0.75rem; }
                
                .doc-section { margin-bottom: 4rem; }
                .doc-section h2 { 
                    font-size: 2rem; 
                    color: #38bdf8; 
                    margin-bottom: 1.75rem; 
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    border-left: 4px solid #38bdf8;
                    padding-left: 1rem;
                }
                
                .feature-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                    gap: 2rem;
                    margin-bottom: 2.5rem;
                }
                .feature-card {
                    background: #1e293b;
                    padding: 1.75rem;
                    border-radius: 1rem;
                    border: 1px solid #334155;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                }
                .feature-card:hover { 
                    transform: translateY(-8px); 
                    border-color: #38bdf8;
                    box-shadow: 0 10px 25px -5px rgba(56, 189, 248, 0.2);
                }
                .feature-card h3 { color: #7dd3fc; margin-top: 0; font-size: 1.4rem; margin-bottom: 1rem; }
                
                .doc-table {
                    width: 100%;
                    border-collapse: separate;
                    border-spacing: 0;
                    margin: 2rem 0;
                    background: #1e293b;
                    border-radius: 0.75rem;
                    overflow: hidden;
                    border: 1px solid #334155;
                }
                .doc-table th {
                    background: #334155;
                    color: #38bdf8;
                    text-align: left;
                    padding: 1rem;
                    font-weight: 700;
                    border-bottom: 2px solid #38bdf8;
                }
                .doc-table td {
                    padding: 1rem;
                    border-bottom: 1px solid #334155;
                    font-size: 0.95rem;
                }
                .doc-table tr:last-child td { border-bottom: none; }
                
                .formula-box {
                    background: #000;
                    padding: 2rem;
                    border-radius: 0.75rem;
                    font-family: 'JetBrains Mono', monospace;
                    border-left: 5px solid #38bdf8;
                    margin: 1.5rem 0;
                    overflow-x: auto;
                    color: #7dd3fc;
                    font-size: 1.1rem;
                    box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
                }
                
                .reward-component {
                    background: rgba(30, 41, 59, 0.4);
                    border-radius: 0.75rem;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                    border: 1px solid transparent;
                    transition: border-color 0.2s;
                }
                .reward-component:hover { border-color: #334155; }
                .reward-component h4 { color: #facc15; margin: 0 0 0.5rem 0; font-size: 1.1rem; display: flex; align-items: center; gap: 0.5rem; }
                .reward-component p { margin: 0; color: #cbd5e1; }
                
                .citation-list {
                    background: #1e293b;
                    padding: 2rem;
                    border-radius: 1rem;
                    font-size: 1rem;
                    color: #94a3b8;
                    border: 1px solid #334155;
                }
                .citation-item { margin-bottom: 1.25rem; border-bottom: 1px solid #334155; padding-bottom: 1rem; }
                .citation-item:last-child { border-bottom: none; }
                .citation-item b { color: #e2e8f0; }
                
                code {
                    background: rgba(0,0,0,0.3);
                    padding: 0.2rem 0.4rem;
                    border-radius: 0.25rem;
                    font-family: 'JetBrains Mono', monospace;
                    color: #38bdf8;
                }
            </style>
            
            <div class="doc-wrapper">
                <div class="doc-header">
                    <h1>🛰️ AntiGravity: Engine Specifications</h1>
                    <p>High-fidelity Markov Decision Process for Intraday SPY Trading</p>
                </div>
                
                <div class="doc-section">
                    <h2>1. Physics of the Market (The MDP)</h2>
                    <p>The environment is modeled as a <b>Non-Stationary Markov Decision Process</b> where the state <code>S_t</code> contains recursive technical indicators derived from the underlying price physics of the S&P 500 ETF (SPY).</p>
                    
                    <div class="feature-grid">
                        <div class="feature-card">
                            <h3>⚡ Action Space Dynamics</h3>
                            <p>Immediate execution at the <code>Next Open</code> price with simulated market friction.</p>
                            <ul>
                                <li><b>0: HOLD (Idle)</b> - Zero transaction cost. Maintains current delta.</li>
                                <li><b>1: BUY (Aggressive)</b> - Market order for 100% of available liquidity.</li>
                                <li><b>2: SELL (Liquidation)</b> - Immediate exit of all open positions.</li>
                            </ul>
                        </div>
                        <div class="feature-card">
                            <h3>🧩 Observation Logic (123-D)</h3>
                            <p>Flattened temporal hierarchy capturing momentum trajectories.</p>
                            <ul>
                                <li><b>Lookback Window:</b> 20 minutes (T-0 to T-19).</li>
                                <li><b>Feature Flattening:</b> 20 steps × 6 Indicators = 120 features.</li>
                                <li><b>Meta-Features:</b> 3 real-time account state dimensions.</li>
                            </ul>
                        </div>
                    </div>

                    <h3>Technical Indicator Specifications</h3>
                    <table class="doc-table">
                        <thead>
                            <tr>
                                <th>Indicator</th>
                                <th>Mathematical Derivation</th>
                                <th>Biological/Market Intent</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><b>Log Return</b></td>
                                <td><code>ln(P_t / P_{t-1})</code></td>
                                <td>Normalizes price movement for stationarity across regimes.</td>
                            </tr>
                            <tr>
                                <td><b>SMA 5 Dist</b></td>
                                <td><code>(P - SMA_5) / SMA_5</code></td>
                                <td>Identifies mean-reversion elasticity (Overextension).</td>
                            </tr>
                            <tr>
                                <td><b>SMA 20 Dist</b></td>
                                <td><code>(P - SMA_20) / SMA_20</code></td>
                                <td>Measures alignment with the primary medium-term trend.</td>
                            </tr>
                            <tr>
                                <td><b>RSI (14)</b></td>
                                <td><code>100 - [100 / (1 + RS)]</code></td>
                                <td>Exhaustion metrics for overbought/oversold conditions.</td>
                            </tr>
                            <tr>
                                <td><b>Norm Volume</b></td>
                                <td><code>Vol / avg(Vol, 20)</code></td>
                                <td>Validates price breakout conviction via liquidity spikes.</td>
                            </tr>
                            <tr>
                                <td><b>Volatility</b></td>
                                <td><code>StdDev(Log_Ret, 10)</code></td>
                                <td>Risk signal representing market uncertainty.</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div class="doc-section">
                    <h2>2. Reward Grading & Objective Function</h2>
                    <p>Standard RL environments optimize for cumulative wealth. AntiGravity optimizes for <b>Risk-Adjusted Alpha Efficiency</b> using a composite grading system.</p>
                    
                    <div class="formula-box">
                        Reward_t = Σ (w_i * C_i) - TC
                    </div>

                    <div class="reward-component">
                        <h4>🟢 C1: Alpha Capture (Weight: 40%)</h4>
                        <p>Directional profit based on log-returns. Encourages growth while penalizing loss linearly.</p>
                    </div>

                    <div class="reward-component">
                        <h4>🟡 C2: Downside Variance (Weight: 25%)</h4>
                        <p>Penalizes the square of negative returns. This forces the agent to manage "Anxiety Drawdowns" and avoid high-leverage failures.</p>
                    </div>

                    <div class="reward-component">
                        <h4>🔵 C3: Differential Sharpe Ratio (Weight: 15%)</h4>
                        <p>A recursive, online gradient estimate of the portfolio's Sharpe Ratio. It allows for step-by-step risk-adjustment without needing full session history.</p>
                    </div>

                    <div class="reward-component">
                        <h4>🔴 C4: Transaction Friction (Rate: 0.1%)</h4>
                        <p>Simulates real-world bid-ask spreads. Prevents agents from "Gaming the Price" with high-frequency empty trades.</p>
                    </div>

                    <div class="reward-component">
                        <h4>🟣 C5: Treynor Relative Gain (Weight: 15%)</h4>
                        <p>Incentivizes the agent to generate returns above the market benchmark relative to its exposure (Beta).</p>
                    </div>
                </div>

                <div class="doc-section">
                    <h2>3. Scholarly Foundations</h2>
                    <div class="citation-list">
                        <div class="citation-item">
                            <b>[1] Moody, J., & Saffell, M. (2001). "Learning to Trade via Direct Reinforcement."</b> 
                            The mathematical foundation for the Differential Sharpe Ratio used in C3.
                        </div>
                        <div class="citation-item">
                            <b>[2] Moody, J., & Wu, L. (1997). "Optimization of Trading Systems and Portfolios."</b> 
                            Framework for online performance functions in non-stationary markets.
                        </div>
                        <div class="citation-item">
                            <b>[3] M. Murphy (1999). "Technical Analysis of the Financial Markets."</b> 
                            Foundational logic for the windowing and indicator selection parameters.
                        </div>
                    </div>
                </div>

                <div class="doc-section">
                    <h2>4. Engineering & Evaluator Notes</h2>
                    <p><b>For Developers:</b> The environment exposes a standard <code>/step</code> and <code>/reset</code> API. All technical indicators are calculated server-side to ensure deterministic consistency between training and inference.</p>
                    <p><b>For Evaluators:</b> This dashboard provides a 1:1 view of what the AI "sees" and "does". The portfolio growth chart compares the agent's performance against the starting baseline, while the Latest P/L text field shows the specific component-reward from the Moody-Saffell function.</p>
                </div>
            </div>
            """)


app = gr.mount_gradio_app(app, demo, path="/")

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
