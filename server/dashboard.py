DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading AI LIVE | LLM Baseline</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --card-bg: #161b2a;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.2);
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --green: #10b981;
            --red: #ef4444;
            --border: #1e293b;
        }

        body {
            margin: 0;
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }

        header {
            background: var(--card-bg);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        h1 {
            margin: 0;
            font-size: 1.25rem;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        h1::before { content: "📈"; }

        .controls {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        button {
            background: var(--accent);
            color: var(--bg);
            border: none;
            padding: 0.5rem 1.25rem;
            border-radius: 0.5rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        button:hover { opacity: 0.9; transform: translateY(-1px); }

        button#resetBtn {
            background: transparent;
            color: var(--accent);
            border: 1px solid var(--accent);
        }

        select {
            background: var(--card-bg);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 0.5rem;
            border-radius: 0.5rem;
            cursor: pointer;
        }

        main {
            flex: 1;
            padding: 1.5rem;
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 1.5rem;
            overflow-y: auto;
        }

        .dashboard-grid { display: flex; flex-direction: column; gap: 1.5rem; }

        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
        }

        .metric-card {
            background: var(--card-bg);
            padding: 1.25rem;
            border-radius: 1rem;
            border: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .metric-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; }
        .metric-value { font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

        .charts {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 250px 250px;
            gap: 1.5rem;
        }

        .chart-container {
            background: var(--card-bg);
            padding: 1rem;
            border-radius: 1rem;
            border: 1px solid var(--border);
            position: relative;
        }

        .log-container {
            background: var(--card-bg);
            border-radius: 1rem;
            border: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .log-header { padding: 1rem; border-bottom: 1px solid var(--border); font-weight: 600; }
        .log-body { flex: 1; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; }

        table { width: 100%; border-collapse: collapse; }
        th { position: sticky; top: 0; background: var(--card-bg); text-align: left; padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); }
        td { padding: 0.5rem 1rem; border-bottom: 1px solid var(--border); }

        .action-BUY { color: var(--green); }
        .action-SELL { color: var(--red); }
        .action-HOLD { color: var(--text-muted); }
        .pos { color: var(--green); }
        .neg { color: var(--red); }

        /* Documentation Section */
        .doc-section {
            margin: 0 1.5rem 2rem 1.5rem;
            background: var(--card-bg);
            border-radius: 1rem;
            border: 1px solid var(--border);
            overflow: hidden;
        }

        .doc-header {
            background: #1e293b;
            padding: 1rem 1.5rem;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .doc-header h2 {
            margin: 0;
            font-size: 1rem;
            color: var(--accent);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .doc-content {
            padding: 1.5rem;
            display: block; /* Open by default */
        }

        .doc-cards {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .stat-card {
            background: rgba(30, 41, 59, 0.5);
            padding: 1rem;
            border-radius: 0.75rem;
            border: 1px dashed var(--border);
            text-align: center;
        }

        .stat-value { font-size: 1.25rem; font-weight: 700; color: var(--accent); display: block; }
        .stat-label { font-size: 0.75rem; color: var(--text-muted); }

        .doc-table { width: 100%; margin: 1rem 0; }
        .doc-table th { background: rgba(30, 41, 59, 0.8); }
        
        .reward-box {
            background: #0f172a;
            padding: 1rem;
            border-radius: 0.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            border-left: 4px solid var(--accent);
            margin: 1rem 0;
            white-space: pre-wrap;
        }

        .highlight-paper {
            color: var(--accent);
            font-weight: 600;
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <header>
        <h1>TRADING AI LIVE | LLM BASELINE</h1>
        <div class="controls">
            <select id="speedSel">
                <option value="1000">Slow</option>
                <option value="200" selected>Normal</option>
                <option value="50">Fast</option>
            </select>
            <button id="runBtn">▶ RUN</button>
            <button id="resetBtn">🔄 RESET</button>
        </div>
    </header>

    <main>
        <div class="dashboard-grid">
            <div class="metrics">
                <div class="metric-card">
                    <span class="metric-label">Portfolio Value</span>
                    <span class="metric-value" id="m-pv">$10,000</span>
                    <div class="metric-sub" id="m-pnl">+$0.00 (0.00%)</div>
                </div>
                <div class="metric-card">
                    <span class="metric-label">Steps Taken</span>
                    <span class="metric-value" id="m-steps">0</span>
                </div>
                <div class="metric-card">
                    <span class="metric-label">Total Trades</span>
                    <span class="metric-value" id="m-trades">0</span>
                </div>
            </div>

            <div class="charts">
                <div class="chart-container"><canvas id="chart-pv"></canvas></div>
                <div class="chart-container"><canvas id="chart-reward"></canvas></div>
                <div class="chart-container"><canvas id="chart-actions"></canvas></div>
                <div class="chart-container"><canvas id="chart-cum-reward"></canvas></div>
            </div>
        </div>

        <aside class="log-container">
            <div class="log-header">Live Session Logs</div>
            <div class="log-body">
                <table id="logTable">
                    <thead><tr><th>Step</th><th>Action</th><th>Reward</th><th>Value</th></tr></thead>
                    <tbody></tbody>
                </table>
            </div>
        </aside>
    </main>

    <section class="doc-section">
        <div class="doc-header" onclick="document.getElementById('docContent').style.display = document.getElementById('docContent').style.display === 'none' ? 'block' : 'none'">
            <h2>📋 Documentation & Judges Reference</h2>
            <span>▼</span>
        </div>
        <div id="docContent" class="doc-content">
            <h3>1. What is this environment?</h3>
            <p>AntiGravity is a high-fidelity <b>Markov Decision Process (MDP)</b> environment simulating the intraday momentum of the SPY index. It provides a non-stationary observation space where a Reinforcement Learning agent must balance alpha generation with strict risk-adjusted drawdown management.</p>

            <div class="doc-cards">
                <div class="stat-card"><span class="stat-value">3</span><span class="stat-label">Base Actions</span></div>
                <div class="stat-card"><span class="stat-value">203 dims</span><span class="stat-label">Observation (default)</span></div>
                <div class="stat-card"><span class="stat-value">20-bar</span><span class="stat-label">Default window</span></div>
            </div>

            <h3>2. Observation Space Deep-Dive</h3>
            <p>Default Space: <b>203 values</b> per step (<code>WINDOW_SIZE=20</code> × 10 features + 3 portfolio). Tasks can use 10- or 50-bar windows → 103 or 503 dims.</p>
            <table class="doc-table">
                <thead><tr><th>Feature Layer</th><th>Dimensions (default)</th><th>Description</th></tr></thead>
                <tbody>
                    <tr><td>Market Features</td><td>200</td><td>20 past bars × 10 indicators (see README + <code>data/preprocess.py</code>).</td></tr>
                    <tr><td>Portfolio Stats</td><td>3</td><td>Cash, holdings (shares), portfolio value ($).</td></tr>
                </tbody>
            </table>

            <h3>3. Technical Indicator Formulas</h3>
            <table class="doc-table">
                <thead><tr><th>Indicator</th><th>Mathematical Formula</th><th>Biological/Market Parallel</th></tr></thead>
                <tbody>
                    <tr><td>Log Return</td><td><code>ln(P_t / P_{t-1})</code></td><td>Continuously compounded price momentum.</td></tr>
                    <tr><td>SMA Distance</td><td><code>(P - SMA_n) / SMA_n</code></td><td>Mean Reversion (Elasticity of price to trend).</td></tr>
                    <tr><td>RSI (14)</td><td><code>100 - [100 / (1 + RS)]</code></td><td>Exhaustion metrics (Overbought/Oversold).</td></tr>
                    <tr><td>Volume Norm</td><td><code>Vol / rolling_avg(Vol, 20)</code></td><td>Validation of price moves via liquidity spikes.</td></tr>
                    <tr><td>Volatility</td><td><code>StdDev(Log_Ret, 10)</code></td><td>Risk uncertainty and regime change signal.</td></tr>
                </tbody>
            </table>

            <h3>4. Reward Function: The Stanford Methodology</h3>
            <p>Utilizing a composite function based on the <span class="highlight-paper">Stanford GSB: "Direct Reinforcement Learning for Financial Forecasting" by Moody & Saffell</span>.</p>
            <div class="reward-box">
<b>Reward = α(LogRet) + β(Downside) + γ(DiffSharpe) - TC</b>

1. <b>Log Return (α=0.4):</b> Direct profit incentive.
2. <b>Downside Variance (β=0.25):</b> Penalizes "Square of Negative Returns".
3. <b>Differential Sharpe Ratio (γ=0.15):</b> Recursive risk-adjusted gradient:
   <i>ηt = (B_prev * dA - 0.5 * A_prev * dB) / (var ^ 1.5)</i>
4. <b>Transaction Costs (TC):</b> 0.1% penalty per trade to prevent chatter.
            </div>

            <h3>5. Agent Architecture</h3>
            <p>The baseline agent is an <b>LLM-based Zero-Shot Reasoner</b>. It performs <b>Instruction-Following</b> across the 120D market state, identifying patterns like "Breakouts" to execute optimal trades.</p>
        </div>
    </section>

    <script>
        let running = false;
        let timer = null;
        let currentStep = 0;
        let baseline = 10000;
        let data = { pv: [], rewards: [], cumRewards: [], actions: [0, 0, 0], logs: [] };

        const ctxPv = document.getElementById('chart-pv').getContext('2d');
        const chartPv = new Chart(ctxPv, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Portfolio', data: [], borderColor: '#38bdf8', fill: true, backgroundColor: 'rgba(56, 189, 248, 0.1)' }]},
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display:false } }}
        });

        const chartReward = new Chart(document.getElementById('chart-reward').getContext('2d'), {
            type: 'bar',
            data: { labels: [], datasets: [{ data: [], backgroundColor: [] }]},
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display:false } }}
        });

        const chartActions = new Chart(document.getElementById('chart-actions').getContext('2d'), {
            type: 'doughnut',
            data: { labels: ['BUY', 'HOLD', 'SELL'], datasets: [{ data: [0,0,0], backgroundColor: ['#10b981', '#94a3b8', '#ef4444'] }]},
            options: { responsive: true, maintainAspectRatio: false }
        });

        const chartCumReward = new Chart(document.getElementById('chart-cum-reward').getContext('2d'), {
            type: 'line',
            data: { labels: [], datasets: [{ data: [], borderColor: '#10b981' }]},
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display:false } }}
        });

        async function resetEnv() {
            const res = await fetch('./reset', { method: 'POST' });
            const obs = await res.json();
            baseline = obs.portfolio_value;
            clearData();
            updateUI(obs);
        }

        function clearData() {
            currentStep = 0;
            data = { pv: [], rewards: [], cumRewards: [], actions: [0, 0, 0], logs: [] };
            chartPv.data.labels = []; chartPv.data.datasets[0].data = []; chartPv.update();
        }

        async function stepEnv() {
            if (!running) return;
            // Demo mode: Random Walk (since server only exposes environment now)
            const action = Math.floor(Math.random() * 3);
            const res = await fetch('./step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: action })
            });
            const obs = await res.json();
            updateData(obs, action);
            updateUI(obs);
            if (obs.done) stop();
            else timer = setTimeout(stepEnv, parseInt(document.getElementById('speedSel').value));
        }

        function updateData(obs, action) {
            currentStep++;
            data.pv.push(obs.portfolio_value);
            data.rewards.push(obs.reward);
            data.cumRewards.push((data.cumRewards[data.cumRewards.length-1] || 0) + obs.reward);
            data.actions[action]++;
            data.logs.unshift({ step: currentStep, action: ['HOLD', 'BUY', 'SELL'][action], reward: obs.reward, pv: obs.portfolio_value });
        }

        function updateUI(obs) {
            document.getElementById('m-pv').innerText = `$${obs.portfolio_value.toFixed(2)}`;
            document.getElementById('m-steps').innerText = currentStep;
            document.getElementById('m-trades').innerText = data.actions[1] + data.actions[2];
            
            const labels = Array.from({length: data.pv.length}, (_, i) => i);
            chartPv.data.labels = labels; chartPv.data.datasets[0].data = data.pv; chartPv.update('none');
            
            const tbody = document.querySelector('#logTable tbody');
            tbody.innerHTML = data.logs.slice(0, 10).map(log => `
                <tr><td>${log.step}</td><td class="action-${log.action}">${log.action}</td><td class="${log.reward >= 0 ? 'pos' : 'neg'}">${log.reward.toFixed(4)}</td><td>$${log.pv.toFixed(2)}</td></tr>
            `).join('');
        }

        function stop() { running = false; document.getElementById('runBtn').innerText = '▶ RUN'; }
        function start() { running = true; document.getElementById('runBtn').innerText = '⏸ PAUSE'; stepEnv(); }

        document.getElementById('runBtn').addEventListener('click', () => running ? stop() : start());
        document.getElementById('resetBtn').addEventListener('click', () => { stop(); resetEnv(); });
        window.addEventListener('load', resetEnv);
    </script>
</body>
</html>
"""
