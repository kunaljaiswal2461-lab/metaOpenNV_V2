import requests
import time

# --- 📡 THE MONITORING HUB ---
# This is where your Space is 'listening'
BASE_URL = "https://kj2461-trading-env.hf.space"

def run_agent_loop():
    """
    Simulates a continuous agent run to show off the Live Browser Graph.
    """
    print(f"🛰️ Connecting to Environment: {BASE_URL}")
    
    # 1. INIT
    try:
        print("🔄 [RESET] Initializing simulation session...")
        res = requests.post(f"{BASE_URL}/reset")
        res.raise_for_status()
        obs = res.json()
        print(f"✅ Connection Stable. Starting with ${obs['port_val']:.2f}")
    except Exception as e:
        print(f"❌ Could not connect to Space: {e}")
        return

    total_reward = 0
    print("\n🎬 SHOWCASE STARTING: Keep your Space Browser window visible!")
    print("-" * 50)

    # 2. THE LOOP (50 Steps)
    for step in range(1, 51):
        # --- BRAIN: Simple Trend Following Heuristic ---
        # Logic: Alternating Buy/Sell every 5 steps to show graph movement
        action = 1 if (step // 5) % 2 == 0 else 2
        
        # EXECUTE
        try:
            res = requests.post(f"{BASE_URL}/step", json={"action": action}).json()
            total_reward += res['reward']
            
            status = "🟢 BUY" if action == 1 else "🔴 SELL"
            print(f"Step {step:02d} | Action: {status} | 💸 Total Value: ${res['port_val']:.2f} | 📈 Reward: {res['reward']:.4f}")
            
            # --- THE MAGIC WAIT ---
            # This 0.8s pause allows the Gradio Timer (2s) to 'catch' 
            # and draw the points on the graph for the audience.
            time.sleep(0.8)
            
        except Exception as e:
            print(f"⚠️ Connection glitch at step {step}: {e}")
            break

    print("-" * 50)
    print(f"🏆 SESSION COMPLETE! Total Reward Accumulated: {total_reward:.4f}")
    print("✨ Check your browser for the final performance curve.")

if __name__ == "__main__":
    run_agent_loop()
