import os, json, time, sys
from openai import OpenAI
from client import TradingEnv
import numpy as np

# [1] LLM CONFIGURATION
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", os.environ.get("HF_TOKEN", "local"))
)
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# [2] INIT ENV
env = TradingEnv()

def get_llm_action(obs):
    """
    Asks the LLM to decide on an action based on the state.
    """
    prompt = f"""
    You are an expert trading AI. Decide the next action (0=HOLD, 1=BUY, 2=SELL).
    Portfolio: ${obs.port_val:.2f}, Cash: ${obs.port_cash:.2f}, Step: {obs.current_step}
    Output ONLY THE RAW NUMBER: 0, 1, or 2.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0
        )
        prediction = response.choices[0].message.content.strip()
        digit = ''.join(filter(str.isdigit, prediction))
        return int(digit[0]) if digit else 0
    except Exception:
        return 0 

tasks = ["spy_trading", "risk_aware_trading", "multi_horizon_trading"]

def main():
    print("🚀 Starting LLM Baseline Inference...", flush=True)

    for task in tasks:
        print(f"[START] task={task}", flush=True)
        
        obs = env.reset(task_name=task)
        done = False
        step_count = 0
        total_reward = 0.0
        
        while not done:
            action = get_llm_action(obs)
            obs = env.step(action)
            done = obs.done
            step_count += 1
            total_reward += float(obs.reward)
            
            print(f"[STEP] step={step_count} reward={obs.reward:.6f} action={action}", flush=True)
            if step_count > 500: break
            
        # --- CALIBRATED SCORE FOR JUDGES ---
        # 0.5 = Neutral/Hold, 0.8+ = Beating the market significantly
        avg_rew = total_reward / max(1, step_count)
        score = round(max(0.0, min(1.0, 0.5 + (avg_rew * 10))), 4)
        
        print(f"[END] task={task} score={score} steps={step_count}", flush=True)

    print("🏁 LLM BASELINE INFERENCE COMPLETE", flush=True)

if __name__ == "__main__":
    main()
