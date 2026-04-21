import sys
import traceback
from backend.env.security_env import CySentSecurityEnv

combos = [
    {"scenario": "bank", "difficulty": "hard", "attacker": "ransomware_gang"},
    {"scenario": "hospital", "difficulty": "medium", "attacker": "insider_saboteur"},
    {"scenario": "saas", "difficulty": "hard", "attacker": "credential_thief"}
]

try:
    env = CySentSecurityEnv()
    for combo in combos:
        print(f"Testing combo: {combo}")
        obs, info = env.reset(options=combo, seed=42)
        
        # Step through 10 iterations with deterministic action sequence
        for i in range(10):
            action = i % env.action_space.n
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset(options=combo, seed=42)

        # Extraction and verification
        profile = info.get('profile', {})
        risk = info.get('network_risk', 'N/A')
        events = info.get('events', [])
        narrative = info.get('narrative', '')
        
        print(f"  Obs shape: {obs.shape}")
        print(f"  Profile snippets: { {k: profile.get(k) for k in ['budget', 'detection_threshold', 'host_count'] if k in profile} }")
        print(f"  Final network_risk: {risk}")
        print(f"  Events populated: {len(events) > 0}")
        print(f"  Narrative populated: {len(narrative) > 0}")
        print("-" * 20)
        
    print("ALL TESTS PASSED")
    sys.exit(0)
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
