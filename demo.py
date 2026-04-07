import numpy as np
from config import RUNS
from pipeline import solve, accuracy, adam

def run_demo():

    adam_results = []
    hlx_results = []

    print("\n🚀 HLX-Adam Demo (Evo Engineering)\n")

    for _ in range(RUNS):

        # Adam baseline
        p1 = np.random.uniform(-10,10,673)
        p1 = adam(p1)
        adam_results.append(accuracy(p1))

        # HLX pipeline
        p2 = solve()
        hlx_results.append(accuracy(p2))

    adam_avg = sum(adam_results)/len(adam_results)
    hlx_avg = sum(hlx_results)/len(hlx_results)

    print("===== RESULTS =====\n")
    print(f"Adam Accuracy:     {adam_avg:.4f}")
    print(f"HLX-Adam Accuracy: {hlx_avg:.4f}")

    improvement = (hlx_avg - adam_avg) * 100
    print(f"\n🔥 Improvement: +{improvement:.2f}%")

    print("\n===================\n")

if __name__ == "__main__":
    run_demo()