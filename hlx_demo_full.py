import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# =========================
# CONFIG
# =========================

RUNS = 8
DIM = 673

HLX_STEPS = 80
REFINE_STEPS = 30
ADAM_STEPS = 100

NOISE = 0.003

# =========================
# DATA
# =========================

def make_data(n=800):
    X = np.random.uniform(-3, 3, (n, 3))
    y = ((np.sin(X[:,0]) + np.cos(X[:,1]) + X[:,2]**2) > 0).astype(float)
    return X, y.reshape(-1,1)

X, y = make_data()

# =========================
# MODEL
# =========================

def unpack(p):
    i = 0
    W1 = p[i:i+96].reshape(3,32); i+=96
    b1 = p[i:i+32].reshape(1,32); i+=32
    W2 = p[i:i+512].reshape(32,16); i+=512
    b2 = p[i:i+16].reshape(1,16); i+=16
    W3 = p[i:i+16].reshape(16,1); i+=16
    b3 = p[i:i+1].reshape(1,1)
    return W1,b1,W2,b2,W3,b3

def forward(p):
    W1,b1,W2,b2,W3,b3 = unpack(p)
    h1 = np.tanh(X @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return h2 @ W3 + b3

# =========================
# LOSS / ACC
# =========================

def loss(p):
    logits = forward(p)
    probs = 1/(1+np.exp(-logits))
    base = np.mean(-(y*np.log(probs+1e-8)+(1-y)*np.log(1-probs+1e-8)))
    return base + np.random.normal(0, NOISE)

def accuracy(p):
    logits = forward(p)
    preds = (logits > 0).astype(float)
    return np.mean(preds == y)

# =========================
# FAST GRAD
# =========================

def fast_grad(x, eps=1e-5, frac=0.3):
    g = np.zeros_like(x)
    idx = np.random.choice(len(x), int(len(x)*frac), replace=False)

    for i in idx:
        x1 = x.copy(); x2 = x.copy()
        x1[i]+=eps; x2[i]-=eps
        g[i]=(loss(x1)-loss(x2))/(2*eps)

    return g

# =========================
# ADAM
# =========================

def adam(p, track=False):
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    history = []

    for t in range(1, ADAM_STEPS+1):
        g = fast_grad(p)

        m = 0.9*m + 0.1*g
        v = 0.999*v + 0.001*(g*g)

        mh = m/(1-0.9**t)
        vh = v/(1-0.999**t)

        p -= 0.01 * mh / (np.sqrt(vh)+1e-8)

        if track:
            history.append(loss(p))

    return p, history

# =========================
# HLX
# =========================

def hlx(p, track=False):
    best = p.copy()
    best_val = loss(p)
    history = []

    for _ in range(HLX_STEPS):
        neighbors = [p + np.random.normal(0,0.15,len(p)) for _ in range(5)]
        vals = [loss(n) for n in neighbors]

        b = neighbors[np.argmin(vals)]
        avg = np.mean(neighbors, axis=0)

        new = 0.4*p + 0.1*avg + 0.5*b
        new += 0.3*(b - new)

        val = loss(new)

        if val < best_val:
            best = new.copy()
            best_val = val
        else:
            new = 0.7*new + 0.3*best

        p = new

        if track:
            history.append(val)

    return p, history

# =========================
# REFINE
# =========================

def refine(p, track=False):
    history = []

    for _ in range(REFINE_STEPS):
        cands = [p + np.random.normal(0,0.02,len(p)) for _ in range(10)]
        vals = [loss(c) for c in cands]
        p = cands[np.argmin(vals)]

        if track:
            history.append(loss(p))

    return p, history

# =========================
# PIPELINE
# =========================

def pipeline(track=False):
    p = np.random.uniform(-10,10,DIM)
    total_hist = []

    p, h = hlx(p, track=True)
    total_hist += h

    p, h = refine(p, track=True)
    total_hist += h

    p, h = adam(p, track=True)
    total_hist += h

    return p, total_hist

# =========================
# UTIL
# =========================

def smooth(x, k=5):
    return np.convolve(x, np.ones(k)/k, mode='valid')

# =========================
# RUN DEMO
# =========================

def run():
    print("\n🚀 HLX-Adam FULL DEMO\n")

    start = time.time()

    adam_acc = []
    hlx_acc = []

    for _ in tqdm(range(RUNS)):

        # Adam
        p1 = np.random.uniform(-10,10,DIM)
        p1, _ = adam(p1)
        adam_acc.append(accuracy(p1))

        # HLX
        p2, _ = pipeline()
        hlx_acc.append(accuracy(p2))

    adam_avg = np.mean(adam_acc)
    hlx_avg = np.mean(hlx_acc)

    print("\n===== RESULTS =====\n")
    print(f"Adam Accuracy:     {adam_avg:.4f}")
    print(f"HLX-Adam Accuracy: {hlx_avg:.4f}")

    improvement = (hlx_avg - adam_avg) * 100
    print(f"\n🔥 Improvement: +{improvement:.2f}%")

    print("\n===================\n")

    print("📉 Generating convergence snapshot...\n")

    _, adam_hist = adam(np.random.uniform(-10,10,DIM), track=True)
    _, hlx_hist = pipeline(track=True)

    plt.figure(figsize=(8,5))

    plt.plot(smooth(adam_hist), label="Adam", linewidth=2)
    plt.plot(smooth(hlx_hist), label="HLX-Adam", linewidth=2)

    plt.yscale("log")
    plt.title("HLX-Adam vs Adam Convergence")
    plt.suptitle("Structure-Conditioned Optimization", fontsize=10)

    plt.xlabel("Step")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/convergence.png")
    plt.savefig("results/convergence_clean.png")

    plt.show()

    print(f"\n⏱ Runtime: {time.time() - start:.2f}s\n")

# =========================

if __name__ == "__main__":
    run()