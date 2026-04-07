import numpy as np
from config import DIM, HLX_STEPS, REFINE_STEPS, ADAM_STEPS, NOISE

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
# LOSS + ACC
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

def adam(p):
    m = np.zeros_like(p)
    v = np.zeros_like(p)

    for t in range(1, ADAM_STEPS+1):
        g = fast_grad(p)

        m = 0.9*m + 0.1*g
        v = 0.999*v + 0.001*(g*g)

        mh = m/(1-0.9**t)
        vh = v/(1-0.999**t)

        lr = 0.01*(0.99**t)
        p -= lr * mh / (np.sqrt(vh)+1e-8)

    return p

# =========================
# HLX
# =========================

def hlx(p):
    best = p.copy()
    best_val = loss(p)

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

    return p

# =========================
# REFINE
# =========================

def refine(p):
    for _ in range(REFINE_STEPS):
        cands = [p + np.random.normal(0,0.02,len(p)) for _ in range(10)]
        vals = [loss(c) for c in cands]
        p = cands[np.argmin(vals)]
    return p

# =========================
# PIPELINE
# =========================

def solve():
    p = np.random.uniform(-10,10,DIM)
    p = hlx(p)
    p = refine(p)
    p = adam(p)
    return p