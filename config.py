# config.py

MODE = "demo"  # "dev" or "demo"

if MODE == "dev":
    RUNS = 3
    DIM = 673
    HLX_STEPS = 60
    REFINE_STEPS = 20
    ADAM_STEPS = 60
    NOISE = 0.003

else:  # demo mode
    RUNS = 10
    DIM = 673
    HLX_STEPS = 80
    REFINE_STEPS = 30
    ADAM_STEPS = 100
    NOISE = 0.005