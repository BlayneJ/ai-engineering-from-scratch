import math
import random


def exponential_pdf(x, lam):
    if x < 0:
        return 0.0
    return lam * math.exp(-lam * x)


def sample_exponential_inverse(lam, n):
    samples = []

    for _ in range(n):
        u = random.random()
        x = -math.log(1 - u) / lam
        samples.append(x)

    return samples


def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]


def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]


def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]


def analyze_log_prob_sequence(log_probs):
    total_log_prob = sum(log_probs)
    raw_probability = math.exp(total_log_prob)
    most_likely_indices = list(range(len(log_probs)))

    return most_likely_indices, total_log_prob, raw_probability


print("=== Exercise 1: Inverse Transform Sampling for Exponential ===")

random.seed(42)
lam = 2.0
samples = sample_exponential_inverse(lam, 10_000)

sample_mean = sum(samples) / len(samples)
true_mean = 1 / lam

sample_variance = sum((x - sample_mean) ** 2 for x in samples) / len(samples)
true_variance = 1 / (lam**2)

print(f"lambda = {lam}")
print(f"sample mean = {sample_mean:.4f}")
print(f"true mean = {true_mean:.4f}")
print(f"sample variance = {sample_variance:.4f}")
print(f"true variance = {true_variance:.4f}")

try:
    import matplotlib.pyplot as plt

    xs = [i / 50 for i in range(250)]
    ys = [exponential_pdf(x, lam) for x in xs]

    plt.hist(samples, bins=60, density=True, alpha=0.6, label="samples")
    plt.plot(xs, ys, label="true PDF")
    plt.title("Exponential Distribution via Inverse Transform Sampling")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend()
    plt.savefig("phase1_lesson06_exponential.png")
    print("saved plot: phase1_lesson06_exponential.png")
except ImportError:
    print("matplotlib not installed; skipping plot.")


print("\n=== Exercise 2: Joint Distribution for Two Loaded Dice ===")

die_a = [0.05, 0.10, 0.15, 0.20, 0.20, 0.30]
die_b = [0.10, 0.10, 0.10, 0.20, 0.20, 0.30]

joint = [
    [pa * pb for pb in die_b]
    for pa in die_a
]

marginal_a = [sum(row) for row in joint]
marginal_b = [sum(joint[i][j] for i in range(6)) for j in range(6)]

print("Marginal A:", [round(x, 4) for x in marginal_a])
print("Original A: ", die_a)
print("Marginal B:", [round(x, 4) for x in marginal_b])
print("Original B: ", die_b)

independent = True

for i in range(6):
    for j in range(6):
        if abs(joint[i][j] - marginal_a[i] * marginal_b[j]) > 1e-10:
            independent = False

print(f"Dice independent? {independent}")


print("\n=== Exercise 3: Cross-Entropy from Logits ===")

logits = [2.0, 0.5, -1.0, 3.0, 0.1]
target_index = 3

probs = softmax(logits)
loss = cross_entropy_loss(logits, target_index)

print(f"logits = {logits}")
print(f"softmax = {[round(p, 6) for p in probs]}")
print(f"target index = {target_index}")
print(f"cross-entropy loss = {loss:.6f}")

try:
    import torch
    import torch.nn as nn

    torch_logits = torch.tensor([logits], dtype=torch.float32)
    torch_target = torch.tensor([target_index])
    torch_loss = nn.CrossEntropyLoss()(torch_logits, torch_target)

    print(f"PyTorch CrossEntropyLoss = {torch_loss.item():.6f}")
except ImportError:
    print("PyTorch not installed; skipping verification.")


print("\n=== Exercise 4: Log Probability Sequence ===")

# 50 words, each with probability 0.01
log_probs = [math.log(0.01)] * 50

indices, total_log_prob, raw_probability = analyze_log_prob_sequence(log_probs)

print(f"sequence length = {len(indices)}")
print(f"single-word probability = 0.01")
print(f"single-word log probability = {math.log(0.01):.6f}")
print(f"total log probability = {total_log_prob:.6f}")
print(f"raw probability = {raw_probability:.6e}")

direct_raw = 0.01**50
print(f"direct 0.01**50 = {direct_raw:.6e}")
print("Log probabilities stay numerically stable even when raw probabilities are tiny.")
