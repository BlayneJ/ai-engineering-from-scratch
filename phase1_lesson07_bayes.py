import math
from collections import defaultdict


def bayes(prior, likelihood, false_positive_rate):
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    return likelihood * prior / evidence


print("=== Exercise 1: Two Independent Positive Tests ===")

prior = 0.0001
likelihood = 0.99
false_positive_rate = 0.01

after_first = bayes(prior, likelihood, false_positive_rate)
after_second = bayes(after_first, likelihood, false_positive_rate)

print(f"Prior:                 {prior:.6f} ({prior * 100:.4f}%)")
print(f"After first positive:  {after_first:.6f} ({after_first * 100:.2f}%)")
print(f"After second positive: {after_second:.6f} ({after_second * 100:.2f}%)")


class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.class_word_totals = defaultdict(int)
        self.vocab = set()

    def train(self, documents, labels):
        for doc, label in zip(documents, labels):
            self.class_counts[label] += 1
            words = doc.lower().split()

            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_totals[label] += 1
                self.vocab.add(word)

    def word_probability(self, word, cls):
        count = self.word_counts[cls].get(word, 0)
        total = self.class_word_totals[cls]
        vocab_size = len(self.vocab)

        if self.smoothing == 0:
            return count / total if total > 0 else 0

        return (count + self.smoothing) / (
            total + self.smoothing * vocab_size
        )

    def top_words(self, cls, n=5):
        probabilities = {}

        for word in self.vocab:
            probabilities[word] = self.word_probability(word, cls)

        return sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:n]

    def predict(self, document):
        words = document.lower().split()
        total_docs = sum(self.class_counts.values())

        best_class = None
        best_score = float("-inf")

        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total_docs)

            for word in words:
                prob = self.word_probability(word, cls)

                if prob == 0:
                    score = float("-inf")
                    break

                score += math.log(prob)

            if score > best_score:
                best_score = score
                best_class = cls

        return best_class, best_score


train_docs = [
    "win free money now",
    "free lottery ticket winner",
    "claim your prize today free",
    "urgent offer free cash",
    "congratulations you won free",
    "meeting tomorrow at noon",
    "project update attached",
    "can we schedule a call",
    "quarterly report review",
    "lunch on thursday sounds good",
    "team standup notes attached",
    "please review the pull request",
]

train_labels = [
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
    "ham",
    "ham",
    "ham",
    "ham",
    "ham",
    "ham",
    "ham",
]


print("\n=== Exercise 2: Smoothing Impact ===")

for smoothing in [0, 0.01, 0.1, 1.0, 10.0]:
    classifier = NaiveBayes(smoothing=smoothing)
    classifier.train(train_docs, train_labels)

    message = "free attached"
    prediction, score = classifier.predict(message)

    print(f"\nSmoothing = {smoothing}")
    print(f"Prediction for '{message}': {prediction}")
    print(f"Score: {score}")
    print("Top spam words:", classifier.top_words("spam", 5))
    print("Top ham words:", classifier.top_words("ham", 5))
    print(
        "P(attached | spam) =",
        classifier.word_probability("attached", "spam"),
    )
    print(
        "P(attached | ham) =",
        classifier.word_probability("attached", "ham"),
    )


class NaiveBayesWithLength(NaiveBayes):
    def __init__(self, smoothing=1.0):
        super().__init__(smoothing=smoothing)
        self.length_counts = defaultdict(lambda: defaultdict(int))
        self.length_totals = defaultdict(int)
        self.length_vocab = {"short", "long"}

    def length_feature(self, document):
        word_count = len(document.lower().split())
        return "short" if word_count <= 4 else "long"

    def train(self, documents, labels):
        super().train(documents, labels)

        for doc, label in zip(documents, labels):
            feature = self.length_feature(doc)
            self.length_counts[label][feature] += 1
            self.length_totals[label] += 1

    def length_probability(self, feature, cls):
        count = self.length_counts[cls].get(feature, 0)
        total = self.length_totals[cls]
        vocab_size = len(self.length_vocab)

        return (count + self.smoothing) / (
            total + self.smoothing * vocab_size
        )

    def predict(self, document):
        words = document.lower().split()
        length_feature = self.length_feature(document)
        total_docs = sum(self.class_counts.values())

        best_class = None
        best_score = float("-inf")

        for cls in self.class_counts:
            score = math.log(self.class_counts[cls] / total_docs)

            for word in words:
                prob = self.word_probability(word, cls)

                if prob == 0:
                    score = float("-inf")
                    break

                score += math.log(prob)

            if score != float("-inf"):
                score += math.log(
                    self.length_probability(length_feature, cls)
                )

            if score > best_score:
                best_score = score
                best_class = cls

        return best_class, best_score


print("\n=== Exercise 3: Naive Bayes With Message Length Feature ===")

classifier = NaiveBayesWithLength(smoothing=1.0)
classifier.train(train_docs, train_labels)

for cls in ["spam", "ham"]:
    print(f"\nClass: {cls}")
    print("P(short | class) =", classifier.length_probability("short", cls))
    print("P(long | class)  =", classifier.length_probability("long", cls))

test_messages = [
    "free money",
    "free money waiting for you",
    "meeting tomorrow",
    "please review the attached quarterly report",
]

for message in test_messages:
    prediction, score = classifier.predict(message)
    print(f"\nMessage: '{message}'")
    print(f"Length feature: {classifier.length_feature(message)}")
    print(f"Prediction: {prediction}")
    print(f"Score: {score:.4f}")


print("\n=== Exercise 4: MAP Estimate With Beta Prior ===")

heads = 7
tails = 3
total = heads + tails

mle = heads / total

alpha = 2
beta = 2

map_estimate = (heads + alpha - 1) / (
    heads + tails + alpha + beta - 2
)

posterior_alpha = alpha + heads
posterior_beta = beta + tails
posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)

print(f"Observed data: {heads} heads, {tails} tails")
print(f"MLE estimate: {mle:.4f}")

print(f"\nPrior: Beta({alpha}, {beta})")
print(f"Prior mean: {alpha / (alpha + beta):.4f}")

print(f"\nPosterior: Beta({posterior_alpha}, {posterior_beta})")
print(f"Posterior mean: {posterior_mean:.4f}")

print(f"\nMAP estimate: {map_estimate:.4f}")
print("The Beta(2,2) prior pulls the estimate toward 0.5.")
