import random
from collections import Counter

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class NearestCentroid:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.centroids = np.array(
            [
                X[y == cls].mean(axis=0)
                for cls in self.classes
            ]
        )

    def predict(self, X):
        distances = np.array(
            [
                np.sqrt(((X - centroid) ** 2).sum(axis=1))
                for centroid in self.centroids
            ]
        )

        nearest_centroid_indices = distances.argmin(axis=0)
        return self.classes[nearest_centroid_indices]


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


print("=== Phase 2 Lesson 01: What Is Machine Learning ===")

print("\nCore idea:")
print("Machine learning learns patterns from data instead of rules written by hand.")


print("\n=== Exercise 1: Train / Validation / Test Split ===")

iris = load_iris()
X = iris.data
y = iris.target

# First split: 70% train, 30% temporary.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

# Second split: split temporary 50/50 into validation and test.
# That gives 15% validation, 15% test.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print(f"Train size:      {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size:       {len(X_test)}")

print("\nWhy not tune on the test set?")
print(
    "Because the test set is supposed to represent truly unseen data. "
    "If we keep changing the model based on test performance, the test set "
    "quietly becomes part of training, and the final score becomes too optimistic."
)


print("\n=== Nearest Centroid Classifier From Scratch ===")

model = NearestCentroid()
model.fit(X_train, y_train)

val_predictions = model.predict(X_val)
test_predictions = model.predict(X_test)

val_acc = accuracy(y_val, val_predictions)
test_acc = accuracy(y_test, test_predictions)

print(f"Validation accuracy: {val_acc:.3f}")
print(f"Test accuracy:       {test_acc:.3f}")


print("\n=== Random Baseline ===")

random.seed(42)
random_predictions = np.array(
    [
        random.choice(list(model.classes))
        for _ in range(len(y_test))
    ]
)

baseline_acc = accuracy(y_test, random_predictions)

print(f"Random baseline accuracy: {baseline_acc:.3f}")
print(
    "A real ML model should beat a simple baseline. "
    "If it does not, the model has not learned anything useful."
)


print("\n=== Majority-Class Baseline ===")

class_counts = Counter(y_train)
majority_class = class_counts.most_common(1)[0][0]

majority_predictions = np.array([majority_class] * len(y_test))
majority_acc = accuracy(y_test, majority_predictions)

print(f"Majority class: {majority_class}")
print(f"Majority baseline accuracy: {majority_acc:.3f}")


print("\n=== Exercise 2: Classify Real-World ML Problems ===")

problems = [
    {
        "problem": "Predict whether an email is spam.",
        "task": "classification",
        "learning_type": "supervised",
        "why": "The output is a discrete label, and training examples have known labels.",
    },
    {
        "problem": "Predict next month's revenue.",
        "task": "regression",
        "learning_type": "supervised",
        "why": "The output is a continuous number, and historical examples provide targets.",
    },
    {
        "problem": "Group customers by shopping behavior.",
        "task": "clustering",
        "learning_type": "unsupervised",
        "why": "There are no labels; the model discovers structure in the data.",
    },
]

for item in problems:
    print(f"\nProblem: {item['problem']}")
    print(f"Task: {item['task']}")
    print(f"Learning type: {item['learning_type']}")
    print(f"Why: {item['why']}")


print("\n=== Exercise 3: Diagnose 99% Train Accuracy, 60% Test Accuracy ===")

print("Diagnosis: overfitting.")
print(
    "The model learned training-specific details or noise instead of general patterns. "
    "That is why it performs very well on training data but poorly on unseen test data."
)

fixes = [
    "Get more training data.",
    "Reduce model complexity.",
    "Add regularization.",
    "Use early stopping based on validation performance.",
    "Check for data leakage or train/test distribution mismatch.",
]

print("\nPossible fixes:")
for fix in fixes:
    print(f"- {fix}")


print("\n=== Summary ===")
print("Fit means learn from training data.")
print("Predict means apply the learned pattern to new data.")
print("Evaluate means measure performance on data the model did not train on.")
print("Baseline comparison tells us whether the model is useful at all.")
