import random


class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.components])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def angle_between(self, other):
        import math

        cos_theta = self.cosine_similarity(other)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.degrees(math.acos(cos_theta))

    def project_onto(self, other):
        scalar = self.dot(other) / other.dot(other)
        return Vector([scalar * x for x in other.components])

    def __repr__(self):
        return f"Vector({self.components})"


def gram_schmidt(vectors):
    orthonormal = []

    for v in vectors:
        w = v

        for u in orthonormal:
            proj = w.project_onto(u)
            w = w - proj

        if w.magnitude() < 1e-10:
            continue

        orthonormal.append(w.normalize())

    return orthonormal


class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector(
                [
                    sum(
                        self.rows[i][j] * other.components[j]
                        for j in range(self.shape[1])
                    )
                    for i in range(self.shape[0])
                ]
            )

        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(
                    sum(
                        self.rows[i][k] * other.rows[k][j]
                        for k in range(self.shape[1])
                    )
                )
            rows.append(row)

        return Matrix(rows)

    def rank(self):
        rows = [row[:] for row in self.rows]
        m, n = self.shape
        r = 0

        for col in range(n):
            pivot = None

            for row in range(r, m):
                if abs(rows[row][col]) > 1e-10:
                    pivot = row
                    break

            if pivot is None:
                continue

            rows[r], rows[pivot] = rows[pivot], rows[r]
            scale = rows[r][col]
            rows[r] = [x / scale for x in rows[r]]

            for row in range(m):
                if row != r and abs(rows[row][col]) > 1e-10:
                    factor = rows[row][col]
                    rows[row] = [
                        rows[row][j] - factor * rows[r][j]
                        for j in range(n)
                    ]

            r += 1

        return r

    def __repr__(self):
        return f"Matrix({self.rows})"


print("=== Exercise 1: Angle Between Vectors ===")
a = Vector([1, 0])
b = Vector([0, 1])
c = Vector([1, 1])

print(f"Angle between {a} and {b}: {a.angle_between(b):.1f} degrees")
print(f"Angle between {a} and {c}: {a.angle_between(c):.1f} degrees")


print("\n=== Exercise 2: Scaling Matrix ===")
scaling = Matrix([[2, 0], [0, 3]])
v = Vector([1, 1])
scaled = scaling @ v

print(f"Scaling matrix: {scaling}")
print(f"Original vector: {v}")
print(f"Scaled vector: {scaled}")


print("\n=== Exercise 3: Most Similar Word-Like Vectors ===")
random.seed(42)

words = {
    f"word_{i}": Vector([random.gauss(0, 1) for _ in range(50)])
    for i in range(5)
}

best_pair = None
best_score = -1

names = list(words)

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        score = words[names[i]].cosine_similarity(words[names[j]])
        print(f"{names[i]} vs {names[j]}: cosine={score:.4f}")

        if score > best_score:
            best_score = score
            best_pair = (names[i], names[j])

print(f"Most similar pair: {best_pair}, cosine={best_score:.4f}")


print("\n=== Exercise 4: Verify Gram-Schmidt Orthonormality ===")
vectors = [
    Vector([1, 1, 0]),
    Vector([1, 0, 1]),
    Vector([0, 1, 1]),
]

basis = gram_schmidt(vectors)

for i, u in enumerate(basis):
    print(f"u{i + 1} = {u}")
    print(f"|u{i + 1}| = {u.magnitude():.6f}")

for i in range(len(basis)):
    for j in range(i + 1, len(basis)):
        print(f"u{i + 1} dot u{j + 1} = {basis[i].dot(basis[j]):.6f}")


print("\n=== Exercise 5: Rank-2 Matrix ===")
rank_2 = Matrix(
    [
        [1, 2, 3],
        [2, 4, 6],
        [0, 1, 1],
    ]
)

print(f"Matrix: {rank_2}")
print(f"Rank: {rank_2.rank()}")
print("The columns span a 2D plane inside 3D space.")


print("\n=== Exercise 6: Projection ===")
p = Vector([1, 2, 3])
direction = Vector([1, 1, 1])
projection = p.project_onto(direction)
residual = p - projection

print(f"Vector: {p}")
print(f"Direction: {direction}")
print(f"Projection: {projection}")
print(f"Residual: {residual}")
print(f"Residual dot direction: {residual.dot(direction):.6f}")
print("Geometrically, the projection is the shadow of [1, 2, 3] onto the line x = y = z.")
