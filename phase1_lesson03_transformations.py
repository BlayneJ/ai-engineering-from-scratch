import math


def rotation_2d(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]


def scaling_2d(sx, sy):
    return [[sx, 0], [0, sy]]


def shearing_2d(kx, ky):
    return [[1, kx], [ky, 1]]


def mat_vec_mul(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]


def mat_mul(a, b):
    rows_a, cols_b = len(a), len(b[0])
    cols_a = len(a[0])

    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]


def det_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def distance(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a))) ** 0.5


def eigenvalues_2x2(matrix):
    a, b = matrix[0]
    c, d = matrix[1]

    trace = a + d
    det = a * d - b * c
    discriminant = trace**2 - 4 * det

    if discriminant < 0:
        real = trace / 2
        imag = (-discriminant) ** 0.5 / 2
        return (complex(real, imag), complex(real, -imag))

    sqrt_disc = discriminant**0.5
    return ((trace + sqrt_disc) / 2, (trace - sqrt_disc) / 2)


def eigenvector_2x2(matrix, eigenvalue):
    a, b = matrix[0]
    c, d = matrix[1]

    if abs(b) > 1e-10:
        v = [b, eigenvalue - a]
    elif abs(c) > 1e-10:
        v = [eigenvalue - d, c]
    else:
        if abs(a - eigenvalue) < 1e-10:
            v = [1, 0]
        else:
            v = [0, 1]

    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    return [v[0] / mag, v[1] / mag]


def fmt(value, decimals=4):
    if isinstance(value, list):
        return [round(x, decimals) for x in value]

    return round(value, decimals)


print("=== Exercise 1: Transform a Unit Square ===")

square = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
]

rotation = rotation_2d(math.pi / 4)
scaling = scaling_2d(2, 0.5)
shearing = shearing_2d(0.5, 0)

for name, transform in [
    ("rotation 45 degrees", rotation),
    ("scaling sx=2 sy=0.5", scaling),
    ("shearing kx=0.5", shearing),
]:
    print(f"\n{name}")

    transformed = [mat_vec_mul(transform, point) for point in square]

    for before, after in zip(square, transformed):
        print(f"{fmt(before)} -> {fmt(after)}")

    original_edge = distance(square[0], square[1])
    transformed_edge = distance(transformed[0], transformed[1])

    print(f"original edge length: {original_edge:.4f}")
    print(f"transformed edge length: {transformed_edge:.4f}")

print("\nRotation should preserve the unit edge length.")


print("\n=== Exercise 2: Eigenvalues of [[4, 2], [1, 3]] ===")

A = [[4, 2], [1, 3]]

a, b = A[0]
c, d = A[1]
trace = a + d
det = a * d - b * c

print(f"trace = {trace}")
print(f"determinant = {det}")
print("characteristic equation: lambda^2 - 7lambda + 10 = 0")
print("roots: lambda = 5 and lambda = 2")

values = eigenvalues_2x2(A)
print(f"from-scratch eigenvalues: {values}")

for value in values:
    vector = eigenvector_2x2(A, value)
    transformed = mat_vec_mul(A, vector)
    scaled = [value * vector[0], value * vector[1]]

    print(f"\nlambda = {fmt(value)}")
    print(f"eigenvector = {fmt(vector)}")
    print(f"A @ v = {fmt(transformed)}")
    print(f"lambda * v = {fmt(scaled)}")

try:
    import numpy as np

    np_values, np_vectors = np.linalg.eig(np.array(A, dtype=float))
    print(f"\nNumPy eigenvalues: {np_values}")
    print(f"NumPy eigenvectors as columns:\n{np_vectors}")
except ImportError:
    print("\nNumPy not installed; skipping NumPy verification.")


print("\n=== Exercise 3: Compose Transformations on Circle Points ===")

circle_points = []

for i in range(8):
    angle = 2 * math.pi * i / 8
    circle_points.append([math.cos(angle), math.sin(angle)])

R = rotation_2d(math.radians(30))
S = scaling_2d(1.5, 0.8)
Sh = shearing_2d(0.3, 0)

# Apply rotate, then scale, then shear.
# Matrix composition is written in reverse application order: Sh @ S @ R.
composed = mat_mul(Sh, mat_mul(S, R))

print("\nBefore -> After")
for point in circle_points:
    transformed = mat_vec_mul(composed, point)
    print(f"{fmt(point)} -> {fmt(transformed)}")

det_R = det_2x2(R)
det_S = det_2x2(S)
det_Sh = det_2x2(Sh)
det_composed = det_2x2(composed)

print("\nDeterminants")
print(f"det(rotation) = {det_R:.4f}")
print(f"det(scale) = {det_S:.4f}")
print(f"det(shear) = {det_Sh:.4f}")
print(f"product = {det_R * det_S * det_Sh:.4f}")
print(f"det(composed) = {det_composed:.4f}")
