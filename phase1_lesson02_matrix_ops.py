import random


class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x**2 for x in self.data) ** 0.5


class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix(
            [
                [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def __sub__(self, other):
        return Matrix(
            [
                [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def scalar_multiply(self, scalar):
        return Matrix(
            [
                [self.data[i][j] * scalar for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def element_wise_multiply(self, other):
        return Matrix(
            [
                [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def matmul(self, other):
        return Matrix(
            [
                [
                    sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                    for j in range(other.cols)
                ]
                for i in range(self.rows)
            ]
        )

    def transpose(self):
        return Matrix(
            [
                [self.data[j][i] for j in range(self.rows)]
                for i in range(self.cols)
            ]
        )

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]

        if self.shape == (2, 2):
            return (
                self.data[0][0] * self.data[1][1]
                - self.data[0][1] * self.data[1][0]
            )

        det = 0
        for j in range(self.cols):
            minor = Matrix(
                [
                    [self.data[i][k] for k in range(self.cols) if k != j]
                    for i in range(1, self.rows)
                ]
            )
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()

        return det

    def inverse_2x2(self):
        det = self.determinant()

        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")

        return Matrix(
            [
                [self.data[1][1] / det, -self.data[0][1] / det],
                [-self.data[1][0] / det, self.data[0][0] / det],
            ]
        )

    def minor(self, row_to_remove, col_to_remove):
        return Matrix(
            [
                [self.data[i][j] for j in range(self.cols) if j != col_to_remove]
                for i in range(self.rows)
                if i != row_to_remove
            ]
        )

    def cofactor_matrix(self):
        return Matrix(
            [
                [
                    ((-1) ** (i + j)) * self.minor(i, j).determinant()
                    for j in range(self.cols)
                ]
                for i in range(self.rows)
            ]
        )

    def inverse_3x3(self):
        if self.shape != (3, 3):
            raise ValueError("inverse_3x3 only supports 3x3 matrices")

        det = self.determinant()

        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")

        cofactors = self.cofactor_matrix()
        adjugate = cofactors.transpose()

        return adjugate.scalar_multiply(1 / det)

    @staticmethod
    def identity(n):
        return Matrix(
            [
                [1 if i == j else 0 for j in range(n)]
                for i in range(n)
            ]
        )


def relu_matrix(matrix):
    return Matrix(
        [
            [max(0, value) for value in row]
            for row in matrix.data
        ]
    )


if __name__ == "__main__":
    print("=== Base Matrix Operations ===")

    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])

    print("\nA + B")
    print(A + B)

    print("\nA element-wise B")
    print(A.element_wise_multiply(B))

    print("\nA @ B")
    print(A.matmul(B))

    print("\nA transpose")
    print(A.transpose())

    print("\ndet(A)")
    print(A.determinant())

    print("\nA inverse")
    print(A.inverse_2x2())

    print("\nA @ A^-1")
    print(A.matmul(A.inverse_2x2()))

    print("\n=== Exercise 1: Verify Inverses ===")

    matrices = [
        Matrix([[1, 2], [3, 4]]),
        Matrix([[2, 0], [0, 5]]),
        Matrix([[4, 7], [2, 6]]),
    ]

    for idx, matrix in enumerate(matrices, start=1):
        inverse = matrix.inverse_2x2()
        product = matrix.matmul(inverse)

        print(f"\nMatrix {idx}")
        print(matrix)
        print("Inverse")
        print(inverse)
        print("Matrix @ inverse")
        print(product)

    print("\nSingular matrix test")
    singular = Matrix([[1, 2], [2, 4]])
    print(f"det(singular) = {singular.determinant()}")

    try:
        singular.inverse_2x2()
    except ValueError as error:
        print(f"Expected error: {error}")

    print("\n=== Exercise 2: 3x3 Inverse ===")

    C = Matrix(
        [
            [1, 2, 3],
            [0, 1, 4],
            [5, 6, 0],
        ]
    )

    C_inverse = C.inverse_3x3()

    print("C")
    print(C)
    print("det(C)")
    print(C.determinant())
    print("C inverse")
    print(C_inverse)
    print("C @ C_inverse")
    print(C.matmul(C_inverse))

    print("\n=== Exercise 3: Two-Layer Neural Network ===")

    random.seed(42)

    x = Matrix(
        [
            [0.5],
            [0.8],
            [0.2],
        ]
    )

    W1 = Matrix(
        [
            [random.uniform(-1, 1) for _ in range(3)]
            for _ in range(4)
        ]
    )

    b1 = Matrix(
        [
            [0.1],
            [0.1],
            [0.1],
            [0.1],
        ]
    )

    W2 = Matrix(
        [
            [random.uniform(-1, 1) for _ in range(4)]
            for _ in range(2)
        ]
    )

    b2 = Matrix(
        [
            [0.1],
            [0.1],
        ]
    )

    hidden_pre_activation = W1.matmul(x) + b1
    hidden_activation = relu_matrix(hidden_pre_activation)

    output_pre_activation = W2.matmul(hidden_activation) + b2
    output = relu_matrix(output_pre_activation)

    print(f"Input shape: {x.shape}")
    print(f"W1 shape: {W1.shape}")
    print(f"Hidden pre-activation shape: {hidden_pre_activation.shape}")
    print(f"Hidden activation shape: {hidden_activation.shape}")
    print(f"W2 shape: {W2.shape}")
    print(f"Output shape: {output.shape}")
    print("Output")
    print(output)
