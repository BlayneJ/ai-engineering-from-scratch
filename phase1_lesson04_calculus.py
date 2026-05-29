def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_second_derivative(f, x, h=1e-5):
    return numerical_derivative(lambda t: numerical_derivative(f, t, h), x, h)


def numerical_gradient(f, point, h=1e-7):
    gradient = []

    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)

        point_plus[i] += h
        point_minus[i] -= h

        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)

    return gradient


def exercise_1_second_derivative():
    print("=== Exercise 1: Numerical Second Derivative ===")

    def f(x):
        return x**3

    x = 2.0
    numerical = numerical_second_derivative(f, x)
    analytical = 6 * x

    print(f"f(x) = x^3")
    print(f"x = {x}")
    print(f"numerical f''(x) = {numerical:.6f}")
    print(f"analytical f''(x) = {analytical:.6f}")


def exercise_2_gradient_descent_2d():
    print("\n=== Exercise 2: Gradient Descent in 2D ===")

    def f(point):
        x, y = point
        return (x - 3) ** 2 + (y + 1) ** 2

    point = [0.0, 0.0]
    lr = 0.1

    for step in range(50):
        grad = numerical_gradient(f, point)
        point = [p - lr * g for p, g in zip(point, grad)]
        loss = f(point)

        if step % 10 == 0 or step == 49:
            print(
                f"step {step:2d} "
                f"point=({point[0]:.6f}, {point[1]:.6f}) "
                f"loss={loss:.8f}"
            )

    print("Expected minimum: (3, -1)")


def gradient_descent_1d(f_prime, x0, lr, steps):
    x = x0
    history = []

    for _ in range(steps):
        grad = f_prime(x)
        x = x - lr * grad
        history.append(x)

    return history


def momentum_gradient_descent_1d(f_prime, x0, lr, momentum, steps):
    x = x0
    velocity = 0.0
    history = []

    for _ in range(steps):
        grad = f_prime(x)
        velocity = momentum * velocity - lr * grad
        x = x + velocity
        history.append(x)

    return history


def exercise_3_momentum():
    print("\n=== Exercise 3: Momentum Gradient Descent ===")

    def f(x):
        return x**4 - 3 * x**2

    def f_prime(x):
        return 4 * x**3 - 6 * x

    x0 = 0.5
    lr = 0.02
    steps = 40

    plain_history = gradient_descent_1d(f_prime, x0, lr, steps)
    momentum_history = momentum_gradient_descent_1d(
        f_prime,
        x0,
        lr,
        momentum=0.9,
        steps=steps,
    )

    print("Function: f(x) = x^4 - 3x^2")
    print("Critical minima are near x = +/- sqrt(1.5) = +/- 1.2247")
    print()

    print("step   plain_x   plain_f     momentum_x   momentum_f")
    print("-" * 58)

    for step in [0, 1, 2, 5, 10, 20, 39]:
        plain_x = plain_history[step]
        momentum_x = momentum_history[step]

        print(
            f"{step:4d} "
            f"{plain_x:9.5f} {f(plain_x):10.6f} "
            f"{momentum_x:12.5f} {f(momentum_x):12.6f}"
        )


if __name__ == "__main__":
    exercise_1_second_derivative()
    exercise_2_gradient_descent_2d()
    exercise_3_momentum()
