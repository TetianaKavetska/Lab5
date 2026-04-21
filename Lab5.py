import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


a = 0
b = 24


# Метод Сімпсона
def simpson(f, a, b, n_steps):
    if n_steps % 2 != 0:
        n_steps += 1
    h = (b - a) / n_steps
    x_nodes = np.linspace(a, b, n_steps + 1)
    y_nodes = f(x_nodes)
    s = y_nodes[0] + y_nodes[-1] + 4 * np.sum(y_nodes[1:-1:2]) + 2 * np.sum(y_nodes[2:-2:2])
    return s * h / 3


I0 = simpson(f, a, b, 100000)
eps = 1e-12
n_current = 10
while True:
    i_n_val = simpson(f, a, b, n_current)
    error_val = abs(i_n_val - I0)
    if error_val < eps or n_current > 5000:
        break
    n_current += 2

n0_res = n_current

print(f"Точне значення інтегралу I0 = {I0:.15f}")
print("-" * 45)
print(f"1. МЕТОД СІМПСОНА (Базовий):")
print(f"   Кількість кроків N0 = {n0_res}")
print(f"   Значення I(N0)      = {i_n_val:.15f}")
print(f"   Абсолютна похибка   = {error_val:.2e}")

print("-" * 45)
print(f"2. МЕТОД РУНГЕ-РОМБЕРГА (Уточнення):")
i_half_n0 = simpson(f, a, b, n0_res // 2)
i_runge_n0 = i_n_val + (i_n_val - i_half_n0) / 15
print(f"   Значення I_rr       = {i_runge_n0:.15f}")
print(f"   Похибка уточнення   = {abs(i_runge_n0 - I0):.2e}")

print("-" * 45)
print(f"3. МЕТОД ЕЙТКЕНА (Екстраполяція):")
n1, n2, n3 = n0_res, n0_res // 2, n0_res // 4
if n3 % 2 != 0: n3 += 1

s1, s2, s3 = simpson(f, a, b, n1), simpson(f, a, b, n2), simpson(f, a, b, n3)
denom = s2 - s1

if abs(denom) > 1e-18:
    p_val_final = np.log(abs((s3 - s2) / denom + 1e-20)) / np.log(2)
    i_aitken_final = s1 + (s1 - s2) / (2 ** p_val_final - 1)
    print(f"   Порядок збіжності p = {p_val_final:.2f}")
    print(f"   Значення I_aitken   = {i_aitken_final:.15f}")
    print(f"   Похибка уточнення   = {abs(i_aitken_final - I0):.2e}")
else:
    print("   Метод Ейткена: Недостатньо різниці для розрахунку")
print("-" * 45)

n_range = np.arange(10, 110, 10)
err_simpson = []
err_runge = []
err_aitken = []

for n in n_range:

    i_s = simpson(f, a, b, n)
    err_simpson.append(abs(i_s - I0))

    i_h = simpson(f, a, b, n // 2)
    err_runge.append(abs(i_s + (i_s - i_h) / 15 - I0))

    if n >= 20:
        a1, a2, a3 = n, n // 2, n // 4
        if a3 % 2 != 0: a3 += 1
        v1, v2, v3 = simpson(f, a, b, a1), simpson(f, a, b, a2), simpson(f, a, b, a3)
        d = v2 - v1
        if abs(d) > 1e-18:
            p = np.log(abs((v3 - v2) / d + 1e-20)) / np.log(2)
            res_ait = v1 + (v1 - v2) / (2 ** p - 1)
            err_aitken.append(abs(res_ait - I0))
        else:
            err_aitken.append(None)
    else:
        err_aitken.append(None)

print("\n[INFO] Закрийте вікна графіків, щоб завершити програму.")

plt.figure(figsize=(8, 4))
plt.plot(np.linspace(a, b, 1000), f(np.linspace(a, b, 1000)))
plt.title("Графік функції навантаження")
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.semilogy(n_range, err_simpson, 'r-o', label='Метод Сімпсона')
plt.semilogy(n_range, err_runge, 'g-s', label='Рунге-Ромберг')
plt.semilogy(n_range, err_aitken, 'b-d', label='Ейткен')
plt.title("Порівняльний аналіз похибок")
plt.xlabel("N")
plt.ylabel("Похибка (log)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()


# Адаптивний метод Сімпсона
def adaptive_simpson(f, a, b, eps, total_calls=0):
    c = (a + b) / 2
    h = b - a

    fa, fb, fc = f(a), f(b), f(c)
    total_calls += 3

    def recursive_step(a, b, eps, fa, fb, fc):
        nonlocal total_calls
        c = (a + b) / 2
        h = b - a
        d = (a + c) / 2
        e = (c + b) / 2
        fd, fe = f(d), f(e)
        total_calls += 2

        s_whole = h * (fa + 4 * fc + fb) / 6
        s_left = (h / 2) * (fa + 4 * fd + fc) / 6
        s_right = (h / 2) * (fc + 4 * fe + fb) / 6
        s_sum = s_left + s_right

        if abs(s_sum - s_whole) <= 15 * eps:
            return s_sum + (s_sum - s_whole) / 15

        return (recursive_step(a, c, eps / 2, fa, fc, fd) +
                recursive_step(c, b, eps / 2, fc, fb, fe))

    result = recursive_step(a, b, eps, fa, fb, fc)
    return result, total_calls


eps_values = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]
results_table = []

for e in eps_values:
    val, calls = adaptive_simpson(f, a, b, e)
    results_table.append((e, calls, abs(val - I0)))

print("\n" + "=" * 50)
print("ДОСЛІДЖЕННЯ АДАПТИВНОГО АЛГОРИТМУ")
print(f"{'Точність (eps)':<15} | {'Викликів f(x)':<15} | {'Реальна похибка':<15}")
print("-" * 50)
for e, c, err in results_table:
    print(f"{e:<15.0e} | {c:<15} | {err:<15.2e}")
print("=" * 50)
