
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
from typing import List, Tuple, Optional

Edge = Tuple[int, int]
AdjList = List[List[int]]

# 1.Побудова списку суміжності з переліку ребер.
def create_graph(n: int, edges: List[Edge], directed: bool = False) -> AdjList: 
    
    graph: AdjList = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        if not directed:
            graph[v].append(u)
    return graph

# 2. Генерація випадкових ребер для графа з n вершинами
def generate_random_edges(n: int, density: float, directed: bool) -> List[Edge]: 
   
    edges: List[Edge] = []
    if directed:
        for u in range(n):
            for v in range(n):
                if u != v and random.random() < density:
                    edges.append((u, v))
    else:
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < density:
                    edges.append((u, v))
    return edges

# 3. Безпечне читання цілого числа з опційною перевіркою меж
def read_int(prompt: str, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    
    while True:
        try:
            value = int(input(prompt))
            if min_value is not None and value < min_value:
                print(f"Значення має бути ≥ {min_value}. Спробуйте ще раз.")
                continue
            if max_value is not None and value > max_value:
                print(f"Значення має бути ≤ {max_value}. Спробуйте ще раз.")
                continue
            return value
        except ValueError:
            print("Помилка: введіть ціле число.")

# 4.Безпечне читання дійсного числа з опційною перевіркою меж
def read_float(prompt: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    
    while True:
        try:
            value = float(input(prompt))
            if min_value is not None and value < min_value:
                print(f"Значення має бути ≥ {min_value}. Спробуйте ще раз.")
                continue
            if max_value is not None and value > max_value:
                print(f"Значення має бути ≤ {max_value}. Спробуйте ще раз.")
                continue
            return value
        except ValueError:
            print("Помилка: введіть число.")

# 5. Зчитування режиму обчислень: paths / cycles / both
def read_mode() -> str:
    
    valid_modes = {"paths", "cycles", "both"}
    while True:
        mode = input("Режим (paths / cycles / both): ").strip().lower()
        if mode in valid_modes:
            return mode
        print("Некоректний режим. Введіть paths, cycles або both.")

# 6. Зчитування графа з клавіатури або автоматична генерація
def read_input() -> tuple[int, List[Edge], bool, str]:
    
    print("Виберіть режим введення графа:")
    print("1 - Ввести вручну")
    print("2 - Автоматично згенерувати")
    choice = read_int("Ваш вибір: ", 1, 2)

    # РУЧНЕ ВВЕДЕННЯ 
    if choice == 1:
        n = read_int("Введіть кількість вершин n: ", 1)
        m = read_int("Введіть кількість ребер m: ", 0)
        directed = read_int("Орієнтований граф? (1 - так, 0 - ні): ", 0, 1) == 1

        print("Введіть ребра (u v) від 1 до n:")
        edges: List[Edge] = []
        for i in range(m):
            while True:
                try:
                    u_str = input(f"Ребро #{i + 1}: ")
                    u, v = map(int, u_str.split())
                    if not (1 <= u <= n and 1 <= v <= n):
                        print("Помилка: вершини мають бути в межах [1, n]. Спробуйте ще раз.")
                        continue
                    if u == v:
                        print("Попередження: петля (u = v) ігнорується. Спробуйте ще раз.")
                        continue
                    edges.append((u - 1, v - 1))
                    break
                except ValueError:
                    print("Помилка: введіть два цілі числа через пробіл.")
        mode = read_mode()
        return n, edges, directed, mode

    # АВТОМАТИЧНА ГЕНЕРАЦІЯ 
    n = read_int("Кількість вершин n: ", 1)
    directed = read_int("Орієнтований граф? (1 - так, 0 - ні): ", 0, 1) == 1
    density = read_float("Ймовірність ребра (0..1, напр. 0.4): ", 0.0, 1.0)

    edges = generate_random_edges(n, density, directed)
    print(f"Згенеровано {len(edges)} ребер.")
    mode = read_mode()
    return n, edges, directed, mode

# 7. Пошук усіх гамільтонових шляхів у графі методом повернення назад
def find_hamiltonian_paths(graph: AdjList) -> List[List[int]]:
    
    n = len(graph)
    used = [False] * n
    path: List[int] = []
    all_paths: List[List[int]] = []

    def backtrack(v: int, depth: int) -> None:
        path.append(v)
        used[v] = True

        if depth == n:
            all_paths.append(path.copy())
        else:
            for to in graph[v]:
                if not used[to]:
                    backtrack(to, depth + 1)

        used[v] = False
        path.pop()

    # Запускаємо пошук з кожної можливої стартової вершини
    for start in range(n):
        backtrack(start, 1)

    return all_paths

# 8. Пошук усіх гамільтонових циклів у графі методом повернення назад
def find_hamiltonian_cycles(graph: AdjList, directed: bool = False) -> List[List[int]]:
    
    n = len(graph)
    used = [False] * n
    path: List[int] = [0]  # починаємо з вершини 0
    used[0] = True
    all_cycles: List[List[int]] = []

    def backtrack(v: int, depth: int) -> None:
        if depth == n:
            # остання вершина повинна мати ребро до 0
            if 0 in graph[v]:
                cycle = path.copy() + [0]
                if not directed:
                    if path[1] < path[-1]:
                        all_cycles.append(cycle)
                else:
                    all_cycles.append(cycle)
            return

        for to in graph[v]:
            if not used[to]:
                used[to] = True
                path.append(to)
                backtrack(to, depth + 1)
                path.pop()
                used[to] = False

    backtrack(0, 1)
    return all_cycles

# 9. Малювання графа за допомогою networkx
def draw_graph(n: int, edges: List[Edge], directed: bool = False, title: str = "Граф") -> None:
    
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    for u, v in edges:
        G.add_edge(u + 1, v + 1)

    pos = nx.circular_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=800,
        font_weight="bold",
        arrows=directed,
    )
    plt.title(title)
    plt.show()


def draw_path_on_graph(
    n: int,
    edges: List[Edge],
    path: List[int],
    directed: bool = False,
    title: str = "Шлях",
) -> None:
# Малювання графа з виділенням заданого шляху / циклу червоним кольором
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    for u, v in edges:
        G.add_edge(u + 1, v + 1)

    pos = nx.circular_layout(G)

    # малюємо граф
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=800,
        font_weight="bold",
        arrows=directed,
    )

    # path: масив типу [0, 1, 3, 2, 4]
    red_edges = [(path[i] + 1, path[i + 1] + 1) for i in range(len(path) - 1)]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=red_edges,
        edge_color="red",
        width=3,
        arrows=directed,
    )

    plt.title(title)
    plt.show()

# 10. Збереження списку шляхів / циклів у текстовий файл
def save_paths(filename: str, paths: List[List[int]], header: str) -> None:
   
    with open(filename, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for p in paths:
            f.write(" -> ".join(str(v + 1) for v in p) + "\n")
        f.write(f"Всього: {len(paths)}\n")


def main() -> None:
    print("----- Пошук гамільтонових шляхів і циклів (метод backtracking) -----\n")

    n, edges, directed, mode = read_input()
    graph = create_graph(n, edges, directed)

    draw_graph(n, edges, directed)

    #  ШЛЯХИ 
    if mode in ("paths", "both"):
        start_time = time.time()
        paths = find_hamiltonian_paths(graph)
        elapsed = time.time() - start_time

        print("\n--- Гамільтонові шляхи ---")
        if not paths:
            print("Немає жодного гамільтонового шляху.")
        else:
            for p in paths:
                print(" -> ".join(str(v + 1) for v in p))
            print("Всього шляхів:", len(paths))
            elapsed_ms = elapsed * 1000
            print(f"Час пошуку шляхів: {elapsed_ms:.3f} мс")

            save_paths("paths.txt", paths, "Гамільтонові шляхи")
            print("Шляхи збережено у файл paths.txt")

            # показуємо перший знайдений шлях
            draw_path_on_graph(n, edges, paths[0], directed, "Перший знайдений гамільтонів шлях")

    # 2. ЦИКЛИ
    if mode in ("cycles", "both"):
        start_time = time.time()
        cycles = find_hamiltonian_cycles(graph, directed)
        elapsed = time.time() - start_time

        print("\n--- Гамільтонові цикли ---")
        if not cycles:
            print("Немає жодного гамільтонового циклу.")
        else:
            for c in cycles:
                print(" -> ".join(str(v + 1) for v in c))
            print("Всього циклів:", len(cycles))
            elapsed_ms = elapsed * 1000
            print(f"Час пошуку циклів: {elapsed_ms:.3f} мс")

            save_paths("cycles.txt", cycles, "Гамільтонові цикли")
            print("Цикли збережено у файл cycles.txt")

            # показуємо перший знайдений цикл
            draw_path_on_graph(n, edges, cycles[0], directed, "Перший знайдений гамільтонів цикл")

# MAIN
if __name__ == "__main__":
    main()
