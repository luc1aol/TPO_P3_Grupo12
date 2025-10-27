inf = float('inf')
grafo_distancias = [[inf for _ in range(12)] for _ in range(12)]

for i in range(12):
        grafo_distancias[i][i] = 0.0

for p in grafo_distancias:
    print(p)
