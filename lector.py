#!/usr/bin/env python3

import sys
from dataclasses import dataclass, field
import time
from typing import List, Optional, Dict, Set
import copy
import random

# ===========================================================
# CLASES
# ===========================================================
@dataclass
class Nodo:
    id: int
    x: int
    y: int


@dataclass
class Hub:
    id_nodo: int
    costo_activacion: float


@dataclass
class Paquete:
    id: int
    id_nodo_origen: int
    id_nodo_destino: int

class Problema:
    def __init__(self):
        self.num_nodos: int = 0
        self.num_hubs: int = 0
        self.num_paquetes: int = 0
        self.capacidad_camion: int = 0
        self.deposito_id: int = 0
        
        self.nodos: List[Nodo] = []
        self.hubs: List[Hub] = []
        self.paquetes: List[Paquete] = []
        self.grafo_distancias: List[List[float]] = []
        
# ===========================================================
# LECTURA
# ===========================================================

def eliminar_comentario(linea: str) -> str:
    """Elimina comentarios de una línea."""
    if "//" in linea:
        return linea[:linea.index("//")].strip()
    return linea.strip()

def leer_archivo(nombre_archivo: str) -> Optional[Problema]:
    """Lee un archivo de problema y retorna un objeto Problema."""
    try:
        with open(nombre_archivo, 'r') as f:
            lineas = f.readlines()
    except FileNotFoundError:
        print(f"Error: No se pudo abrir el archivo '{nombre_archivo}'")
        return None

    p = Problema()

    # --- LEER CONFIGURACIÓN (primeras líneas) ---
    idx = 0
    while idx < len(lineas):
        linea = eliminar_comentario(lineas[idx])
        
        if not linea:
            idx += 1
            continue
        
        if linea.startswith("NODOS"):
            p.num_nodos = int(linea.split()[1])
        elif linea.startswith("HUBS"):
            p.num_hubs = int(linea.split()[1])
        elif linea.startswith("PAQUETES"):
            p.num_paquetes = int(linea.split()[1])
        elif linea.startswith("CAPACIDAD_CAMION"):
            p.capacidad_camion = int(linea.split()[1])
        elif linea.startswith("DEPOSITO_ID"):
            p.deposito_id = int(linea.split()[1])
            idx += 1
            break  # <-- BREAK aquí después de encontrar DEPOSITO_ID
        
        idx += 1

    # Inicializar matriz de distancias
    inf = float('inf')
    p.grafo_distancias = [[inf for _ in range(p.num_nodos)] for _ in range(p.num_nodos)]

    for i in range(p.num_nodos):
        p.grafo_distancias[i][i] = 0.0

    # --- ENCONTRAR Y LEER CADA SECCIÓN ---
    
    # Buscar encabezado de NODOS
    start_nodos = -1
    for i in range(idx, len(lineas)):
        if "NODOS" in lineas[i] and "---" in lineas[i]:
            start_nodos = i + 1
            break
    
    # Leer NODOS
    if start_nodos > 0:
        nodos_leidos = 0
        for i in range(start_nodos, len(lineas)):
            if nodos_leidos >= p.num_nodos:
                break
            linea = eliminar_comentario(lineas[i])
            if not linea:
                continue
            try:
                partes = linea.split()
                nodo = Nodo(id=int(partes[0]), x=int(partes[1]), y=int(partes[2]))
                p.nodos.append(nodo)
                nodos_leidos += 1
            except (ValueError, IndexError):
                pass

    # Buscar encabezado de HUBS
    start_hubs = -1
    for i in range(len(lineas)):
        if "HUBS" in lineas[i] and "---" in lineas[i]:
            start_hubs = i + 1
            break
    
    # Leer HUBS
    if start_hubs > 0:
        hubs_leidos = 0
        for i in range(start_hubs, len(lineas)):
            if hubs_leidos >= p.num_hubs:
                break
            linea = eliminar_comentario(lineas[i])
            if not linea:
                continue
            try:
                partes = linea.split()
                hub = Hub(id_nodo=int(partes[0]), costo_activacion=float(partes[1]))
                p.hubs.append(hub)
                hubs_leidos += 1
            except (ValueError, IndexError):
                pass

    # Buscar encabezado de PAQUETES
    start_paquetes = -1
    for i in range(len(lineas)):
        if "PAQUETES" in lineas[i] and "---" in lineas[i]:
            start_paquetes = i + 1
            break
    
    # Leer PAQUETES
    if start_paquetes > 0:
        paquetes_leidos = 0
        for i in range(start_paquetes, len(lineas)):
            if paquetes_leidos >= p.num_paquetes:
                break
            linea = eliminar_comentario(lineas[i])
            if not linea:
                continue
            try:
                partes = linea.split()
                paquete = Paquete(id=int(partes[0]), 
                                 id_nodo_origen=int(partes[1]), 
                                 id_nodo_destino=int(partes[2]))
                p.paquetes.append(paquete)
                paquetes_leidos += 1
            except (ValueError, IndexError):
                pass

    # Buscar encabezado de ARISTAS
    start_aristas = -1
    for i in range(len(lineas)):
        if "ARISTAS" in lineas[i] and "---" in lineas[i]:
            start_aristas = i + 1
            break
    
    # Leer ARISTAS
    if start_aristas > 0:
        for i in range(start_aristas, len(lineas)):
            linea = eliminar_comentario(lineas[i])
            if not linea:
                continue
            try:
                partes = linea.split()
                if len(partes) >= 3:
                    u, v, peso = int(partes[0]), int(partes[1]), float(partes[2])
                    if u < p.num_nodos and v < p.num_nodos:
                        p.grafo_distancias[u][v] = peso
                        p.grafo_distancias[v][u] = peso
            except (ValueError, IndexError):
                pass

    return p

def imprimir_problema(p: Problema) -> None:
    """Imprime un resumen del problema cargado."""
    print("\n============== RESUMEN DEL PROBLEMA CARGADO ===============")
    
    print("\n--- CONFIGURACION ---")
    print(f"Total de Nodos:\t\t{p.num_nodos}")
    print(f"Total de Hubs:\t\t{p.num_hubs}")
    print(f"Total de Paquetes:\t{p.num_paquetes}")
    print(f"Capacidad del Camión:\t{p.capacidad_camion}")
    print(f"ID del Depósito:\t\t{p.deposito_id}")
    
    print("\n--- NODOS ---")
    for nodo in p.nodos:
        print(f"  Nodo {nodo.id:2d}: (x={nodo.x:4d}, y={nodo.y:4d})")
    
    print("\n--- HUBS ---")
    for hub in p.hubs:
        print(f"  Hub en Nodo {hub.id_nodo:2d}: Costo de Activación = {hub.costo_activacion:.2f}")
    
    print("\n--- PAQUETES ---")
    for paquete in p.paquetes:
        print(f"  Paquete {paquete.id:2d}: Origen={paquete.id_nodo_origen} -> Destino={paquete.id_nodo_destino}")
    
    print("\n--- MUESTRA DEL GRAFO (MATRIZ DE ADYACENCIA) ---")
    imprimir_matriz(p)

def imprimir_matriz(p:Problema):    
    print("      ", end="")
    for j in range(min(10, p.num_nodos)):
        print(f"{j:7d} ", end="")
    print()
    
    print("----", end="")
    for j in range(min(10, p.num_nodos)):
        print("--------", end="")
    print()
    
    for i in range(min(10, p.num_nodos)):
        print(f"{i:4d}| ", end="")
        for j in range(min(10, p.num_nodos)):
            print(f"{p.grafo_distancias[i][j]:7.2f} ", end="")
        print()
    
    print("===========================================================\n")

# ===========================================================
# RESOLUCION
# ===========================================================

# Para n<=100
def floyd_warshall(p: Problema):
    """
    Calcular todos los pares de caminos mínimos.
    """

    dist = p.grafo_distancias
    
    # intermedio
    for k in range(p.num_nodos):
        # origen
        for i in range(p.num_nodos):
            # destino
            for j in range(p.num_nodos):
                
                costo_via_k = dist[i][k] + dist[k][j]
                
                if dist[i][j] > costo_via_k:
                    dist[i][j] = costo_via_k

# Para n>100 se debería usar Dijkstra desde cada nodo

@dataclass
class Camion:
    """
    Representa el vehículo y su estado durante la simulación.
    Diseñada para ser fácilmente copiable y compatible con backtracking o heurísticas.
    """
    capacidad_maxima: int
    deposito_id: int
    dict_hubs: Dict[int, float]  # id_hub -> costo de activación
    
    # --- Estado dinámico ---
    nodo_actual: int = field(init=False)
    carga_actual: int = field(init=False)
    ruta_actual: List[int] = field(default_factory=list)
    paquetes_pendientes: List = field(default_factory=list)
    hubs_activados: Set[int] = field(default_factory=set)
    costo_total: float = 0.0    
    costo_hubs: float = 0.0
    distancia_recorrida: float = 0.0

    def __post_init__(self):
        # Inicializa en el depósito con el camión lleno
        self.nodo_actual = self.deposito_id
        self.carga_actual = self.capacidad_maxima
        self.ruta_actual = [self.deposito_id]

    # ---------------------------------------------------------------
    # Métodos de acción
    # ---------------------------------------------------------------

    def clonar(self):
        """Crea una copia profunda del estado actual (para backtracking o heurísticas)."""
        return copy.deepcopy(self)
    
    def entregar_paquete(self, paquete, problema: Problema):
        """
        Entrega un paquete al destino correspondiente.
        """
        nodo_destino = paquete.id_nodo_destino
        dist = problema.grafo_distancias[self.nodo_actual][nodo_destino]

        # Actualiza estado
        self.distancia_recorrida += dist
        self.costo_total += dist
        self.nodo_actual = nodo_destino
        self.carga_actual -= 1
        self.ruta_actual.append(nodo_destino)
        self.paquetes_pendientes.remove(paquete)
    
    def recargar(self, nodo_destino, problema):
        """
        Recarga el camión en un hub o en el depósito.
        Si el hub no estaba activado, paga su costo de activación.
        """
        dist = problema.grafo_distancias[self.nodo_actual][nodo_destino]
        self.distancia_recorrida += dist
        self.costo_total += dist
        self.nodo_actual = nodo_destino
        self.carga_actual = self.capacidad_maxima
        self.ruta_actual.append(nodo_destino)

        # Si es un hub no activado
        if nodo_destino in self.dict_hubs and nodo_destino not in self.hubs_activados:
            costo_hub = self.dict_hubs[nodo_destino]
            self.costo_total += costo_hub
            self.costo_hubs += costo_hub
            self.hubs_activados.add(nodo_destino)

    # ---------------------------------------------------------------
    # Métodos utilitarios
    # ---------------------------------------------------------------

    def calcular_costo_retorno(self, problema: Problema):
        """Devuelve el costo de volver al depósito desde el nodo actual."""
        return problema.grafo_distancias[self.nodo_actual][self.deposito_id]
    
    def puede_entregar(self):
        """Devuelve True si aún tiene paquetes en carga."""
        return self.carga_actual > 0
    
    def necesita_recargar(self):
        """Devuelve True si la carga está vacía."""
        return self.carga_actual == 0

        if activacion_de_hub:
            self.hubs_activados_actual.remove(nodo_destino_recarga)
            self.costo_total_actual -= costo_activacion
            self.costo_hubs_actual -= costo_activacion
        self.ruta_actual.pop()
        self.carga_actual = carga_anterior
        self.nodo_actual = nodo_anterior
        self.costo_total_actual -= dist_viaje
        self.distancia_recorrida_actual -= dist_viaje
@dataclass
class Solucion:
    """
    Representa una solución completa a un caso de ruteo.
    Es independiente del método usado para generarla.
    """
    ruta: List[int] = field(default_factory=list)
    hubs_activados: Set[int] = field(default_factory=set)
    costo_total: float = float("inf")
    distancia_recorrida: float = 0.0
    costo_hubs: float = 0.0

    def actualizar_desde_camion(self, camion, problema: Problema):
        """
        Actualiza los datos de la solución basándose en el estado final del camión.
        """
        dist_retorno = problema.grafo_distancias[camion.nodo_actual][problema.deposito_id]
        costo_final = camion.costo_total + dist_retorno

        self.ruta = camion.ruta_actual + [problema.deposito_id]
        self.distancia_recorrida = camion.distancia_recorrida + dist_retorno
        self.costo_hubs = camion.costo_hubs
        self.hubs_activados = set(camion.hubs_activados)
        self.costo_total = costo_final

    def es_mejor_que(self, otra_solucion) -> bool:
        """Compara esta solución con otra según el costo total."""
        return self.costo_total < otra_solucion.costo_total
    
 
def nearest_neighbor_greedy(problema: Problema):
    """
    Construye una ruta inicial utilizando el heurístico del vecino más cercano.
    No considera inventario, solo distancia y capacidad.
    Retorna una lista de nodos visitados.
    """
    # Inicialización del camión y paquetes pendientes
    camion = Camion(
        capacidad_maxima=problema.capacidad_camion,
        deposito_id=problema.deposito_id,
        dict_hubs={hub.id_nodo: hub.costo_activacion for hub in problema.hubs}
    )
    camion.paquetes_pendientes = problema.paquetes.copy() # Asignar todos los paquetes al camión

    # Mientras haya paquetes pendientes, buscar el siguiente movimiento óptimo
    while camion.paquetes_pendientes:
        if camion.necesita_recargar():
            # buscar el hub o depósito más cercano
            destinos = [problema.deposito_id] + [hub.id_nodo for hub in problema.hubs]
        else:
            # buscar destino de paquete más cercano
            destinos = [p.id_nodo_destino for p in camion.paquetes_pendientes]

        mejor_nodo = None
        mejor_dist = float('inf')
        
        # encontrar el destino más cercano
        for destino in destinos:
            if destino == camion.nodo_actual:
                continue # evitar quedarse en el mismo nodo
            dist = problema.grafo_distancias[camion.nodo_actual][destino]
            if dist < mejor_dist: # actualizar mejor opción
                mejor_dist = dist
                mejor_nodo = destino

        if mejor_nodo is None: # no hay más movimientos posibles
            break

        # aplicar movimiento
        if mejor_nodo in [p.id_nodo_destino for p in camion.paquetes_pendientes]:
            # es entrega
            paquete = next(p for p in camion.paquetes_pendientes if p.id_nodo_destino == mejor_nodo)
            camion.entregar_paquete(paquete, problema)
        else:
            # es recarga
            camion.recargar(mejor_nodo, problema)
    
    # retornar al depósito
    camion.ruta_actual.append(problema.deposito_id)
    return camion.ruta_actual
# ---------------------------------
def dos_opt(ruta, problema: Problema):
    """
    Aplica la optimización 2-opt para mejorar la ruta dada.
    Cortar dos aristas de la ruta y reconectarlas de otra forma que reduzca el costo total.
    """
    mejorado = True
    mejor_ruta = ruta.copy()
    while mejorado:
        mejorado = False
        for i in range(1, len(mejor_ruta) - 2):
            for j in range(i + 1, len(mejor_ruta) - 1):
                if j - i == 1: # aristas adyacentes, no tiene sentido
                    continue
                # calcular costo antes y después del swap
                # Aristas: (a-b) y (c-d) se convierten en (a-c) y (b-d)
                a, b = mejor_ruta[i-1], mejor_ruta[i]
                c, d = mejor_ruta[j], mejor_ruta[j+1]
                dist_antes = problema.grafo_distancias[a][b] + problema.grafo_distancias[c][d]
                dist_despues = problema.grafo_distancias[a][c] + problema.grafo_distancias[b][d]
                if dist_despues < dist_antes:
                    mejor_ruta[i:j+1] = reversed(mejor_ruta[i:j+1])
                    mejorado = True
    return mejor_ruta

def evaluar_ruta(ruta, problema:Problema):
    """
    Calcula el costo total de una ruta.
    Suma las distancias entre nodos consecutivos.
    """
    costo_total = 0.0
    for i in range(len(ruta) - 1):
        costo_total += problema.grafo_distancias[ruta[i]][ruta[i + 1]]
    return costo_total

def generar_vecinos(ruta, cantidad_vecinos=20):
    """
    Genera rutas vecinas intercambiando dos nodos al azar.
    """
    vecinos = []
    for _ in range(cantidad_vecinos):
        i, j = sorted(random.sample(range(1, len(ruta) - 1), 2))
        nueva = ruta.copy()
        nueva[i], nueva[j] = nueva[j], nueva[i]
        vecinos.append(nueva)
    return vecinos

def tabu_search(ruta_inicial, problema: Problema, iteraciones=200, tamaño_tabu=20):
    """
    Aplica la metaheurística Tabu Search a una ruta.
    """
    mejor_ruta = ruta_inicial.copy()
    mejor_costo = evaluar_ruta(mejor_ruta, problema)
    ruta_actual = mejor_ruta.copy()
    lista_tabu = []

    for _ in range(iteraciones):
        vecinos = generar_vecinos(ruta_actual)
        vecinos_validos = [v for v in vecinos if v not in lista_tabu]

        if not vecinos_validos:
            continue

        # Elegir el vecino con menor costo
        mejor_vecino = min(vecinos_validos, key=lambda r: evaluar_ruta(r, problema))
        costo_vecino = evaluar_ruta(mejor_vecino, problema)

        # Si mejora, actualizamos
        if costo_vecino < mejor_costo:
            mejor_ruta = mejor_vecino.copy()
            mejor_costo = costo_vecino

        # Actualizar lista Tabú
        lista_tabu.append(mejor_vecino)
        if len(lista_tabu) > tamaño_tabu:
            lista_tabu.pop(0)

        # Mover a ese vecino
        ruta_actual = mejor_vecino.copy()

    return mejor_ruta

# ===========================================================
# MAIN
# ===========================================================

def main():
    if len(sys.argv) != 2:
        print(f"Uso: {sys.argv[0]} <nombre_del_archivo.txt>")
        sys.exit(1)

    nombre_archivo = sys.argv[1]
    print(f"Leyendo el archivo de problema: {nombre_archivo}")

    problema = leer_archivo(nombre_archivo)
    if problema is None:
        print("\n>> Hubo un error al leer o procesar el archivo. Revisa el formato.")
        sys.exit(1)

    print("\n¡Archivo leído y procesado con éxito!")
    imprimir_problema(problema)

    inicio = time.time()

    print("Iniciando Floyd-Warshall...")
    floyd_warshall(problema)

    print("--- MUESTRA DEL GRAFO (MATRIZ DE CAMINOS MINIMOS) ---")
    imprimir_matriz(problema)

    # estado_inicial_greedy = Camion(problema)
    # camion = Camion(problema)
    # mejor_solucion = encontrar_solucion_greedy(estado_inicial_greedy, problema)

    # print("Iniciando backtracking...")
    # resolver_backtracking(camion, mejor_solucion, problema)

    print("Generando solución inicial con vecino más cercano...")
    ruta_inicial = nearest_neighbor_greedy(problema)
    print(f"Ruta inicial generada: {ruta_inicial}")

    print("Mejorando ruta con 2-opt...")
    ruta_mejorada = dos_opt(ruta_inicial, problema)
    print(f"Ruta mejorada: {ruta_mejorada}")

    print("Aplicando búsqueda tabú...")
    ruta_final = tabu_search(ruta_mejorada, problema)
    print(f"Ruta final después de búsqueda tabú: {ruta_final}")
    
    # Construir la solución final
    solucion = Solucion(
        ruta=ruta_final,
        costo_total=evaluar_ruta(ruta_final, problema),
    )
    
    fin = time.time()

    tiempo = fin - inicio

    print("\n================ RESULTADO FINAL ================")
    print(f"Ruta inicial (Greedy):      {ruta_inicial}")
    print(f"Ruta tras 2-opt:           {ruta_mejorada}")
    print(f"Ruta final (Tabu Search):  {ruta_final}")
    print("------------------------------------------------")
    print(f"Distancia total: {evaluar_ruta(ruta_final, problema):.2f}")
    print("================================================\n")

    print(f"Tiempo de ejecucion: {tiempo:.5f} segundos")
    print(f"Mejor ruta: {solucion.ruta}")
    print(f"Costo total: {solucion.costo_total:.5f}")
    print(f"Costo activacion de hubs: {solucion.costo_hubs:.5f}")
    print(f"Distancia total recorrida: {solucion.distancia_recorrida:.5f}")
    print(f"Hubs activados: {solucion.hubs_activados}")

if __name__ == "__main__":
    main()
