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
    mejor_costo, es_viable = evaluar_ruta_completa(mejor_ruta, problema)

    if not es_viable:
        print("Error: La ruta inicial del greedy no es viable.")
        return ruta # Devuelve la ruta original sin tocar
    
    while mejorado:
        mejorado = False
        for i in range(1, len(mejor_ruta) - 2):
            for j in range(i + 1, len(mejor_ruta) - 1):
                if j - i == 1: # aristas adyacentes, no tiene sentido
                    continue
                ruta_propuesta = mejor_ruta.copy()
                ruta_propuesta[i:j+1] = reversed(ruta_propuesta[i:j+1])
                
                nuevo_costo, es_viable = evaluar_ruta_completa(ruta_propuesta, problema)
                
                if es_viable and nuevo_costo < mejor_costo:
                    mejor_ruta = ruta_propuesta
                    mejor_costo = nuevo_costo
                    mejorado = True
    return mejor_ruta

def evaluar_ruta_completa(ruta: List[int], problema:Problema):
    """
    Evalúa una ruta completa simulando un camión.
    Devuelve (costo_total_opcion_B, es_viable)
    """
    capacidad_max = problema.capacidad_camion
    carga_actual = capacidad_max
    distancia_total = 0.0
    costo_hubs = 0.0
    hubs_activados = set()
    
    dict_hubs = {h.id_nodo: h.costo_activacion for h in problema.hubs}

    # Extraer el conjunto de TODOS los nodos de destino de los paquetes
    # Nota: Esto es una simplificación. Asumimos que CUALQUIER visita a un nodo
    # de destino entrega UN paquete, y que la ruta visita cada nodo
    # la cantidad de veces necesaria.

    # UNA MEJOR APROXIMACIÓN: Contar cuántos paquetes van a cada destino
    paquetes_por_entregar = {}
    for p in problema.paquetes:
        dest = p.id_nodo_destino
        paquetes_por_entregar[dest] = paquetes_por_entregar.get(dest, 0) + 1
        
    # El primer nodo debe ser el depósito
    if not ruta or ruta[0] != problema.deposito_id:
        return float('inf'), False

    for i in range(len(ruta) - 1):
        nodo_origen = ruta[i]
        nodo_destino = ruta[i+1]
        
        dist_viaje = problema.grafo_distancias[nodo_origen][nodo_destino]
        if dist_viaje == float('inf'):
            return float('inf'), False # Ruta imposible
        
        distancia_total += dist_viaje
        
        # --- Lógica de Carga y Entrega ---
        es_recarga = (nodo_destino == problema.deposito_id or nodo_destino in dict_hubs)
        es_entrega = (nodo_destino in paquetes_por_entregar and paquetes_por_entregar[nodo_destino] > 0)

        if es_recarga:
            carga_actual = capacidad_max
            # Activar hub si es necesario
            if nodo_destino in dict_hubs and nodo_destino not in hubs_activados:
                costo_hubs += dict_hubs[nodo_destino]
                hubs_activados.add(nodo_destino)
                
        elif es_entrega:
            if carga_actual == 0:
                return float('inf'), False # Inválido: Intento de entrega sin carga
            
            carga_actual -= 1
            paquetes_por_entregar[nodo_destino] -= 1
            # Si ya se entregaron todos, se saca del dict
            if paquetes_por_entregar[nodo_destino] == 0:
                del paquetes_por_entregar[nodo_destino]
        
        # Si no es recarga ni entrega (ej. nodo de paso), la carga no cambia.

    # --- Verificación Final ---
    # ¿Se entregaron todos los paquetes?
    if paquetes_por_entregar:
        return float('inf'), False # Inválido: No se entregaron todos los paquetes
    
    costo_total = distancia_total + costo_hubs
    return costo_total, True



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
    (mejor_costo, es_viable, 
     mejor_dist, mejor_hubs, 
     mejor_hubs_set) = evaluar_ruta_completa(mejor_ruta, problema)
    
    if not es_viable:
        print("ADVERTENCIA: La ruta inicial para Tabu Search no es viable. Deteniendo.")
        # Devuelve datos vacíos para que el main los reporte
        return Solucion(costo_total=float('inf'))

    ruta_actual = mejor_ruta.copy()
    lista_tabu = []

    for _ in range(iteraciones):
        # 1. Generar vecinos (usa tu función existente)
        vecinos_rutas = generar_vecinos(ruta_actual) 
        
        # 2. Filtrar por lista tabú
        vecinos_no_tabu = [v for v in vecinos_rutas if v not in lista_tabu]

        if not vecinos_no_tabu:
            continue # No hay movimientos no-tabú, saltar iteración

        # 3. Evaluar a todos los vecinos
        vecinos_evaluados = []
        for ruta_vecina in vecinos_no_tabu:
            (costo_vecino, es_viable_vecino, 
             dist_vecino, hubs_costo_vecino, 
             hubs_set_vecino) = evaluar_ruta_completa(ruta_vecina, problema)
            
            if es_viable_vecino:
                # Guardamos todos los datos relevantes
                vecinos_evaluados.append((
                    costo_vecino, ruta_vecina, dist_vecino, 
                    hubs_costo_vecino, hubs_set_vecino
                ))
        
        if not vecinos_evaluados:
            continue # Ningún vecino no-tabú fue viable

        # 4. Encontrar el mejor vecino viable (Criterio de Aspiración Básico)
        # El "mejor" es el de menor costo, INCLUSO si es peor que ruta_actual
        # (esto permite a Tabu Search escapar de mínimos locales)
        (mejor_vecino_costo, mejor_vecino_ruta, 
         mejor_vecino_dist, mejor_vecino_hubs, 
         mejor_vecino_hubs_set) = min(vecinos_evaluados, key=lambda item: item[0])
        
        # 5. Actualizar la MEJOR SOLUCIÓN GLOBAL (si aplica)
        # Comparamos el costo de este vecino con el mejor costo global encontrado
        if mejor_vecino_costo < mejor_costo:
            mejor_ruta = mejor_vecino_ruta
            mejor_costo = mejor_vecino_costo
            mejor_dist = mejor_vecino_dist
            mejor_hubs = mejor_vecino_hubs
            mejor_hubs_set = mejor_vecino_hubs_set

        # 6. Moverse al mejor vecino (ESTO ES TABU SEARCH)
        ruta_actual = mejor_vecino_ruta
        
        # 7. Actualizar lista Tabú
        lista_tabu.append(ruta_actual) # Añadir la ruta a la que nos movimos
        if len(lista_tabu) > tamaño_tabu:
            lista_tabu.pop(0) # Mantener el tamaño

    # Al final del bucle, devolvemos la mejor solución encontrada
    return Solucion(
        ruta=mejor_ruta,
        hubs_activados=mejor_hubs_set,
        costo_total=mejor_costo,
        distancia_recorrida=mejor_dist,
        costo_hubs=mejor_hubs
    )

def verificar_inicio_fin(ruta, deposito_id):
    return ruta[0] == deposito_id and ruta[-1] == deposito_id

def verificar_repeticiones(ruta, hubs_permitidos, deposito_id):
    vistos = set()
    for nodo in ruta:
        if nodo in vistos and nodo not in hubs_permitidos and nodo != deposito_id:
            return False
        vistos.add(nodo)
    return True

def verificar_capacidad(ruta, problema):
    capacidad = problema.capacidad_camion
    carga = capacidad
    for nodo in ruta:
        # si es entrega
        if nodo in [p.id_nodo_destino for p in problema.paquetes]:
            carga -= 1
        # si es hub o depósito -> recarga completa
        if nodo == problema.deposito_id or nodo in [h.id_nodo for h in problema.hubs]:
            carga = capacidad
        if carga < 0:
            return False
    return True

def verificar_costo(ruta, problema, costo_reportado):
    costo_calculado = 0
    for i in range(len(ruta) - 1):
        costo_calculado += problema.grafo_distancias[ruta[i]][ruta[i+1]]
    return abs(costo_calculado - costo_reportado) < 1e-3



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

    print("Generando solución inicial con vecino más cercano...")
    ruta_inicial = nearest_neighbor_greedy(problema)
    print(f"Ruta inicial generada: {ruta_inicial}")

    print("Mejorando ruta con 2-opt...")
    ruta_mejorada = dos_opt(ruta_inicial, problema)
    print(f"Ruta mejorada: {ruta_mejorada}")

    print("Aplicando búsqueda tabú...")
    solucion_final = tabu_search(ruta_mejorada, problema, iteraciones=200, tamaño_tabu=20)

    fin = time.time()

    tiempo = fin - inicio

    # --- Impresión de Resultados ---
    print("\n================ RESULTADO FINAL ================")
    if solucion_final.costo_total == float('inf'):
        print("NO SE ENCONTRÓ SOLUCIÓN VIABLE.")
    else:
        print(f"Ruta final (resumida): {solucion_final.ruta[:10]}... (Total pasos: {len(solucion_final.ruta)})")
        print("------------------------------------------------")
        print(f"Costo total: {solucion_final.costo_total:.2f}")
        print(f"Distancia total recorrida: {solucion_final.distancia_recorrida:.2f}")
        print(f"Costo activacion de hubs: {solucion_final.costo_hubs:.2f}")
        print(f"Hubs activados: {solucion_final.hubs_activados or 'Ninguno'}")
        print(f"Tiempo de ejecucion: {tiempo:.5f} segundos")
    print("================================================\n")

    # (Aquí iría tu código para escribir solucion.txt)

if __name__ == "__main__":
    main()
