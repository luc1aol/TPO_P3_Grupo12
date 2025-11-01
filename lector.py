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

@dataclass
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
    
#IMPLEMENTAR BACKTRACKING PARA n<50

def probar_configuracion_hubs(problema: Problema, 
                              hubs_a_probar: Set[int], 
                              desc: str) -> Solucion:
    """
    Función helper: Genera una ruta greedy para un set de hubs 
    y la optimiza con 2-Opt.
    Esta función es usada tanto por el solver heurístico como por el exhaustivo.
    """
    print(f"  Probando config: '{desc}' (Hubs: {hubs_a_probar or 'Ninguno'})...")
    
    # 1. Generar ruta inicial con el Greedy (Tarea 2.2)
    solucion_greedy = generar_ruta_greedy(problema, hubs_a_probar)
    
    if solucion_greedy.costo_total == float('inf'):
        print("    -> Greedy no encontró ruta viable.")
        return solucion_greedy # Devolver solución inválida
        
    print(f"    -> Costo Greedy: {solucion_greedy.costo_total:.2f}")
    
    # 2. Pulir ruta con 2-Opt (Tarea 1.2)
    solucion_optimizada = dos_opt(solucion_greedy.ruta, problema)
    print(f"    -> Costo 2-Opt: {solucion_optimizada.costo_total:.2f}")
    
    return solucion_optimizada


def generar_combinaciones_hubs(hubs_disponibles: List[int]) -> List[Set[int]]:
    """
    Genera todas las 2^M combinaciones de hubs usando recursión (backtracking).
    Cumple con el requisito de usar backtracking.
    Devuelve una lista de sets, cada set es una combinación de hubs.
    """
    todas_las_combinaciones = []
    
    def backtrack_hubs(index: int, combinacion_actual: Set[int]):
        # Caso base: hemos considerado todos los hubs
        if index == len(hubs_disponibles):
            todas_las_combinaciones.append(combinacion_actual.copy())
            return
        
        # --- Opción 1: NO incluir el hub[index] en esta combinación ---
        backtrack_hubs(index + 1, combinacion_actual)
        
        # --- Opción 2: INCLUIR el hub[index] en esta combinación ---
        hub_id = hubs_disponibles[index]
        combinacion_actual.add(hub_id)
        
        # Llamada recursiva
        backtrack_hubs(index + 1, combinacion_actual)
        
        # Deshacer (backtrack) para la siguiente iteración
        combinacion_actual.remove(hub_id) 
    
    backtrack_hubs(0, set())
    return todas_las_combinaciones


def solver_exhaustivo(problema: Problema) -> Solucion:
    """
    Ejecuta el solver en modo EXHAUSTIVO, probando TODAS las
    combinaciones de hubs. Cumple el requisito de backtracking del TPO.
    Ideal para casos pequeños (n < 50).
    """
    print(f"Iniciando solver exhaustivo (Backtracking de {problema.num_hubs} hubs)...")
    
    mejor_solucion_global = Solucion(costo_total=float('inf'))
    
    # 1. Obtener la lista de IDs de hubs disponibles
    hubs_disponibles = [hub.id_nodo for hub in problema.hubs]
    
    # 2. Tarea 3.1: Generar todas las 2^M combinaciones
    todas_las_combinaciones = generar_combinaciones_hubs(hubs_disponibles)
    print(f"Se probarán {len(todas_las_combinaciones)} combinaciones de hubs.")

    # 3. Iterar sobre CADA combinación (la "fuerza bruta")
    for i, hubs_a_probar in enumerate(todas_las_combinaciones):
        
        # 4. Probar y pulir la ruta para esta combinación
        desc = f"Combinación {i+1}/{len(todas_las_combinaciones)}"
        solucion_actual = probar_configuracion_hubs(problema, hubs_a_probar, desc)
        
        # 5. Actualizar el "campeón"
        if solucion_actual.costo_total < mejor_solucion_global.costo_total:
            mejor_solucion_global = solucion_actual
            print(f"  *** NUEVA MEJOR SOLUCIÓN GLOBAL: {mejor_solucion_global.costo_total:.2f} ***")

    print(f"Solver exhaustivo finalizado. Mejor costo: {mejor_solucion_global.costo_total:.2f}")
    return mejor_solucion_global

#---------------------------------
#HEURISTICAS PARA n>=50 
#---------------------------------
def seleccionar_hubs_heuristico(problema: Problema) -> Set[int]:
    """
    Calcula un puntaje para cada hub basado en el ahorro de distancia
    potencial vs. su costo de activación.
    Devuelve un set con los IDs de los hubs que vale la pena activar.
    Esto es para no probar todas las 2^M combinaciones de hubs.
    Puntaje = (Ahorro total de distancia) - (Costo de activación)
    """
    hubs_seleccionados = set()
    puntajes_hubs = []
    
    # Datos que usaremos repetidamente
    matriz_dist = problema.grafo_distancias
    depot_id = problema.deposito_id
    
    # 1. Calcular el puntaje para cada hub
    for hub in problema.hubs:
        ahorro_total = 0.0
        
        # 2. Iterar sobre CADA paquete para ver cuánto ahorra este hub
        for paquete in problema.paquetes:
            dest_id = paquete.id_nodo_destino
            
            # Costo de la ruta directa desde el depósito
            dist_directa = matriz_dist[depot_id][dest_id]
            
            # Costo de la ruta pasando por este hub
            dist_deposito_a_hub = matriz_dist[depot_id][hub.id_nodo]
            dist_hub_a_destino = matriz_dist[hub.id_nodo][dest_id]
            
            dist_via_hub = dist_deposito_a_hub + dist_hub_a_destino
            
            # Calculamos el ahorro (solo si es positivo)
            ahorro = dist_directa - dist_via_hub
            if ahorro > 0:
                ahorro_total += ahorro
        
        # 3. Puntaje final = Ahorro total menos el costo de activarlo
        puntaje = ahorro_total - hub.costo_activacion
        puntajes_hubs.append((puntaje, hub.id_nodo)) # (puntaje, id)
        
    # 4. Ordenar los hubs por su puntaje, de mayor a menor
    puntajes_hubs.sort(key=lambda x: x[0], reverse=True)
    
    # 5. Activar todos los hubs que tengan puntaje positivo
    for puntaje, hub_id in puntajes_hubs:
        if puntaje > 0:
            hubs_seleccionados.add(hub_id)
        else:
            # Como está ordenado, si este es <= 0, los demás también
            break 
            
    return hubs_seleccionados

def find_nearest_reload_point(from_node: int, 
                              active_hubs: Set[int], 
                              problema: Problema) -> int:
    """
    Encuentra el punto de recarga (depósito o hub activo) más cercano
    al 'from_node'.
    """
    best_point = problema.deposito_id
    best_dist = problema.grafo_distancias[from_node][problema.deposito_id]
    
    for hub_id in active_hubs:
        dist = problema.grafo_distancias[from_node][hub_id]
        if dist < best_dist:
            best_dist = dist
            best_point = hub_id
            
    return best_point

def generar_ruta_greedy(problema: Problema, active_hubs: Set[int]) -> Solucion:
    """
    Construye una ruta inicial viable usando una heurística greedy (vecino más cercano)
    que fuerza las recargas cuando la capacidad llega a 0.
    
    (Versión CORREGIDA basada en la recomendación del análisis)
    """
    # 1. Configuración inicial
    matriz_dist = problema.grafo_distancias
    depot_id = problema.deposito_id
    capacidad = problema.capacidad_camion
    dict_hubs = {h.id_nodo: h.costo_activacion for h in problema.hubs}
    
    # Paquetes que faltan por entregar (copia de trabajo)
    packages_remaining = {}
    for pkg in problema.paquetes:
        packages_remaining[pkg.id_nodo_destino] = packages_remaining.get(pkg.id_nodo_destino, 0) + 1

    # Estado de la ruta
    path = [depot_id]
    total_distance = 0.0
    current_node = depot_id
    current_load = capacidad # Empezamos llenos
    
    hubs_activados_en_ruta = set()
    total_hub_cost = 0.0

    # 2. Bucle principal: continuar mientras queden paquetes
    while packages_remaining:
        
        # 3. Lógica de decisión: ¿Recargar o Entregar?
        
        if current_load == 0:
            # --- FORZAR RECARGA ---
            # No tenemos carga, debemos ir al punto de recarga más cercano
            
            nearest_reload = find_nearest_reload_point(current_node, active_hubs, problema)
            dist_viaje = matriz_dist[current_node][nearest_reload]
            
            if dist_viaje == float('inf'):
                 return Solucion(costo_total=float('inf')) # Error: no puede recargar

            total_distance += dist_viaje
            current_node = nearest_reload
            path.append(current_node)
            current_load = capacidad # Recargado
            
            # Activar el hub si es nuevo
            if current_node in dict_hubs and current_node not in hubs_activados_en_ruta:
                hubs_activados_en_ruta.add(current_node)
                total_hub_cost += dict_hubs[current_node]

        else:
            # --- ENTREGAR PAQUETE ---
            # Tenemos carga, buscamos el paquete pendiente más cercano
            
            next_dest = -1
            next_dist = float('inf')
            
            for dest in packages_remaining.keys():
                dist = matriz_dist[current_node][dest]
                if dist < next_dist:
                    next_dist = dist
                    next_dest = dest
            
            if next_dest == -1:
                # No quedan paquetes, pero el bucle while debería haber terminado
                # Esto puede pasar si un nodo es inalcanzable (dist=inf)
                break 

            # Viajar y entregar 1 paquete
            total_distance += next_dist
            current_node = next_dest
            path.append(current_node)
            
            current_load -= 1
            packages_remaining[next_dest] -= 1
            if packages_remaining[next_dest] == 0:
                del packages_remaining[next_dest]

    # 4. Volver al depósito al finalizar
    if current_node != depot_id:
        total_distance += matriz_dist[current_node][depot_id]
        path.append(depot_id)
        
    # 5. Crear el objeto Solucion
    total_cost_final = total_distance + total_hub_cost
    
    # Verificación final
    if packages_remaining:
        return Solucion(costo_total=float('inf')) # El greedy falló

    return Solucion(
        ruta=path,
        hubs_activados=hubs_activados_en_ruta,
        costo_total=total_cost_final,
        distancia_recorrida=total_distance,
        costo_hubs=total_hub_cost
    )
# ---------------------------------

def solver_heuristico(problema: Problema) -> Solucion:
    """
    Ejecuta el solver en modo HEURÍSTICO, probando un conjunto
    limitado pero "inteligente" de configuraciones de hubs.
    Ideal para casos medianos y grandes.
    Estrategia:
    1) Sin hubs (línea base)
    2) Hubs heurísticos (suposición inteligente)
    3) Búsqueda local ("flipping" de hubs)
    Cada configuración genera una ruta greedy que luego se pule con 2-Opt.
    """
    print("Iniciando solver heurístico...")
    # La mejor solución encontrada hasta ahora
    mejor_solucion_global = Solucion(costo_total=float('inf'))

    # 1. Experimento 1: Sin hubs (Línea Base)
    hubs_vacios = set()
    sol_sin_hubs = probar_configuracion_hubs(problema, hubs_vacios, "Sin Hubs")
    if sol_sin_hubs.costo_total < mejor_solucion_global.costo_total:
        mejor_solucion_global = sol_sin_hubs

    # 2. Experimento 2: Hubs Heurísticos (Suposición Inteligente)
    #
    hubs_inteligentes = seleccionar_hubs_heuristico(problema)
    
    # Solo ejecutar si la lista no es la misma que ya probamos (vacía)
    if hubs_inteligentes != hubs_vacios:
        sol_hubs_intel = probar_configuracion_hubs(problema, hubs_inteligentes, "Hubs Heurísticos")
        if sol_hubs_intel.costo_total < mejor_solucion_global.costo_total:
            mejor_solucion_global = sol_hubs_intel

    # 3. Experimento 3: Búsqueda Local ("Flipping")
    #
    print(f"  Experimento: Probando 'Búsqueda Local' (flipping {len(problema.hubs)} hubs)...")
    for hub in problema.hubs:
        hubs_a_probar = hubs_inteligentes.copy()
        
        if hub.id_nodo in hubs_a_probar:
            hubs_a_probar.remove(hub.id_nodo)
            desc = f"Flipping (Quitar Hub {hub.id_nodo})"
        else:
            hubs_a_probar.add(hub.id_nodo)
            desc = f"Flipping (Añadir Hub {hub.id_nodo})"

        sol_flip = probar_configuracion_hubs(problema, hubs_a_probar, desc)
        if sol_flip.costo_total < mejor_solucion_global.costo_total:
            mejor_solucion_global = sol_flip

    print(f"Solver heurístico finalizado. Mejor costo encontrado: {mejor_solucion_global.costo_total:.2f}")
    return mejor_solucion_global



def dos_opt(ruta_inicial: List[int], problema: Problema) -> Solucion:
    """
    Aplica la optimización 2-opt para mejorar la ruta dada.
    Llama a 'evaluar_ruta_completa' para CADA nueva ruta candidata
    para asegurar que sea viable (Opción B) y que el costo total
    (distancia + hubs) haya mejorado.
    """
    
    # 1. Evaluar la ruta inicial para obtener la solución base
    mejor_solucion = evaluar_ruta_completa(ruta_inicial, problema)
    
    if mejor_solucion.costo_total == float('inf'):
        print("[2-Opt ADVERTENCIA] La ruta Greedy inicial no es viable. Saltando optimización.")
        return mejor_solucion # Devuelve la solución inválida

    # Usamos mejor_ruta_lista para iterar y manipular la lista
    mejor_ruta_lista = ruta_inicial
    
    mejorado = True
    while mejorado:
        mejorado = False
        
        # Iterar de 1 a len-2 para no tocar el depósito de inicio/fin
        for i in range(1, len(mejor_ruta_lista) - 2):
            for j in range(i + 1, len(mejor_ruta_lista) - 1):
                
                # 1. Crear la nueva ruta candidata invirtiendo el segmento [i, j]
                ruta_propuesta_lista = mejor_ruta_lista[:i] + list(reversed(mejor_ruta_lista[i:j+1])) + mejor_ruta_lista[j+1:]
                
                # 2. Evaluar la nueva ruta (viabilidad Y costo total)
                solucion_propuesta = evaluar_ruta_completa(ruta_propuesta_lista, problema)
                
                # 3. Comparar el COSTO TOTAL (distancia + hubs)
                # (Usamos un pequeño épsilon para la comparación de floats)
                if solucion_propuesta.costo_total < (mejor_solucion.costo_total - 1e-5):
                    # ¡Mejora encontrada!
                    mejor_solucion = solucion_propuesta
                    mejor_ruta_lista = ruta_propuesta_lista
                    mejorado = True
                    
                    # Romper los bucles internos para reiniciar la búsqueda
                    # con la nueva mejor ruta
                    break
            if mejorado:
                break
    
    # Devuelve el objeto Solucion completo
    return mejor_solucion

def evaluar_ruta_completa(ruta: List[int], problema:Problema) -> Solucion:
    """
    Evalúa una ruta completa simulando un camión (Opción B).
    Devuelve un objeto Solucion, si la ruta es inválida, el objeto tendrá costo_total=inf
    Si es válida, contendrá todos los datos.
    """
    # 1. Preparar datos y estado inicial
    capacidad_max = problema.capacidad_camion
    carga_actual = capacidad_max
    distancia_total = 0.0
    costo_hubs = 0.0
    hubs_activados = set()
    
    dict_hubs = {h.id_nodo: h.costo_activacion for h in problema.hubs}
    
    paquetes_por_entregar = {}
    for p in problema.paquetes:
        dest = p.id_nodo_destino
        paquetes_por_entregar[dest] = paquetes_por_entregar.get(dest, 0) + 1
    
    # --- 2. Verificaciones iniciales de la ruta ---
    if not ruta or ruta[0] != problema.deposito_id or ruta[-1] != problema.deposito_id:
        return Solucion(costo_total=float('inf'))

    # --- 3. Simulación del viaje ---
    for i in range(len(ruta) - 1):
        nodo_origen = ruta[i]
        nodo_destino = ruta[i+1]
        
        dist_viaje = problema.grafo_distancias[nodo_origen][nodo_destino]
        if dist_viaje == float('inf'):
            return Solucion(costo_total=float('inf'))
        
        distancia_total += dist_viaje
        
        es_recarga = (nodo_destino == problema.deposito_id or nodo_destino in dict_hubs)
        es_entrega = (nodo_destino in paquetes_por_entregar and paquetes_por_entregar[nodo_destino] > 0)

        if es_recarga:
            carga_actual = capacidad_max
            if nodo_destino in dict_hubs and nodo_destino not in hubs_activados:
                costo_hubs += dict_hubs[nodo_destino]
                hubs_activados.add(nodo_destino)
                
        elif es_entrega:
            
            # --- INICIO DE LA CORRECCIÓN ---
            # En lugar de entregar 1 paquete, entregamos TODOS los que podamos.
            
            # Cuántos paquetes quedan por entregar en este destino
            paquetes_pendientes_en_nodo = paquetes_por_entregar[nodo_destino]
            
            # Cuántos podemos entregar (limitado por la carga y los pendientes)
            paquetes_a_entregar = min(carga_actual, paquetes_pendientes_en_nodo)
            
            if paquetes_a_entregar == 0:
                # Si llegamos aquí, significa que la ruta nos trajo
                # a un destino sin carga en el camión.
                return Solucion(costo_total=float('inf')) 
            
            carga_actual -= paquetes_a_entregar
            paquetes_por_entregar[nodo_destino] -= paquetes_a_entregar

            if paquetes_por_entregar[nodo_destino] == 0:
                del paquetes_por_entregar[nodo_destino]
            # --- FIN DE LA CORRECCIÓN ---

    # --- 4. Verificación Final ---
    if paquetes_por_entregar:
        return Solucion(costo_total=float('inf')) 
    
    # --- 5. Éxito: La ruta es válida ---
    costo_total = distancia_total + costo_hubs
    
    return Solucion(
        ruta=ruta,
        hubs_activados=hubs_activados,
        costo_total=costo_total,
        distancia_recorrida=distancia_total,
        costo_hubs=costo_hubs
    )
# ===========================================================
def escribir_solucion_txt(solucion: Solucion, tiempo_ejecucion: float, nombre_archivo="solucion.txt"):
    """
    Genera el archivo solucion.txt con el formato exacto requerido por el TPO.
   
    """
    try:
        with open(nombre_archivo, 'w') as f:
            # --- HUBS ACTIVADOS ---
            f.write("// --- HUBS ACTIVADOS ---\n")
            if not solucion.hubs_activados:
                f.write("Ninguno\n")
            else:
                # Escribir IDs ordenados
                for hub_id in sorted(list(solucion.hubs_activados)):
                    f.write(f"{hub_id}\n")
            
            # --- RUTA OPTIMA ---
            f.write("\n// --- RUTA OPTIMA ---\n")
            if not solucion.ruta:
                f.write("No se encontró ruta válida\n")
            else:
                f.write(" -> ".join(map(str, solucion.ruta)) + "\n")
            
            # --- METRICAS ---
            f.write("\n// --- METRICAS ---\n")
            f.write(f"COSTO_TOTAL: {solucion.costo_total:.2f}\n")
            f.write(f"DISTANCIA_RECORRIDA: {solucion.distancia_recorrida:.2f}\n")
            f.write(f"COSTO_HUBS: {solucion.costo_hubs:.2f}\n")
            f.write(f"TIEMPO_EJECUCION: {tiempo_ejecucion:.6f} segundos\n")
            
    except IOError as e:
        print(f"Error al escribir el archivo de solución: {e}")

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
    # imprimir_problema(problema) # Descomentar para debug

    print("Iniciando Floyd-Warshall...")
    floyd_warshall(problema)

    # print("--- MUESTRA DEL GRAFO (MATRIZ DE CAMINOS MINIMOS) ---")
    # imprimir_matriz(problema) # Descomentar para debug

    # --- INICIO DEL SOLVER HÍBRIDO ---
    inicio_solver = time.time()
    
    # Decidimos qué solver usar basado en el tamaño del problema
    # (n=50 es un buen umbral, n=20 es el caso pequeño)
    if problema.num_nodos < 50:
        print("\n[MODO: EXHAUSTIVO] (n < 50) -> Buscando solución óptima.")
        solucion_final = solver_exhaustivo(problema)
    else:
        print("\n[MODO: HEURÍSTICO] (n >= 50) -> Buscando solución eficiente.")
        solucion_final = solver_heuristico(problema)

    fin_solver = time.time()
    tiempo_ejecucion = fin_solver - inicio_solver
    # --- FIN DEL SOLVER HÍBRIDO ---

    # --- Impresión de Resultados ---
    print("\n================ RESULTADO FINAL ================")
    if solucion_final.costo_total == float('inf'):
        print("NO SE ENCONTRÓ SOLUCIÓN VIABLE.")
    else:
        # --- MODIFICACIÓN ---
        # Se eliminó el 'if' que acortaba la ruta.
        ruta_str = " -> ".join(map(str, solucion_final.ruta))
        # --- FIN DE LA MODIFICACIÓN ---
            
        print(f"Ruta final (expandida): {ruta_str} (Total pasos: {len(solucion_final.ruta)})")
        print("------------------------------------------------")
        print(f"Costo total: {solucion_final.costo_total:.2f}")
        print(f"Distancia total recorrida: {solucion_final.distancia_recorrida:.2f}")
        print(f"Costo activacion de hubs: {solucion_final.costo_hubs:.2f}")
        print(f"Hubs activados: {solucion_final.hubs_activados or 'Ninguno'}")
        print(f"Tiempo de ejecucion: {tiempo_ejecucion:.5f} segundos")
    print("================================================\n")

    # --- Escritura del Archivo de Salida ---
    try:
        escribir_solucion_txt(solucion_final, tiempo_ejecucion)
        print("Archivo 'solucion.txt' generado con éxito.")
    except Exception as e:
        print(f"Error al escribir 'solucion.txt': {e}")


if __name__ == "__main__":
    main()