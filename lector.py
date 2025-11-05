#!/usr/bin/env python3

import sys
from dataclasses import dataclass
import time
from typing import List, Optional, Dict, Tuple

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

class Camion:
    """
    Estado actual del camion
    """
    def __init__(self, p: Problema):
        # inicia en el deposito
        self.ruta_actual: List[int] = [p.deposito_id] 
        self.nodo_actual = p.deposito_id

        # comienza lleno
        self.capacidad_maxima = p.capacidad_camion
        self.carga_actual = p.capacidad_camion

        self.costo_total_actual = 0.0
        self.distancia_recorrida_actual = 0.0
        self.paquetes_pendientes_actual = p.paquetes.copy()
        self.hubs_activados_actual: List[int] = []
        self.dict_hubs = {hub.id_nodo: hub.costo_activacion for hub in p.hubs}
        self.costo_hubs_actual: float = 0.0

    def aplicar_entrega(self, nodo_destino_entrega: Paquete, problema: Problema, dist_viaje: float):
        self.distancia_recorrida_actual += dist_viaje
        self.costo_total_actual += dist_viaje
        self.nodo_actual = nodo_destino_entrega.id_nodo_destino
        self.carga_actual -= 1
        self.paquetes_pendientes_actual.remove(nodo_destino_entrega)
        self.ruta_actual.append(nodo_destino_entrega.id_nodo_destino)

    def deshacer_entrega(self, nodo_destino_entrega: Paquete, nodo_anterior: int, dist_viaje: float):
        self.ruta_actual.pop()
        self.paquetes_pendientes_actual.append(nodo_destino_entrega)
        self.carga_actual += 1
        self.nodo_actual = nodo_anterior
        self.costo_total_actual -= dist_viaje
        self.distancia_recorrida_actual -= dist_viaje

    def aplicar_recarga(self, nodo_destino_recarga: int, dist_viaje: float):
        activacion_de_hub = False
        costo_activacion = None

        self.distancia_recorrida_actual += dist_viaje
        self.costo_total_actual += dist_viaje
        self.nodo_actual = nodo_destino_recarga
        self.carga_actual = self.capacidad_maxima
        self.ruta_actual.append(nodo_destino_recarga)

        # Si es un hub
        if nodo_destino_recarga in self.dict_hubs.keys() and nodo_destino_recarga not in self.hubs_activados_actual:
            activacion_de_hub = True
            costo_activacion = self.dict_hubs.get(nodo_destino_recarga)

            self.costo_hubs_actual += costo_activacion
            self.costo_total_actual += costo_activacion
            self.hubs_activados_actual.append(nodo_destino_recarga)

        return activacion_de_hub, costo_activacion
    
    def deshacer_recarga(self, activacion_de_hub: bool, nodo_destino_recarga: int, costo_activacion: float, carga_anterior: int, nodo_anterior: int, dist_viaje: float):
        if activacion_de_hub:
            self.hubs_activados_actual.remove(nodo_destino_recarga)
            self.costo_total_actual -= costo_activacion
            self.costo_hubs_actual -= costo_activacion
        self.ruta_actual.pop()
        self.carga_actual = carga_anterior
        self.nodo_actual = nodo_anterior
        self.costo_total_actual -= dist_viaje
        self.distancia_recorrida_actual -= dist_viaje
class Solucion:
    """La mejor solución completa encontrada hasta ahora."""
    def __init__(self):
        self.ruta: List[int] = []
        self.costo_total: float = float("inf")
        self.distancia_recorrida = 0.0
        self.paquetes_pendientes: List[Paquete] = []
        self.costo_hubs: float = 0.0
        self.hubs_activados: List[int] = []

    def actualizar_solucion(self, camion: Camion, problema: Problema, costo_final: float, dist_retorno: float):
        self.costo_total = costo_final
        self.ruta = camion.ruta_actual.copy() + [problema.deposito_id]
        self.distancia_recorrida = camion.distancia_recorrida_actual + dist_retorno
        self.costo_hubs = camion.costo_hubs_actual
        self.hubs_activados = camion.hubs_activados_actual.copy()
        self.paquetes_pendientes = []

def encontrar_solucion_greedy(problema: Problema):
    camion = Camion(problema)
    solucion = Solucion()
    while len(camion.paquetes_pendientes_actual) > 0:
        dist_minima = float("inf")
        paquete_mas_cercano = None
        hub_mas_cercano = None

        # SI ESTA LLENO
        if camion.carga_actual == camion.capacidad_maxima:
            for paquete in camion.paquetes_pendientes_actual:
                nodo_destino = paquete.id_nodo_destino

                if camion.nodo_actual == nodo_destino: # no viajar al nodo actual
                    continue

                dist = problema.grafo_distancias[camion.nodo_actual][nodo_destino]
                if dist < dist_minima:
                    dist_minima = dist
                    paquete_mas_cercano = paquete

            if paquete_mas_cercano:
                camion.aplicar_entrega(paquete_mas_cercano, problema, dist_minima)
        
        # SI ESTA VACIO
        elif camion.carga_actual == 0:
            opciones = [0] + [hub.id_nodo for hub in problema.hubs]

            for hub in opciones:

                if camion.nodo_actual == hub: # no viajar al nodo actual
                    continue

                dist = problema.grafo_distancias[camion.nodo_actual][hub]
                if dist < dist_minima:
                    dist_minima = dist
                    hub_mas_cercano = hub

            if hub_mas_cercano:
                camion.aplicar_recarga(hub_mas_cercano, dist_minima)

        # SI AUN TIENE ESPACIO
        elif camion.carga_actual > 0:

            # paquete mas cercano
            dist_minima_paquete = float("inf")
            for paquete in camion.paquetes_pendientes_actual:
                nodo_destino = paquete.id_nodo_destino

                if camion.nodo_actual == nodo_destino: # no viajar al nodo actual
                    continue

                dist = problema.grafo_distancias[camion.nodo_actual][nodo_destino]
                if dist < dist_minima_paquete:
                    dist_minima_paquete = dist
                    paquete_mas_cercano = paquete

            # hub/deposito mas cercano
            dist_minima_recarga = float("inf")
            opciones = [0] + [hub.id_nodo for hub in problema.hubs]
            for hub in opciones:

                if camion.nodo_actual == hub: # no viajar al nodo actual
                    continue

                dist = problema.grafo_distancias[camion.nodo_actual][hub]
                if dist < dist_minima_recarga:
                    dist_minima_recarga = dist
                    hub_mas_cercano = hub
            
            # compara paquete vs recarga
            if dist_minima_paquete < dist_minima_recarga:
                dist_minima = dist_minima_paquete

                camion.aplicar_entrega(paquete_mas_cercano, problema, dist_minima)

            else:
                dist_minima = dist_minima_recarga

                camion.aplicar_recarga(hub_mas_cercano, dist_minima)


    dist_retorno = problema.grafo_distancias[camion.nodo_actual][problema.deposito_id]
    costo_final = camion.costo_total_actual + dist_retorno
    solucion.actualizar_solucion(camion, problema, costo_final, dist_retorno)
    
    return solucion

# NUEVO: Función para estimar costo restante (ajustada: menos conservadora, ignora hubs para permitir exploración)
def estimar_costo_restante(camion: Camion, problema: Problema) -> float:
    """
    Estima el costo mínimo restante: suma distancias mínimas a paquetes pendientes + retorno al depósito.
    Ahora ignora costos de hubs para ser menos conservadora y permitir más exploración.
    """
    if not camion.paquetes_pendientes_actual:
        return problema.grafo_distancias[camion.nodo_actual][problema.deposito_id]
    
    costo_estimado = 0.0
    for paquete in camion.paquetes_pendientes_actual:
        costo_estimado += problema.grafo_distancias[camion.nodo_actual][paquete.id_nodo_destino]
    costo_estimado += problema.grafo_distancias[camion.nodo_actual][problema.deposito_id]
    return costo_estimado  # Sin sumar costos de hubs para ser menos restrictiva

def resolver_backtracking(camion: Camion, solucion: Solucion, problema: Problema, memo: Dict[Tuple, float], k_vecinos: int):
    estado = (
        frozenset(p.id for p in camion.paquetes_pendientes_actual),
        camion.carga_actual,
        camion.nodo_actual,
        frozenset(camion.hubs_activados_actual)
    )
    
    # Verificar memo
    if estado in memo and camion.costo_total_actual >= memo[estado]:
        return
    memo[estado] = camion.costo_total_actual
    
    # Poda por estimación de costo restante
    estimacion = estimar_costo_restante(camion, problema)
    if camion.costo_total_actual + estimacion * 0.8 >= solucion.costo_total:
        return
    
    # --- Casos base ---
    if len(camion.paquetes_pendientes_actual) == 0:
        dist_retorno = problema.grafo_distancias[camion.nodo_actual][problema.deposito_id]
        print("[DEBUG] Caso base - Costo total:", camion.costo_total_actual + dist_retorno)
        costo_final = dist_retorno + camion.costo_total_actual
        if costo_final < solucion.costo_total:
            solucion.actualizar_solucion(camion, problema, costo_final, dist_retorno)
        return
    
    # Entregar (si hay carga)
    if camion.carga_actual > 0:
        opciones_con_distancia = []
        
        for paquete in camion.paquetes_pendientes_actual:
            if paquete.id_nodo_destino != camion.nodo_actual:
                dist = problema.grafo_distancias[camion.nodo_actual][paquete.id_nodo_destino]
                opciones_con_distancia.append((dist, paquete))
        
        opciones_con_distancia.sort(key=lambda tupla: tupla[0])
        opciones_recortadas = opciones_con_distancia[:k_vecinos]
        
        for dist_viaje_entrega, nodo_destino_entrega in opciones_recortadas:
            nodo_anterior = camion.nodo_actual
            camion.aplicar_entrega(nodo_destino_entrega, problema, dist_viaje_entrega)
            
            resolver_backtracking(camion, solucion, problema, memo, k_vecinos)
            
            camion.deshacer_entrega(nodo_destino_entrega, nodo_anterior, dist_viaje_entrega)
    
    # Recargar (si no está lleno)
    if camion.carga_actual < camion.capacidad_maxima:
        opciones_recarga_con_distancia = []
        nodos_de_recarga = [problema.deposito_id] + [hub.id_nodo for hub in problema.hubs]
        
        for id_nodo in nodos_de_recarga:
            if id_nodo != camion.nodo_actual:
                dist = problema.grafo_distancias[camion.nodo_actual][id_nodo]
                opciones_recarga_con_distancia.append((dist, id_nodo))
        
        opciones_recarga_con_distancia.sort(key=lambda tupla: tupla[0])
        opciones_recortadas = opciones_recarga_con_distancia[:k_vecinos]
        
        for dist_viaje_recarga, nodo_destino_recarga in opciones_recortadas:
            carga_anterior = camion.carga_actual
            nodo_anterior = camion.nodo_actual
            activacion_de_hub, costo_activacion = camion.aplicar_recarga(nodo_destino_recarga, dist_viaje_recarga)
            
            resolver_backtracking(camion, solucion, problema, memo, k_vecinos)
            
            camion.deshacer_recarga(activacion_de_hub, nodo_destino_recarga, costo_activacion, carga_anterior, nodo_anterior, dist_viaje_recarga)

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

    camion = Camion(problema)

    mejor_solucion = Solucion()
    
    # Ejecutar greedy por separado para comparación
    solucion_greedy = encontrar_solucion_greedy(problema)
    
    memo: Dict[Tuple, float] = {}
    
    print("Iniciando backtracking con memoization, estimación y DFS...")
    resolver_backtracking(camion, mejor_solucion, problema, memo, 10)
    
    print(f"[DEBUG] Solución greedy - Costo: {solucion_greedy.costo_total:.2f}")
    if mejor_solucion.costo_total < solucion_greedy.costo_total:
        print("[DEBUG] Backtracking encontró una mejor solución")
    else: 
        print("[DEBUG] Backtracking NO mejoró greedy")
        mejor_solucion = solucion_greedy

    fin = time.time()
    tiempo = fin - inicio

    print(f"\nTiempo de ejecucion: {tiempo:.5f} segundos")
    print(f"Mejor ruta: {mejor_solucion.ruta}")
    print(f"Costo total: {mejor_solucion.costo_total:.5f}")
    print(f"Costo activacion de hubs: {mejor_solucion.costo_hubs:.5f}")
    print(f"Distancia total recorrida: {mejor_solucion.distancia_recorrida:.5f}")
    print(f"Hubs activados: {mejor_solucion.hubs_activados}")

if __name__ == "__main__":
    main()