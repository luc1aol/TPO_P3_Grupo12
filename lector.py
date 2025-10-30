#!/usr/bin/env python3

import sys
from dataclasses import dataclass
import time
from typing import List, Optional

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
class Solucion:
    """La mejor solución completa encontrada hasta ahora."""
    def __init__(self):
        self.ruta: List[int] = []
        self.costo_total: float = float("inf")
        self.distancia_recorrida = 0.0
        self.paquetes_pendientes: List[Paquete] = []
        self.costo_hubs: float = 0.0
        self.hubs_activados: List[int] = []

def encontrar_solucion_greedy(camion: Camion, problema: Problema):
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
                camion.distancia_recorrida_actual += dist_minima
                camion.costo_total_actual += dist_minima
                camion.nodo_actual = paquete_mas_cercano.id_nodo_destino
                camion.ruta_actual.append(paquete_mas_cercano.id_nodo_destino)
                camion.carga_actual -= 1
                camion.paquetes_pendientes_actual.remove(paquete_mas_cercano)
        
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
                camion.distancia_recorrida_actual += dist_minima
                camion.costo_total_actual += dist_minima
                camion.nodo_actual = hub_mas_cercano
                camion.ruta_actual.append(hub_mas_cercano)
                camion.carga_actual = camion.capacidad_maxima
                if hub_mas_cercano in camion.dict_hubs.keys() and hub_mas_cercano not in camion.hubs_activados_actual:
                    costo_activacion = camion.dict_hubs.get(hub_mas_cercano)
                    camion.costo_hubs_actual += costo_activacion
                    camion.costo_total_actual += costo_activacion
                    camion.hubs_activados_actual.append(hub_mas_cercano)

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

                camion.distancia_recorrida_actual += dist_minima
                camion.costo_total_actual += dist_minima
                camion.nodo_actual = paquete_mas_cercano.id_nodo_destino
                camion.ruta_actual.append(paquete_mas_cercano.id_nodo_destino)
                camion.carga_actual -= 1
                camion.paquetes_pendientes_actual.remove(paquete_mas_cercano)

            else:
                dist_minima = dist_minima_recarga

                camion.distancia_recorrida_actual += dist_minima
                camion.costo_total_actual += dist_minima
                camion.nodo_actual = hub_mas_cercano
                camion.ruta_actual.append(hub_mas_cercano)
                camion.carga_actual = camion.capacidad_maxima
                if hub_mas_cercano in camion.dict_hubs.keys() and hub_mas_cercano not in camion.hubs_activados_actual:
                    costo_activacion = camion.dict_hubs.get(hub_mas_cercano)
                    camion.costo_hubs_actual += costo_activacion
                    camion.costo_total_actual += costo_activacion
                    camion.hubs_activados_actual.append(hub_mas_cercano)


    dist_retorno = problema.grafo_distancias[camion.nodo_actual][problema.deposito_id]
    
    solucion.costo_total = camion.costo_total_actual + dist_retorno
    solucion.distancia_recorrida = camion.distancia_recorrida_actual + dist_retorno
    solucion.costo_hubs = camion.costo_hubs_actual
    solucion.hubs_activados = camion.hubs_activados_actual.copy()
    solucion.ruta = camion.ruta_actual.copy() + [problema.deposito_id]
    solucion.paquetes_pendientes = []
    
    return solucion



def resolver_backtracking(camion: Camion, solucion: Solucion, problema: Problema):
    """
    Función recursiva de backtracking (Opción B: Ruteo Simple).
    """

    # --- Casos base ---
    # Poda
    if camion.costo_total_actual >= solucion.costo_total:
        return
    
    # Exito
    if len(camion.paquetes_pendientes_actual) == 0:
        dist_retorno = problema.grafo_distancias[camion.nodo_actual][problema.deposito_id]
        costo_final = dist_retorno + camion.costo_total_actual
        if costo_final < solucion.costo_total:
            # actualizar solucion
            solucion.costo_total = costo_final
            solucion.ruta = camion.ruta_actual.copy() + [problema.deposito_id]
            solucion.distancia_recorrida = camion.distancia_recorrida_actual + dist_retorno
            solucion.costo_hubs = camion.costo_hubs_actual
            solucion.hubs_activados = camion.hubs_activados_actual.copy()
            solucion.paquetes_pendientes = []
        return
    
    # Entregar
    if camion.carga_actual > 0:
        # Añadir paquetes pendientes a opciones
        # Opciones
        opciones = camion.paquetes_pendientes_actual

        for nodo_destino_entrega in opciones.copy():
            # Aplicar
            nodo_anterior = camion.nodo_actual
            dist_viaje = problema.grafo_distancias[camion.nodo_actual][nodo_destino_entrega.id_nodo_destino]
            camion.distancia_recorrida_actual += dist_viaje
            camion.costo_total_actual += dist_viaje
            camion.nodo_actual = nodo_destino_entrega.id_nodo_destino
            camion.carga_actual -= 1
            camion.paquetes_pendientes_actual.remove(nodo_destino_entrega)
            camion.ruta_actual.append(nodo_destino_entrega.id_nodo_destino)

            resolver_backtracking(camion, solucion, problema) # Llamada recursiva con el nuevo estado del camion

            # Deshacer
            camion.ruta_actual.pop()
            camion.paquetes_pendientes_actual.append(nodo_destino_entrega)
            camion.carga_actual += 1
            camion.nodo_actual = nodo_anterior
            camion.costo_total_actual -= dist_viaje
            camion.distancia_recorrida_actual -= dist_viaje
    
    # Recargar
    if camion.carga_actual < camion.capacidad_maxima:
        # Opciones
        opciones = [0] # 0 = deposito central
        # Añadir hubs a opciones
        for i in problema.hubs:
            opciones.append(i.id_nodo)

        for nodo_destino_recarga in opciones:
            if camion.nodo_actual != nodo_destino_recarga:
                # Aplicar
                activacion_de_hub = False
                carga_anterior = camion.carga_actual
                nodo_anterior = camion.nodo_actual
                dist_viaje = problema.grafo_distancias[camion.nodo_actual][nodo_destino_recarga]

                camion.distancia_recorrida_actual += dist_viaje
                camion.costo_total_actual += dist_viaje
                camion.nodo_actual = nodo_destino_recarga
                camion.carga_actual = camion.capacidad_maxima
                camion.ruta_actual.append(nodo_destino_recarga)

                # Si es un hub
                if nodo_destino_recarga in camion.dict_hubs.keys() and nodo_destino_recarga not in camion.hubs_activados_actual:
                    activacion_de_hub = True
                    costo_activacion = camion.dict_hubs.get(nodo_destino_recarga)

                    camion.costo_hubs_actual += costo_activacion
                    camion.costo_total_actual += costo_activacion
                    camion.hubs_activados_actual.append(nodo_destino_recarga)

                resolver_backtracking(camion, solucion, problema) # Llamada recursiva con el nuevo estado del camion

                # Deshacer
                if activacion_de_hub:
                    camion.hubs_activados_actual.remove(nodo_destino_recarga)
                    camion.costo_total_actual -= costo_activacion
                    camion.costo_hubs_actual -= costo_activacion
                camion.ruta_actual.pop()
                camion.carga_actual = carga_anterior
                camion.nodo_actual = nodo_anterior
                camion.costo_total_actual -= dist_viaje
                camion.distancia_recorrida_actual -= dist_viaje

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

    estado_inicial = Camion(problema)
    mejor_solucion = encontrar_solucion_greedy(estado_inicial, problema)

    print("Iniciando backtracking...")
    resolver_backtracking(estado_inicial, mejor_solucion, problema)
    
    fin = time.time()

    tiempo = fin - inicio

    print(f"Tiempo de ejecucion: {tiempo:.5f} segundos")
    print(f"Mejor ruta: {mejor_solucion.ruta}")
    print(f"Costo total: {mejor_solucion.costo_total:.5f}")
    print(f"Distancia total recorrida: {mejor_solucion.distancia_recorrida:.5f}")

if __name__ == "__main__":
    main()
