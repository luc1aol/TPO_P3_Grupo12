#!/usr/bin/env python3

import sys
from dataclasses import dataclass
from typing import List, Optional
import time
from collections import Counter

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

# ===========================================================
# CLASES DE ESTADO PARA BACKTRACKING
# ===========================================================
class solucion_optima:
    """
    Almacena la mejor solución completa encontrada hasta ahora.
    Esto permite la poda y guardar el resultado final.
    """
    def __init__(self):
        self.costo_total = float('inf')
        self.distancia_recorrida = float('inf')
        self.costo_hubs = 0.0
        self.hubs_activos = set()
        self.ruta = []

    def actualizar(self, estado_final, costo_retorno_deposito, dist_retorno_deposito, id_deposito):
        """Actualiza esta solución si la nueva es mejor."""
        
        costo_final = estado_final.costo_total_actual + costo_retorno_deposito
        
        if costo_final < self.costo_total:
            self.costo_total = costo_final
            self.distancia_recorrida = estado_final.distancia_recorrida_actual + dist_retorno_deposito
            self.costo_hubs = estado_final.costo_hubs_actual
            self.hubs_activos = set(estado_final.hubs_activos) # Copiar
            self.ruta = list(estado_final.ruta_actual) + [id_deposito] # Copiar y añadir retorno
            
            # print(f"  -> Nueva mejor solución: Costo {self.costo_total:.2f}") # Descomentar para debug
class estado:
    """
    Representa el estado actual del problema en la recursión (el "explorador").
    """
    def __init__(self, capacidad_max: int, id_deposito: int, paquetes_counter: Counter):
        self.nodo_actual = id_deposito
        self.carga_actual = capacidad_max
        self.capacidad_max = capacidad_max
        
        # Un diccionario {id_destino: cantidad_paquetes}
        self.paquetes_pendientes = paquetes_counter
        
        self.costo_total_actual = 0.0
        self.distancia_recorrida_actual = 0.0
        self.costo_hubs_actual = 0.0
        self.hubs_activos = set()
        
        self.ruta_actual = [id_deposito]

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

# ===========================================================
# BACKTRACKING
# ===========================================================

from functools import lru_cache

def buscar_solucion_recursiva(
    estado_actual: estado,
    mejor_solucion: solucion_optima,
    matriz_dist,
    hubs_info: dict,
    id_deposito: int,
    cache=None
):
    """
    Versión optimizada del backtracking (Opción B del TPO).
    Usa heurísticas, memorización y recarga selectiva.
    """
    if cache is None:
        cache = {}

    # --- Generar una clave de estado para la memorización ---
    clave = (
        estado_actual.nodo_actual,
        estado_actual.carga_actual,
        tuple(sorted(estado_actual.paquetes_pendientes.items())),
        tuple(sorted(estado_actual.hubs_activos))
    )

    # Si ya se visitó este estado con un costo menor o igual → poda
    if clave in cache and cache[clave] <= estado_actual.costo_total_actual:
        return
    cache[clave] = estado_actual.costo_total_actual

    # --- Heurística más informada ---
    if estado_actual.paquetes_pendientes:
        # h(s): promedio de distancias a los destinos pendientes + costo de volver al depósito
        dist_restantes = [
            matriz_dist[estado_actual.nodo_actual][dest] for dest in estado_actual.paquetes_pendientes.keys()
        ]
        h = (sum(dist_restantes) / len(dist_restantes)) + matriz_dist[estado_actual.nodo_actual][id_deposito]
    else:
        h = matriz_dist[estado_actual.nodo_actual][id_deposito]

    if estado_actual.costo_total_actual + h >= mejor_solucion.costo_total:
        return  # poda por cota inferior

    # --- CASO BASE: Todos los paquetes entregados ---
    if not estado_actual.paquetes_pendientes:
        dist_retorno = matriz_dist[estado_actual.nodo_actual][id_deposito]
        mejor_solucion.actualizar(estado_actual, dist_retorno, dist_retorno, id_deposito)
        return

    nodo_previo = estado_actual.nodo_actual

    # --- Opción A: ENTREGAR si tengo carga ---
    if estado_actual.carga_actual > 0:
        destinos_ordenados = sorted(
            estado_actual.paquetes_pendientes.keys(),
            key=lambda d: matriz_dist[nodo_previo][d]
        )

        for nodo_destino in destinos_ordenados:
            dist_viaje = matriz_dist[nodo_previo][nodo_destino]
            nuevo_costo = estado_actual.costo_total_actual + dist_viaje
            if nuevo_costo >= mejor_solucion.costo_total:
                continue  # poda temprana

            # Aplicar entrega
            estado_actual.distancia_recorrida_actual += dist_viaje
            estado_actual.costo_total_actual = nuevo_costo
            estado_actual.nodo_actual = nodo_destino
            estado_actual.carga_actual -= 1
            estado_actual.paquetes_pendientes[nodo_destino] -= 1

            completado = estado_actual.paquetes_pendientes[nodo_destino] == 0
            if completado:
                del estado_actual.paquetes_pendientes[nodo_destino]

            estado_actual.ruta_actual.append(nodo_destino)
            buscar_solucion_recursiva(estado_actual, mejor_solucion, matriz_dist, hubs_info, id_deposito, cache)

            # Deshacer
            estado_actual.ruta_actual.pop()
            if completado:
                estado_actual.paquetes_pendientes[nodo_destino] = 1
            else:
                estado_actual.paquetes_pendientes[nodo_destino] += 1
            estado_actual.carga_actual += 1
            estado_actual.nodo_actual = nodo_previo
            estado_actual.costo_total_actual -= dist_viaje
            estado_actual.distancia_recorrida_actual -= dist_viaje

    # --- Opción B: RECARGAR si está vacío ---
    else:
        puntos_recarga = [id_deposito] + list(hubs_info.keys())

        for nodo_recarga in puntos_recarga:
            if nodo_recarga == nodo_previo:
                continue

            dist_viaje = matriz_dist[nodo_previo][nodo_recarga]
            costo_hub = 0.0
            es_hub = nodo_recarga in hubs_info

            if es_hub and nodo_recarga not in estado_actual.hubs_activos:
                costo_hub = hubs_info[nodo_recarga]

            nuevo_costo = estado_actual.costo_total_actual + dist_viaje + costo_hub
            if nuevo_costo >= mejor_solucion.costo_total:
                continue  # poda

            # Aplicar recarga
            estado_actual.distancia_recorrida_actual += dist_viaje
            estado_actual.costo_total_actual = nuevo_costo
            estado_actual.nodo_actual = nodo_recarga
            estado_actual.carga_actual = estado_actual.capacidad_max
            estado_actual.ruta_actual.append(nodo_recarga)

            if costo_hub > 0:
                estado_actual.costo_hubs_actual += costo_hub
                estado_actual.hubs_activos.add(nodo_recarga)

            buscar_solucion_recursiva(estado_actual, mejor_solucion, matriz_dist, hubs_info, id_deposito, cache)

            # Deshacer
            if costo_hub > 0:
                estado_actual.hubs_activos.remove(nodo_recarga)
                estado_actual.costo_hubs_actual -= costo_hub

            estado_actual.ruta_actual.pop()
            estado_actual.carga_actual = 0
            estado_actual.nodo_actual = nodo_previo
            estado_actual.costo_total_actual -= (dist_viaje + costo_hub)
            estado_actual.distancia_recorrida_actual -= dist_viaje

def escribir_solucion_txt(nombre_archivo: str, sol: solucion_optima, tiempo_ejecucion: float):
    """
    Genera el archivo solucion.txt con el formato exacto requerido.
    """
    try:
        with open(nombre_archivo, 'w') as f:
            # --- HUBS ACTIVADOS --- 
            f.write("// HUBS ACTIVADOS\n")
            if not sol.hubs_activos:
                f.write("Ninguno\n")
            else:
                for hub_id in sorted(list(sol.hubs_activos)):
                    f.write(f"{hub_id}\n")
            
            # --- RUTA OPTIMA --- 
            f.write("// RUTA OPTIMA\n")
            f.write(" -> ".join(map(str, sol.ruta)) + "\n")
            
            # --- METRICAS --- 
            f.write("// METRICAS\n")
            f.write(f"COSTO TOTAL: {sol.costo_total:.2f}\n")
            f.write(f"DISTANCIA_RECORRIDA: {sol.distancia_recorrida:.2f}\n")
            f.write(f"COSTO_HUBS: {sol.costo_hubs:.2f}\n")
            f.write(f"TIEMPO EJECUCION: {tiempo_ejecucion:.6f} segundos\n")
            
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
    print(f"Leyendo el archivo de problema: {nombre_archivo}") # Nombre del archivo de salida

    nombre_archivo_salida = "solucion.txt"

    # --- 1. LECTURA Y PREPROCESAMIENTO ---
    inicio_total = time.time()
    problema = leer_archivo(nombre_archivo)
    if problema is None:
        print("\n>> Hubo un error al leer o procesar el archivo. Revisa el formato.")
        sys.exit(1)

    print("\n¡Archivo leído y procesado con éxito!")
    imprimir_problema(problema)

    floyd_warshall(problema)
    print("--- MUESTRA DEL GRAFO (MATRIZ DE CAMINOS MINIMOS) ---")
    imprimir_matriz(problema)

    # --- 2. PREPARAR DATOS PARA BACKTRACKING ---
    # "Traducir" los datos del parser a estructuras optimizadas
    matriz_dist = problema.grafo_distancias
    hubs_info = {h.id_nodo: h.costo_activacion for h in problema.hubs}
    
    # Contar cuántos paquetes van a cada destino
    paquetes_counter = Counter(p.id_nodo_destino for p in problema.paquetes)
    
    id_deposito_const = problema.deposito_id
    cap_camion_const = problema.capacidad_camion

    # --- 3. INICIAR BÚSQUEDA ---
    estado_inicial = estado(cap_camion_const, id_deposito_const, paquetes_counter)
    mejor_solucion = solucion_optima()

    print(f"Iniciando búsqueda de ruta óptima (Backtracking)...")
    
    inicio_backtracking = time.time()
    
    buscar_solucion_recursiva(
        estado_inicial,
        mejor_solucion,
        matriz_dist,
        hubs_info,
        id_deposito_const
    )
    
    fin_backtracking = time.time()
    tiempo_ejecucion_algoritmo = fin_backtracking - inicio_backtracking
    
    print("Búsqueda finalizada.")

    # --- 4. MOSTRAR Y ESCRIBIR RESULTADOS ---
    if mejor_solucion.costo_total == float('inf'):
        print("\n>> NO SE ENCONTRÓ NINGUNA SOLUCIÓN VÁLIDA <<")
    else:
        print(f"\n============== SOLUCIÓN ÓPTIMA ENCONTRADA ===============")
        print(f"  COSTO TOTAL:\t\t{mejor_solucion.costo_total:.2f}")
        print(f"  Distancia Recorrida:\t{mejor_solucion.distancia_recorrida:.2f}")
        print(f"  Costo de Hubs:\t{mejor_solucion.costo_hubs:.2f}")
        print(f"  Hubs Activados:\t{mejor_solucion.hubs_activos or 'Ninguno'}")
        print(f"  Tiempo del Algoritmo:\t{tiempo_ejecucion_algoritmo:.6f} seg")
        print(f"  Ruta (primeros 20 nodos):\n  {' -> '.join(map(str, mejor_solucion.ruta[:20]))}...")
        
        # Escribir el archivo de salida oficial
        escribir_solucion_txt(nombre_archivo_salida, mejor_solucion, tiempo_ejecucion_algoritmo)
        print(f"\nArchivo '{nombre_archivo_salida}' generado con éxito.")
    
    fin_total = time.time()
    print(f"Tiempo total del programa: {(fin_total - inicio_total):.6f} seg")

    print("Memoria liberada correctamente.")

if __name__ == "__main__":
    main()

