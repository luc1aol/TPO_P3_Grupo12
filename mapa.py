#Antes de que el explorador (backtracking) pueda hacer un solo movimiento, necesita el 
# "mapa" de distancias mínimas. El TPO sugiere Floyd-Warshall o Dijkstra. 
# Floyd-Warshall es ideal para esto.

# 1. Cargar Datos: Lee caso.txt y guarda todo en estructuras claras.
# 2. Construir Matriz de Adyacencia: Crea una matriz donde matriz[i][j] sea la distancia 
#       directa (arista) o inf si no hay conexión.
# 3. Ejecutar Floyd-Warshall: Esta es la función clave del preprocesamiento.

import sys
import time

# --- Constantes ---
ID_DEPOSITO = 0
INF = float('inf')

def cargar_datos(nombre_archivo):
    """
    Lee el archivo caso.txt y devuelve las estructuras de datos.
    NECESITARÁS IMPLEMENTAR ESTA LÓGICA DE PARSEO.
    """
    print(f"Cargando datos desde {nombre_archivo}...")
    # Ejemplo de estructuras que necesitas llenar:
    num_nodos = 0
    capacidad_camion = 0
    # Un set con los IDs de los nodos que son destinos
    paquetes_pendientes_inicial = set()
    # Un dict {id_hub: costo_activacion}
    hubs_info = {}
    # Una lista de tuplas (origen, destino, distancia)
    aristas_directas = []
    
    # ... Lógica de parseo del archivo ...

    # Ejemplo de datos (basado en el PDF de caso de estudio [cite: 81-86])
    num_nodos = 5 # DC(0), H(1), D1(2), D2(3), D3(4)
    capacidad_camion = 2 # [cite: 82]
    paquetes_pendientes_inicial = {2, 3, 4} # Asumiendo IDs 2, 3, 4
    hubs_info = {1: 10.0} # [cite: 83]
    aristas_directas = [
        (0, 1, 8), (1, 0, 8), (0, 2, 10), (2, 0, 10), (0, 3, 15), (3, 0, 15),
        (0, 4, 25), (4, 0, 25), (1, 2, 3), (2, 1, 3), (1, 3, 4), (3, 1, 4),
        (1, 4, 10), (4, 1, 10), (2, 3, 5), (3, 2, 5), (3, 4, 7), (4, 3, 7)
    ]
    # NOTA: Tu parser real leerá esto del archivo.

    return num_nodos, capacidad_camion, paquetes_pendientes_inicial, hubs_info, aristas_directas

