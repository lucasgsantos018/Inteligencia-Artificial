def bestFirst(labirinto, h=100):
    inicio = (labirinto.rows,labirinto.cols)

    obstaculos = [h.cost]

    naoVisitado = {n:float('inf') for n in labirinto.grid}
    naoVisitado[inicio] = 0

    visitado = {}
    revPath = {}

    while naoVisitado:
        vertice = min(naoVisitado, key = naoVisitado.get)
        visitado[vertice] = naoVisitado[vertice]
        if vertice == labirinto._goal:
            print("Best-First: objetivo encontrado")
            break
        
        movimentos = ["E", "S", "N", "W"]

        for d in movimentos:
            if labirinto.maze_map[vertice][d]==True:
                if d=='E':
                    vizinho=(vertice[0],vertice[1]+1)
                elif d=='W':
                    vizinho=(vertice[0],vertice[1]-1)
                elif d=='S':
                    vizinho=(vertice[0]+1,vertice[1])
                elif d=='N':
                    vizinho=(vertice[0]-1,vertice[1])
                if vizinho in visitado:
                    continue
                tempDist= naoVisitado[vertice]+1

                for obstaculo in obstaculos:
                    if obstaculo[0] == vertice:
                        tempDist += obstaculo[1]

                if tempDist < naoVisitado[vizinho]:
                    naoVisitado[vizinho]=tempDist
                    revPath[vizinho]=vertice
        naoVisitado.pop(vertice)
    
    bestFirstPath = {}
    cell = labirinto._goal
    while cell != inicio:
        bestFirstPath[revPath[cell]] = cell
        cell = revPath[cell]
    
    return bestFirstPath
    




# def dijkstra(m,*h,inicio=None):
    