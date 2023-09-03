from queue import PriorityQueue

def h(vertice1, vertice2):
    x1,y1 = vertice1
    x2,y2 = vertice2

    return abs(x1-x2) + abs(y1-y2)

def aStar(labirinto):
    inicio = (labirinto.rows, labirinto.cols)
    
    g_score = {vertice:float('inf') for vertice in labirinto.grid}
    g_score[inicio] = 0

    f_score = {vertice:float('inf') for vertice in labirinto.grid}
    f_score[inicio] = h(inicio, labirinto._goal)

    pilha = PriorityQueue()
    pilha.put((h(inicio, labirinto._goal), h(inicio, labirinto._goal), inicio))

    
    aStarPath = {}

    while not pilha.empty():
        vertice = pilha.get()[2]
        if vertice == labirinto._goal:
            print("A*: objetivo encontrado")
            break
        
        movimentos=["E", "S", "N", "W"]

        for d in movimentos:
            if labirinto.maze_map[vertice][d]==True:
                if d=='E':
                    vizinho = (vertice[0],vertice[1]+1)
                if d=='W':
                    vizinho = (vertice[0],vertice[1]-1)
                if d=='N':
                    vizinho = (vertice[0]-1,vertice[1])
                if d=='S':
                    vizinho = (vertice[0]+1,vertice[1])

                temp_g_score=g_score[vertice]+1
                temp_f_score=temp_g_score+h(vizinho,labirinto._goal)

                if temp_f_score < f_score[vizinho]:
                    g_score[vizinho]= temp_g_score
                    f_score[vizinho]= temp_f_score
                    pilha.put((temp_f_score,h(vizinho,labirinto._goal),vizinho))
                    aStarPath[vizinho]=vertice

    path = {}
    cell = labirinto._goal
    while cell != inicio:
        path[aStarPath[cell]] = cell
        cell = aStarPath[cell]
    return path