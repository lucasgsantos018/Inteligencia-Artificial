def dfs(labirinto):
    inicio=(labirinto.rows, labirinto.cols)
    explorado=[inicio]
    fronteira=[inicio]

    dfsPath = {}
   
    while len(fronteira) > 0:
        vertice = fronteira.pop(0)
        
        if vertice == labirinto._goal:
            print("DFS: objetivo encontrado")
            break
        
        movimentos=["E", "S", "N", "W"]

        for d in movimentos:
            if labirinto.maze_map[vertice][d]==True:
                if d=='E':
                    vizinho=(vertice[0],vertice[1]+1)
                if d=='W':
                    vizinho=(vertice[0],vertice[1]-1)
                if d=='N':
                    vizinho=(vertice[0]-1,vertice[1])
                if d=='S':
                    vizinho=(vertice[0]+1,vertice[1])
  
                if vizinho in explorado:
                    continue
                
                explorado.append(vizinho)
                fronteira.append(vizinho)
                dfsPath[vizinho] = vertice
    fwdPath={}
    cell = labirinto._goal
    while cell != inicio:
        fwdPath[dfsPath[cell]] = cell
        cell = dfsPath[cell]
    return fwdPath