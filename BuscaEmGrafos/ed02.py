from pyamaze import maze, agent, COLOR
from random import randint

#from bfs import *
#from dfs import *
from bestFirst import *
#from aStar import *

def execucaoMaze(tamanho=30, possibilidadeCaminhos=100):
    
    goalX, goalY = randint(1,tamanho), 1
    
    m=maze(tamanho,tamanho)
    m.CreateMaze(goalX, goalY, loopPercent=possibilidadeCaminhos)
    
    a=agent(m,footprints=True, color='red', filled=True)
    b=agent(m,footprints=True, color='cyan', shape='arrow')
    c=agent(m,footprints=True, color='yellow', shape='arrow')
    d=agent(m,footprints=True, color='green', filled=True)

    c.cost = 100

    
    print("Executando busca")
    #path1 = bfs(m)
    #path2 = dfs(m)
    path3 = bestFirst(m)
    #path4 = aStar(m)

  
    m.tracePath({c:path3})   #a:path1, a:path1, b:path2, d:path4
    m.run()

if __name__=='__main__':
    execucaoMaze(tamanho=10)