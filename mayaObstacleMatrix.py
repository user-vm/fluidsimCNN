import maya.cmds as cmds

cubesFile = open("/home/vmiu/2018/cnn_fluid/fluidsim/obstacleMatrix.txt","r")

gridSize = 64
gridDim = 10.0

cellSize = gridDim/gridSize

while(True):
    currentCube = cmds.polyCube(width=cellSize, height=cellSize, depth = cellSize)
    
    posList = cubesFile.readline().split(" ")
    
    if(len(posList)<3):
        break
    
    cmds.move(int(posList[0])*cellSize-gridDim/2+cellSize/2,
              int(posList[1])*cellSize-gridDim/2+cellSize/2,
              int(posList[2])*cellSize-gridDim/2+cellSize/2)