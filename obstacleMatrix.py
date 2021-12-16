import os

outFiles = os.listdir('.')

outFiles.sort()

lenPrefix = len("fluidsim-out")

for i in xrange(len(outFiles)-1,-1,-1):
	if(outFiles[i][:lenPrefix] == "fluidsim-out"):
		outFileName = outFiles[i]
		break

print outFileName

import sys

#sys.exit()

theFile = open(outFileName,"r") #open("fluidsim-out-24-04-18_14-29-17.txt","r")
outFile = open("obstacleMatrix.txt","w")

#obstacleMatrix = [[[0]*64]*64]*64
n = 64 #size in x,y and z of the matrix

obstacleMatrix = [[[0 for k in xrange(n)] for j in xrange(n)] for i in xrange(n)]

while(True):
	s = theFile.readline()

	sList = s.split(" ")
	
	if len(sList) < 3:
		break

	obstacleMatrix[int(sList[0])][int(sList[1])][int(sList[2])] = 1;

print obstacleMatrix

for i in xrange(n):
	for j in xrange(n):
		for k in xrange(n):
			if(obstacleMatrix[i][j][k]):
				outFile.write("%d %d %d\n"%(i,j,k))
