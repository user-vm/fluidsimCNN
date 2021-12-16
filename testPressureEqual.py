# Compares outputs of projection neural network between Torch7 (Lua) and CPP

luaFilename = "/home/vmiu/2018/cnn_fluid/FluidNet/torch/pPred_UPred_out_24-09-2018_19-28.txt"
cppFilename = "/home/vmiu/2018/cnn_fluid/fluidsim/UPred_pPred_CPP.txt"

cppFile = open(cppFilename,"r")
luaFile = open(luaFilename,"r")

while(cppFile.readline()[:5] != "pPred"):
	pass

while(luaFile.readline()[:5] != "pPred"):
	pass

outFile = open("div_pPred.txt", "w")

while(True):

	luapPredString = luaFile.readline()
	cpppPredString = cppFile.readline()

	if((not(bool(luapPredString)))^(not(bool(cpppPredString)))):
		outFile.write("The number of pPred values mismatch\n")
		outFile.close()
		break

	if(not(bool(luapPredString)) and not(bool(cpppPredString))):
		outFile.close()
		break

	outFile.write(str(float(cpppPredString)/float(luapPredString)))
	outFile.write("\n")
