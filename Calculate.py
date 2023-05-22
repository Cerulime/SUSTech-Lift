import argparse
import numpy as np
import sys

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-r","--raw",type=str,required=True,help="input the raw file path as str")
    args=parser.parse_args()
    with open(args.raw,"r",encoding="utf-8") as file:
	    lines=file.readlines()
    data=np.array([[0.0,0.0,0.0]])
    for line in lines:
        data=np.append(data,[list(map(float,line.strip().split(",")))],axis=0)
    data=np.delete(data,0,axis=0)
    x=np.array([])
    y1=np.array([])
    y2=np.array([])
    for row in data:
        x=np.append(x,row[2])
        y1=np.append(y1,row[2]-row[0])
        y2=np.append(y2,row[1]-row[2])
    result=np.polyfit(x,y1,1)
    print("Decelerating acceleration="+str(result[0]))
    print("Correction="+str(result[1]/result[0]))
    result=np.polyfit(x,y2,1)
    print("Accelerating acceleration="+str(result[0]))
    print("Correction="+str(result[1]/result[0]))