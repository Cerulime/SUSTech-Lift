import argparse
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sys

def cal_integral(x,y):
	integrals=[]
	for i in range(len(y)):
		integrals.append(scipy.integrate.trapz(y[:i+1],x[:i+1]))
	return integrals
def filter(y):
	sum=0
	cnt=0
	for item in y:
		sum+=item
		cnt+=1
		if cnt>=60:
			break
	sum=sum/cnt
	return sum,[(item-sum)*9.7887/(sum+args.cor) for item in y]
def create(size,rank):
	x=[]
	for i in range(2*size+1):
		m=i-size
		row=[m**j for j in range(rank)]
		x.append(row) 
	x=np.mat(x)
	return x
def check(data,window_size,rank):
	m=(window_size-1)//2
	odata=list(data)
	for i in range(m):
		odata.insert(0,odata[0])
		odata.insert(len(odata),odata[len(odata)-1])
	x=create(m, rank)
	b=(x*(x.T*x).I)*x.T
	a0=b[m]
	a0=a0.T
	for i in range(len(data)):
		y=[odata[i+j] for j in range(window_size)]
		y1=np.mat(y)*a0
		y1=float(y1)
		if abs(y[m]-y1)>args.check:
			print(str(i)+"-> raw: "+str(y[m])+" prediction: "+str(y1))
	return

if __name__=="__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument("-r","--raw",type=str,required=True,help="input the raw file path as str")
	parser.add_argument("-cor","--cor",type=int,default=0,help="input the correction of the balance")
	parser.add_argument("-f","--figure",action="store_true",help="input for a figure output")
	parser.add_argument("-d","--detail",action="store_true",help="input for a detail output")
	parser.add_argument("-c","--check",type=int,default=0,help="input limitation for checking input(Advice: 6)")
	args=parser.parse_args()
	with open(args.raw,"r",encoding="utf-8") as file:
		lines=file.readlines()
	x=[]
	y=[]
	for line in lines:
		temp=line.strip().split(";")
		if "." not in temp[0]:
			continue
		x.append(float(temp[0]))
		y.append(float(temp[1]))
	if args.detail==True:
		low=np.percentile(y,2)
		high=np.percentile(y,98)
		mid=np.percentile(y,50)
		print("low 2% of all: "+str(low))
		print("high 2% of all: "+str(high))
		print("middle: "+str(mid))
		with open("detail.csv","a") as file:
			file.write("{},{},{}\n".format(low,high,mid))
	if args.check>0:
		check(y,53,3)
	sum,y=filter(y)
	print("Average in first 60 items: "+str(sum))
	y=scipy.signal.savgol_filter(y,53,3)
	v=cal_integral(x,y)
	d=cal_integral(x,v)
	print("Height: "+str(np.ptp(d)))
	if args.figure==False:
		sys.exit(0)
	plt.figure(1)
	plt.xlabel("Time(s)")
	plt.ylabel("Acceleration(m/s^2)")
	plt.plot(x,y)
	plt.show()
	
	plt.figure(2)
	plt.xlabel("Time(s)")
	plt.ylabel("Velocity(m/s)")
	plt.plot(x,v)
	plt.show()
	
	plt.figure(3)
	plt.xlabel("Time(s)")
	plt.ylabel("Displacement(m)")
	plt.plot(x,d)
	plt.show()
	