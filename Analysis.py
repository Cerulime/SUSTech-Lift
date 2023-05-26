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
def fit(s,t,x,y):
    tempx=[]
    tempy=[]
    for i in range(0,len(x)):
        if s<=x[i] and x[i]<=t:
            tempx.append(x[i])
            tempy.append(y[i])
    z=np.polyfit(tempx,tempy,1)
    p=np.poly1d(z)
    print(p)
    yvals=np.polyval(z,tempx)
    r2=1-np.sum((tempy-yvals)**2)/np.sum((tempy-np.mean(tempy))**2)
    print(r2)
    return tempx,yvals
def logfit(s,t,x,y):
    tempx=[]
    tempy=[]
    for i in range(0,len(x)):
        if s<=x[i] and x[i]<=t:
            tempx.append(x[i]-s)
            tempy.append(y[i])
    flag=1
    if tempy[0]<0:
        flag=-1
        tempy=[-item for item in tempy]
    logy=np.log(tempy)
    z=np.polyfit(tempx,logy,1)
    print(z)
    yvals=[np.exp(z[1])*np.exp(z[0]*item) for item in tempx]
    fz=0
    fm=0
    for i in range(0,len(tempx)):
        fz+=(tempy[i]-yvals[i])**2
        fm+=(tempy[i]-np.mean(tempy))**2
    r2=1-fz/fm
    print(r2)
    return [item+s for item in tempx],[flag*item for item in yvals]

if __name__=="__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument("-r","--raw",type=str,required=True,help="input the raw file path as str")
	parser.add_argument("-cor","--cor",type=float,default=0.0,help="input the correction of the balance")
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
		mid=np.median(y)
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
	print("Height: "+str(np.sign(d[-1]-d[0]))+"*"+str(np.ptp(d)))
	if args.figure==False:
		sys.exit(0)
	plt.figure(1)
	plt.xlabel("Time(s)")
	plt.ylabel("Acceleration(m/s^2)")
	plt.plot(x,y,"r")
	'''
	print("linear fit1:")
	x1,y1=fit(1.8,2.7,x,y)
	plt.plot(x1,y1,"b")
	print("linear fit2:")
	x2,y2=fit(4.5,5.35,x,y)
	plt.plot(x2,y2,"b")
	print("linear fit3:")
	x3,y3=fit(10.2,11,x,y)
	plt.plot(x3,y3,"b")
	print("linear fit4:")
	x4,y4=fit(12.7,13.5,x,y)
	plt.plot(x4,y4,"b")
	print("logfit:")
	x5,y5=logfit(13.5,14.75,x,y)
	plt.plot(x5,y5,"b")
	'''
	plt.show()
	
	sys.exit(0)

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
	