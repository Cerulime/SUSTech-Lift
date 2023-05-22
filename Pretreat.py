import argparse
import cv2
from tqdm import tqdm
from paddleocr import PaddleOCR

def GetFrame():
	success,image=video.read()
	if success:
		return image

if __name__=="__main__":
	parser=argparse.ArgumentParser()
	parser.add_argument("-r","--raw",type=str,help="input the raw file path as str")
	parser.add_argument("-o","--output",type=str,default="./",help="input the output path as str")
	parser.add_argument("-t","--threshold",type=int,default=170,help="input the threshold during binarization as int")
	parser.add_argument("-s","--start",type=int,default=0,help="input the start time in second as int")
	parser.add_argument("-e","--end",type=int,default=-1,help="input the end time in second as int")
	args=parser.parse_args()
	video=cv2.VideoCapture(args.raw)
	fps=int(video.get(cv2.CAP_PROP_FPS))
	tick=1.0/fps
	print("fps: ",fps)
	StartFrame=int(args.start*fps)
	print("StartFrame: ",StartFrame)
	EndFrame=int(args.end*fps)
	if EndFrame<0:
		EndFrame=int(video.get(7))
	print("EndFrame: ",EndFrame)
	video.set(cv2.CAP_PROP_POS_FRAMES,StartFrame)
	txt=[]
	ocr=PaddleOCR(use_gpu=False)
	for i in tqdm(range(StartFrame,EndFrame)):
		image=GetFrame()
		assert(image is not None)
		image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		image=255-cv2.threshold(image,args.threshold,255,cv2.THRESH_BINARY_INV)[1]
		'''
		cv2.imshow("image",image)
		cv2.waitKey(0)
		'''
		out=ocr.ocr(image)
		print(out)
		if len(out[-1])>0:
			print(out[-1][-1][-1][0])
			txt.append(out[-1][-1][-1][0])
		else:
			txt.append("er")
	with open(args.output+"result.csv","w") as file:
		file.write("Time(s);Weight(g)\n")
		for i in range(StartFrame,EndFrame):
			now=txt[i-StartFrame]
			now=now.replace(".","")
			now=now.replace("*","")
			now=now.replace(" ","")
			now=now.replace(":","2")
			now=now.replace("-","4")
			if len(now)<=4:
				continue
			now=now[:-2]+"."+now[-2:]
			file.write(str(round(tick*i,3))+";"+now+"\n")
		