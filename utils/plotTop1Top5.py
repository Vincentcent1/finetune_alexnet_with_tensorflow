import sys
import matplotlib.pyplot as plt
'''
Example validation:
Validation with 0.0% occlusion
accuracy type occlusionRatio correctPrediction numberOfImages accuracy
top1 0.0 27207 50000 0.54414
top5 0.0 39010 50000 0.7802
'''

textFile = sys.argv[1]
title = sys.argv[2]
top1Arr = []
top5Arr = []


with open(textFile, 'r') as f:
	name = f.readline().strip('\n')
	for line in f:
		lineArr = line.split()
		if lineArr[0] == 'top1':
			top1Arr.append((lineArr[0], lineArr[1], lineArr[2], lineArr[3], lineArr[4]))
		elif lineArr[0] == 'top5':
			top5Arr.append((lineArr[0], lineArr[1], lineArr[2], lineArr[3], lineArr[4]))
		else:
			continue
top1x = [top1Tuple[1] for top1Tuple in top1Arr]
top1y = [1 - float(top1Tuple[4]) for top1Tuple in top1Arr]
top5x = [top5Tuple[1] for top5Tuple in top5Arr]
top5y = [1 - float(top5Tuple[4]) for top5Tuple in top5Arr]
plt.plot(top1x, top1y, 'r.', label="top1")
plt.plot(top5x, top5y, 'b.', label="top5")

for a,b in zip(top1x, top1y):
	plt.text(a, b, str(round(b,3)))
for a,b in zip(top5x, top5y):
	plt.text(a, b, str(round(b,3)))
plt.legend(loc='best')
plt.xlabel('Occlusion')
plt.ylabel('Error-rate')
plt.title(title, loc = 'left')
# plt.show()
plt.savefig(name + "error" + ".png")
plt.gcf().clear()

top1x = [top1Tuple[1] for top1Tuple in top1Arr]
top1y = [float(top1Tuple[4]) for top1Tuple in top1Arr]
top5x = [top5Tuple[1] for top5Tuple in top5Arr]
top5y = [float(top5Tuple[4]) for top5Tuple in top5Arr]
plt.plot(top1x, top1y, 'r.', label="top1")
plt.plot(top5x, top5y, 'b.', label="top5")

for a,b in zip(top1x, top1y):
	plt.text(a, b, str(round(b,3)))
for a,b in zip(top5x, top5y):
	plt.text(a, b, str(round(b,3)))
plt.legend(loc='best')
plt.xlabel('Occlusion')
plt.ylabel('Accuracy')
plt.title(title, loc = 'left')
# plt.show()
plt.savefig(name + "accuracy" + ".png")

