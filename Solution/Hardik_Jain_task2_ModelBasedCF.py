from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import sys,time

def lineSplit(line):
	l1 = line.split(",")
	if len(l1)==2:
		return ((int(l1[0]),int(l1[1])),1)	
	return ((int(l1[0]),int(l1[1])),float(l1[2]))

start = time.time()
ratingFile = sys.argv[1]
testingFile = sys.argv[2]

sc = SparkContext('local[*]','recommendation')
rate_file = sc.textFile(ratingFile)
header = rate_file.first()
train_data = rate_file.filter(lambda line: line != header)
training_data = train_data.flatMap(lambda line: line.split('\n')).map(lambda word: lineSplit(word))

test_file = sc.textFile(testingFile)
header1 = test_file.first()
test_data = test_file.filter(lambda line: line != header1)
test_data = test_data.flatMap(lambda line: line.split('\n')).map(lambda word: lineSplit(word))

train_data = training_data.subtractByKey(test_data)

ratings=train_data.map(lambda l: Rating(l[0][0],l[0][1],l[1]))

rank = 6
itn = 10
model = ALS.train(ratings, rank, itn)

test_data = test_data.map(lambda l: (l[0][0],l[0][1]))
predictions = model.predictAll(test_data).map(lambda r: ((r[0], r[1]), r[2])).sortByKey().cache()

x = training_data.subtractByKey(train_data).collect()
true_dataset = {}
for item in x:
	true_dataset[item[0]] = item[1]

prediction_set = set(predictions.map(lambda l : (l[0][0],l[0][1])).collect())

outFile = open("./Hardik_Jain_ModelBasedCF.txt","w")
count1,count2,count3,count4,count5 = 0,0,0,0,0
sum = 0
for i in predictions.collect():
	outFile.write(str(i[0][0])+","+str(i[0][1])+","+str(i[1])+"\n")
	dif = abs(i[1]-true_dataset[i[0]])
	if dif<1:
		count1+=1
	elif dif<2:
		count2+=1
	elif dif<3:
		count3+=1
	elif dif<4:
		count4+=1
	else:
		count5+=1
	sum += dif**2
outFile.close()
rmse = (sum/len(prediction_set))**0.5

print(len(prediction_set))
print(">=0 and <1: "+str(count1))
print(">=1 and <2: "+str(count2))
print(">=2 and <3: "+str(count3))
print(">=3 and <4: "+str(count4))
print(">=4: "+str(count5))
print("RMSE: "+str(rmse))
print("Time: "+str(time.time()-start))