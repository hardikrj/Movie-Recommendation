from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import sys,time

def lineSplit(line):
	l1 = line.split(",")
	if len(l1)==2:
		return ((int(l1[0]),int(l1[1])),1)	
	return ((int(l1[0]),int(l1[1])),float(l1[2]))

def listSplit(line):
	b = set()
	b.add(int(line[0][1]))
	return (int(line[0][0]),b)

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

user_movie_rating = {}
user_dict = {}
for i in train_data.collect():
	user_movie_rating[i[0]] = i[1]
	if i[0][0] in user_dict:
		user_dict[i[0][0]].append(i[0][1])
	else:
		user_dict[i[0][0]] = [i[0][1]]

user_list = train_data.map(lambda word: listSplit(word)).reduceByKey(lambda x,y: x.union(y)).sortByKey().collect()

w = {}
r = {}
l = len(user_list)
for i in range(0,l):
	u1 = user_list[i][0]
	setA = user_list[i][1]
	for j in range(i+1, l):
		u2 = user_list[j][0]
		setB = user_list[j][1]
		intersect = list(setA.intersection(setB))
		if len(intersect)==0:
			w[(u1,u2)] = 0
			w[(u2,u1)] = 0
			r[(u1,u2)] = 0
			r[(u2,u1)] = 0	
			continue
		sum1 = 0
		sum2 = 0
		for id in intersect:
			sum1 += user_movie_rating[(u1,id)]
			sum2 += user_movie_rating[(u2,id)]
		r1 = sum1/len(intersect)
		r2 = sum2/len(intersect)
		r[(u1,u2)] = r2
		r[(u2,u1)] = r1
		num_sum = 0
		den_sum1 = 0
		den_sum2 = 0
		for id in intersect:
			num_sum += ((user_movie_rating[(u1,id)]-r1)*(user_movie_rating[(u2,id)]-r2))
			den_sum1 += (user_movie_rating[(u1,id)]-r1)**2
			den_sum2 += (user_movie_rating[(u2,id)]-r2)**2
		if den_sum1==0 or den_sum2==0:
			w[(u1,u2)] = 0
			w[(u2,u1)] = 0
		else:
			val = num_sum/((den_sum1**0.5)*(den_sum2**0.5))
			w[(u1,u2)] = val
			w[(u2,u1)] = val

predictions = []

for i in test_data.collect():
	u = i[0][0]
	m = i[0][1]
	Ru = 0
	sum = 0
	num = 0
	den = 0
	for mid in user_dict[u]:
		sum += user_movie_rating[(u,mid)]
	Ru = sum/len(user_dict[u])
	for user in user_list:
		if m in user[1]:
			num += (user_movie_rating[(user[0],m)]-r[(u,user[0])])*w[(u,user[0])]
			den += abs(w[(u,user[0])])
	if den!=0:
		predictions.append(((u,m),(Ru+(num/den))))
	else:
		predictions.append(((u,m),Ru))

predictions = sc.parallelize(predictions).sortByKey().collect()

training_data = training_data.subtractByKey(train_data).collect()

true_dataset = {}
for item in training_data:
	true_dataset[item[0]] = item[1]

outFile = open("./Hardik_Jain_UserBasedCF.txt","w")
count1,count2,count3,count4,count5 = 0,0,0,0,0
sum = 0
for i in predictions:
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
rmse = (sum/len(predictions))**0.5

print(">=0 and <1: "+str(count1))
print(">=1 and <2: "+str(count2))
print(">=2 and <3: "+str(count3))
print(">=3 and <4: "+str(count4))
print(">=4: "+str(count5))
print("RMSE: "+str(rmse))
print("Time: "+str(time.time()-start))
