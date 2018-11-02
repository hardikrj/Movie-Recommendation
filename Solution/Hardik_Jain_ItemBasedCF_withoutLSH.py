from pyspark import SparkContext
import sys,random,time

def lineSplit(line):
	l1 = line.split(",")
	return ((int(l1[0]),int(l1[1])),float(l1[2]))

def listSplit(line):
	b = set()
	b.add(int(line[0][0]))
	return (int(line[0][1]),b)

start = time.time()
inputFile = sys.argv[1]

sc = SparkContext('local[*]','LSH')

text_file = sc.textFile(inputFile)
header = text_file.first()
train_data = text_file.filter(lambda line: line != header).cache()
training_data = train_data.flatMap(lambda line: line.split('\n')).map(lambda word: lineSplit(word))

test_file = sc.textFile(sys.argv[2])
header1 = test_file.first()
test_data = test_file.filter(lambda line: line != header1)
test_data = test_data.flatMap(lambda line: line.split('\n')).map(lambda word: ((int(word.split(",")[0]),int(word.split(",")[1])),1))

train_data = training_data.subtractByKey(test_data)

user_movie_rating = {}
movie_dict = {}
for i in train_data.collect():
	user_movie_rating[i[0]] = i[1]
	if i[0][1] in movie_dict:
		movie_dict[i[0][1]].append(i[0][0])
	else:
		movie_dict[i[0][1]] = [i[0][0]]

movie_rdd = train_data.map(lambda word: listSplit(word)).reduceByKey(lambda x,y: x.union(y)).sortByKey().cache()
movie_list = movie_rdd.collect()

user_list = train_data.map(lambda word: (word[0][0],[word[0][1]])).reduceByKey(lambda x,y: x+y).sortByKey().collect()
user_dict = {i[0]:i[1] for i in user_list}

w = {}
l = len(movie_list)
for i in range(0,l):
	m1 = movie_list[i][0]
	setA = movie_list[i][1]
	for j in range(i+1, l):
		m2 = movie_list[j][0]
		setB = movie_list[j][1]
		intersect = list(setA.intersection(setB))
		if len(intersect)==0:
			w[(m1,m2)] = 0
			w[(m2,m1)] = 0
			continue
		sum1 = 0
		sum2 = 0
		for id in intersect:
			sum1 += user_movie_rating[(id,m1)]
			sum2 += user_movie_rating[(id,m2)]
		r1 = sum1/len(intersect)
		r2 = sum2/len(intersect)
		num_sum = 0
		den_sum1 = 0
		den_sum2 = 0
		for id in intersect:
			num_sum += ((user_movie_rating[(id,m1)]-r1)*(user_movie_rating[(id,m2)]-r2))
			den_sum1 += (user_movie_rating[(id,m1)]-r1)**2
			den_sum2 += (user_movie_rating[(id,m2)]-r2)**2
		if den_sum1==0 or den_sum2==0:
			w[(m1,m2)] = 0
			w[(m2,m1)] = 0
		else:
			val = num_sum/((den_sum1**0.5)*(den_sum2**0.5))
			w[(m1,m2)] = val
			w[(m2,m1)] = val

prediction = {}
for i in test_data.collect():
	u = i[0][0]
	m = i[0][1]
	num = 0
	den = 0
	for n in user_dict[u]:
		if (m,n) in w and w[(m,n)]>0:
			num += user_movie_rating[(u,n)]*w[(m,n)]
			den += abs(w[(m,n)])
		else:
			break
	if den!=0:
		prediction[(u,m)] = num/den
	else:
		sum=0
		for n in user_dict[u]:
			sum+=user_movie_rating[(u,n)]
		prediction[(u,m)] = sum/len(user_dict[u])

prediction = [(k,v) for k,v in prediction.items()]

prediction = sc.parallelize(prediction).sortByKey().collect()

training_data = training_data.subtractByKey(train_data).collect()

true_dataset = {}
for item in training_data:
	true_dataset[item[0]] = item[1]

outFile = open("./Hardik_Jain_ItemBasedCF_withoutLSH.txt","w")
count1,count2,count3,count4,count5 = 0,0,0,0,0
sum = 0
for i in prediction:
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
rmse = (sum/len(prediction))**0.5

print(">=0 and <1: "+str(count1))
print(">=1 and <2: "+str(count2))
print(">=2 and <3: "+str(count3))
print(">=3 and <4: "+str(count4))
print(">=4: "+str(count5))
print("RMSE: "+str(rmse))
print("Time: "+str(time.time()-start))