from pyspark import SparkContext
import sys,random,time

def lineSplit(line):
	l1 = line.split(",")
	a = int(l1[1])
	b = set()
	b.add(int(l1[0]))
	return (a,b)

def listSplit(word):
	w = word.split(",")
	return (int(w[0]),int(w[1]))

def signature_hash(iterator):
	itn = 0
	part = [i for i in iterator]
	hashes = {}
	
	for i in part:
		x = i[1][0]
		itn = 0
		hashes[i[0]] = []
		while itn<60:
			h = ((a[itn]*x)+b[itn])%m
			hashes[i[0]].append(h)
			itn+=1
	
	hashes = [(k,v) for k,v in hashes.items()]

	return hashes

def bucket_hash(iterator):
	part = []
	for i in iterator:
		for j in i:
			part.append(j)
	buckets = {}
	l = len(part[1][1])
	for i in range(0,l):
		total = part[0][1][i] + part[1][1][i] + part[2][1][i]
		h = ((2*total)+5)%m
		if h in buckets:
			buckets[h].add(i+1)
		else:
			buckets[h] = set()
			buckets[h].add(i+1)
	
	buckets = [(k,v) for k,v in buckets.items()]

	return buckets
	
def similarity(iterator):
	part =  [i for i in iterator]
	movies = { i[0]:i[1] for i in movie_list}
	output = {}

	for i in part:
		l = len(i[1])
		for x in range(0,l):
			m1 = sorted_list[i[1][x]-1][0]
			setA = movies[m1]
			for y in range(x+1,l):
				m2 = sorted_list[i[1][y]-1][0]
				if (m1,m2) not in output:
					setB = movies[m2]
					jac=len(setA.intersection(setB))/len(setA.union(setB))
					output[(m1,m2)] = jac

	output = [(k,v) for k,v in output.items() if v>=0.1]

	return output		

start = time.time()
inputFile = sys.argv[1]

sc = SparkContext('local[*]','LSH')

text_file = sc.textFile(inputFile)
header = text_file.first()
data = text_file.filter(lambda line: line != header).cache()

movie_rdd = data.flatMap(lambda line: line.split('\n')).map(lambda word: lineSplit(word)).reduceByKey(lambda x,y: x.union(y)).cache()
movie_list = movie_rdd.collect()

sorted_rdd = movie_rdd.map(lambda x: (x[0],sorted(x[1]))).sortByKey()
sorted_list = sorted_rdd.collect()

a = random.sample(range(1,100),60)
b = random.sample(range(1,100),60)
m = 65

map_output = sorted_rdd.mapPartitions(signature_hash).sortByKey().collect()

signatures = {}
for item in map_output:
	i=0
	while i<60:
		key = "h"+str(i+1)
		if key in signatures:
			signatures[key].append(item[1][i])
		else:
			signatures[key] = [item[1][i]]
		i+=1

signatures = [(k,v) for k,v in signatures.items()]

numHashes = len(signatures)
count,i=0,0
sign_band=[]
while count<numHashes:
	sign_band.append([signatures[count],signatures[count+1],signatures[count+2]])
	count+=3
	i+=1

map_rdd = sc.parallelize(sign_band,20)

reduce_output = map_rdd.mapPartitions(bucket_hash).reduceByKey(lambda a,b : a.union(b))
reduce_output = reduce_output.map(lambda x: (x[0],sorted(x[1])))

jac_sim = reduce_output.mapPartitions(similarity).reduceByKey(lambda a,b : a).collect()

jac_dict = {}
for i in jac_sim:
	jac_dict[(i[0][0],i[0][1])]=i[1]
	jac_dict[(i[0][1],i[0][0])]=i[1]

test_file = sc.textFile(sys.argv[2])
header1 = test_file.first()
test_data = test_file.filter(lambda line: line != header1)
test_data = test_data.flatMap(lambda line: line.split('\n')).map(lambda word: ((int(word.split(",")[0]),int(word.split(",")[1])),1))

data = data.map(lambda l: ((int(l.split(",")[0]),int(l.split(",")[1])),float(l.split(",")[2])))

data_dict = {(i[0][0],i[0][1]):i[1] for i in data.collect()}

data = data.subtractByKey(test_data)

user_list = data.map(lambda l: (l[0][0],[l[0][1]])).reduceByKey(lambda a,b: a+b).collect()
user_list = {i[0]:i[1] for i in user_list}

prediction = {}
for i in test_data.collect():
	u = i[0][0]
	m = i[0][1]
	num = 0
	den = 0
	for n in user_list[u]:
		if (m,n) in jac_dict:
			num += data_dict[(u,n)]*jac_dict[(m,n)]
			den += abs(jac_dict[(m,n)])
	if den!=0:
		prediction[(u,m)] = num/den
	else:
		sum=0
		for n in user_list[u]:
			sum+=data_dict[(u,n)]
		prediction[(u,m)] = sum/len(user_list[u])

outFile = open("./Hardik_Jain_ItemBasedCF.txt","w")
count1,count2,count3,count4,count5 = 0,0,0,0,0
sum = 0
for i in prediction:
	outFile.write(str(i[0])+","+str(i[1])+","+str(prediction[i])+"\n")
	dif = abs(prediction[i]-data_dict[i])
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