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
		while itn<num_hash:
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

	output = [(k,v) for k,v in output.items() if v>=0.5]

	return output		

start = time.time()
inputFile = sys.argv[1]

sc = SparkContext('local[*]','LSH')

text_file = sc.textFile(inputFile)
header = text_file.first()
data = text_file.filter(lambda line: line != header)

movie_rdd = data.flatMap(lambda line: line.split('\n')).map(lambda word: lineSplit(word)).reduceByKey(lambda x,y: x.union(y)).cache()
movie_list = movie_rdd.collect()

sorted_rdd = movie_rdd.map(lambda x: (x[0],sorted(x[1]))).sortByKey()
sorted_list = sorted_rdd.collect()

a = random.sample(range(1,100),60)
b = random.sample(range(1,100),60)
m = 65
num_hash = 60

map_output = sorted_rdd.mapPartitions(signature_hash).sortByKey().collect()

signatures = {}
for item in map_output:
	i=0
	while i<num_hash:
		key = "h"+str(i+1)
		if key in signatures:
			signatures[key].append(item[1][i])
		else:
			signatures[key] = [item[1][i]]
		i+=1

signatures = [(k,v) for k,v in signatures.items()]

count=0
sign_band=[]
while count<num_hash :
	sign_band.append([signatures[count],signatures[count+1],signatures[count+2]])
	count+=3

map_rdd = sc.parallelize(sign_band,20)

reduce_output = map_rdd.mapPartitions(bucket_hash).reduceByKey(lambda a,b : a.union(b))
reduce_output = reduce_output.map(lambda x: (x[0],sorted(x[1]))).sortByKey()

jac_sim = reduce_output.mapPartitions(similarity).reduceByKey(lambda a,b : a).sortByKey().collect()

outFile = open("./Hardik_Jain_SimilarMovies_Jaccard.txt","w")
for i in jac_sim:
	outFile.write(str(i[0][0])+","+str(i[0][1])+","+str(i[1])+"\n")
outFile.close()

print(time.time()-start)