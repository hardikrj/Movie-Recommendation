import org.apache.spark.{SparkConf, SparkContext}
import java.io._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating

object JaccardLSH{
  def main(args: Array[String]): Unit = {
    val start = System.nanoTime()
    val conf = new SparkConf()
    conf.setAppName("Assignment3")
    conf.setMaster("local[*]")

    val sc = new SparkContext(conf)
    val text = sc.textFile(args(0))
    val header=text.first()
    val data = text.filter(row => row!=header)

    val movie_rdd = data.flatMap(line => line.split("\n")).map(word => (word.split(",")(1).toInt,Set(word.split(",")(0).toInt))).reduceByKey((a,b)=>a union b).cache()
    val movie_list = movie_rdd.collect()

    var movies:Map[Int,Set[Int]] = Map()
    for(x <- movie_list){
      movies += (x._1 -> x._2)
    }

    val sorted_rdd = movie_rdd.map(x => (x._1,x._2.toList.sorted)).sortByKey()
    val sorted_list = sorted_rdd.collect()

    val r = scala.util.Random
    var a:List[Int] = List()
    var b:List[Int] = List()
    for(i <- 1 to 60) {
      a = r.nextInt(100) :: a
      b = r.nextInt(100) :: b
    }
    var m = 65
    var num_hash = 60

    var map_output = sorted_rdd.mapPartitions (iterator => {
      var part = iterator.toList
      var itn = 0
      var hashes:Map[Int,List[Int]] = Map()

      for(i <- part){
        var x = i._2(0)
        itn = 0
        hashes += (i._1 -> List())
        while (itn<60){
          var h = ((a(itn)*x)+b(itn))%m
          hashes += (i._1 -> (hashes(i._1) :+ h))
          itn+=1
        }
      }

      hashes.toIterator
    })

    map_output = map_output.sortByKey()

    var signatures:Map[String,List[Int]] = Map()

    for(item <- map_output.collect()){
      var i = 0
      while (i<num_hash){
        var key = "h"+(i+1)
        if (signatures contains key){
          signatures += (key -> (signatures(key) :+ item._2(i)))
        }
        else{
          signatures += (key -> List(item._2(i)))
        }
        i+=1
      }
    }

    var count = 0

    var sign_band:List[List[(String,List[Int])]] = List()
    while(count<num_hash){
      sign_band = sign_band :+ List(("h"+(count+1),signatures("h"+(count+1))),("h"+(count+2),signatures("h"+(count+2))),("h"+(count+3),signatures("h"+(count+3))))
      count += 3
    }

    val map_rdd = sc.parallelize(sign_band,20)

    var reduce_output = map_rdd.mapPartitions(iterator => {
      var part = iterator.toList
      var buckets:Map[Int,Set[Int]] = Map()
      var l = part(0)(0)._2.length

      for(i <- 0 to l-1){
        var total:Int = part(0)(0)._2(i)  + part(0)(1)._2(i) + part(0)(2)._2(i)
        var h = ((2*total)+5)%65

        if (buckets contains h){
          buckets += (h -> (buckets(h) ++ Set(i+1)))
        }
        else{
          buckets += (h -> Set(i+1))
        }
      }

      buckets.toIterator
    })

    var red = reduce_output.reduceByKey((a,b)=>a union b)
    var reduce = red.map(x => (x._1,x._2.toList.sorted)).sortByKey()

    var jac_sim = reduce.mapPartitions(iterator => {
      var part = iterator.toList

      var output:Map[(Int,Int),Float] = Map()

      for(i <- part) {
        var l = i._2.length
        for (x <- 0 until l) {
          var m1 = sorted_list(i._2(x) - 1)._1
          var setA = movies(m1)
          for (y <- x + 1 until l) {
            var m2 = sorted_list(i._2(y) - 1)._1
            if (!(output contains(m1, m2))) {
              var setB = movies(m2)
              var jac = (setA intersect setB).size.toFloat / (setA union setB).size
              if (jac >= 0.5) {
                output += ((m1, m2) -> jac)
              }
            }
          }
        }
      }
      output.toIterator
    })

    var jac_out = jac_sim.reduceByKey((a,b)=>a).sortByKey().collect()

    val outFile = new PrintWriter(new File("./Hardik_Jain_SimilarMovies_Jaccard.txt"))
    for(x<-jac_out){
      outFile.write(x._1._1 + "," + x._1._2 + "," + x._2 + "\n")
    }
    outFile.close()

    println("Time: "+(System.nanoTime()-start)/1000000000.0)

  }

}

object ModelBasedCF{
  def main(args: Array[String]): Unit = {
    val start = System.nanoTime()
    val conf = new SparkConf()
    conf.setAppName("Assignment3")
    conf.setMaster("local[*]")

    val sc = new SparkContext(conf)

    val text = sc.textFile(args(0))
    val header=text.first()
    var training_data = text.filter(row => row!=header).flatMap(line => line.split('\n')).map(word => ((word.split(",")(0).toInt,word.split(",")(1).toInt),word.split(",")(2).toFloat))

    val test_file = sc.textFile(args(1))
    val header1 = test_file.first()
    val testing_data = test_file.filter(row => row!=header1).flatMap(line => line.split('\n')).map(word => ((word.split(",")(0).toInt,word.split(",")(1).toInt),1))

    val train_data = training_data.subtractByKey(testing_data)

    val model = ALS.train(train_data.map(l => Rating(l._1._1,l._1._2,l._2)),6,10)

    val predictions = model.predict(testing_data.map(l => (l._1._1,l._1._2))).map { case Rating(user, product, rate) => ((user, product), rate) }.sortByKey()

    val outFile = new PrintWriter(new File("./Hardik_Jain_ModelBasedCF.txt"))

    for (x <- predictions.collect()){
      outFile.write(x._1._1 + "," + x._1._2 + "," + x._2 + "\n")
    }
    outFile.close()

    println("Time: "+(System.nanoTime()-start)/1000000000.0)
  }

}