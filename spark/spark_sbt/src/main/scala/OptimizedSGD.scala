import java.security.MessageDigest

import breeze.linalg.DenseVector
import org.apache.spark.examples.ml.TestALS.Movie
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable.HashMap
import scala.collection.mutable.Map
import scala.util.Random

///home/Downloads/spark/bin/spark-submit --class TestGradientDescent target/scala-2.10/spark_sbt_2.10-1.0.jar
/**
  * Created by les on 19/02/16.
  */
object OptimizedSGD {
    // val rand = RandomUtils.getRandom(42L)
    val rand = new Random(42L)
    val learningRate = 0.1
    val preventOverFitting = 0.1
    val randomNoise = 0.1
    val learningRateDecay = 1.0

    val user_bias_index = 1
    val item_bias_index = 2
    val feature_offset = 3

    val bias_learning_rate = 0.5
    val biasReg = 0.1
    val numFeatures = 10
    val numIterations = 50


    def main(args: Array[String]) {

        val conf = new SparkConf().setAppName(s"TestGradient")
        val sc = new SparkContext(conf)
        sc.setLogLevel("WARN")
        // Load training data in LIBSVM format.
        val sqlContext = new SQLContext(sc)
        import sqlContext.implicits._


        //test on efficency
        /**
          * 1. use dot product between densevectors, great improvement for big numbers 39138 vs 4948115
          * 2. apply has the same cost
          * 3. dict is super faster then using rdd 132080 vs 48
          *
          */


        val ratingsPath = "ml-latest/ratings-1m.dat"
        //val movies = sc.textFile("ml-latest/movies.csv").map(Movie.parseMovie).toDF()

        val ratings = sc.textFile(ratingsPath).map(Rating.parseRating).cache()
        val totUsers = ratings.map(_.userId).distinct().takeOrdered(500)
        //val totUsers = ratings.map(_.userId).distinct().collect()
        val movies = ratings.map(_.movieId).distinct()

        val splits = ratings.randomSplit(Array(0.8, 0.2), 0L)

        val globalAvg = ratings.map(x => x.rating).collect().sum / ratings.count().toInt

        val (cachedUsers, cachedItems,userDict) = cache(ratings, totUsers)


        val training = splits(0).cache()


        val numUsers = ratings.map(_.userId).distinct().count()

        var cachedRatings = new HashMap[String, Double]

        ratings.collect.foreach {
            r => val digest = md5(r.userId.toString + " " + r.movieId.toString)
                cachedRatings += digest -> r.rating.toDouble
        }





        var userMap= new HashMap[Int,VectorFactorItem]
        totUsers.foreach{
            user =>
                val tmpVect = DenseVector.fill[Double](numFeatures){Random.nextDouble() * randomNoise}
                userMap += user -> new VectorFactorItem(globalAvg.toDouble,0.0,1.0,numFeatures,joinVectors(Array(globalAvg,0.0,1.0),tmpVect))
        }
        var itemMap =  new HashMap[Int, Array[Double]]
        movies.collect.foreach{
            item =>
                val tmpVect = Array.fill[Double](numFeatures){Random.nextDouble() * randomNoise}
                itemMap += item -> joinVectors(Array(1.0,1.0,0.0),tmpVect)
        }


        val userMatrix = totUsers.map {
            user =>
                val v1 = Array(globalAvg.toDouble, 0.0, 1.0)
                joinVectors(v1, Array.fill(numFeatures) {
                    rand.nextDouble() * randomNoise
                })
        }

        val itemMatrix = movies.map {
            item =>
                val v2 = Array(1.0, 1.0, 0.0)
                joinVectors(v2, Array.fill(numFeatures) {
                    rand.nextDouble() * randomNoise
                })
        }.collect()



        var currentLearningRate = learningRate

        println("Real value for user with item " + cachedUsers.apply(0), cachedItems.apply(0) + " value: " + ratings.filter(f => f.movieId == cachedItems.apply(0) && f.userId == cachedUsers.apply(0)).first().rating)
        println("Current prediction")
        println(predictRating(userMatrix.apply(cachedUsers.apply(0)), itemMatrix.apply(cachedItems.apply(0))))


        for (iteration <- 0 to numIterations) {
            println("rmse training "+rmse(userMatrix,itemMap,training,cachedUsers,cachedItems))
            println("rmse test "+rmse(userMatrix,itemMap,splits(1),cachedUsers,cachedItems))
            println(predictRating(userMatrix.apply(cachedUsers.apply(0)), itemMatrix.apply(cachedItems.apply(0))))

            updateUser(userMatrix,itemMap,cachedRatings,cachedUsers,cachedItems,currentLearningRate)

            /**
              * to remember:
              * take slices of the array using a dictionary to remember the number of preferences dict[id_user] = preferencesCount
              */
            currentLearningRate *= learningRateDecay


        }

        testOutput(userMatrix,itemMatrix,ratings,cachedUsers,cachedItems)
    }



    def updateUser(userMatrix:Array[Array[Double]],itemMatrix:HashMap[Int,Array[Double]],ratings:HashMap[String,Double], cachedUsers:Array[Int], cachedItems:Array[Int], currentLearningRate:Double): Unit= {
        userMatrix.zipWithIndex.foreach {
            userRow =>
                val loopIndex = userRow._2
                val uid = cachedUsers.apply(loopIndex)
                val iid = cachedItems.apply(loopIndex)
                val userVector = userMatrix.apply(uid)
                val itemVector = itemMatrix.apply(iid)


                val pr_rating = predictRating(userVector, itemVector)
                val value = ratings.apply(uid.toString+" "+iid.toString)
                val err = value - pr_rating

                userVector.update(user_bias_index, userVector.apply(user_bias_index) + bias_learning_rate * (err - biasReg * preventOverFitting * userVector.apply(user_bias_index)))
                itemVector.update(item_bias_index, itemVector.apply(item_bias_index) + bias_learning_rate * (err - biasReg * preventOverFitting * itemVector.apply(item_bias_index)))


                for (featureIndex <- feature_offset to numFeatures) {
                    val uF = userVector.apply(featureIndex)
                    val iF = itemVector.apply(featureIndex)

                    val deltaUserFeature = err * iF - preventOverFitting * uF
                    userVector.update(featureIndex, userVector.apply(featureIndex) + currentLearningRate * deltaUserFeature)


                    val deltaItemFeature = err * uF - preventOverFitting * iF
                    itemVector.update(featureIndex, itemVector.apply(featureIndex) + currentLearningRate * deltaItemFeature)

                }


            }

    }
    def rmse (userMatrix:Array[Array[Double]],itemMatrix:HashMap[Int,Array[Double]],ratings:RDD[Rating], cachedUsers:Array[Int], cachedItems:Array[Int]): Double={
        var numRatings = 0
        val res = cachedUsers.zipWithIndex.map{
            userRow=>
                val loopIndex = userRow._2
                val uid = cachedUsers.apply(loopIndex)
                val iid = cachedItems.apply(loopIndex)
                val v1 = userMatrix.apply(uid)
                val v2 = itemMatrix.apply(iid)

                val pr_rating = predictRating(v1,v2)
                val temp = ratings.filter(r=> r.movieId == iid && r.userId == uid)

                if(temp.collect().length > 0) {
                    val realValue = temp.first().rating
                    numRatings += 1
                    math.pow(realValue - pr_rating,2)
                }
                else
                    0



        }.sum

        Math.sqrt(res / numRatings)
    }
    def testOutput(userMatrix:Array[Array[Double]],itemMatrix:Array[Array[Double]], ratings:RDD[Rating], cachedUsers:Array[Int], cachedItems:Array[Int]): Unit={
        println("Test prediction on all users one item each")

        userMatrix.zipWithIndex.foreach{
            userRow=>
                val loopIndex = userRow._2
                val uid = cachedUsers.apply(loopIndex)
                val iid = cachedItems.apply(loopIndex)

                val pr_rating = predictRating(userMatrix.apply(uid),itemMatrix.apply(iid))
                val realValue = ratings.filter(r=> r.movieId == iid && r.userId == uid).first().rating

                println("Prediction for user "+uid+" for item "+iid+" predicted value "+pr_rating+" real value "+realValue)
        }
    }
    def joinVectors(v1:Array[Double],v2:Array[Double]): Array[Double] ={
        Array(v1.toArray,v2.toArray).flatten
    }
    def predictRating(userVector:DenseVector[Double], itemVector:DenseVector[Double]): Double ={
        userVector dot itemVector
    }
    def predictRating(users:Array[Double], items:Array[Double]): Double ={

        assert(users.size == items.size)
        /** val tempItems = items.slice(3, items.size)
        users.slice(3,users.size).zipWithIndex.map{
            u=> u._1 * tempItems.apply(u._2)
        }.sum**/

        users.zipWithIndex.map{
            u=> u._1 * items.apply(u._2)
        }.sum

    }
    def cache(ratings: RDD[Rating], users:Array[Int]): (Array[Int],Array[Int],HashMap[Int,Int]) ={

        val numPreferences = ratings.map(x=>x.rating).collect().size
        val cachedItems = Array.fill(numPreferences)(0)
        val cachedUsers = Array.fill(numPreferences)(0)

        var uDict = new HashMap[Int,Int]

        var index = 0
        var offset = 0

        users.foreach{
            uid=>
                /**too bad this doesn't work. cachedItems and cachedUsers are not updated throughout the cycle and reinitialized each time
                  * val sum= ratings.filter(r=> r.userId == uid).map{
                    rating=>
                        cachedUsers.update(index+offset, uid)
                        cachedItems.update(index+offset,rating.movieId)
                        //index.update(0, index.apply(0)+1)
                        index+=1

                }.collect().size
                offset+=sum*/
                val rating = ratings.filter(r=> r.userId == uid).collect()

                for(rid <- rating.indices){
                    cachedUsers.update(index, uid)
                    cachedItems.update(index,rating.apply(rid).movieId)
                    index+=1
                }
                uDict += uid -> rating.length
                if(index == Int.MaxValue -1)
                    println("Warning overflow!")

        }


        shuffleCachedItems(cachedUsers,cachedItems)
        (cachedUsers,cachedItems,uDict)

    }
    def shuffleCachedItems(users:Array[Int],items:Array[Int]): Unit={

        for(i <- ((users.size -1) to 0).reverse){
            val toSwap = rand.nextInt(i + 1)

            //swapping the values
            val tempU = users.apply(i)
            val tempI = items.apply(i)

            users.update(i,users.apply(toSwap))
            items.update(i,items.apply(toSwap))

            users.update(toSwap,tempU)
            items.update(toSwap,tempI)

        }

    }
    def md5(s: String): String = {
        MessageDigest.getInstance("MD5").digest(s.getBytes).map("%02x".format(_)).mkString
    }
    def joinVectors(v1:Array[Double],v2:DenseVector[Double]): DenseVector[Double] ={
        val lastArray = Array(v1,v2.toArray).flatten
        DenseVector(lastArray)
    }
    
}
