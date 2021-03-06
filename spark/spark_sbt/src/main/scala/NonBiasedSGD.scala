import java.security.MessageDigest

import breeze.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable.HashMap
import scala.collection.{Map, breakOut}
import scala.util.Random

///home/Downloads/spark/bin/spark-submit --class TestGradientDescent target/scala-2.10/spark_sbt_2.10-1.0.jar
/**
  * Created by les on 19/02/16.
  */
object NonBiasedSGD {

    // val rand = RandomUtils.getRandom(42L)
    val rand = new Random(42L)
    val learningRate = 0.1
    val preventOverFitting = 0.1
    val randomNoise = 0.1
    val learningRateDecay = 1.0

    val user_bias_index = 1
    val item_bias_index = 2
    val feature_offset = 3

    val bias_learning_rate = 0.01
    //val bias_learning_rate = 0.5
    val biasReg = 0.1
    val numFeatures = 20
    val numIterations = 25
    var pool = java.util.concurrent.Executors.newFixedThreadPool(1500)
    var threadPool = List




    def main(args: Array[String]) {

        val conf = new SparkConf().setAppName(s"TestGradient")
        val sc = new SparkContext(conf)
        sc.setLogLevel("WARN")
        val sqlContext = new SQLContext(sc)


        //val ratingsPath  = "ml-latest/ratings.csv"
        val ratingsPath  = "ml-latest/ratings-1m.dat"
        //val movies = sc.textFile("ml-latest/movies.csv").map(Movie.parseMovie)


        val ratings = sc.textFile(ratingsPath).map(Rating.parseRating).cache()


        val totUsers = ratings.map(_.userId).distinct().collect()
        // val totUsers = ratings.map(_.userId).distinct().takeOrdered(1000)
        val movies = ratings.map(_.movieId).distinct()
        val splits = ratings.randomSplit(Array(0.8, 0.2), 0L)
        //val globalAvg = ratings.map(x=>x.rating).collect().sum / ratings.count().toInt
        val training = splits(0).cache()



        var userMap= new HashMap[Int,VectorFactorItem]
        totUsers.foreach{
            user =>
                val tmpVect = DenseVector.fill[Double](numFeatures){Random.nextDouble() * randomNoise}
                userMap += user -> new VectorFactorItem(numFeatures,0,0,0,tmpVect)
        }
        var itemMap =  new HashMap[Int, VectorFactorItem]
        movies.collect.foreach{
            item =>
                val tmpVect = DenseVector.fill[Double](numFeatures){Random.nextDouble() * randomNoise}
                itemMap += item -> new VectorFactorItem(1.0,1.0,0.0,numFeatures,tmpVect)
        }

        val userItemMatrix: Map[Int, Seq[Int]] = ratings.groupBy(x=>x.userId).map(x=> x._2.head.userId -> x._2.map(x=>x.movieId).toSeq).collectAsMap()
        val cachedRatings: Map[String, Double] = ratings.collect{
            case r: Rating => val digest = md5(r.userId.toString+" "+r.movieId.toString)
                (digest , r.rating.toDouble)
        }.map(x=>x).collect().toMap
        var currentLearningRate = learningRate


        println(s"Serial running with: $numIterations iterations  $numFeatures features $learningRateDecay learningDecay $preventOverFitting lambda")
        val time = System.nanoTime()
        for(iteration <- 0 to numIterations ) {

            /**just debug **/
            if(iteration % 5 == 0) {
                println("rmse training " + rmse(userMap, itemMap, training))
                println("rmse test " + rmse(userMap, itemMap, splits(1)))
            }

            userItemMatrix.foreach {
                userItem =>
                    val uid = userItem._1
                    val preferencesVector = userItem._2
                    updateUser(userItemMatrix, preferencesVector, userMap,itemMap,cachedRatings, uid, currentLearningRate)


            }

            currentLearningRate *= learningRateDecay

        }
        val micros = (System.nanoTime - time) / 1000
        println("%d microseconds".format(micros))

        println("rmse training "+rmse(userMap,itemMap,training))
        println("rmse test "+rmse(userMap,itemMap,splits(1)))





    }
    def updateUser(userMatrix:Map[Int,Seq[Int]],preferencesVector:Seq[Int],userMap:HashMap[Int,VectorFactorItem],itemMap:HashMap[Int,VectorFactorItem],ratings:Map[String,Double],uid:Int, currentLearningRate:Double): Unit= {
        preferencesVector.foreach {
            iid =>

                val user = userMap.apply(uid)
                val item = itemMap.apply(iid)
                val digest = md5(uid.toString + " " + iid.toString)
                val pr_rating = predictRating(user.factors, item.factors)


                val value = ratings.apply(digest)
                val userVector = user.factors
                val itemVector = item.factors

                val err = value - pr_rating
                //userVector.update(user_bias_index, user.userBias + bias_learning_rate * (err - biasReg * preventOverFitting * user.userBias))
                //itemVector.update(item_bias_index, item.itemBias + bias_learning_rate * (err - biasReg * preventOverFitting * item.itemBias))
                //user.userBias = userVector.apply(user_bias_index)
                //item.itemBias = itemVector.apply(item_bias_index)
                for (featureIndex <- 0 to numFeatures-1) {

                    val uF = userVector.apply(featureIndex)
                    val iF = itemVector.apply(featureIndex)
                    val deltaUserFeature = err * iF - preventOverFitting * uF
                    userVector.update(featureIndex, userVector.apply(featureIndex) + currentLearningRate * deltaUserFeature)
                    val deltaItemFeature = err * uF - preventOverFitting * iF
                    itemVector.update(featureIndex, itemVector.apply(featureIndex) + currentLearningRate * deltaItemFeature)
                }




        }

    }
    def md5(s: String): String = {
        MessageDigest.getInstance("MD5").digest(s.getBytes).map("%02x".format(_)).mkString
    }
    def rmse(userMatrix:HashMap[Int,VectorFactorItem],itemMatrix:HashMap[Int,VectorFactorItem],ratings:RDD[Rating]): Double={
        val res = ratings.map{
            rating=>
                val pr_val = predictRating(userMatrix.apply(rating.userId).factors, itemMatrix.apply(rating.movieId).factors)
                Math.pow(rating.rating - pr_val, 2)
        }.mean()
        Math.sqrt(res)

    }
    def rmse_test (userMatrix:HashMap[Int,VectorFactorItem],itemMatrix:HashMap[Int,VectorFactorItem],ratings:RDD[Rating]): Double={
        var counter = 0
        val res = ratings.collect.map{
            rating=>
                if(rating.userId <= 1000) {
                    val pr_val = predictRating(userMatrix.apply(rating.userId).factors, itemMatrix.apply(rating.movieId).factors)
                    counter += 1
                    Math.pow(rating.rating - pr_val, 2)
                }
                else
                    0
        }.sum
        Math.sqrt(res / counter)


    }
    def testOutput(userMatrix:Array[Array[Double]],itemMatrix:Array[Array[Double]], ratings:RDD[Rating], cachedUsers:Array[Int], cachedItems:Array[Int]): Unit={
        println("Test prediction on all users one item each")

        userMatrix.zipWithIndex.foreach{
            userRow=>
                val loopIndex = userRow._2
                val uid = cachedUsers.apply(loopIndex)
                val iid = cachedItems.apply(loopIndex)

            //val pr_rating = predictRating(userMatrix.apply(uid),itemMatrix.apply(iid))
            //val realValue = ratings.filter(r=> r.movieId == iid && r.userId == uid).first().rating

            //println("Prediction for user "+uid+" for item "+iid+" predicted value "+pr_rating+" real value "+realValue)
        }
    }
    def joinVectors(v1:Array[Double],v2:DenseVector[Double]): DenseVector[Double] ={
        val lastArray = Array(v1,v2.toArray).flatten
        DenseVector(lastArray)
    }
    def cacheOpt(ratings: RDD[Rating], users:Array[Int]): HashMap[Int,Seq[Int]]={

        var userItem = new HashMap[Int,Seq[Int]]()
        users.foreach{
            user=>
                val rating = ratings.filter(r=> r.userId == user).map(u=>u.movieId).collect().toSeq
                userItem += user -> rating
        }
        userItem
    }
    def predictRating(userVector:DenseVector[Double], itemVector:DenseVector[Double]): Double ={
        userVector dot itemVector
    }

    
}
