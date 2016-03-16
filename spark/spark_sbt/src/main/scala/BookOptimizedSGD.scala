import java.security.MessageDigest
import java.util.concurrent.TimeUnit

import breeze.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable.HashMap
import scala.collection.parallel.mutable.{ParHashMap, ParMap}
import scala.collection.{Map, breakOut}
import scala.util.Random

///home/Downloads/spark/bin/spark-submit --class TestGradientDescent target/scala-2.10/spark_sbt_2.10-1.0.jar
/**
  * Created by les on 19/02/16.
  */
object BookOptimizedSGD {

    case class Book(userId: Int, isbn: String, rating: Int)

    object Book {
        def parseBook(str: String): Book = {
            val fields = str.split(";").map(x=> x.replace("\"",""))
            assert(fields.size == 3)

            Book(fields(0).toInt, fields(1), fields(2).toInt)
        }
    }



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
    val biasReg = 0.1
    val numFeatures = 20
    val numIterations = 25
    var pool = java.util.concurrent.Executors.newFixedThreadPool(1000)
    var threadPool = List

    val test_users = 10000




    def main(args: Array[String]) {

        val conf = new SparkConf().setAppName(s"TestGradient")
        val sc = new SparkContext(conf)
        sc.setLogLevel("WARN")
        val sqlContext = new SQLContext(sc)


        val ratingsPath  = "ml-latest/BX-Book-Ratings.csv"
        //val ratingsPath  = "ml-latest/ratings-1m.dat"
        //val movies = sc.textFile("ml-latest/movies.csv").map(Movie.parseMovie)

        //val movies =Array(1,2,3,4,5,6,7,8,9,10)
        //val totUsers =Array(1,2,3,4,5,6,7,8,9,10)


        val ratings = sc.textFile(ratingsPath).map(Book.parseBook).filter(x=>x.rating > 0).cache()



        val totUsers = ratings.map(_.userId).distinct().collect()
        //val totUsers = ratings.map(_.userId).distinct().takeOrdered(test_users)
        val books = ratings.map(_.isbn).distinct()

        val splits = ratings.randomSplit(Array(0.8, 0.2), 0L)

        val globalAvg = ratings.map(x=>x.rating).collect().sum / ratings.count().toInt


        val training = splits(0).cache()


        //val numUsers = ratings.map(_.userId).distinct().count()
        //var userItem: ParHashMap[Int,Seq[String]] = users.par.collect{case u: Int => val rating = ratings.filter(r=> r.userId == u).map(u=>u.isbn).collect().toSeq; (u,rating)}.map(x=>x)(breakOut)



        //var userMap =  Map.empty[Int,VectorFactorItem]
        var userMap= new HashMap[Int,VectorFactorItem]
        totUsers.foreach{
            user =>
                val tmpVect = DenseVector.fill[Double](numFeatures){Random.nextDouble() * randomNoise}
                userMap += user -> new VectorFactorItem(globalAvg.toDouble,0.0,1.0,numFeatures,joinVectors(Array(globalAvg,0.0,1.0),tmpVect))
        }
        var itemMap =  new HashMap[String, VectorFactorItem]
        books.collect.foreach{
            item =>
                val tmpVect = DenseVector.fill[Double](numFeatures){Random.nextDouble() * randomNoise}
                itemMap += item -> new VectorFactorItem(1.0,1.0,0.0,numFeatures,joinVectors(Array(1.0,1.0,0.0),tmpVect))
        }

        //[TODO] check shuffling
        val userItemMatrix: Map[Int, Seq[String]] = ratings.groupBy(x=>x.userId).map(x=> x._2.head.userId -> x._2.map(x=>x.isbn).toSeq).collectAsMap()
        //val userItemMatrix = cacheOpt(training,rand.shuffle(totUsers.toSeq).toArray)
        val cachedRatings: Map[String, Double] = ratings.collect{
            case r: Book => val digest = md5(r.userId.toString+" "+r.isbn.toString)
                (digest , r.rating.toDouble)
        }.map(x=>x).collect().toMap


        var currentLearningRate = learningRate
        /**val testUser = userMap.apply(userItemMatrix.head._1)
        val testItem = itemMap.apply(userItemMatrix.head._2.head)
        val i = userItemMatrix.head._1
        val j = userItemMatrix.head._2.head
        val md5val = md5(i.toString+" "+j.toString)**/


        println("Serial running with 15 iterations and 30 features")
        //println("Test prediction for user "+userItemMatrix.head._1+" with item "+userItemMatrix.head._2.head+" and real value "+cachedRatings.apply(md5val))
        val time = System.nanoTime()
        for(iteration <- 0 to numIterations ) {
            //println("Iteration " + iteration + " out of " + numIterations)
            //println("Current prediction")
            //println(predictRating(testUser.factors
            // , testItem.factors))

            if(iteration % 5 == 0) {
                println("rmse training " + rmse(userMap, itemMap, training))
                println("rmse test " + rmse(userMap, itemMap, splits(1)))
            }

            userItemMatrix.foreach {
                userItem =>
                    val uid = userItem._1
                    val preferencesVector = userItem._2
                    /**pool.execute(new Runnable {
                        override def run(): Unit = {
                            updateUserMap(userItemMatrix, preferencesVector, userMap,itemMap,cachedRatings, uid, currentLearningRate)
                        }
                    })**/
                    updateUserMap(userItemMatrix, preferencesVector, userMap,itemMap,cachedRatings, uid, currentLearningRate)




            }
            /**println("Waiting for tasks to terminate")
            pool.shutdown()
            pool.awaitTermination(10000,TimeUnit.SECONDS)
            pool = java.util.concurrent.Executors.newFixedThreadPool(1000)**/

            currentLearningRate *= learningRateDecay

        }
        val micros = (System.nanoTime - time) / 1000
        println("%d microseconds".format(micros))

        println("rmse training "+rmse(userMap,itemMap,training))
        println("rmse test "+rmse(userMap,itemMap,splits(1)))
        //testOutput(splits(1), cachedRatings, userMap,itemMap)





    }
    def updateUserMap(userMatrix:Map[Int,Seq[String]],preferencesVector:Seq[String],userMap:HashMap[Int,VectorFactorItem],itemMap:HashMap[String,VectorFactorItem],ratings:Map[String,Double],uid:Int, currentLearningRate:Double): Unit= {
        preferencesVector.foreach {
            iid =>

                val user = userMap.apply(uid)
                val item = itemMap.apply(iid)
                val digest = md5(uid.toString + " " + iid.toString)


                item.synchronized {
                    val pr_rating = predictRating(user.factors, item.factors)


                    val value = ratings.apply(digest)
                    val userVector = user.factors
                    val itemVector = item.factors

                    val err = value - pr_rating
                    userVector.update(user_bias_index, user.userBias + bias_learning_rate * (err - biasReg * preventOverFitting * user.userBias))
                    itemVector.update(item_bias_index, item.itemBias + bias_learning_rate * (err - biasReg * preventOverFitting * item.itemBias))
                    user.userBias = userVector.apply(user_bias_index)
                    item.itemBias = itemVector.apply(item_bias_index)
                    for (featureIndex <- feature_offset to numFeatures) {

                        val uF = userVector.apply(featureIndex)
                        val iF = itemVector.apply(featureIndex)
                        val deltaUserFeature = err * iF - preventOverFitting * uF
                        userVector.update(featureIndex, userVector.apply(featureIndex) + currentLearningRate * deltaUserFeature)
                        val deltaItemFeature = err * uF - preventOverFitting * iF
                        itemVector.update(featureIndex, itemVector.apply(featureIndex) + currentLearningRate * deltaItemFeature)
                    }
                    item.notifyAll()

                }

        }
        //println("Task completed for user "+uid)

    }
    def updateUser(userMatrix:ParHashMap[Int,Seq[String]],preferencesVector:Seq[String],userMap:HashMap[Int,VectorFactorItem],itemMap:HashMap[String,VectorFactorItem],ratings:HashMap[String,Double],uid:Int, currentLearningRate:Double): Unit= {
        preferencesVector.foreach {
            iid =>

                val user = userMap.apply(uid)
                val item = itemMap.apply(iid)
                val digest = md5(uid.toString + " " + iid.toString)


                item.synchronized {
                    val pr_rating = predictRating(user.factors, item.factors)


                    val value = ratings.apply(digest)
                    val userVector = user.factors
                    val itemVector = item.factors

                    val err = value - pr_rating
                    userVector.update(user_bias_index, user.userBias + bias_learning_rate * (err - biasReg * preventOverFitting * user.userBias))
                    itemVector.update(item_bias_index, item.itemBias + bias_learning_rate * (err - biasReg * preventOverFitting * item.itemBias))
                    user.userBias = userVector.apply(user_bias_index)
                    item.itemBias = itemVector.apply(item_bias_index)
                    for (featureIndex <- feature_offset to numFeatures) {

                        val uF = userVector.apply(featureIndex)
                        val iF = itemVector.apply(featureIndex)
                        val deltaUserFeature = err * iF - preventOverFitting * uF
                        userVector.update(featureIndex, userVector.apply(featureIndex) + currentLearningRate * deltaUserFeature)
                        val deltaItemFeature = err * uF - preventOverFitting * iF
                        itemVector.update(featureIndex, itemVector.apply(featureIndex) + currentLearningRate * deltaItemFeature)
                    }
                    item.notifyAll()

                }

        }
        //println("Task completed for user "+uid)

    }

    def md5(s: String): String = {
        MessageDigest.getInstance("MD5").digest(s.getBytes).map("%02x".format(_)).mkString
    }
    def rmse(userMatrix:HashMap[Int,VectorFactorItem],itemMatrix:HashMap[String,VectorFactorItem],ratings:RDD[Book]): Double={
        val res = ratings.map{
            rating=>
                val pr_val = predictRating(userMatrix.apply(rating.userId).factors, itemMatrix.apply(rating.isbn).factors)
                Math.pow(rating.rating - pr_val, 2)
        }.mean()
        Math.sqrt(res)

    }
    def rmse_test (userMatrix:HashMap[Int,VectorFactorItem],itemMatrix:HashMap[String,VectorFactorItem],ratings:RDD[Book]): Double={
        var counter = 0
        val res:Double = ratings.collect.map{
            rating=>
                if(rating.userId <= test_users) {
                    val pr_val = predictRating(userMatrix.apply(rating.userId).factors, itemMatrix.apply(rating.isbn).factors)
                    counter += 1
                    Math.pow(rating.rating - pr_val, 2)
                }
                else
                    0
            //val pr_val = predictRating(userMatrix.apply(rating.userId).factors, itemMatrix.apply(rating.movieId).factors)
            //Math.pow(rating.rating - pr_val, 2)
        }.sum

        Math.sqrt(res / counter)
        //Math.sqrt(res)

    }
    def testOutput(ratings:RDD[Book], cachedRatings: Map[String,Double], userMatrix:HashMap[Int,VectorFactorItem], itemMatrix:HashMap[String,VectorFactorItem]): Unit={

        ratings.foreach{
            r=> val pr_val = predictRating(userMatrix.apply(r.userId).factors, itemMatrix.apply(r.isbn).factors)
                val rating = r.rating
                println(s"Prediction for $rating  is $pr_val")
        }
    }
    def joinVectors(v1:Array[Double],v2:DenseVector[Double]): DenseVector[Double] ={
        val lastArray = Array(v1,v2.toArray).flatten
        DenseVector(lastArray)
    }
    def cacheOpt(ratings: RDD[Book], users:Array[Int]): ParHashMap[Int,Seq[String]]={

        //var userItem = new HashMap[Int,Seq[String]]()
        var userItem: ParHashMap[Int,Seq[String]] = users.par.collect{case u: Int => val rating = ratings.filter(r=> r.userId == u).map(u=>u.isbn).collect().toSeq; (u,rating)}.map(x=>x)(breakOut)

        /**users.foreach{
            user=>
                val rating = ratings.filter(r=> r.userId == user).map(u=>u.isbn).collect().toSeq
                userItem += user -> rating
        }**/
        userItem
    }
    def predictRating(userVector:DenseVector[Double], itemVector:DenseVector[Double]): Double ={
        userVector.dot(itemVector)
    }

    
}
