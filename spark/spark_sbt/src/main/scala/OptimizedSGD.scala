import java.security.MessageDigest

import breeze.linalg.DenseVector
import org.apache.spark.examples.ml.TestALS.Movie
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.immutable.HashMap
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
    val numFeatures = 20
    val numIterations = 25



    def main(args: Array[String]) {

        val conf = new SparkConf().setAppName(s"TestGradient")
        val sc = new SparkContext(conf)
        sc.setLogLevel("WARN")
        val sqlContext = new SQLContext(sc)
        import sqlContext.implicits._


        val ratingsPath  = "ml-latest/ratings-1m.dat"
        val movies = sc.textFile("ml-latest/movies.csv").map(Movie.parseMovie)

        //val movies =Array(1,2,3,4,5,6,7,8,9,10)
        //val totUsers =Array(1,2,3,4,5,6,7,8,9,10)



        val ratings = sc.textFile(ratingsPath).map(Rating.parseRating).cache()


        val totUsers = ratings.map(_.userId).distinct().collect()

        val splits = ratings.randomSplit(Array(0.8, 0.2), 0L)

        val globalAvg = ratings.map(x=>x.rating).collect().sum / ratings.count().toInt


        val training = splits(0).cache()


        //val numUsers = ratings.map(_.userId).distinct().count()




        //var userMap =  Map.empty[Int,VectorFactorItem]
        var userMap= new HashMap[Int,VectorFactorItem]
        totUsers.foreach{
            user =>
                val tmpVect = DenseVector.fill[Double](numFeatures){Random.nextDouble() * randomNoise}
                userMap += user -> new VectorFactorItem(globalAvg.toDouble,0.0,1.0,numFeatures,joinVectors(Array(globalAvg,0.0,1.0),tmpVect))
        }
        var itemMap =  new HashMap[Int, VectorFactorItem]
        movies.collect.foreach{
            item =>
                val tmpVect = DenseVector.fill[Double](numFeatures){Random.nextDouble() * randomNoise}
                itemMap += item.movieId -> new VectorFactorItem(1.0,1.0,0.0,numFeatures,joinVectors(Array(1.0,1.0,0.0),tmpVect))
        }



        //[TODO] check shuffling
        val userItemMatrix = cacheOpt(ratings,rand.shuffle(totUsers.toSeq).toArray)
        var cachedRatings = new HashMap[String,Double]

        ratings.collect.foreach{
            r=> val digest = md5(r.userId.toString+" "+r.movieId.toString)
                cachedRatings += digest -> r.rating.toDouble
        }

        //[B@5c5e62e1 2163 2997
        var currentLearningRate = learningRate
        val testUser = userMap.apply(userItemMatrix.head._1)
        val testItem = itemMap.apply(userItemMatrix.head._2.head)
        val i = userItemMatrix.head._1
        val j = userItemMatrix.head._2.head
        val md5val = md5(i.toString+" "+j.toString)


        println("Test prediction for user "+userItemMatrix.head._1+" with item "+userItemMatrix.head._2.head+" and real value"+cachedRatings.apply(md5val))
        val time = System.nanoTime()
        for(iteration <- 0 to numIterations ) {
            println("Iteration " + iteration + " out of " + numIterations)
            println("Current prediction")
            println(predictRating(testUser.factors, testItem.factors))
            userItemMatrix.foreach {
                userItem =>
                    val uid = userItem._1
                    val preferencesVector = userItem._2

                    preferencesVector.foreach {
                        iid=>
                            val user = userMap.apply(uid)
                            val item = itemMap.apply(iid)
                            val pr_rating = predictRating(user.factors, item.factors)
                            val digest = md5(uid.toString + " " + iid.toString)

                            val test = cachedRatings.keySet.par.find(k => k == digest)
                            if (!test.isEmpty) {
                                val value = test.get.toDouble
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

                            }
                    }
                    currentLearningRate *= learningRateDecay
            }
        }
        val micros = (System.nanoTime - time) / 1000
        println("%d microseconds".format(micros))

        //println("rmse training "+rmse(userMatrix,itemMatrix,training,cachedUsers,cachedItems))
        //println("rmse test "+rmse(userMatrix,itemMatrix,splits(1),cachedUsers,cachedItems))





    }
    def md5(s: String): String = {
        MessageDigest.getInstance("MD5").digest(s.getBytes).map("%02x".format(_)).mkString
    }
    def rmse (userMatrix:Array[Array[Double]],itemMatrix:Array[Array[Double]],ratings:RDD[Rating], cachedUsers:Array[Int], cachedItems:Array[Int]): Double={
        /**val res = userMatrix.zipWithIndex.map{
            userRow=>
              /**  val loopIndex = userRow._2
                val uid = cachedUsers.apply(loopIndex)
                val iid = cachedItems.apply(loopIndex)

                val pr_rating = predictRating(userMatrix.apply(uid),itemMatrix.apply(iid))
                val temp = ratings.filter(r=> r.movieId == iid && r.userId == uid)

                if(temp.collect().length > 0) {
                    val realValue = temp.first().rating
                    math.pow(realValue - pr_rating,2)
                }
                else
                    0**/



        }.sum

        //Math.sqrt(res / cachedUsers.length)**/
        0
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
    def cacheOpt(ratings: RDD[Rating], users:Array[Int]): Map[Int,Seq[Int]]={

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
