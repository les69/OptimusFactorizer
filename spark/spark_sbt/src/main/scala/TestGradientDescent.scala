import breeze.linalg.{DenseMatrix, DenseVector, Counter, randomDouble}
import breeze.optimize.StochasticGradientDescent
import org.apache.spark.examples.ml.TestALS.Movie
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{RowMatrix, MatrixEntry, CoordinateMatrix}
import org.apache.spark.mllib.optimization.{SimpleUpdater, LogisticGradient, GradientDescent}
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

import scala.util.Random

///home/Downloads/spark/bin/spark-submit --class TestGradientDescent target/scala-2.10/spark_sbt_2.10-1.0.jar
/**
  * Created by les on 19/02/16.
  */
object TestGradientDescent {
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
    val numIterations = 100



    def main(args: Array[String]) {

        val conf = new SparkConf().setAppName(s"TestGradient")
        val sc = new SparkContext(conf)
        sc.setLogLevel("WARN")
        // Load training data in LIBSVM format.
        val sqlContext = new SQLContext(sc)
        import sqlContext.implicits._
        val time = System.nanoTime()

        val ratingsPath  = "ml-latest/ratings-1m.dat"
        val movies = sc.textFile("ml-latest/movies.csv").map(Movie.parseMovie).toDF()

        //val movies =Array(1,2,3,4,5,6,7,8,9,10)
        //val totUsers =Array(1,2,3,4,5,6,7,8,9,10)



        val ratings = sc.textFile(ratingsPath).map(Rating.parseRating).cache()


        val totUsers = ratings.map(_.userId).distinct().collect()

        val splits = ratings.randomSplit(Array(0.8, 0.2), 0L)

        val globalAvg = ratings.map(x=>x.rating).collect().sum / ratings.count().toInt

        val (cachedUsers,cachedItems)=  cache(ratings,totUsers)


        val training = splits(0).cache()


        val numUsers = ratings.map(_.userId).distinct().count()


        /**val ratingMatrix = new CoordinateMatrix(ratings.map{
            case Rating(user,item,rating) => MatrixEntry(user,item,rating)
        }).toBlockMatrix().toIndexedRowMatrix().toRowMatrix().rows.map{
            row=>
                row.toArray
        }.toArray()**/





        val userMatrix = totUsers.map{
            user =>
                val v1 = Array(globalAvg.toDouble, 0.0,1.0)
                joinVectors(v1,Array.fill(numFeatures) {rand.nextDouble() * randomNoise})
        }

        val itemMatrix = movies.map{
            item =>
                val v2 = Array(1.0,1.0,0.0)
                joinVectors(v2,Array.fill(numFeatures){rand.nextDouble() * randomNoise})
        }.collect()



        var currentLearningRate = learningRate

        println("Real value for user with item"+cachedUsers.apply(0),cachedItems.apply(0)+" value: "+ratings.filter(f=> f.movieId == cachedItems.apply(0) && f.userId==cachedUsers.apply(0)).first().rating)
        println("Current prediction")
        println(predictRating(userMatrix.apply(cachedUsers.apply(0)),itemMatrix.apply(cachedItems.apply(0))))


        for(iteration <- 0 to numIterations ){
            println("Iteration "+iteration+" out of "+numIterations)
            println("Current prediction")
            println(predictRating(userMatrix.apply(cachedUsers.apply(0)),itemMatrix.apply(cachedItems.apply(0))))
            userMatrix.zipWithIndex.foreach{
                userRow =>
                    val loopIndex = userRow._2
                    val uid = cachedUsers.apply(loopIndex)
                    val iid = cachedItems.apply(loopIndex)

                    val pr_rating = predictRating(userMatrix.apply(uid),itemMatrix.apply(iid))
                    //val entry = ratingMatrix.entries.filter(e=> e.i == userRow._2 && e.j == userRow._2)
                    //val value = ratings.filter(r=> r.movieId == iid && r.userId == uid).first().rating //[TODO] optimize with a cached dataset
                    val temp = ratings.filter(r=> r.movieId == iid && r.userId == uid)
                    val test= ratings.collect()

                    if(temp.collect().length == 0){
                        println("Non existing"+uid+","+iid)
                    }
                    else {
                        val value = temp.first().rating

                        val userVector = userMatrix.apply(uid)
                        val itemVector = itemMatrix.apply(iid)
                        val err = value - pr_rating


                        userVector.update(user_bias_index,userVector.apply(user_bias_index) + bias_learning_rate * (err - biasReg * preventOverFitting * userVector.apply(user_bias_index)))
                        itemVector.update(item_bias_index,itemVector.apply(item_bias_index) + bias_learning_rate * (err - biasReg * preventOverFitting * itemVector.apply(item_bias_index)))


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




        val micros = (System.nanoTime - time) / 1000
        println("%d microseconds".format(micros))

        testOutput(userMatrix,itemMatrix,ratings,cachedUsers,cachedItems)


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
    def cache(ratings: RDD[Rating], users:Array[Int]): (Array[Int],Array[Int]) ={

        val numPreferences = ratings.map(x=>x.rating).collect().size
        val cachedItems = Array.fill(numPreferences)(0)
        val cachedUsers = Array.fill(numPreferences)(0)


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
                if(index == Int.MaxValue -1)
                    println("Warning overflow!")

        }


        shuffleCachedItems(cachedUsers,cachedItems)
        (cachedUsers,cachedItems)

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
    
}
