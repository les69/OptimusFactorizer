import breeze.linalg.DenseVector
import org.apache.mahout.common.RandomUtils
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry}
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}


/**
  * Created by les on 19/02/16.
  */
object TestGradientDescentOld {
    val rand = RandomUtils.getRandom(42L)
    val learningRate = 0.1
    val preventOverFitting = 0.1
    val randomNoise = 0.1
    val learningRateDecay = 1.0

    val user_bias_index = 1
    val item_bias_index = 2
    val feature_offset = 3

    val bias_learning_rate = 0.5
    val biasReg = 0.1

    def main(args: Array[String]) {

        val conf = new SparkConf().setAppName(s"TestGradient")
        val sc = new SparkContext(conf)
        sc.setLogLevel("WARN")
        // Load training data in LIBSVM format.
        val sqlContext = new SQLContext(sc)

        val ratingsPath  = "ml-latest/ratings-1m.dat"
        //val movies = sc.textFile("ml-latest/movies.csv").map(Movie.parseMovie).toDF()

        val movies =Seq(1,2,3,4,5,6,7,8,9,10)
        val totUsers =Seq(1,2,3,4,5,6,7,8,9,10)

        val ratings = sc.textFile(ratingsPath).map(Rating.parseRating).cache()
        val splits = ratings.randomSplit(Array(0.8, 0.2), 0L)
        val numFeatures = 5
        val numIterations = 100
        val globalAvg = ratings.map(x=>x.rating).collect().sum / ratings.count().toInt

        val training = splits(0).cache()


        val numUsers = ratings.map(_.userId).distinct().count()


        val ratingMatrix = new CoordinateMatrix(ratings.map{
            case Rating(user,item,rating) => MatrixEntry(user,item,rating)
        }).toBlockMatrix().toIndexedRowMatrix().toRowMatrix().rows.map{
            row=>
                row.toArray
        }.toArray()

        val normal01 = breeze.stats.distributions.Gaussian(0,1)

        val userMatrix = totUsers.map{
            user =>
                val v1 = DenseVector(globalAvg.toDouble, 0.0,1.0)
                joinVectors(v1,DenseVector.fill(numFeatures) {rand.nextGaussian() * randomNoise})


        //}.toArray()
        }.toArray

        userMatrix.foreach(println)


        val itemMatrix = movies.map{
            item =>
                val v2 = DenseVector(1.0,1.0,0.0)
                joinVectors(v2,DenseVector.fill(numFeatures){rand.nextGaussian() * randomNoise}.t.inner)
        //}.toArray()
        }.toArray


        predictRating(1,1,userMatrix,itemMatrix)
        var iteration = 0

        var currentLearningRate = learningRate

        //[TODO] shuffle the item vectors

        for(iteration <- 0 to numIterations ){
            userMatrix.zipWithIndex.foreach{
                userRow =>
                    val pr_rating = predictRating(userRow._1,userRow._1)
                    //val entry = ratingMatrix.entries.filter(e=> e.i == userRow._2 && e.j == userRow._2)
                    val value = ratingMatrix.apply(userRow._2).apply(userRow._2)



                    val userVector = userMatrix.apply(userRow._2)
                    val itemVector = itemMatrix.apply(userRow._2)
                    println(value)
                    val err = value - pr_rating


                    userVector.update(user_bias_index, bias_learning_rate * (err - biasReg * preventOverFitting *userVector.apply(user_bias_index)))
                    itemVector.update(item_bias_index, bias_learning_rate * (err - biasReg * preventOverFitting * itemVector.apply(item_bias_index)))


                    for(featureIndex <- feature_offset to numFeatures){
                        val uF = userVector.apply(featureIndex)
                        val iF = itemVector.apply(featureIndex)

                        val deltaUserFeature = err * iF - preventOverFitting * uF
                        userVector.update(featureIndex, userVector.apply(featureIndex) + currentLearningRate * deltaUserFeature)


                        val deltaItemFeature = err * uF - preventOverFitting * iF
                        itemVector.update(featureIndex,itemVector.apply(featureIndex)+ currentLearningRate * deltaItemFeature)
                    }





            }
            currentLearningRate *= learningRateDecay

        }

        userMatrix.foreach(println)
        predictRating(1,1,userMatrix,itemMatrix)







    }
    def joinVectors(v1:DenseVector[Double],v2:DenseVector[Double]): DenseVector[Double] ={
        val lastArray = Array(v1.toArray,v2.toArray).flatten
        DenseVector(lastArray)
    }
    def updateParameters(itemID:Int, userID:Int, rating:Double, currentLearningRate:Double): Unit ={

    }
    def predictRating(userVector:DenseVector[Double], itemVector:DenseVector[Double]): Double ={
        userVector dot itemVector
    }
    def predictRating(itemID:Int, userID:Int,users:Array[DenseVector[Double]], items:Array[DenseVector[Double]]){



        println("not working")

    }
    
}
