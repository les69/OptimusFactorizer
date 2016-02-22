import org.apache.spark.examples.ml.TestALS.{Movie}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, CoordinateMatrix}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

case class Rating(userId: Int, movieId: Int, rating: Float)

object Rating {
    def parseRating(str: String): Rating = {
        //val fields = str.split(",")
        val fields = str.split("::")
        Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat)
    }
}

object TestSGDSVM {
    def  main (args: Array[String]) {

        val conf = new SparkConf().setAppName(s"TestSGD")
        val sc = new SparkContext(conf)
        sc.setLogLevel("WARN")
        // Load training data in LIBSVM format.
        val sqlContext = new SQLContext(sc)
        import sqlContext.implicits._

        val ratingsPath  = "ml-latest/ratings-1m.dat"
        val movies = sc.textFile("ml-latest/movies.csv").map(Movie.parseMovie).toDF()
        val ratings = sc.textFile(ratingsPath).map(Rating.parseRating).cache()

        //val splits = ratings.randomSplit(Array(0.8, 0.2), 0L)


        val numRatings = ratings.count()
        // val numUsers = ratings.map(_.userId).distinct().count()
        // val totUsers = ratings.map(_.userId).distinct().collect()

        val mat = new CoordinateMatrix(ratings.map{
            case Rating(user,item,rating) => MatrixEntry(user,item,rating)
        })
        val rowMat = mat.toBlockMatrix().toIndexedRowMatrix().toRowMatrix()


        val test = rowMat.rows.zipWithIndex().map{
            row =>
                LabeledPoint(row._2.toDouble,row._1.toDense)
        }.cache()

        // Split data into training (60%) and test (40%).
        val splits = test.randomSplit(Array(0.6, 0.4), seed = 11L)
        val training = splits(0).cache()
        training.foreach(println)
        val testsplit = splits(1)

        // Run training algorithm to build the model
        val numIterations = 100
        val model = SVMWithSGD.train(training, numIterations)

        // Clear the default threshold.
        model.clearThreshold()

        // Compute raw scores on the test set.
        val scoreAndLabels = testsplit.map { point =>
            val score = model.predict(point.features)
            (score, point.label)
        }
        scoreAndLabels.foreach(println)

        //Get evaluation metrics.
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        val auROC = metrics.areaUnderROC()

        println("Area under ROC = " + auROC)



        // Save and load model
        //model.save(sc, "myModelPath")
        //val sameModel = SVMModel.load(sc, "myModelPath")**/

    }

}