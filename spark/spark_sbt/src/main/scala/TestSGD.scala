/*
/*
import org.apache.spark.examples.ml.TestALS.{Movie}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, CoordinateMatrix}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils



object TestSGD {


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
        val splits = ratings.randomSplit(Array(0.8, 0.2), 0L)


        val numRatings = ratings.count()
        val numUsers = ratings.map(_.userId).distinct().count()
        val totUsers = ratings.map(_.userId).distinct().collect()


        /**val parsedData =sc.parallelize(totUsers.map{
            user =>
                val userRatings = ratings.filter(x => x.userId == user).map(f=>f.rating.toDouble).collect()
                LabeledPoint(user, Vectors.dense(userRatings).toSparse)
        })*/

        val mat = new CoordinateMatrix(ratings.map{
            case Rating(user,item,rating) => MatrixEntry(user,item,rating)
        })
        val rowMat = mat.toBlockMatrix().toIndexedRowMatrix().toRowMatrix()


        val test = rowMat.rows.zipWithIndex().map{
            row =>
                LabeledPoint(row._2.toDouble,row._1.toDense)
        }.cache()



        // Building the model
        val numIterations = 100
        val model = LinearRegressionWithSGD.train(test, numIterations)

        // Evaluate model on training examples and compute training error
        val valuesAndPreds = test.map { point =>
            val prediction = model.predict(point.features)
            println(point.label, prediction)
            (point.label, prediction)
        }
        val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
        println("training Mean Squared Error = " + MSE)


        // Save and load model
        //model.save(sc, "myModelPath")
        //val sameModel = SVMModel.load(sc, "myModelPath")**/

        for(iteration <- 0 to numIterations ){
            userMatrix.zipWithIndex.foreach{
                userRow =>
                    val loopIndex = userRow._2
                    val pr_rating = predictRating(userRow._1,userRow._1)
                    //val entry = ratingMatrix.entries.filter(e=> e.i == userRow._2 && e.j == userRow._2)
                    val value = ratings.filter(r=> r.movieId == loopIndex && r.userId == loopIndex).first().rating //[TODO] optimize with a cached dataset

                    if(value.toInt > 0) {

                        val userVector = userMatrix.apply(loopIndex)
                        val itemVector = itemMatrix.apply(loopIndex)
                        val err = value - pr_rating


                        userVector.update(user_bias_index, bias_learning_rate * (err - biasReg * preventOverFitting * userVector.apply(user_bias_index)))
                        itemVector.update(item_bias_index, bias_learning_rate * (err - biasReg * preventOverFitting * itemVector.apply(item_bias_index)))


                        for (featureIndex <- feature_offset to numFeatures) {
                            val uF = userVector.apply(featureIndex)
                            val iF = itemVector.apply(featureIndex)

                            val deltaUserFeature = err * iF - preventOverFitting * uF
                            userVector.update(featureIndex, userVector.apply(featureIndex) + currentLearningRate * deltaUserFeature)


                            val deltaItemFeature = err * uF - preventOverFitting * iF
                            itemVector.update(featureIndex, itemVector.apply(featureIndex) + currentLearningRate * deltaItemFeature)
                        }
                    }
                    else
                        numIterations += 1





            }
            currentLearningRate *= learningRateDecay

        }

        println(predictRating(userMatrix.apply(0),itemMatrix.apply(0)))
    }

}*/
/**
  *
  * bank of the nice but unused code
  *
/**val itemMatrix = movies.map{
            item =>
                val v2 = Array(1.0,1.0,0.0)
                joinVectors(v2,Array.fill(numFeatures){rand.nextGaussian() * randomNoise})
        //}.toArray()
        }.toArray.toSeq.transpose.toArray.map{
            seq=>
                val arr = Array[Double](seq.size)
                seq.copyToArray(arr)
                arr
        }**/

   /** this works!
          for(uid <- 1 to users.size){
            val rating = ratings.filter(r=> r.userId == uid).collect()

            for(rid <- 0 to rating.length-1){
                cachedUsers.update(index, uid)
                cachedItems.update(index,rating.apply(rid).movieId)
                index+=1
            }

        }*/
  */
