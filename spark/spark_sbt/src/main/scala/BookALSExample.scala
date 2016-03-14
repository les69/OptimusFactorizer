/**
  * Created by les on 16/02/16.
  */
/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

object BookALSExample {

    case class Book(userId: Int, isbn: String, rating: Int)

    object Book {
        def parseBook(str: String): Book = {
            val fields = str.split(";").map(x=> x.replace("\"",""))
            assert(fields.size == 3)

            Book(fields(0).toInt, fields(1), fields(2).toInt)
        }
    }



    def main(args: Array[String]): Unit = {

        //run(0) 3.7638334016698507
        run(5)
        //run(5) //0.8940511755361474
        //run(10) //0.8716105777929495
        //run(15) //0.8656929171850316
        //run(20) //0.863464071167232
        //run(25) stack overflow
        //run(30)
        //run(10) max dataset 0.81233459384
    }

    def run(maxIter:Int): Unit={
        val rank = 10
        //val maxIter = 15
        val regParam = 0.1
        val booksPath  = "ml-latest/BX-Book-Ratings.csv"
        val conf = new SparkConf().setAppName(s"BookALS")
        val sc = new SparkContext(conf)
        val sqlContext = new SQLContext(sc)
        import sqlContext.implicits._

        sc.setLogLevel("WARN")
        //val movies = sc.textFile("ml-latest/movies.csv").map(Movie.parseMovie).toDF()
        val ratings = sc.textFile(booksPath).map(Book.parseBook).cache()
       
        val t0 = System.nanoTime()

        val numRatings = ratings.count()
        val numUsers = ratings.map(_.userId).distinct().count()
        val numBooks = ratings.map(_.isbn).distinct().count()

        println(s"Got $numRatings ratings from $numUsers users on $numBooks books.")

        val splits = ratings.randomSplit(Array(0.8, 0.2), 0L)
        val training = splits(0).cache()
        val test = splits(1).cache()

        val numTraining = training.count()
        val numTest = test.count()
        println(s"Training: $numTraining, test: $numTest.")

        ratings.unpersist(blocking = false)

        val als = new ALS()
            .setUserCol("userId")
            .setItemCol("isbn")
            .setRank(rank)
            .setMaxIter(maxIter)
            .setRegParam(regParam)
            .setNumBlocks(10)

        val model = als.fit(training.toDF())


        val predictions = model.transform(test.toDF()).cache()

        // Evaluate the model.
        // TODO: Create an evaluator to compute RMSE.
        val mse = predictions.select("rating", "prediction").rdd
            .flatMap { case Row(rating: Float, prediction: Float) =>
                val err = rating.toDouble - prediction
                val err2 = err * err
                if (err2.isNaN) {
                    None
                } else {
                    Some(err2)
                }
            }.mean()
        val rmse = math.sqrt(mse)
        println(s"Test RMSE = $rmse.")

        // Inspect false positives.
        // Note: We reference columns in 2 ways:
        //  (1) predictions("movieId") lets us specify the movieId column in the predictions
        //      DataFrame, rather than the movieId column in the movies DataFrame.
        //  (2) $"userId" specifies the userId column in the predictions DataFrame.
        //      We could also write predictions("userId") but do not have to since
        //      the movies DataFrame does not have a column "userId."

       /** val falsePositives = predictions.join(movies)
            .where((predictions("movieId") === movies("movieId"))
                && ($"rating" <= 1) && ($"prediction" >= 4))
            .select($"userId", predictions("movieId"), $"title", $"rating", $"prediction")
        val numFalsePositives = falsePositives.count()
        println(s"Found $numFalsePositives false positives")
        if (numFalsePositives > 0) {
            println(s"Example false positives:")
            falsePositives.limit(100).collect().foreach(println)
        }**/

        /**val output = predictions
           .select($"userId",$"movieId",$"prediction")
           .where(predictions("userId") === 1).collect().foreach(println)**/
     val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) + "ns")
        sc.stop()
    }
}
// scalastyle:on println
