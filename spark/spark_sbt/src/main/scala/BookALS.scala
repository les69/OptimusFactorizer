/**
  * Created by les on 13/03/16.
  */

import org.apache.spark.ml.recommendation.{ALSModel, ALS}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{IntegerType, FloatType}

/**
class BookALS extends ALS {
    import BookALSExample.Book
    def this() = this(Identifiable.randomUID("als"))

    override def fit(dataset: DataFrame): ALSModel = {
        import dataset.sqlContext.implicits._
        val r = if ($(ratingCol) != "") col($(ratingCol)).cast(FloatType) else lit(1.0f)
        val ratings = dataset
            .select(col($(userCol)).cast(IntegerType), col($(itemCol)).cast(IntegerType), r)
            .map { row =>
                Book(row.getInt(0), row.getString(1), row.getInt(2))
            }
        val (userFactors, itemFactors) = ALS.train(ratings, rank = $(rank),
            numUserBlocks = $(numUserBlocks), numItemBlocks = $(numItemBlocks),
            maxIter = $(maxIter), regParam = $(regParam), implicitPrefs = $(implicitPrefs),
            alpha = $(alpha), nonnegative = $(nonnegative),
            checkpointInterval = $(checkpointInterval), seed = $(seed))
        val userDF = userFactors.toDF("id", "features")
        val itemDF = itemFactors.toDF("id", "features")
        val model = new ALSModel(uid, $(rank), userDF, itemDF).setParent(this)
        copyValues(model)
    }

}**/
