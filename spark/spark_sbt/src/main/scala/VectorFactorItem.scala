import breeze.linalg.DenseVector

import scala.util.Random

/**
  * Created by les on 26/02/16.
  */
class VectorFactorItem(globalAverage:Double, uBias:Double, iBias:Double, numFeatures:Int,factorVector:DenseVector[Double]) extends Serializable{


    var avg:Double = globalAverage
    var itemBias:Double = iBias
    var userBias:Double = uBias
    val factors = factorVector


}
