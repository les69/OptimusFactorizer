/**
  * Created by les on 16/02/16.
  */
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object TestWrdCount {
    def main(args: Array[String]) {
        val txtFile = "src/main/scala/TestWrdCount.scala"
        val conf = new SparkConf().setAppName("Sample Application")
        val sc = new SparkContext(conf)
        val txtFileLines = sc.textFile(txtFile , 2).cache()
        val numAs = txtFileLines .filter(line => line.contains("val")).count()
        println("Lines with val: %s".format(numAs))
    }
}
