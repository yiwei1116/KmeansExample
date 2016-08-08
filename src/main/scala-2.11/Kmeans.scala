/**
  * Created by yiwei on 2016/8/7.
  */

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
object Kmeans {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setMaster("local[*]")
      .set("spark.ui.port", "4040")
      .setAppName("Simple Application")
    val sc = new SparkContext(conf)
    val data = sc.textFile("/home/yiwei/spark/data/mllib/kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    // Cluster the data into two classes using KMeans
    val numClusters = 3
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)
    clusters.clusterCenters.foreach(println)
    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("Within Set Sum of Squared Errors = " + WSSSE)

    // Save and load model
    clusters.save(sc, "/home/yiwei/spark/target/org/apache/spark/KMeansExample/KMeansModel")
    val sameModel = KMeansModel.load(sc, "/home/yiwei/spark/target/org/apache/spark/KMeansExample/KMeansModel")
  }}
