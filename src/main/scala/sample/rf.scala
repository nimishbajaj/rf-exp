package sample

import java.time.LocalDateTime

import ml.combust.bundle.BundleFile
import ml.combust.bundle.serializer.SerializationFormat
import org.apache.spark.ml.Pipeline
import ml.combust.mleap.spark.SparkSupport._
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.mleap.SparkUtil
import resource._

object rf {
  def main(arg: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder
      .master("local")
      .appName("rf-exp-1")
      .getOrCreate()


    // TODO: Figure out the data preprocessing bit for the random forest model
    // Load and parse the data file, converting it to a DataFrame.
    val data = spark.read.format("libsvm").load("data/__default__/user/current/sample_libsvm_data.txt")

    // Automatically identify categorical features, and index them.
    // Here, we treat features with > 4 distinct values as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    // Chain indexer and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, rf))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    print(predictions.select("prediction", "label", "features").show(5))

    //TODO: Add proper logging
    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
    println(s"Learned regression forest model:\n ${rfModel.toDebugString}")

    val modelPipeline = SparkUtil.createPipelineModel(uid = "pipeline",
      Array(featureIndexer, rfModel))

//    // Model persistence using MLeap start
//    val sbc = SparkBundleContext().withDataset(predictions)
//
//    // Have a look here
//    // https://docs.databricks.com/_static/notebooks/mleap-model-export-demo-scala.html
//    // figure out the best method for model persistence
//    // TODO: change location of this file, make it relative
//    for(bundle <- managed(BundleFile("jar:file:/tmp/mleap-models/rf-exp-1.zip" ))) {
//      modelPipeline.writeBundle.save(bundle)(sbc)
//    }
//    // Model persistence using MLeap start

    modelPipeline.write.overwrite().save("/tmp/mleal-models/rf-exp-2")
  }
}