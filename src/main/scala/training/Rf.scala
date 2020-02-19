package training

import java.util.Properties

import org.apache.log4j.Logger
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.SparkSession

object Rf {

  def main(arg: Array[String]): Unit = {

    // Fetch properties
    val properties = new Properties
    properties.load(getClass.getResourceAsStream("/config.properties"))
    val application_name = properties.get("application.name")

    @transient lazy val log = Logger.getLogger(getClass.getName)

    log.error(s"Application name is $application_name")

    val spark: SparkSession = SparkSession
      .builder
      .master("local")
      .appName(application_name.toString)
      .getOrCreate()

    // Config variables
    // TODO: Fix the file read issue while running from jar
    val source_uri = "data/sample_libsvm_data.txt"
    val label_column = "label"
    val train_split = 0.7

    log.info(s"source_uri $source_uri")
    log.error(s"train_split $train_split")

    // TODO: Figure out the data preprocessing bit for the random forest model
    // TODO: Enable picking up model from a location stuff
    // Load and parse the data file, converting it to a DataFrame.
    // TODO: Enable exception handling here
    // TODO: Test CSV file connector
    // TODO: Decouple file reading from this class, keep the output format consistent
    // TODO: Test with Hive connection
    val data = spark.read.format("libsvm").load(source_uri)

    log.error(s"Data has been successfully read from $source_uri")
    log.error(s"Number of rows in the dataset ${data.count}")

    // Automatically identify categorical features, and index them.
    // Here, we treat features with > 4 distinct values as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(train_split, 1.0 - train_split))
    log.error(s"Train test splits generated $train_split")


    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setLabelCol(label_column)
      .setFeaturesCol("indexedFeatures")

    // Chain indexer and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, rf))
    log.error("model pipeline successfully created")

    // Train model. This also runs the indexer.
    log.error("model training started")
    val model = pipeline.fit(trainingData)
    log.error("model training finished")


    // Make predictions.
    log.error("scoring the model against test data")
    val predictions = model.transform(testData)
    log.error("scoring fininshed")


    // Select example rows to display.
    log.error(predictions.select("prediction", "label", "features").show(5))

    // TODO: Add proper logging
    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    log.error(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
    log.error(s"Learned regression forest model:\n ${rfModel.toDebugString}")

    val model_location = "/tmp/mleal-models/rf-exp-2"
    model.write.overwrite().save(model_location)
    log.error(s"Model successfully saved in the location $model_location")
  }
}
