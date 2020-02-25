package training

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import training.Rf.{log, target_variable}

object Utilities {

  def listFeatures(data: DataFrame, label_column: String): Array[String] = {
    // Fetch all column names
    log.error("The control reaches here")
    val feature_cols = data
      .columns
      .toList
      .filter(x => x != label_column)
      .toArray

    log.error(s"Label column $label_column")
    log.error("Feature columns \n " + feature_cols)

    feature_cols
  }


  def dataAssember(data: DataFrame, feature_cols: Array[String]): DataFrame = {
    val assembler = new VectorAssembler()
      .setInputCols(feature_cols)
      .setOutputCol("features")

    assembler.transform(data)
  }

  def readDataCsv(source_uri: String, spark: SparkSession): DataFrame = {
    // Load and parse the data file, converting it to a DataFrame.
    val data = spark.read.format("csv")
      .option("header", value = true)
      .option("delimiter", ",")
      .option("mode", "DROPMALFORMED")
      .option("inferSchema", "true")
      .load(source_uri)
      .cache()

    log.error(data.printSchema())
    log.error(s"Data has been successfully read from $source_uri")
    log.error(s"Number of rows in the dataset ${data.count}")
    log.error(s"Number of columns in the dataset ${data.columns.length}")

    data
  }

  def scoreModel(model: PipelineModel, testData: Dataset[Row]): Unit = {
    // Make predictions.
    log.error("scoring the model against test data")
    val predictions = model.transform(testData)
    log.error("scoring fininshed")

    // Select example rows to display.
    log.error(predictions.select("prediction", target_variable, "features").show(5))

    val evaluator = new RegressionEvaluator()
      .setLabelCol(target_variable)
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    log.error(s"Root Mean Squared Error (RMSE) on test data = $rmse")
  }
}
