package training

import java.io.{File, FileWriter, IOException}
import java.util.{Calendar, Properties}

import com.google.gson.Gson
import org.apache.log4j.Logger
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer, VectorIndexerModel}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.param.shared.HasFeaturesCol
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.codehaus.jackson.map.ObjectMapper

import scala.util.parsing.json.JSONObject

object Rf {
  @transient lazy val log: Logger = Logger.getLogger(getClass.getName)

  // Default model name is the algorithm name + timestamp at which it was created
  val model_name: String = "RF-" + Calendar.getInstance.getTimeInMillis
  val model_location: String = "models/" + model_name

  def main(arg: Array[String]): Unit = {

    // Fetch properties
    val properties = new Properties
    properties.load(getClass.getResourceAsStream("/config.properties"))
    val application_name: String = properties.get("application.name") + ""

    log.error(s"Application name is $application_name")

    val spark: SparkSession = SparkSession
      .builder
      .master("local")
      .appName(application_name)
      .getOrCreate()

    // Config variables
    val source_uri = "data/credit.csv"
    val label_column = "age"
    val train_split = 0.7

    log.info(s"source_uri $source_uri")
    log.error(s"train_split $train_split")

    // Setup the working directory for the model
    log.error(s"Creating model working directory at $model_location")
    val file = new File(model_location)
    file.mkdirs()
    log.error(s"Model directory created")

    // TODO: P2 Enable picking up model from a location stuff
    // TODO: P1 Enable exception handling while reading data
    // TODO: P2 Test with Hive connection

    val data = readDataCsv(source_uri, spark)

    val feature_cols = listFeatures(data, label_column)

    // Combine features into a feature vector
    val pdata = dataAssember(data, feature_cols)

    // Automatically identify categorical features, and index them.
    // Here, we treat features with > 4 distinct values as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(pdata)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = pdata.randomSplit(Array(train_split, 1.0 - train_split))
    log.error(s"Train test splits generated $train_split")


    val model = trainModel(trainingData, label_column, featureIndexer, feature_cols)
    scoreModel(model, testData, label_column)

    model.write.overwrite().save(model_location + "/model")
    log.error(s"Model successfully saved in the location $model_location")
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


  def trainModel(trainingData: Dataset[Row], label_column: String,
                 featureIndexer: VectorIndexerModel, featuresCol: Array[String]): PipelineModel = {
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

    val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
    // log.error(s"Learned regression forest model:\n ${rfModel.toDebugString}")

    /**
     * Generate feature importance chart
     * 1. Find feature importance
     * 2. Fetch the top 10 features from it
     * 3. Convert the feature importance into a json and store
     * 4. Try plotting a radar chart using these feature importance values
      */

    // 1
    val featureImportance = {featuresCol zip rfModel.featureImportances.asInstanceOf[SparseVector].values}.toMap
    log.error(s"Feature importance is $featureImportance")

    // 2
    val jo = new JSONObject(featureImportance).toString()
    val fileWriter = new FileWriter(model_location + "/featureImportance.json")
    try {
      fileWriter.write(jo)
      log.error("Successfully saved the feature importance json")
    } catch {
      case e: IOException => log.error(e)
    } finally {
      fileWriter.close()
      log.error("File closed")
    }

    model
  }


  def scoreModel(model: PipelineModel, testData: Dataset[Row], label_column: String): Unit = {
    // Make predictions.
    log.error("scoring the model against test data")
    val predictions = model.transform(testData)
    log.error("scoring fininshed")

    // Select example rows to display.
    log.error(predictions.select("prediction", label_column, "features").show(5))

    // TODO: P1 Add proper logging
    // TODO: P1 Add a model interpretation utility, use LIME or similar packages for this
    // TODO: P1 Generate graphs and store the data as json files
    // TODO: P3 Spark job as service
    // TODO: P2 Test cases
    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol(label_column)
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    log.error(s"Root Mean Squared Error (RMSE) on test data = $rmse")
  }
}
