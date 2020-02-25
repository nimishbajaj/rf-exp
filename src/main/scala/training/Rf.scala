package training

import java.io.{File, FileWriter, IOException}
import java.nio.file.attribute.PosixFilePermissions
import java.nio.file.{Files, Paths}
import java.util.{Calendar, Properties}

import org.apache.log4j.Logger
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer, VectorIndexerModel}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

import scala.collection.immutable.ListMap
import scala.util.parsing.json.JSONObject

import Utilities._

object Rf {
  @transient lazy val log: Logger = Logger.getLogger(getClass.getName)

  // All variables initialized with default configuration
  val experiment_name: String = "churn_prediction"
  val model_name: String = "RF"
  val version_name: String = "1.0"
  val output_location = "/tmp/quaero/experiments"
  val model_location: String = output_location + "/" +
    experiment_name + "/" +
    model_name + "/" +
    version_name + "/"
  val n_key_features: Int = 15
  val source_uri = "data/credit.csv"
  val target_variable = "age"
  val train_split = 0.7 // TODO: P3 Look at time series data split
  val rf_max_depth = 5
  val rf_num_tree = 20
  val universal_seed = 3

  // TODO: set status to finished after model training in mysql also set the time of update
  // TODO: enable unique application id, so that the job can be stopped
  // TODO: Add chart name in the JSON
  // TODO: Figure out the trigger for the training job in backend
  // TODO: Figure out the trigger for the deployment in backend

  def printDefaults(): Unit = {
    val vars = getClass.getDeclaredFields
    log.error("Printing default values")
    for(v <- vars) {
      v.setAccessible(true)
      log.error(s"${v.getName}: ${v.get(this)}")
    }
  }


  def main(arg: Array[String]): Unit = {

    // TODO: Parameterize the code
    // Fetch properties
    printDefaults()

    // TODO: Remove dependency on config properties
    val properties = new Properties
    properties.load(getClass.getResourceAsStream("/config.properties"))
    val application_name: String = properties.get("application.name") + ""

    log.error(s"Application name is $application_name")

    val spark: SparkSession = SparkSession
      .builder
      .master("local")
      .appName(application_name)
      .getOrCreate()

    log.info(s"source_uri $source_uri")
    log.error(s"train_split $train_split")

    // Setup the working directory for the model
    log.error(s"Creating model working directory at $model_location")

    // TODO: Handle exception here - The code below will throw an exception if the directory cannot be created
    Files.createDirectories(Paths.get(model_location),
      PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString("rwxr-x---")))

    log.error(s"Model directory created")

    val data = readDataCsv(source_uri, spark)

    val feature_cols = listFeatures(data, target_variable)

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
    val Array(trainingData, testData) = pdata.randomSplit(Array(train_split, 1.0 - train_split), seed = universal_seed)
    log.error(s"Train test splits generated $train_split")

    val model = trainModel(trainingData, featureIndexer, feature_cols)
    scoreModel(model, testData)

    model.write.overwrite().save(model_location + "/model")
    log.error(s"Model successfully saved in the location $model_location")
  }


  def trainModel(trainingData: Dataset[Row], featureIndexer: VectorIndexerModel,
                 featuresCol: Array[String]): PipelineModel = {
    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setLabelCol(target_variable)
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(rf_max_depth)
      .setNumTrees(rf_num_tree)
      .setSeed(universal_seed)

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
     * 2. Fetch the top 15 features from it
     * 3. Convert the feature importance into a json and store
     */

    // 1
    val featureImportance = {
      featuresCol zip rfModel.featureImportances.asInstanceOf[SparseVector].values
    }.toMap

    // 2
    // sort the map values and take at max top 15 features
    val sortedFeatureImportance = ListMap(featureImportance.toSeq.sortBy(_._2).reverse: _*).take(n_key_features)

    log.error(s"Feature importance is $sortedFeatureImportance")

    // 3
    val jo = new JSONObject(sortedFeatureImportance).toString()
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




}
