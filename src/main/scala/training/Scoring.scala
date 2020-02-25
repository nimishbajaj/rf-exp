package training

import org.apache.log4j.Logger
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import Utilities._

object Scoring {

  @transient lazy val log: Logger = Logger.getLogger(getClass.getName)

  val model_location = "/tmp/quaero/experiments/churn_prediction/RF/1.0/model"
  val source_uri = "data/credit.csv"
  val target_variable = "age"

  def main(arg: Array[String]): Unit = {

    val spark: SparkSession = SparkSession
      .builder
      .master("local")
      .appName("rf-scoring-2")
      .getOrCreate()

    // TODO: Hive connector to write the data in the tables

    val data = readDataCsv(source_uri, spark)
    val feature_cols = listFeatures(data, target_variable)

    // Combine features into a feature vector
    val pdata = dataAssember(data, feature_cols)

    // load the model
    val sameModel = PipelineModel.load(model_location)

    // score the data against the model
    scoreModel(sameModel, pdata)
  }
}
