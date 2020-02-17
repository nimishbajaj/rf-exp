package sample

import org.apache.spark.sql.SparkSession

object indianpincode {
  def main(arg: Array[String]): Unit = {
    val sparkSession: SparkSession = SparkSession
      .builder
      .master("local")
      .appName("rf-exp-1")
      .getOrCreate()

    val csvPO = sparkSession.read.option("inferSchema", true)
      .option("header", true)
      .csv("data/__default__/user/current/all_india_PO.csv")

    csvPO.createOrReplaceTempView("tabPO")
    val count = sparkSession.sql("select * from tabPO").count()
    print(count)
  }
}
