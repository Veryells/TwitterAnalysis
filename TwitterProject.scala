package edu.ucr.cs.cs167.amaha018

import org.apache.spark.SparkConf
import org.apache.spark.beast.SparkSQLRegistration
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, StringIndexer, Tokenizer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{concat_ws, lit, row_number}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}


object TwitterProject {

  def main(args: Array[String]) {

        /////// TASK 1

    val inputFile: String = args(0)

    val conf = new SparkConf
    if (!conf.contains("spark.master"))
      conf.setMaster("local[*]")
    println(s"Using Spark master '${conf.get("spark.master")}'")

    val spark = SparkSession
      .builder()
      .appName("Twitter Task 1")
      .config(conf)
      .getOrCreate()

    try {
      val input = spark.read.format("json")
        .option("sep", "\t")
        .option("inferSchema", "true")
        .option("header", "true")
        .load(inputFile)
      //input.printSchema()
      import org.apache.spark.sql.functions._
      import spark.implicits._
      input.createOrReplaceTempView("root")

      // PART 1 - CLEAN DATA
      val selectQuery: String = "SELECT id, text, entities.hashtags.text AS hashtags, user.description AS user_description, retweet_count, reply_count, quoted_status_id FROM root;"
      val df: DataFrame = spark.sql(selectQuery)
      df.printSchema()
      df.write.json("tweets_clean.json")

      // PART 2 - TOP 20 HASHTAGS
      df.withColumn("tag", explode($"hashtags")).drop("hashtags").createOrReplaceTempView("exploded")
      val hashtagQuery: String = "SELECT tag, COUNT(*) AS cnt FROM exploded GROUP BY tag ORDER BY cnt DESC LIMIT 20;"
      spark.sql(hashtagQuery).show()
    }

          /////// TASK 2
    // Initialize Spark context

    val conf2 = new SparkConf().setAppName("Twitter Task 2")
    // Set Spark master to local if not already set
    if (!conf2.contains("spark.master"))
      conf2.setMaster("local[*]")

    val spark2: SparkSession.Builder = SparkSession.builder().config(conf2)

    val sparkSession = spark2.getOrCreate()
    val sparkContext = sparkSession.sparkContext
    SparkSQLRegistration.registerUDT

    val inputFile2: String = args(1)
    val outputFile: String = args(2)

    val tweetsDFView = sparkSession.read.format("json")
      .option("sep", "\t")
      .option("inferSchema", "true")
      .option("header", "true")
      .load(inputFile2)
      .createOrReplaceTempView("tweetsDF")

    // Remove records with hashtags that aren't in the top 20 array ( or empty )
    val noEmptyHashtagDF = sparkSession.sql(
      s"""SELECT *
         |FROM tweetsDF
         |WHERE exists(hashtags , x -> array_contains(array("ALDUBxEBLoveis","FurkanPalalı","no309","LalOn","chien","job","Hiring","sbhawks","Top3Apps","perdu","trouvé","CareerArc","Job","trumprussia","trndnl","Jobs","ShowtimeLetsCelebr8","hiring","impeachtrumppence","music"), x))
         |""".stripMargin
    ).withColumn("temp1", lit(1)).withColumn("row1",row_number.over(Window.partitionBy("temp1").orderBy("temp1")))
    noEmptyHashtagDF.createOrReplaceTempView("noEmptyHashtag")

    // Get column with first element of intersection of relevant hashtags array with the top 20 array
    val topicDF = sparkSession.sql(
      s"""SELECT element_at(
         |  array_intersect(
         |   hashtags,
         |   array("ALDUBxEBLoveis","FurkanPalalı","no309","LalOn","chien","job","Hiring","sbhawks","Top3Apps","perdu","trouvé","CareerArc","Job","trumprussia","trndnl","Jobs","ShowtimeLetsCelebr8","hiring","impeachtrumppence","music")
         |  ) ,
         |  1
         |)
         |AS topic FROM noEmptyHashtag
         |""".stripMargin
    ).withColumn("temp2", lit(1)).withColumn("row2",row_number.over(Window.partitionBy("temp2").orderBy("temp2")))

    // Join topic column with main DataFrame
    val newDF = noEmptyHashtagDF.join(topicDF, noEmptyHashtagDF("row1") === topicDF("row2"), "inner")

    // 1) Drop hashtags column,
    // 2) Reorder columns
    val finalDF = newDF.drop("hashtags")
      .select("id","text","topic","user_description","retweet_count","reply_count","quoted_status_id")

    // Output file
    finalDF.write.json(outputFile)

      //////////// TASK 3
    val conf3 = new SparkConf().setAppName("Twitter Task 3")
    val inputfile = args(2)
    if (!conf3.contains("spark.master"))
      conf3.setMaster("local[*]")
    println(s"Using Spark master '${conf3.get("spark.master")}'")

    val spark3 = SparkSession
      .builder()
      .appName("Twitter Task 3")
      .config(conf)
      .getOrCreate()

    val t1 = System.nanoTime
    try {
      val tweetPrimer : DataFrame = spark3.read.format("json")
        .option("inferSchema", "true")
        .option("header","true")
        .load(inputfile)

      val tweetData = tweetPrimer.withColumn("topic user_desc", concat_ws(" ", tweetPrimer("user_description"), tweetPrimer("text")))

      tweetData.printSchema()
      tweetData.show()

      val tokenizer = new Tokenizer().setInputCol("topic user_desc").setOutputCol("words")

      val hashingTF = new HashingTF()
        .setInputCol("words").setOutputCol("features")

      val stringIndexer = new StringIndexer()
        .setInputCol("topic")
        .setOutputCol("label")
        .setHandleInvalid("skip")

      val logisticRegression = new LogisticRegression()

      val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, stringIndexer, logisticRegression))

      val paramGrid: Array[ParamMap] = new ParamGridBuilder()
        .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
        .addGrid(logisticRegression.regParam, Array(0.01, 0.1, 0.3, 0.8))
        .build()

      val cv = new TrainValidationSplit()
        .setEstimator(pipeline)
        .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("label"))
        .setEstimatorParamMaps(paramGrid)
        .setTrainRatio(0.8)
        .setParallelism(2)

      val Array(trainingData: Dataset[Row], testData: Dataset[Row]) = tweetData.randomSplit(Array(0.8, 0.2))

      // Run cross-validation, and choose the best set of parameters.
      val logisticModel: TrainValidationSplitModel = cv.fit(trainingData)

      // val numFeatures: Int = logisticModel.bestModel.asInstanceOf[PipelineModel].stages(1).asInstanceOf[HashingTF].getNumFeatures
      // val regParam: Double = logisticModel.bestModel.asInstanceOf[PipelineModel].stages(3).asInstanceOf[LogisticRegressionModel].getRegParam
      // println(s"Number of features in the best model = $numFeatures")
      // println(s"RegParam the best model = $regParam")

      val predictions: DataFrame = logisticModel.transform(testData)
      predictions.select("id","text", "topic", "user_description","label", "prediction").show()

      val multiclassClassificationEvaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")

      val accuracy: Double = multiclassClassificationEvaluator.evaluate(predictions)
      println(s"Accuracy of the test set is $accuracy")

      val t2 = System.nanoTime
      println(s"Applied sentiment analysis algorithm on input $inputfile in ${(t2 - t1) * 1E-9} seconds")
    } finally {
      spark.stop
    }
  }
}
