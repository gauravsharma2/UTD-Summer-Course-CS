# Databricks notebook source
# MAGIC %md
# MAGIC Information on the DataSet used in the project:
# MAGIC The U.S. Department of Transportation's (DOT) Bureau of Transportation Statistics tracks the on-time performance of domestic flights operated by large air carriers. Summary information on the number of on-time, delayed, canceled, and diverted flights is published in DOT's monthly Air Travel Consumer Report and in this dataset of 2015 flight delays and cancellations.
# MAGIC
# MAGIC Source: https://www.kaggle.com/datasets/usdot/flight-delays
# MAGIC Size: 593 MB 
# MAGIC Class: Big Data

# COMMAND ----------

# dbfs:/FileStore/shared_uploads/shalinronakkumar.kaji@utdallas.edu/airports.csv
# dbfs:/FileStore/shared_uploads/shalinronakkumar.kaji@utdallas.edu/airlines.csv
# Loading and Studying the Dataset.
Airlines_SRC = spark.read\
                .option("header","true")\
                .option("inferSchema","true")\
                .csv("dbfs:/FileStore/shared_uploads/shalinronakkumar.kaji@utdallas.edu/airlines.csv")
Airports_SRC = spark.read\
                .option("header","true")\
                .option("inferSchema","true")\
                .csv("dbfs:/FileStore/shared_uploads/shalinronakkumar.kaji@utdallas.edu/airports.csv")
Flights_SRC = spark.read\
                .option("header","true")\
                .option("inferSchema","true")\
                .csv("dbfs:/FileStore/shared_uploads/shalinronakkumar.kaji@utdallas.edu/flights.csv")

# COMMAND ----------

display(Airlines_SRC)

# COMMAND ----------

display(Airports_SRC)

# COMMAND ----------

display(Flights_SRC)

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.sql.functions import when, col, count, isnan, concat, lit
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf, to_timestamp, date_format

# COMMAND ----------

# Slicing original DF to include necessary colums for faster computation using Spark SQL.
DelayByFlight = Flights_SRC.select('AIRLINE','DEPARTURE_DELAY')
display(DelayByFlight)

# COMMAND ----------

# In this code, we use when() and otherwise() functions to create a new column delay_category, which categorizes the departure delays based on the given constraints. Then, we group the DataFrame by AIRLINE and delay_category, count the occurrences, and finally pivot the data to have delay categories as separate columns for each airline. The result will be a DataFrame showing the counts for ON-TIME, NORM_FD, and DISRUPT_FD for each airline.
# For this we will segregate the TimeDelays(td) into three categories.
# Category 1: Flight Running On-Time: ON_TIME: td < 5 mins.
# Category 2: Flight Delayed (reasonable circumstances): NORM_FD: td |= [5,30]
# Category 3: Flight Delayed (Disruption): DISRUPT_FD: td > 30 mins.

# Use conditional expressions to categorize departure delays
delay_category = (
    when(col("DEPARTURE_DELAY") < 5, "ON-TIME")
    .when(col("DEPARTURE_DELAY") < 30, "NORM_FD")
    .otherwise("DISRUPT_FD")
)

# Group by AIRLINE and delay_category, and count the occurrences
grouped_data = (
    DelayByFlight
    .withColumn("delay_category", delay_category)
    .groupBy("AIRLINE", "delay_category")
    .agg(count("*").alias("count"))
)

# Pivot the data to have delay categories as separate columns
pivoted_data = grouped_data.groupBy("AIRLINE").pivot("delay_category").sum("count").fillna(0)

# Show the final result
pivoted_data.show()

# COMMAND ----------

# Visualising the Delay Statistics for each Airline present in the DataSet.
# Using Matplotlib and Pandas.
# Convert to Pandas DataFrame for plotting
pandas_pivoted_data = pivoted_data.toPandas()

# Plotting grouped Bar Plot
ax = pandas_pivoted_data.plot(
    kind="bar",
    figsize=(15, 6),      # Adjust the figure size as needed
    width=0.5,           # Set the width of each bar
)

# Set the labels for x and y axes
ax.set_xlabel("AIRLINE")
ax.set_ylabel("Count")

# Create a legend for the delay categories
ax.legend(title="Delay_Category")

# Set the positions and labels for x-axis ticks
ax.set_xticks(range(len(pandas_pivoted_data.index)))
ax.set_xticklabels(pandas_pivoted_data["AIRLINE"], rotation=45, ha="right")

# Annotate the count of delay categories for each bar
for i, patch in enumerate(ax.patches):
    x = patch.get_x() + patch.get_width() / 2
    y = patch.get_height()
    ax.annotate(f"{int(y)}", (x, y), textcoords="offset points", xytext=(0, 5), ha='center', rotation=90)


# Show the plot
plt.tight_layout()
plt.show()

# COMMAND ----------

# To create the Punctuality_RankDF DataFrame with the calculated APDF (AirlinePunctualityDelayFactor) column and print it in descending order.

# Calculate the APDF for each airline
pivoted_data = pivoted_data.withColumn(
    "APDF",
    (col("ON-TIME") / (col("DISRUPT_FD") + (0.5*col("NORM_FD")))) # Assigning Penalty Factor of 1 to DISRUPT_FD and 0.5 to NORM_FD
)

# Select the required columns for Punctuality_RankDF
punctuality_rank_df = pivoted_data.select("AIRLINE", "APDF")

# Sort the DataFrame in descending order based on APDF
punctuality_rank_df = punctuality_rank_df.orderBy(col("APDF").desc())

# Show the resulting DataFrame
punctuality_rank_df.show()

# COMMAND ----------

display(punctuality_rank_df)

# COMMAND ----------

# Pre-Processing the Flights Data to get rid off attributes with large no. of missing values.
NOFA = len(Flights_SRC.columns)
print("Original Number of Attributes: ",NOFA)

# COMMAND ----------

AttrLst = Flights_SRC.columns
AttrStat = Flights_SRC.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in AttrLst])
display(AttrStat)

# COMMAND ----------

print("Number of Records we are dealing with : ",Flights_SRC.count())

# COMMAND ----------

# Cleaning the Dataset. We will drop the columns which play no significant role in prediction.
col_insig = ['YEAR','MONTH','DAY_OF_WEEK','TAIL_NUMBER','SCHEDULED_TIME','CANCELLED','CANCELLATION_REASON','FLIGHT_NUMBER','WHEELS_OFF','WHEELS_ON',]
Flight_Clean = Flights_SRC.drop(*col_insig)
display(Flight_Clean)

# COMMAND ----------

temp_drop = ['DESTINATION_AIRPORT','SCHEDULED_DEPARTURE','AIR_SYSTEM_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY']
PCorr_Attr = Flight_Clean.drop(*temp_drop)
# Convert PySpark DataFrame to Pandas DataFrame
pandas_df = PCorr_Attr.toPandas()

# Calculate the correlation matrix
correlation_matrix = pandas_df.corr()

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# COMMAND ----------

# Joining the Flight_Clean DF with AirportDF to draw further insights.
MegaData = Airports_SRC.join(Flight_Clean, Airports_SRC["IATA_CODE"] == Flight_Clean["ORIGIN_AIRPORT"], "right")
display(MegaData)

# COMMAND ----------

# Attribute Selection for MegaData.
drop_attr = ['IATA_CODE','DESTINATION_AIRPORT','TAXI_OUT','ELAPSED_TIME','TAXI_IN','DISTANCE','COUNTRY','LATITUDE','LONGITUDE','WEATHER_DELAY']
FLPredict = MegaData.drop(*drop_attr)
display(FLPredict)

# COMMAND ----------

Flight_Clean.columns

# COMMAND ----------

coldrop = ['AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY']
FLML_Maker = Flight_Clean.drop(*coldrop)

# COMMAND ----------

Fl_Regres = FLML_Maker.join(punctuality_rank_df, on="AIRLINE", how="inner")

# COMMAND ----------

display(Fl_Regres)

# COMMAND ----------

# MAGIC %md
# MAGIC ##*Exploratory Data Analysis finished. Relevant Insights drawn. Visualization produced.*
# MAGIC ##*Machine Learning on Flight-Data commences.*

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pyspark.ml.feature import StringIndexer

# COMMAND ----------

# MAGIC %md
# MAGIC *Applying Label Encoder to convert Text to Numbers.*

# COMMAND ----------

# List of categorical columns to label encode
categorical_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DAY']

# Loop through each categorical column and apply StringIndexer
for col_name in categorical_cols:
    indexer = StringIndexer(inputCol=col_name, outputCol=col_name + "_index")
    Fl_Regres = indexer.fit(Fl_Regres).transform(Fl_Regres)

# Drop the original categorical columns, if needed
Fl_Regres = Fl_Regres.drop(*categorical_cols)


# COMMAND ----------

X = Fl_Regres.drop('ARRIVAL_DELAY')
print("Records : {} and Attributes : {}".format(X.count(), X.columns))

# COMMAND ----------

y = Fl_Regres.select('ARRIVAL_DELAY')
display(y)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Split into train and test sets.

# COMMAND ----------

# Define the test size proportion (20% test size)
test_size = 0.20

# Randomly split X and y DataFrames into train and test sets
X_train, X_test = X.randomSplit([1 - test_size, test_size], seed=2)
y_train, y_test = y.randomSplit([1 - test_size, test_size], seed=2)

X_train = X_train.na.drop()
X_test = X_test.na.drop()
y_train = y_train.na.drop()
y_test = y_test.na.drop()

# COMMAND ----------

import random

# Sample 1,000,000 records from X_train and y_train dataframes (1M)
sample_size = 1000000
X_train_sampled = X_train.sample(withReplacement=False, fraction=sample_size/X_train.count(), seed=42)
y_train_sampled = y_train.sample(withReplacement=False, fraction=sample_size/y_train.count(), seed=42)

# COMMAND ----------

import random

# Step 2: Convert sampled X_train and y_train dataframes to RDDs
X_train_rdd = X_train_sampled.rdd.zipWithIndex().map(lambda row: (row[1], row[0].asDict()))
y_train_rdd = y_train_sampled.rdd.zipWithIndex().map(lambda row: (row[1], row[0]['ARRIVAL_DELAY']))

# Step 3: Initialize model weights (coefficients)
num_features = len(X_train.columns)
weights = [random.random() for _ in range(num_features)]

# Step 4: Implement MapReduce functions

# Function to calculate the gradient and update the weights for each partition
def gradient_and_update(iterator):
    local_gradients = [0.0] * num_features
    num_samples = 0

    for row in iterator:
        x_features = row[1][0]
        y_label = row[1][1]

        # Convert feature values to float if they are numeric, otherwise, ignore them
        x_features = [float(x) if x.isdigit() else 0.0 for x in x_features]

        # Check if all feature values are numeric, only then proceed with the prediction
        if all(isinstance(x, float) for x in x_features):
            prediction = sum(w * x for w, x in zip(weights, x_features))
            error = prediction - y_label

            for i in range(num_features):
                local_gradients[i] += error * x_features[i]

            num_samples += 1

    yield (local_gradients, num_samples)

# Function to merge the gradients from all partitions
def merge_gradients(g1, g2):
    gradients1, samples1 = g1
    gradients2, samples2 = g2

    total_gradients = [g1 + g2 for g1, g2 in zip(gradients1, gradients2)]
    total_samples = samples1 + samples2

    return (total_gradients, total_samples)

# Step 5: Train the model using MapReduce (Gradient Descent)

learning_rate = 0.1
num_iterations = 100

for _ in range(num_iterations):
    gradients_and_samples = X_train_rdd.join(y_train_rdd).mapPartitions(gradient_and_update)
    total_gradients, total_samples = gradients_and_samples.reduce(merge_gradients)
    
    # Update the weights using the average gradient
    weights = [weight - (learning_rate / total_samples) * grad for weight, grad in zip(weights, total_gradients)]

# Step 6: Evaluate the model on the test data

# Convert X_test and y_test dataframes to RDDs
X_test_rdd = X_test.rdd.zipWithIndex().map(lambda row: (row[1], row[0].asDict()))
y_test_rdd = y_test.rdd.zipWithIndex().map(lambda row: (row[1], row[0]['ARRIVAL_DELAY']))

# Function to make predictions on test data
def predict(iterator):
    for row in iterator:
        x_features = row[1]
        # Convert all feature values to float if they are numeric, otherwise, ignore them
        x_features = [float(x) if isinstance(x, (int, float)) else 0.0 for x in x_features]
        prediction = sum(w * x for w, x in zip(weights, x_features))
        yield prediction

# Make predictions on the test data
predictions_rdd = X_test_rdd.mapPartitions(predict)

# Collect the predicted values and actual values from y_test
predicted_values = predictions_rdd.collect()
actual_values = y_test_rdd.map(lambda x: x[1]).collect()  # Extract the actual ARRIVAL_DELAY values from the RDD


# COMMAND ----------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Combine actual_values and predicted_values RDDs using zip
zipped_values = y_test_rdd.zip(predictions_rdd)

# Filter out elements where either actual_value or predicted_value is missing
zipped_values = zipped_values.filter(lambda x: x[0][1] is not None and x[1] is not None)

# Extract the actual ARRIVAL_DELAY values and predicted values from the filtered RDD
actual_values_rdd = zipped_values.map(lambda x: x[0][1]).map(float)
predicted_values_rdd = zipped_values.map(lambda x: x[1]).map(float)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(actual_values_rdd.collect(), predicted_values_rdd.collect())

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(actual_values_rdd.collect(), predicted_values_rdd.collect())

# Calculate R-squared (Coefficient of Determination)
r_squared = r2_score(actual_values_rdd.collect(), predicted_values_rdd.collect())

# Print the evaluation metrics
print("Evaluation Metrics of Regression Model with Stochastic Gradient Descent:")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r_squared)
