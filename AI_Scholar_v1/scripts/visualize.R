library(ggplot2)
library(jsonlite)

# Load dataset
# confirm path to dataset
dataset <- fromJSON("/home/cmejo/arxiv-dataset/dataset.json")

# Convert to data frame
df <- data.frame(
  Title = sapply(dataset, function(x) x$title),
  Date = as.Date(sapply(dataset, function(x) x$date))
)

# Plot the number of documents over time
ggplot(df, aes(x = Date)) +
  geom_histogram(binwidth = 30) +
  labs(title = "Number of Documents Over Time",
       x = "Date",
       y = "Number of Documents")

