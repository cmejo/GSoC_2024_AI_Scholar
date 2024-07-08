library(ggplot2)
library(jsonlite)

# Load dataset
# confirm path to dataset
dataset <- fromJSON("/home/cmejo/arxiv-dataset/dataset.json")

# Convert to data frame
df <- data.frame(
  Title = sapply(dataset, function(x) x$title),
  Date = as.Date(sapply(dataset, function(x) x$date))
  TextLength = sapply(dataset, function(x) nchar(x$text))

)

# Plot the number of documents over time
ggplot(df, aes(x = Date)) +
  geom_histogram(binwidth = 30, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Number of Documents Over Time",
       x = "Date",
       y = "Number of Documents") +
  theme_minimal()

# Save the plot
ggsave("number_of_documents_over_time.png")

# Plot 2: Distribution of Text Lengths
ggplot(df, aes(x = TextLength)) +
  geom_histogram(binwidth = 100, fill = "green", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Text Lengths",
       x = "Text Length (characters)",
       y = "Frequency") +
  theme_minimal()

# Save the plot
ggsave("distribution_of_text_lengths.png")


# Plot 3: Documents Over Time with Text Length
ggplot(df, aes(x = Date, y = TextLength)) +
  geom_point(color = "red", alpha = 0.5) +
  labs(title = "Documents Over Time with Text Length",
       x = "Date",
       y = "Text Length (characters)") +
  theme_minimal()

# Save the plot
ggsave("documents_over_time_with_text_length.png")

