# Exploratory analysis in R

library(ggplot2)

clean <- read.csv("data/processed/clean_ticks.csv")
clean$ts <- as.POSIXct(clean$ts)

p <- ggplot(clean[1:5000, ], aes(x = ts, y = (bid + ask) / 2)) +
  geom_line() +
  ggtitle("Mid price (first 5k ticks)")

print(p)
