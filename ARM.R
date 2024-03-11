# Load the arules package
# install.packages("arules")
# install.packages("arulesViz", dependencies = TRUE)
# Load the arules package
library(viridis)
library(arules)
library(TSP)
library(data.table)
library(dplyr)
library(devtools)
library(purrr)
library(tidyr)
library(arulesViz)

#Setting working directory
setwd("/Users/vamsigontu/Documents/TM/Project/")
# Documents/TM/Project/transactions.csv

# Load your data
data <- read.transactions("transactions.csv",
                                       rm.duplicates = FALSE,
                                       format = "basket",  ##if you use "single" also use cols=c(1,2)
                                       sep=",",  ## csv file
                                       cols=NULL)
inspect(data)
rules <- arules::apriori(data, parameter = list(support=0.01, confidence=0.01, minlen=2))
summary(rules)
# Display the rules
inspect(rules)

########### Item Frequency

itemFrequencyPlot(data, topN=15, type="absolute", cex.names = 1, main = "Main Data Item Frequency Plot")
# ## Plot of which items are most frequent
itemFrequencyPlot(items(rules), topN=15, type="absolute", main = "Rules Item Frequency Plot")

############ Sort rules by confidence in descending order
sorted_rules_conf <- sort(rules, by = "confidence", decreasing = TRUE)
# Print the sorted rules
inspect(sorted_rules_conf[1:15])

############ Sort rules by support in descending order
sorted_rules_sup <- sort(rules, by = "support", decreasing = TRUE)
# Print the sorted rules
inspect(sorted_rules_sup[1:15])

############ Sort rules by lift in descending order
sorted_rules_lift <- sort(rules, by = "lift", decreasing = TRUE)
# Print the sorted rules
inspect(sorted_rules_lift[1:15])

######## Filter rules to include only those with one of the specified labels on the left-hand side
# labels <- c("US Elections", "Trump", "Biden", "Democracy", "Republic", "Politics", "United States")
rules_filtered_L1 <- subset(rules, lhs = 'US Elections')
rules_filtered_L2 <- subset(rules, lhs = Trump)
rules_filtered_L3 <- subset(rules, lhs = Democracy)
rules_filtered_R1 <- subset(rules, rhs = Politics)
rules_filtered_R2 <- subset(rules, rhs = Biden)
rules_filtered_R3 <- subset(rules, rhs = Republic)

# Visualize top rules sorted on the basis of lift.sorted_rules_lift
subrules_lift_l <- head(sort(rules_filtered_L1, by="lift"), 15)
subrules_lift_r <- head(sort(rules_filtered_R1, by="lift"), 15)
# Visualize top rules sorted on the basis of confidence.sorted_rules_confidence
subrules_conf_l <- head(sort(rules_filtered_L2, by="confidence"), 15)
subrules_conf_r <- head(sort(rules_filtered_R2, by="confidence"), 15)
# Visualize top rules sorted on the basis of confidence.sorted_rules_confidence
subrules_sup_l <- head(sort(rules_filtered_L3, by="support"), 15)
subrules_sup_r <- head(sort(rules_filtered_R3, by="support"), 15)

###### Plotting the html
plot(subrules_lift_l, method="graph", engine="htmlwidget", main="Lift lhs Graph")
plot(subrules_lift_r, method="graph", engine="htmlwidget", main="Lift rhs Graph")
plot(subrules_conf_l, method="graph", engine="htmlwidget", main="Confidence lhs Graph")
plot(subrules_conf_r, method="graph", engine="htmlwidget", main="Confidence rhs Graph")
plot(subrules_sup_l, method="graph", engine="htmlwidget", main="Support lhs Graph")
plot(subrules_sup_r, method="graph", engine="htmlwidget", main="Support rhs Graph")
