## Load necessary libraries
library(ggplot2)
library(dplyr)
library(corrplot)
library(GGally)
library(gridExtra)
library(cluster)
library(factoextra)
library(caret)
library(rmarkdown)
library(shiny)


## Load the dataset
data(mtcars)
df <- mtcars


## Basic overview
cat("First few rows of the dataset:\n")
print(head(df))

cat("Summary of dataset:\n")
print(summary(df))

cat("Structure of dataset:\n")
print(str(df))


## Checking for missing values
cat("Checking for missing values:\n")
print(sum(is.na(df)))  # Should be 0 as mtcars has no missing values


## Convert categorical variables
df$cyl <- as.factor(df$cyl)
df$gear <- as.factor(df$gear)
df$am <- as.factor(df$am)
df$vs <- as.factor(df$vs)
df$carb <- as.factor(df$carb)


## Hypothesis Testing
cat("Performing ANOVA on MPG across Cylinder categories:\n")
anova_result <- aov(mpg ~ cyl, data = df)
print(summary(anova_result))

cat("Chi-Square test between transmission (am) and cylinders (cyl):\n")
tab <- table(df$am, df$cyl)
print(chisq.test(tab))


## Univariate Analysis
cat("Distribution of numerical features:\n")
num_cols <- names(df)[sapply(df, is.numeric)]
par(mfrow = c(3, 3))  # Arrange plots in a grid
for (col in num_cols) {
  hist(df[[col]], main = paste("Histogram of", col), col = "skyblue", border = "black")
}
par(mfrow = c(1, 1))  # Reset layout


## Boxplots for categorical variables
cat("Boxplots of numerical variables grouped by cylinder:\n")
ggplot(df, aes(x = cyl, y = mpg, fill = cyl)) +
  geom_boxplot() + theme_minimal() +
  labs(title = "MPG by Cylinder")


## Violin Plot
ggplot(df, aes(x = cyl, y = mpg, fill = cyl)) +
  geom_violin() + theme_minimal() +
  labs(title = "MPG Distribution by Cylinder")


## Pairplot to visualize relationships
ggpairs(df, aes(color = cyl, alpha = 0.5))


## Correlation Analysis
cat("Correlation matrix:\n")
corr_matrix <- cor(df[sapply(df, is.numeric)])
print(corr_matrix)
corrplot(corr_matrix, method = "circle", type = "upper", tl.cex = 0.7)


## Heatmap Visualization
heatmap(as.matrix(corr_matrix), col = terrain.colors(10), scale = "column", margins = c(5,5))


## Clustering Analysis
set.seed(123)
kmeans_result <- kmeans(df[sapply(df, is.numeric)], centers = 3)
df$cluster <- as.factor(kmeans_result$cluster)

fviz_cluster(kmeans_result, data = df[sapply(df, is.numeric)])


## Predictive Modeling
set.seed(123)
trainIndex <- createDataPartition(df$mpg, p = 0.8, list = FALSE)
trainData <- df[trainIndex, ]
testData <- df[-trainIndex, ]


## Linear Regression Model
lm_model <- lm(mpg ~ hp + wt + cyl, data = trainData)
predictions <- predict(lm_model, newdata = testData)
cat("Linear Regression Model Summary:\n")
print(summary(lm_model))


## Decision Tree Model
library(rpart)
tree_model <- rpart(mpg ~ hp + wt + cyl, data = trainData, method = "anova")
pred_tree <- predict(tree_model, newdata = testData)
cat("Decision Tree Model Summary:\n")
print(summary(tree_model))


## Save Plots
png("correlation_plot.png")
corrplot(corr_matrix, method = "circle", type = "upper", tl.cex = 0.7)
dev.off()


## Automated Reporting using R Markdown
rmarkdown::render("report.Rmd")


## Shiny App Integration
ui <- fluidPage(
  titlePanel("MTCARS Data Analysis"),
  sidebarLayout(
    sidebarPanel(selectInput("xvar", "X Variable:", choices = names(df))),
    mainPanel(plotOutput("scatterPlot"))
  )
)

server <- function(input, output) {
  output$scatterPlot <- renderPlot({
    ggplot(df, aes_string(x = input$xvar, y = "mpg")) +
      geom_point() + theme_minimal()
  })
}

shinyApp(ui, server)

cat("EDA Completed Successfully!\n")
