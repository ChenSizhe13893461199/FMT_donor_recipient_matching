library(tidyr)
library(dplyr)
library(readr)
setwd("Your Directory")

folder_path <- "Your Directory"

bracken_files1 <- list.files(folder_path, pattern = "bacteria_bracken_report.txt$", full.names = TRUE)
print(bracken_files1)

data_list1 <- lapply(bracken_files1, function(file) {
  df <- read_tsv(file)
  df$Sample <- gsub("bacteria_bracken_report", "", basename(file))  # Extract sample name from file title
  return(df)
})
x1 <- do.call(rbind, data_list1)

# Reshape the data from long to wide format
y <- pivot_wider(x1, id_cols = c(name,taxonomy_id), 
                 names_from = Sample, values_from = new_est_reads)
x <- y[,-c(1,2)]

rownames(x) <- y$name
x <- t(x)
x <- as.data.frame(x)
write.csv(x, "bacteria.csv", row.names = TRUE, fileEncoding = "UTF-8") # bacteria


bracken_files1 <- list.files(folder_path, pattern = "archaea_bracken_report.txt$", full.names = TRUE)
print(bracken_files1)

data_list1 <- lapply(bracken_files1, function(file) {
  df <- read_tsv(file)
  df$Sample <- gsub("archaea_bracken_report", "", basename(file))  # Extract sample name from file title
  return(df)
})
x1 <- do.call(rbind, data_list1)

# Reshape the data from long to wide format
y <- pivot_wider(x1, id_cols = c(name,taxonomy_id), 
                 names_from = Sample, values_from = new_est_reads)
x <- y[,-c(1,2)]
rownames(x) <- y$name
x <- t(x)
x <- as.data.frame(x)
write.csv(x, "archaea.csv", row.names = TRUE, fileEncoding = "UTF-8") # archaea

bracken_files1 <- list.files(folder_path, pattern = "fungi_bracken_report.txt$", full.names = TRUE)
print(bracken_files1)

data_list1 <- lapply(bracken_files1, function(file) {
  df <- read_tsv(file)
  df$Sample <- gsub("fungi_bracken_report", "", basename(file))  # Extract sample name from file title
  return(df)
})
x1 <- do.call(rbind, data_list1)

# Reshape the data from long to wide format
y <- pivot_wider(x1, id_cols = c(name,taxonomy_id), 
                 names_from = Sample, values_from = new_est_reads)
x <- y[,-c(1,2)]
rownames(x) <- y$name
x <- t(x)
x <- as.data.frame(x)
write.csv(x, "fungi.csv", row.names = TRUE, fileEncoding = "UTF-8") # fungi

bracken_files1 <- list.files(folder_path, pattern = "viral_bracken_report.txt$", full.names = TRUE)

# 打印找到的文件
print(bracken_files1)

data_list1 <- lapply(bracken_files1, function(file) {
  df <- read_tsv(file)
  df$Sample <- gsub("viral_bracken_report", "", basename(file))  # Extract sample name from file title
  return(df)
})
x1 <- do.call(rbind, data_list1)

# Reshape the data from long to wide format
y <- pivot_wider(x1, id_cols = c(name,taxonomy_id), 
                 names_from = Sample, values_from = new_est_reads)
x <- y[,-c(1,2)]
rownames(x) <- y$name
x <- t(x)
x <- as.data.frame(x)
write.csv(x, "viral.csv", row.names = TRUE, fileEncoding = "UTF-8") # virus
