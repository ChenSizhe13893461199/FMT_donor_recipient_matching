library(tidyr)
library(dplyr)
library(readr)

setwd("Your Directory") # please replace it with your directory


folder_path <- "Your Directory"

bracken_files1 <- list.files(folder_path, pattern = "_KOs.tsv$", full.names = TRUE)
print(bracken_files1)

data_list1 <- lapply(bracken_files1, function(file) {
  df <- read_tsv(file)
  sample_name <- gsub("_KOs.tsv$", "", basename(file))
  
  if (ncol(df) >= 2) {
    colnames(df)[2] <- "GeneFamily"  
    df <- cbind(df, sample = sample_name)  
    return(df)
  } else {
    warning(paste("File", file, "does not have enough columns."))
    return(NULL)
  }
})

x1 <- do.call(rbind, data_list1)

x1_filtered <- x1 %>% filter(!grepl("unclassified|UNGROUPED\\|g__|\\|g__|UNMAPPED", `# Gene Family`))

# Reshape the data from long to wide format

y <- pivot_wider(x1_filtered, id_cols = c('# KO'), 
                 names_from = sample, values_from = 'KO')
y1 <- t(y)
y1 <- as.data.frame(y1)
colnames(y1) <- as.character(y$`# Gene Family`)  
y1 <- y1[-1, ]



write.csv(y1, "ko.csv", row.names = TRUE, fileEncoding = "UTF-8")



