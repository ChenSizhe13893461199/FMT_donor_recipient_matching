library(tidyr)
library(dplyr)
library(readr)

setwd("Your Directory")

folder_path <- "Your Directory"
bracken_files1 <- list.files(folder_path, pattern = "_pathabundance.tsv$", full.names = TRUE)
print(bracken_files1)

data_list1 <- lapply(bracken_files1, function(file) {
  df <- read_tsv(file)
  
  sample_name <- gsub("_pathabundance.tsv$", "", basename(file))
  
  if (ncol(df) >= 2) {
    colnames(df)[2] <- "abundance"  
    df <- cbind(df, sample = sample_name)  
    return(df)
  } else {
    warning(paste("File", file, "does not have enough columns."))
    return(NULL)
  }
})

x1 <- do.call(rbind, data_list1)

x1_filtered <- x1 %>% filter(!grepl("unclassified|Unclassified|UNCLASSIFIED|UNINTEGRATED|UNMAPPED", `# Pathway`))
#g__


x1_filtered <- x1 %>% filter(!grepl("unclassified|Unclassified|UNCLASSIFIED|UNINTEGRATED|UNMAPPED", `# Gene Family`))

# Reshape the data from long to wide format
y <- pivot_wider(x1_filtered, id_cols = c('# Pathway'), 
                 names_from = sample, values_from = 'abundance')

# Reshape the data from long to wide format
y <- pivot_wider(x1_filtered, id_cols = c('# Gene Family'), 
                 names_from = sample, values_from = 'abundance')
y1 <- t(y)
y1 <- as.data.frame(y1)
colnames(y1) <- as.character(y$`# Pathway`) 
y1 <- y1[-1, ]

write.csv(y1, "pathway.csv", row.names = TRUE, fileEncoding = "UTF-8")
# pathway.csv will be further processed in local device
