library(tidyr)
library(dplyr)
library(readr)

# 设置工作目录
setwd("F:/ko")

# 设置文件夹路径
folder_path <- "F:/ko"

# 列出所有以 '_pathabundance.tsv' 结尾的文件   _pathabundance.tsv$
bracken_files1 <- list.files(folder_path, pattern = "_KOs.tsv$", full.names = TRUE)

# 打印找到的文件
print(bracken_files1)

data_list1 <- lapply(bracken_files1, function(file) {
  df <- read_tsv(file)
  
  # 提取样本名称
  sample_name <- gsub("_KOs.tsv$", "", basename(file))
  
  # 确保数据框包含至少两列
  if (ncol(df) >= 2) {
    colnames(df)[2] <- "GeneFamily"  # 将第二列重命名为abundance
    df <- cbind(df, sample = sample_name)  # 添加第三列并命名为sample
    return(df)
  } else {
    warning(paste("File", file, "does not have enough columns."))
    return(NULL)
  }
})

x1 <- do.call(rbind, data_list1)

# 过滤掉未分类的 pathway

x1_filtered <- x1 %>% filter(!grepl("unclassified|Unclassified|UNCLASSIFIED|UNINTEGRATED|UNMAPPED", `# Gene Family`))

# Reshape the data from long to wide format

# Reshape the data from long to wide format
y <- pivot_wider(x1_filtered, id_cols = c('# Gene Family'), 
                 names_from = sample, values_from = 'GeneFamily')
y1 <- t(y)
y1 <- as.data.frame(y1)
colnames(y1) <- as.character(y$`# Gene Family`)  # 设置列名
y1 <- y1[-1, ]

# 输出 CSV 文件
write.csv(y1, "ko.csv", row.names = TRUE, fileEncoding = "UTF-8")