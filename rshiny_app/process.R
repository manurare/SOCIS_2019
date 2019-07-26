# Title     : TODO
# Objective : TODO
# Created by: manuel
# Created on: 26/7/19
df = read.csv("correlation.csv", stringsAsFactors = F)
df <- tidyr::separate(data=df,
                      col=value,
                      into="val",
                      sep=",",
                      remove=FALSE)
df <- tidyr::separate(data=df,
                      col=long,
                      into="long",
                      sep=",",
                      remove=FALSE)
df <- tidyr::separate(data=df,
                      col=lat,
                      into="lat",
                      sep=",",
                      remove=FALSE)

df$val <- as.numeric(df$val)
df$lat <- as.numeric(df$lat)
df$long <- as.numeric(df$long)

scalar1 <- function(x) {9*((x-min(x))/max(x)-min(x))+1}
df$val = scalar1(df$val)

saveRDS(df, "./correlation.rds")
# sample_data <- df[c(1:1000),]
# saveRDS(sample_data, "./correlation.rds")