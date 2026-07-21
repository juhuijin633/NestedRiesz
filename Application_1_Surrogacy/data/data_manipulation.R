##### LOAD PACKAGES ######

# list of packages
list.of.packages <- c("R.matlab", 
                      "dplyr", 
                      "stargazer", 
                      "ggplot2", 
                      "tidyr", 
                      "gridExtra", 
                      "stringr",
                      "knitr", 
                      "kableExtra", 
                      "foreach", 
                      "doParallel",
                      "glmnet",
                      "xtable",
                      "tidyverse")

# install packages if needed
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

# load packages
invisible(lapply(list.of.packages, library, character.only = TRUE))


##### DATA MANIPULATION #####

# read data
quarterly <- readMat('quarterly.mat')

# convert to dataframe
quarterly_df <- data.frame(quarterly)

# rename variables from pctedd to employ and tcedd to earn 
quarterly_df <- quarterly_df %>%
  rename_with(~sub("ptcedd", "employ", .), starts_with("ptcedd")) %>%
  rename_with(~sub("tcedd", "earn", .), starts_with("tcedd"))

# create a variable named "zero" which takes the value 0 for all rows
quarterly_df <- quarterly_df %>%
  mutate(zero = 0.001)


# filter out only riverside county
df_river <- quarterly_df %>% filter(river == 1)

# filter out other counties
df_others <- quarterly_df %>% filter(river == 0)

## EMPLOYMENT DATA 

# compute 9-year mean for employment (ptcedd1-36) for riverside
employment_cols <- paste0("employ", 1:36)
employ_df <- df_river %>% 
  mutate(Y_employ = rowMeans(select(., all_of(employment_cols))))

# compute 9-year mean for employment (ptcedd1-36) for other counties
employ_df_others <- df_others %>% 
  mutate(Y_employ = rowMeans(select(., all_of(employment_cols)))) %>%
  select(-e)

## EARNINGS DATA

# compute 9-year mean for earnings (tcedd1-36) for riverside
earn_cols <- paste0("earn", 1:36)
earn_df <- df_river %>% 
  mutate(Y_earn = rowMeans(select(., all_of(earn_cols))))

# compute 9-year mean for earnings (tcedd1-36) for other counties
earn_df_others <- df_others %>% 
  mutate(Y_earn = rowMeans(select(., all_of(earn_cols)))) %>%
  select(e)


## EARNINGS AND EMPLOYMENT OUTCOME ##

# list of employment (ptcedd1-36) and earnings (tcedd1-36) surrogates
employment_cols <- paste0("employ", 1:36)
earn_cols <- paste0("earn", 1:36)

# list of pre-treatment variables (lagged variables)
pretreat_vars <- c(paste0("paid", 1:4), # 4 lagged values for aid
                   paste0("tcpp", 1:10), # 10 lagged values for employment
                   paste0("tcprn", 1:10)) # 10 lagged values for earnings
                   

# list of used covariates
covariates <- c("xsexf", "xhsdip", "xchld05", "single",  
                "grd1720", "grade16", "grd1315", "grade12", "grde911", "white", 
                "hisp", "black", "age", pretreat_vars)

# compute 9-year mean for employment and earnings for riverside
river_data <- df_river %>% 
  mutate(Y_employ = rowMeans(select(., all_of(employment_cols))), 
         Y_earn = rowMeans(select(., all_of(earn_cols)))) %>%
  select(e, all_of(covariates), all_of(employment_cols),
         all_of(earn_cols), Y_employ, Y_earn, all_of(paste0("aid", 1:36)))

# compute 9-year mean for employment (ptcedd1-36) for other counties
# observational data should have no treatment indicator
others_data <- df_others %>% 
  mutate(Y_employ = rowMeans(select(., all_of(employment_cols))),
         Y_earn = rowMeans(select(., all_of(earn_cols)))) %>%
  select(e, all_of(covariates), all_of(employment_cols),
         all_of(earn_cols), Y_employ, Y_earn, all_of(paste0("aid", 1:36)))

write_csv(river_data, "river_data.csv")
write_csv(others_data, "others_data.csv")



