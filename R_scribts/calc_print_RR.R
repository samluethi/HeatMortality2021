# calc RR

# laod data
load("/Users/sam/OneDrive - ETH Zurich/WCR/Projects/Heat/Data/Swiss/data_3cantons_19892017.Rdata")
source("/Users/sam/OneDrive - ETH Zurich/WCR/Projects/Heat/Scripts/R_code/return_RR.R")

names(data_zurich) <- c("date","deaths","T")
RR_ZRH <- return_RR(data_zurich)
write.csv(RR_ZRH,"/Users/sam/OneDrive - ETH Zurich/WCR/Projects/Heat/Data/Swiss/RR_ZRH.csv")

names(data_basel) <- c("date","deaths","T")
RR_BSL <- return_RR(data_basel)
write.csv(RR_BSL,"/Users/sam/OneDrive - ETH Zurich/WCR/Projects/Heat/Data/Swiss/RR_BSL.csv")

names(data_gva) <- c("date","deaths","T")
RR_GVA <- return_RR(data_gva)
write.csv(RR_ZRH,"/Users/sam/OneDrive - ETH Zurich/WCR/Projects/Heat/Data/Swiss/RR_GVA.csv")
