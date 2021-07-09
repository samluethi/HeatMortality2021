################################################################################
# FUNCTION FOR CROSSBASIS FROM DLNM USING PYTHON
#   REQUIRES dlnm VERSION 2.2.0 AND ON
################################################################################
return_RR <- function(data) {
  
  # load required libraries
  library(dlnm) ; library(mixmeta) ; library(splines) ; library(tsModel)
  library(mgcv) 
  library(data.table); library(reshape2); library(ggplot2)

  ## DEFINITON OF THE PARAMETERS
  # SPECIFICATION OF THE EXPOSURE FUNCTION
  varfun <- "bs"
  vardegree <- 2
  varper <- c(10,75,90)
  
  # SPECIFICATION OF THE LAG FUNCTION
  lag <- 21
  lagnk <- 3
  
  # DEGREE OF FREEDOM FOR SEASONALITY
  dfseas <- 8
  
  # DEFINE THE OUTCOME
  out <- "deaths"
  
  # MODEL FORMULA (original includes dow)
  formula <- deaths ~ cb +  ns(date,df=round(dfseas*length(date)/365.25)) # adjusted for dow
  
  #### - Insitu  ####
  # DEFINE THE CROSSBASIs
  argvar <- list(fun=varfun, degree=vardegree,
                 knots=quantile(data$T,varper/100,na.rm=T), 
                 Bound=c(-20,40))
  
  
  arglag <- list(fun="ns",knots=logknots(lag,lagnk))
  cb <- crossbasis(data$T,lag=lag,argvar=argvar,arglag=arglag)
  
  # RUN THE MODEL AND OBTAIN PREDICTIONS
  model <- glm(formula,data,family=quasipoisson,na.action="na.exclude")
  
  # REDUCTION TO OVERALL CUMULATIVE
  reduced <- crossreduce(cb,model,cen=mean(data$T,na.rm=T))
  
  # RETURN RR values
  RR <- data.frame(reduced$predvar,unname(reduced$RRfit),unname(reduced$RRlow),unname(reduced$RRhigh))
  names(RR) <- c("temp","RRfit","RRlow","RRhigh")
  
  return(RR)
}