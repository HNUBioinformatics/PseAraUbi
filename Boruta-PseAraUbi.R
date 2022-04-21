
if(!requireNamespace('Boruta',quietly=T))
  install.packages('Boruta')
library('Boruta')


if(!requireNamespace('mlbench',quietly=T))
  install.packages('mlbench')
library('mlbench') 
data('PseAraUbi')
help(PseAraUbi)  

summary(PseAraUbi)  
rawdim<-dim(PseAraUbi) 
head(PseAraUbi) 
PseAraUbi <- na.omit(PseAraUbi) 
head(PseAraUbi) 
rawdim[1]-dim(PseAraUbi)[1]  


set.seed(1)  
Boruta.PseAraUbi <- Boruta(V4 ~ ., data = PseAraUbi, doTrace = default, maxRuns = 500,getImp = getImpRfZ)  

Boruta.PseAraUbi  
str(Boruta.PseAraUbi)  

Boruta.PseAraUbi$finalDecision  
Boruta.PseAraUbi[["finalDecision"]][["V1"]]  
Boruta.PseAraUbi[["ImpHistory"]]  
write.table(Boruta.PseAraUbi$finalDecision,'FinalDecision.txt',sep="\t",quote=F,col.names=F)

confirmedFormula<-getConfirmedFormula(Boruta.PseAraUbi) 
Confirmed.Boruta.PseAraUbi <- Boruta(confirmedFormula, data = PseAraUbi, doTrace = default, maxRuns = 600,getImp = getImpRfZ)
Confirmed.Boruta.PseAraUbi  
