sobolpickfreeze_th <- function(y1,y2,nboot,ypredmean,ypredsd){
	# cat(var(ypredmean), mean(ypredsd^2), "\n")
	output <- (mean(y1*y2) - mean(y1)*mean(y2))/(var(ypredmean)+mean(ypredsd^2))
	# output <- (mean(y1*y2) - mean(y1)*mean(y2))/var(y1)
	# output <- (apply(y1*y2,1,mean) - apply(y1,1,mean)*apply(y2,1,mean))/apply(y1,1,var)
	if(nboot > 1){
		n <- dim(y1)[2]
		for (i in 1:nboot){
			b <- sample(n,replace = TRUE)
			output <- rbind(output,(apply(y1[,b]*y2[,b],1,mean) - apply(y1[,b],1,mean)*apply(y2[,b],1,mean))/apply(y1[,b],1,var))
		}
	}
	return(output)
}
