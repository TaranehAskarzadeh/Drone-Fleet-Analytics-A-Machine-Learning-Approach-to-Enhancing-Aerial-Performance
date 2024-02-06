

### Stat 712 Lab 2: classification methods: Logistic regression,  LDA, QDA and KNN
#########################################################################################


	##  data set:  Smarket (The Stock Market Data)  in ISLR library
	##  This data set consists of percentage returns for the S&P 500 stock index over 1, 250 days, from the
      ##  beginning of 2001 until the end of 2005.

	##  For each date, we have recorded the percentage returns for each of the five previous trading days, Lag1
      ##  through Lag5.
	##  Volume: the number of shares traded  on the previous day, in billions
	##  Today:the percentage return on the date in question
	##  Direction: whether the market was Up or Down on this date


	library(ISLR)     ## data is in this library 
	names(Smarket)    ##  show the variable names

	dim(Smarket)	##  show sample size and number of variables
	summary (Smarket) ## five number summary of each variable.


	## cor():  used to produce correlation matrix

	cor(Smarket) 	## error occurs, since Direction is not numeric
	cor(Smarket [,-9]) ## do not calculate correlation of Direction.

## lag variables and Today has almost zero correlation:  it is not possible to predict stock price using previous days returns

## Volume and Year are substantial correlated. Plot Volume over time.

	attach (Smarket )
	plot(Volume )	  ## Volume is increasing over time: the average number of shares traded daily increased from 2001 to 2005




##########################
##  logistic regression
##########################


	## Use Lag1 to Lag5, Volume to predict Direction.
	## glm(): fit logistic regression
	## syntax is similar to lm(),  but we need to specify 'family=binomial': the response follows binomial(bernoulli)

	glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume , data=Smarket, family =binomial )
		## family=binomial  is to fit logistic regression.
	summary (glm.fit )

	## all the p-values are pretty large, this means it is not possible to accurately predict stock price.

	coef(glm.fit)   ## only show the fitted coefficient

	## predict(): can be used to predict the probability that the market will go up, given values of the predictors
	## type="response":   used to output the probabilities  Pr(Y=1|X).
	## If no input is provided, predict() output the predicted probabilities at training data

	glm.probs =predict (glm.fit , type="response")
	glm.probs [1:10]		##  show the first 10 outputs





	## contrasts() function indicates that R has created a dummy variable with a 1 for Up.

	contrasts(Direction)    ## show the coding of response variable.


	## transfer the fitted probabilities to Down and UP

	glm.pred=rep ("Down",1250)
	glm.pred[glm.probs >.5]="Up"	## if probability > 0.5 is predicted to be Up


	## table():  can be used to produce a confusion matrix in order to determine how many
	##           observations were correctly or incorrectly classified.


	table(glm.pred,Direction) 		## confusion matrix

	mean(glm.pred==Direction)		## proportion correctly predicted


	## evaluate test error:  split the data into Training data and Test data. 

	## data from 2001 to 2004 as training data;  data in 2005 as test data

	train =(Year <2005)			## training data
	Smarket.2005= Smarket [!train ,]     ## test data:  !train means not in train
	dim(Smarket.2005)

	Direction.2005=Direction [!train]    ## get the true label of the test data


	## use 'subset' argument to fit model based a subset of the data

	glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket ,family =binomial,subset =train )
	glm.probs =predict (glm.fit,Smarket.2005 , type="response")

	glm.pred=rep("Down",252)
	glm.pred[glm.probs>.5]="Up"
	table(glm.pred,Direction.2005)

	mean(glm.pred==Direction.2005)

	mean(glm.pred!= Direction.2005)	## prediction error: glm.pred not equal to the true label




	## all p-values are very large. We delete several variables to see whether the prediction error reduces or not
	## only keep Lag1 and Lag2

	glm.fit=glm(Direction~Lag1+Lag2 ,data=Smarket ,family =binomial ,subset =train)
	glm.probs =predict(glm.fit ,Smarket.2005, type="response")
	glm.pred=rep("Down" ,252)
	glm.pred[glm.probs >.5]="Up"
	table(glm.pred,Direction.2005)
	mean(glm.pred==Direction.2005)   ## prediction accuracy improved

	## predict probabilities at given Lag1 and Lag2
	predict(glm.fit ,newdata =data.frame(Lag1=c(1.2 ,1.5) , Lag2=c(1.1 , -0.8) ),type ="response")






#########################################
##  LDA  Linear Discriminant Analysis
###########################################




	## lda():   is used to fif LDA.  is in MASS library.
	## syntax is similar to glm() and lm().

	library (MASS)
	lda.fit=lda(Direction~Lag1+Lag2,data=Smarket,subset =train)
	lda.fit     ## coefficient estimates are the coefficient in the delta_k(x)

	plot(lda.fit)  ## plot estimated delta_k(x) for each k.

	lda.pred=predict (lda.fit , Smarket.2005)    ## predict(): contains classes of response variable; posterior co
	names(lda.pred)  

		##posterior: is a matrix whose kth column contains the
		## posterior probability that the corresponding observation belongs to the kth class
		## x: contains the linear discriminants delta_k(x).

	lda.pred$posterior
	lda.pred$x

	lda.class =lda.pred$class    ## class: contains predicted class
	table(lda.class,Direction.2005)
	mean(lda.class == Direction.2005)

	sum(lda.pred$posterior[ ,1] >=.5)  ## use 0.5 as threshold
	sum(lda.pred$posterior[,1]<.5)


	sum(lda.pred$posterior[,1]>.9)    ## we can change threshold 0.5. For example change it to 0.9







#########################################
##  QDA  Quadratic Discriminant Analysis
###########################################


	## qda():  fit QDA to data

	qda.fit=qda(Direction~Lag1+Lag2 ,data=Smarket ,subset =train)
	qda.fit


	qda.class =predict (qda.fit ,Smarket.2005)$class
	table(qda.class,Direction.2005)

	mean(qda.class == Direction.2005)








#########################################
##  KNN  
###########################################


	## knn(): fit KNN to data.  many packages available for knn. We use 'class' library
	## knn(): has 4 arguments
	##  train.X: a matrix contains training inputs
	##  test.X:  a matrix contains testing inputs
	##  train.Y: a vector of the training response
	##  K:  the number of  neighbors


	library (class)   ## knn() is in this library 
	train.X=cbind(Lag1,Lag2)[train ,]      ## cbind(): column combine two vectors.   
	test.X=cbind(Lag1 ,Lag2)[!train ,]
	train.Direction =Direction [train]


	set.seed(1)   ## random split the ties in the training data, to guarantee reproducity.
	knn.pred=knn (train.X,test.X,train.Direction ,k=1)
	table(knn.pred ,Direction.2005)




	knn.pred=knn (train.X,test.X,train.Direction ,k=3)
	table(knn.pred ,Direction.2005)






#########################################
##  Application of KNN  to data Caravan
###########################################





	## We  apply the KNN approach to the Caravan data set in the ISLR library

	## n=5,822    p=85.
	## response variable: Purchase, indicating whether an individual purchases caravan insurance policy.


	dim(Caravan)
	attach (Caravan )
	summary(Purchase )


	## The KNN classifier predicts the class of a given test observation by
	## identifying the observations that are nearest to it, the scale of the variables matters.

	## In practice, it is better to standardize the data so that all
	## variables are given a mean of zero and a standard deviation of one.
	## Then all variables will be on a comparable scale

	standardized.X=scale(Caravan [,-86])   ## scale(): can standardize the variable
	var(Caravan [,1])
	var(Caravan [,2])
	var( standardized.X[,1])
	var( standardized.X[,2])


	test =1:1000                          ## first 1000 observations as testing data set
	train.X=standardized.X[-test,]       ## the rest observations as training data set
	test.X=standardized.X[test,]
	train.Y=Purchase[-test]
	test.Y=Purchase[test]

	set.seed (1)
	knn.pred=knn(train.X,test.X,train.Y,k=1)
	mean(test.Y!= knn.pred)
	mean(test.Y!="No")

	table(knn.pred, test.Y)    ## confusion matrix



	knn.pred=knn (train.X,test.X,train.Y,k=3)
	table(knn.pred ,test.Y)


	knn.pred=knn (train.X,test.X,train.Y,k=5)
	table(knn.pred,test.Y)


	## compare with logistic regression
	glm.fit=glm(Purchase~.,data=Caravan,family=binomial,subset =-test)
	glm.probs =predict(glm.fit ,Caravan [test ,], type="response")
	glm.pred=rep ("No" ,1000)
	glm.pred[glm.probs >.5]="Yes"     ## use 0.5 as threshold
	table(glm.pred ,test.Y)



	glm.pred=rep("No" ,1000)
	glm.pred[glm.probs >.25]=" Yes"    ## use 0.25 as threshold
	table(glm.pred ,test.Y)































































