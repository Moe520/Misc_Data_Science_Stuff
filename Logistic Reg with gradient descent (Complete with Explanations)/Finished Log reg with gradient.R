
set.seed(1234567)

setwd("C:/Users/PC/Dropbox/AQM Class Work/AQM Nov 30/Logistic Reg with gradient descent/")

cancer <- read.csv("BreastCancer.csv")[-1]

attach(cancer)


# First let's remove incomplete cases as they can screw with our functions and generate NA's


cancer <- cancer[complete.cases(cancer),]

#our output value is the binary variable "Class" which must take a value of 0 or 1

# We'll call this T because it's our target vector that we're trying to predict

T <- cancer$Class

#our predictors are columns 1:9 in the dataset

X <- cancer[,1:9]

# We need to add the beta0 (intercept) which is just a column of 1's:

# generate a vector of 1's that's the same length as a column of X and call it Xb to denote that it has this xtra column

beta0 = rep(1,length(X$Cell.size))

Xb = cbind(beta0,X)

#Now we initialize the weights by making a vector that is as many elements as Xb has columns
# We pick random numbers for the weights

w = rnorm(length(Xb))

# The standard linear regression function we work with is yhat = (Xb)(w)

# The function we will use is the logistic version of it : 1 / 1 + e^ - (yhat)

# So first let's put together the original linear function, and so as not to confuse yhat of this function with the 
# logistic function that we're trying to make, we'll refer to the linear yhat as "z"

#By default if we tell R to multiply Xb by w it will just multiply element (row wise) by element and then output a matrix

# We want it instead to multiply each element of vector w by the corresponding column in Xb
# so we want the output to be scalar z = w[1]*Xb[,1] + w[2]*Xb[,2] + ....... + w[n] * Xb[,n]

#Since R will take each ROW of what we give it and multiply that by w[i], we will trick R into doing colwise multiplication by transposing Xb
# Then we will sum all those terms to get the scalar z 

z = colSums(  t(Xb) * w )
z = as.numeric(z)
# Now we need to transform each z to its logistic (sigmoid) version that is gotten by logistic_yhat = 1 / 1 + e^ - (old_yhat) where the old yhat is our z

#Since we may be doing this action more than once, it is time-saving to have a function that takes each yhat you give it and 
#converts that into the sigmoid version of yhat which we will call Y

sigmoid = function(z){
  
  return( 1  /  (1+exp(-z))  )
}


# Run each iteration of Z through the sigmoid generator and append it to Y

Y = sigmoid(z)

#for(i in length(z)){
#  Y[i]=sigmoid(z[i])
#}



# Now we build the cross entropy function that calculates our error:

# In linear regression we used standard error E = sum(t - y)^2 which was fine because it was assumed to be gaussian distributed

# But in logistic regression we can't make that assumption because all targets are either 1 or 0 and all outputs of the sigmoid function are between 0 and 1

# Therefore we use the cross-entropy function:

# When the actual target T is 1, the error at that iteration is - (log Yhat) 

# When the actual target T is 0, the error at that iteration is - ( 1 - (log yhat)      )

# When the the Target is equal to our Yhat : The output of the cross-entropy is 0 

# When the the Target differs from Yhat : The output of the cross-entropy is an error that reflects how far off target this prediction was 

cross_entropy = function(T,Y){
 E <<- 0  # initialize E at 0 ; Then each iteration will either do nothing to it (if guess is correct) , or (if incorrect) decrease it by an amount proportional to how far off we were 
  for(i in length(Y)){
   if (T[i] == 1){E <<- E - log(Y[i])} 
   else { E <<- E -  log( 1- Y[i] )  }
  }
 return(E)
 print(E)
}

cross_entropy(T,Y)

# Now we will run gradient descent for 100 iterations at a learning rate of  0.01

learningrate= 0.01

  
cross_entropy(T,Y)

for(i in 1:100){
  cross_entropy(T,Y)
  print(as.numeric(E))
  
  w <<- w + learningrate * ( colSums(   t(t(Xb))*(T-Y) )  )
  Y <<- sigmoid( colSums( t(Xb) * w )  )
}

cross_entropy(T,Y)

# Print the most recent error

as.numeric(E)


#print the final weights

w




  