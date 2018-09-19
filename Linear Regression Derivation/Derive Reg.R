
#The goal is to prove the R's lm() model is correct by deriving the slopes of a reg model using linear algebra

# A regression model's slope is defined as:

# beta( the slope) = (Xt * x)^-1 * (Xt * Y) where ^-1 is the inverse and Xt is the transpose of x




tv <- read.csv("Advertising.csv")[-1] #read in the advertising dataset and remove the first column

# In our case the Y is the output variable sales 

Y <- tv$Sales

# The X is a matrix containing all data except the Y variable

X <- tv[-4]

# We add a column of 1's to the X matrix to give it an intercept (otherwise it would be regression thru the origin)

X <- cbind(1,X)

# Convert the X to a proper R matrix so that R knows to do matrix operations with it

X <- as.matrix(X)

# Take the transpose of X so we can use it in our equation for beta

xtrans <- t(X)

# beta( the slope) = (Xt * x)^-1 * (Xt * Y) where ^-1 is the inverse and Xt is the transpose of x

beta <- (solve(xtrans %*% X)) %*% (xtrans %*% Y)


beta


#Now we have the coefficients that we manually calculated

#Lets see if R's lm function gets the same result



blackbox <- lm(tv$Sales ~ ., data=tv) # the "." symbol menas all other variables

summary(blackbox) 


blackbox$coefficients # Show the beta's that R calculated

list(blackbox$coefficients,beta) # They Match!
