
# A regression equation is defined as y= bX + E
#b is the slope , X is the set of predictors and E is a gaussian error term to represent the inherent error in a model

# First we simulate E

# Because E is composed of Independently Identiacally Distributed draws (i.e each observation is one draw from an independent distribution ) ,
# We cant simply sample 1000 numbers from a normal distribution
# We have to sample 1 number from 1 different distribution, 1000 times

E <- vector()
for(i in 1:1000){
  E[i] <- rnorm(1)  
}

# X is just taken as 1000 random numbers

x <- 1:1000

#in this example the slope of .25 is taken

y = x*.25 + E 



plot(y)

hist(E)


reg <- lm(y~x)

plot(predict(reg),resid(reg))
