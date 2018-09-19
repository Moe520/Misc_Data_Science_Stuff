#manipulate package is required for this applet to work
#install.packages("manipulate")
#install.packages("gapminder")


library(gapminder)
library(plyr)
library(dplyr)
library(manipulate)
library(ggplot2)

options(scipen=999) # Remove Annoying Scientific notation

gapminder <-gapminder

#gapminder <- read.csv("gapminder.tsv",sep="\t") #read in the gapminder dataset

attach(gapminder)


yearslist <- unique(gapminder$year) #create list of all years covered
continentlist <- c("Africa" ,"Americas" ,"Asia", "Europe", "Oceania") #create list of all continents for the automation to work




applet <- manipulate( #Start with the ggplot itself and replace every changable parameter with an alias
  # Use aes_string instead of aes because we want to be able to programmatically feed variables into the ggplot code
  ggplot(data=gapminder[year==Year & continent %in% Continent,], aes_string(x=Variable_1,y=Variable_2, color=ifelse(ccoding==TRUE,"factor(continent)","FALSE")))+
    geom_point()+
    ggtitle(expression(atop("Gapminder Explorer", atop(italic("Click the Settings Button on the Top-Left to Pick Variables"), ""))))+
    theme(plot.title = element_text(size=20 ,face="bold", margin =margin(10,0,10,0)))+
    theme(axis.title.x = element_text(color="forestgreen", vjust=-0.35),axis.title.y = element_text(color="cadetblue" , vjust=0.35))+
    theme(panel.background = element_rect(fill = '#e3e1d8'))+
    theme(legend.title=element_blank())+
    theme(legend.key=element_rect(fill='#aaaab0'))+
    #theme(plot.background = element_rect(fill = '#f0f5f2'))+ #exclude this line if using the "scale_color_brewer" 
    guides(colour = guide_legend(override.aes = list(size=5)))+
    scale_color_brewer(palette="Set1")+
    {if(logxswitch)scale_x_continuous(trans = "log")}+
    {if(logyswitch)scale_y_continuous(trans = "log")}+
    {if(reglineswitch)geom_smooth(method = "lm")}+
    theme(axis.text.x=element_text(angle=50, size=12, vjust=0.5)), #vjust creates distance between the axis labels and the graph itself
  
  #Now we add the controls so that the user can manipulate the gapminder set
  #for Changable parameter we put in the ggplot, we put a corresponding control below
  
  Year = picker(All=yearslist,1952,1962,1972,1982,1992,2002,2007),  # Drop-down list for choosing the year
  Variable_1 = picker( GdpPerCapita = "gdpPercap",LifeExpectancy = "lifeExp" ,Population ="pop"),  #Drop-down list for choosing 1st variable  
  Variable_2 = picker( LifeExpectancy = "lifeExp",GdpPerCapita = "gdpPercap",Population = "pop"), #Drop-down list for choosing 1st variable 
  ccoding=checkbox(TRUE, "Color Code By Continent"), #Choose whether to color code by continent
  Continent = picker( ALL= continentlist, "Africa", "Americas", "Asia", "Europe", "Oceania"),#Drop-down list for continent
  logxswitch = checkbox(FALSE , "Log the X variable"),
  logyswitch = checkbox(FALSE , "Log the Y variable"),
  reglineswitch = checkbox(FALSE , "Plot Line of Best Fit For Each Active Group")
)





