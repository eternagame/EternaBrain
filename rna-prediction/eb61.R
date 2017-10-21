# install.packages("reshape2")
library("ggplot2")
library(reshape2)

#data = read.table("/Users/rohankoodli/Desktop/results_5timeout.txt")
#new.df = melt(data, id.vars='name', variable.name='name', value.name='solved')
#ggplot(data) + geom_tile(aes(x=algorithm, y=name, fill=solved)) + scale_x_discrete("", labels=names) + scale_y_discrete("") + scale_fill_manual(values = c("#DD3333", "#33DD33"), na.value="gray50")+ theme(axis.text.x = element_text(angle = 90, colour=cols), legend.position="none")

png("/eb100-2.png",height=495,width = 315)

df = read.table('~/Desktop/combined_6timeout.txt', header=T)
new.df = melt(df, id.vars='name', variable.name='algorithm', value.name='solved')
ggplot(new.df) + geom_tile(aes(x=algorithm, y=name, fill=solved == "0")) + scale_x_discrete("") + 
  scale_y_discrete("") + scale_fill_manual(values = c("#33DD33", "#DD3333"), na.value="gray50") + 
  theme(axis.text.x = element_text(angle = 90), legend.position="none")

#dev.off()

#df2 = read.table('/Users/rohankoodli/Desktop/results_7timeout.txt', header=T)
#new2.df = melt(df2, id.vars='name', variable.name='algorithm', value.name='solved')
#ggplot(new2.df) + geom_tile(aes(x=algorithm, y=name, fill=solved == "0")) + scale_x_discrete("") + 
#  scale_y_discrete("") + scale_fill_manual(values = c("#33DD33", "#DD3333"), na.value="gray50") + #  theme(axis.text.x = element_text(angle = 90), legend.position="none")

ggsave('~/Desktop/eb_1-50.png')


