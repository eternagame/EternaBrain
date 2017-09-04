library(ggplot2)
library(reshape2)

df = read.table('/combined_7timeout.txt', header=T)
new.df = melt(df, id.vars='name', variable.name='algorithm', value.name='solved')
ggplot(new.df) + geom_tile(aes(x=algorithm, y=name, fill=solved == "0")) + scale_x_discrete("") + 
  scale_y_discrete("") + scale_fill_manual(values = c("#33DD33", "#DD3333"), na.value="gray50") + 
  theme(axis.text.x = element_text(angle = 90), legend.position="none")
  
ggsave('/eb100-2.png')