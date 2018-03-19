library(ggplot2)
library(reshape2)

png("/hyper_50-100.png",height=495,width = 330)

df = read.table('~/Documents/EternaBrain-data-archive/sapresults2.txt', header=T)
new.df = melt(df, id.vars='name', variable.name='algorithm', value.name='solved')
ggplot(new.df) + geom_tile(aes(x=algorithm, y=name, fill=solved == "0")) + scale_x_discrete("") + 
  scale_y_discrete("") + scale_fill_manual(values = c("#33DD33", "#DD3333"), na.value="gray50") + 
  theme(axis.text.x = element_text(angle = 90), legend.position="none")
  
ggsave('~/Desktop/hyper_50-100.png')