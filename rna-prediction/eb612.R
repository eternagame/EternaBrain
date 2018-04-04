library(ggplot2)
library(reshape2)

png("/hyper_50-100.png",height=495,width = 330)

df = read.table('~/Documents/EternaBrain-data-archive/sapresults2.txt', header=T)
df <- df[c("name", "CNN_alone...20", "SAP_alone...50", "No_energy...51", "No_sequence...51", "No_pairmap...52", "No_dropout...57", "No_dot_bracket...58", "EternaBrain...61")]
new.df = melt(df, id.vars='name', variable.name='algorithm', value.name='solved')
ggplot(new.df) + geom_tile(aes(x=algorithm, y=name, fill=solved == "0")) + scale_x_discrete("") + 
  scale_y_discrete("") + scale_fill_manual(values = c("#33DD33", "#DD3333"), na.value="gray50") + 
  theme(axis.text.x = element_text(angle = 90), legend.position="none")
  
ggsave('~/Desktop/hyper_50-100.png')