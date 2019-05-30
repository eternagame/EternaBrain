library(ggplot2)
library(reshape2)

png("/hyper_50-100.png",height=775,width = 465)

df = read.table('~/Desktop/hyperparameterresults1.txt', header=T)

df$name = as.character(df$name)
df$name <- factor(df$name, levels=unique(df$name))

df <- df[c("name", "CNN_alone...20", "SAP_alone...50", "No_energy...51", "No_sequence...51", "No_pairmap...52", "No_dropout...57", "No_dot_bracket...58", "EternaBrain_SAP...61")]
new.df = melt(df, id.vars='name', variable.name='algorithm', value.name='solved')

ggplot(new.df) + geom_tile(aes(x=algorithm, y=name, fill=solved == "0")) + scale_x_discrete("") + 
  scale_y_discrete(limits = rev(levels(df$name))) + scale_fill_manual(values = c("#33DD33", "#DD3333"), na.value="gray50") + 
  theme(axis.text.x = element_text(angle = 90), legend.position="none", text=element_text(size=16))
  
ggsave('~/Desktop/LARGE_hyper100_1_50.svg')