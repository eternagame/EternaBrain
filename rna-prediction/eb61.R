# install.packages("reshape2")
# install.packages("ggplot2")
library(ggplot2)
library(reshape2)

#data = read.table("/Users/rohankoodli/Desktop/results_5timeout.txt")
#new.df = melt(data, id.vars='name', variable.name='name', value.name='solved')
#ggplot(data) + geom_tile(aes(x=algorithm, y=name, fill=solved)) + scale_x_discrete("", labels=names) + scale_y_discrete("") + scale_fill_manual(values = c("#DD3333", "#33DD33"), na.value="gray50")+ theme(axis.text.x = element_text(angle = 90, colour=cols), legend.position="none")

png("eb-2222",height=495,width = 315)
df = read.table('/Volumes/RohanDataVault/EternaBrain-data-archive/cnnresults1.txt', header=TRUE, sep='\t')
df$EteRNABot...27 = NULL
print(colnames(df))

df$name = as.character(df$name)
df$name <- factor(df$name, levels=unique(df$name))

df <- df[c("name", "RNASSD...27", "RNAinverse...28", "DSSOPT...47", "NUPACK...48", "INFORNA...50", "MODENA...54", "EternaBrain...61")]
new.df = melt(df, id.vars='name', variable.name='algorithm', value.name='solved')

ggplot(new.df) + geom_tile(aes(x=algorithm, y=name, fill=solved == "0")) + scale_x_discrete("") +
  scale_y_discrete(limits = rev(levels(df$name))) + scale_fill_manual(values = c("#33DD33", "#DD3333"), na.value="gray50") +
  theme(axis.text.x = element_text(angle = 90), legend.position="none")

#dev.off()

# df2 = read.table('~/Documents/EternaBrain-data-archive/results_7timeout.txt', header=T)
# print(colnames(df2))
# df2 <- df2[c("name", "RNA.SSD...27", "RNAinverse...28", "DSS.Opt...47", "NUPACK...48", "INFO.RNA...50", "MODENA...54", "EternaBrain..61")]
# 
# new2.df = melt(df2, id.vars='name', variable.name='algorithm', value.name='solved')
# ggplot(new2.df) + geom_tile(aes(x=algorithm, y=name, fill=solved == "0")) + scale_x_discrete("") +
#   scale_y_discrete("") + scale_fill_manual(values = c("#33DD33", "#DD3333"), na.value="gray50") +
#   theme(axis.text.x = element_text(angle = 90), legend.position="none")

ggsave('~/Desktop/eb-2222.png')


