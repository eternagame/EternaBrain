import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn; seaborn.set()

# players
x = [214,214,214,104,379,392,400,16,16,22,104,150,312,285,105,400,382,361,315,287,72,116]
yt = [219,273,134,123,346,319,512,32,20,23,104,211,280,330,150,5277,441,479,396,323,65,110]
yb = [518,554,588,303,1009,1119,1201,10,30,40,257,333,600,506,169,982,907,955,882,767,166,274]
#eternabrain
#xe = [400,389,32,16,149,072,104,165,303,400,287,360,284,213,63,111,116,123]
ye = [518,554,588,303,1009,1119,1201,35,25,40,257,333,600,506,169,982,997,907,955,882,767,166,274]
yet = [518,554,588,303,1009,1119,1201,10,30,40,257,333,600,506,169,982,997,907,955,882,767,166,274]
y = [219,273,134,123,346,319,512,32,20,23,104,211,280,330,150,576,555,441,479,396,323,65,110]

# thunderbolt, cuboid, gcplacement, cloudbeta, six legd turtle, water strider, stickshift, martian, tarax, chctrac, frac3, trtmoves
x = [400     , 380   , 16         , 63       , 108            , 103          , 39        , 213    , 282  , 116    , 119  , 316]
yp = [1833   , 1345  , 51         , 211      , 100            , 130          , 40        , 181    , 656  , 500    , 525  , 177]
yb = [1850   , 1535  , 23         , 45       , 50             , 41           , 12        , 165    , 633  , 76     , 75   , 333]
mp,bp = np.polyfit(x,yp,1)
me,be = np.polyfit(x,yb,1)
print bp,'+',mp,'x'
print be,'+',me,'x'
# player_eq = np.poly1d(player_line)
# brain_line = np.poly1d(brain_line)
def player_eq_list(x):
    ylist = []
    for i in x:
        ylist.append(bp + mp * i)
    return ylist

def brain_eq_list(x):
    ylist = []
    for i in x:
        ylist.append(bp + mp * i)
    return ylist

plt.scatter(x,yp,label='Top Eterna Players')
plt.scatter(x,yb,label='EternaBrain')
plt.plot(x,player_eq_list(x),label='Line of best fit for Players            y = -8.78 + 1.18x')
plt.plot(x,brain_eq_list(x),label='Line of best fit for EternaBrain     y = -34.7 + 2.66x')
plt.title("Number of moves needed to solve a puzzle of a given length")
plt.xlabel("Length of the puzzle (in number of nucleotides)")
plt.ylabel("Number of moves needed to solve")
plt.legend()
plt.show()
