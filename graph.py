from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy
import seaborn as sns
from scipy.interpolate import make_interp_spline
import random

plt.rcParams["figure.dpi"] = 200
plt.style.use("ggplot")






def column(A, j):
    return [A[i][j] for i in range(len(A))]

def transpose(A):
    return [column(A, j) for j in range(len(A[0]))]


def regretWeightsGraph(filename, title, u):

   plt.clf()
   #plt.style.use("ggplot")

   with open(filename, 'r') as infile:
      lines = infile.readlines()

   lines = [[eval(x.split(": ")[1]) for x in line.split('\t')] for line in lines]
   data = transpose(lines)

   regret = numpy.array(data[0]) #/10000
   #regretBound = numpy.array(data[1]) #/10000
   weights = numpy.array(transpose(data[1]))
   xs = numpy.array(list(range(len(data[0]))))



   if u == 1:
      plt.ylabel('Cumulative Regret  \n ($\mathregular{x10^{4}}$)')
      #plt.plot(xs, regretBound, label = "Upper Bound", color = "navy", linewidth=2)
      plt.plot(xs, regret, label = "Regret" , color = "orchid", linewidth=2)
   
      #ax1.scatter(xs,regret,marker='o',linewidths=3)
      #ax1.scatter(xs,regretBound,marker='o',linewidths=3)
      plt.xlabel('Time slot ($\mathregular{x10^{3}}$)')
      plt.title(title)
      plt.legend()
      plt.show()

      plt.ylabel('Action Probability')
      s = 0
      for w in weights:
         if s == 25 :
            plt.plot(xs, w, label= "Cell-25",color="orange", linewidth=2)
         if s == 10:
            plt.plot(xs, w, label= "Cell-10",color = "purple", linewidth=2)
         else: 
            plt.plot(xs, w, linewidth=2)
         s= s+1
      
            #ax2.scatter(xs,w,marker='o')
   
      plt.xlabel('Time slot ($\mathregular{x10^{3}}$)')
      plt.legend()
      plt.show()


   if u == 0: 
      s = [] 
      s_1= [] 
      s_bench = [] 
      mm= 1000
      for m in range(len(regret)-mm):
         summ = 0 
         summ_base = 0
         for h in range(mm):
            summ += regret[h+m]
            summ_base += regretBound[h+m]
         s.append(((summ/ mm)) * 120 )
         s_bench.append(( (summ_base/ mm)) * 100 )
         s_1.append(xs[m])
         print(s[m])
      print(len(s))
      print(len(xs))

      benchmarrk = numpy.array([100.0] *len(s_1))


      plt.ylabel('Average Utility % ')
      plt.plot(s_1, s, color = "darkblue", label = "C-Exp3")
      plt.plot(s_1, s_bench, color = "blue", label = "Baseline")
      plt.plot(s_1, benchmarrk, color = "red", label = "Benchmark - Optimal Policy k*")
      
      #plt.plot(xs, regretBound, label = "Best Action $\mathregular{k^{*}}$", color = "red", linewidth=2)
      #plt.fill_between(xs, regret, regretBound, color='grey', alpha='0.5', hatch='/')
      plt.xlabel('Time slot ($\mathregular{x10^{3}}$)')
      plt.legend()
      plt.show()

   if u == 2: 
      X_Y_Spline = make_interp_spline(xs, regret)
      # Returns evenly spaced numbers
      # over a specified interval.
      X_ = numpy.linspace(xs.min(), xs.max(), 500)
      Y_ = X_Y_Spline(X_)
      plt.ylabel('$\mathregular{R_{T}/T}$)')
      plt.plot(X_, Y_, color = "orangered", linewidth=2)
      #plt.plot(xs, regretBound, label = "Best Action $\mathregular{k^{*}}$", color = "red", linewidth=2)
      plt.xlabel('Time slot ($\mathregular{x10^{3}}$)')
      plt.legend()
      plt.show()


#regretWeightsGraph('M3.txt', "", 0)
#regretWeightsGraph('M1.txt', "", 0)
regretWeightsGraph('exp3/alg/results/exp4|E|=2.txt', "", 1)






 
   #plt.ylabel('Cumulative Utility  \n ($\mathregular{x10^{4}}$)')
   #ax1.plot(xs, regret, label = "Action k$\mathregular{_{t}}$" , color = "blue", linewidth=2)
   #ax1.plot(xs, regretBound, label = "Best Action $\mathregular{k^{*}}$", color = "red", linewidth=2)
   #plt.fill_between(xs, regret, regretBound, color='grey', alpha='0.5', hatch='/')


   #plt.ylabel('Cumulative Regret  \n ($\mathregular{x10^{4}}$)')
   #ax1.plot(xs, regret, label = "Regret" , color = "orchid", linewidth=2)
   #ax1.plot(xs, regretBound, label = "Upper Bound", color = "navy", linewidth=2)