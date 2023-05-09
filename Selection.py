import numpy as np
class Selection(object):
    #生成概率矩阵
    def generate_probability(self,fitness,population):
        population_size=np.shape(population)[0]
        probability=np.zeros([population_size,2],dtype='float')
        sum_fitness=sum(fitness)
        for i in range(population_size):
            probability[i][0] = float(fitness[i] / sum_fitness)
            if i==0:
                probability[i][1]=probability[i][0]
            if i!=0:
                probability[i][1]=probability[i][0]+probability[i-1][1]
        return probability
    #轮盘赌选择
    def routewheel(self,probability,population):
        population_size=np.shape(population)[0]
        rand1=np.random.rand()
        idx=1
        for i in range(population_size):
            if i==0:
                if rand1<  probability[0][1]:
                    idx=0
            else:
                if rand1< probability[i][1] and rand1> probability[i-1][1]:
                    idx=i
        return idx

