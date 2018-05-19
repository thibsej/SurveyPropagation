import numpy as np
import networkx as nx
from random import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm

class randomKSAT(object):
    
    def __init__(self, N, M, K, max_iter, eps=1e-3, verbose=False):
        super(randomKSAT, self)
        
        # General features
        self.N = N
        self.M = M
        self.K = K
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.graph = self.initialize_graph()
        self.dgraph = self.graph.copy()
        self.WPstatus = None
        self.SPstatus = None
        self.sat = None
        self.assignment = np.zeros(self.N)
        # for warning propagation
        self.H = np.zeros(self.N)
        self.U = np.zeros((self.N, 2))
        # for survey propagation
        self.W = np.zeros((self.N, 3))
        if(self.verbose):
            print(self.dgraph.edges())
        
    def initialize_graph(self):
        G = nx.Graph()
        G.add_nodes_from(np.arange(self.N + self.M))
        for t in range(self.M):
            B = np.unique(np.random.choice(self.N, self.K))
            J = np.random.binomial(1, .5, size = np.shape(B))
            J[J==0] = -1
            G.add_edges_from([(x, self.N + t) for x in B])
            for i in range(len(B)):
                G[B[i]][self.N + t]['J'] = J[i]
                G[B[i]][self.N + t]['u'] = np.random.binomial(1, .5)
                G[B[i]][self.N + t]['h'] = 0
                G[B[i]][self.N + t]['delta'] = np.random.rand(1)
        return G
    
    def decimate_graph(self):
        for i in range(self.N):
            if self.assignment[i] == 0:
                continue
            if i not in self.dgraph.nodes():
                continue
            l = []
            for a in self.dgraph.neighbors(i):
                if self.dgraph[i][a]['J'] * self.assignment[i] == -1:
                    l.append(a)
            for a in l:
                self.dgraph.remove_node(a)
            self.dgraph.remove_node(i)
            
    def check_truth(self):
        l = []
        for a in range(self.N, self.N + self.M):
            for i in self.graph.neighbors(a):
                if self.graph[i][a]['J'] * self.assignment[i] == -1:
                    l.append(a)
                    break
        return len(l) == self.M
            
    ###########################################################################
    # WARNING PROPAGATION
    ###########################################################################
    
    def warning_prop(self):
        for t in range(self.max_iter):
            d = set(nx.get_edge_attributes(self.dgraph, 'u').items())
            self.wp_update()
            d_ = set(nx.get_edge_attributes(self.dgraph, 'u').items())
            if d == d_:
                self.WPstatus = 'CONVERGED'
                return
        self.WPstatus = 'UNCONVERGED'
        
#    def wp_update(self):
#        # Compute cavity fields
#        for a in range(self.N, self.N + self.M):
#            if a not in self.dgraph.nodes():
#                continue
#            for j in self.dgraph.neighbors(a):
#                self.dgraph[j][a]['h'] = 0
#                for b in self.dgraph.neighbors(j):
#                    if b == a:
#                        continue
#                    self.dgraph[j][a]['h'] += self.dgraph[j][b]['u'] * self.dgraph[j][b]['J']
#        
#        # Compute warnings
#        L = self.dgraph.edges()
#        for (i,a) in L:
#            self.dgraph[i][a]['u'] = 1
#            for j in self.dgraph.neighbors(a):
#                if i == j:
#                    continue
#                self.dgraph[i][a]['u'] *= np.heaviside(- self.dgraph[j][a]['h'] * self.dgraph[j][a]['J'], 0)
        
    def wp_update(self):
        L = list(self.dgraph.edges())
        shuffle(L)
        for (i,a) in L:
            # Compute cavity fields
            for j in self.dgraph.neighbors(a):
                if i == j:
                    continue
                self.dgraph[j][a]['h'] = 0
                for b in self.dgraph.neighbors(j):
                    if b == a:
                        continue
                    self.dgraph[j][a]['h'] += self.dgraph[j][b]['u'] * self.dgraph[j][b]['J']
        
            # Compute warnings
                self.dgraph[i][a]['u'] = 1
                for j in self.dgraph.neighbors(a):
                    if i == j:
                        continue
                    self.dgraph[i][a]['u'] *= np.heaviside(- self.dgraph[j][a]['h'] * self.dgraph[j][a]['J'], 0)
                    
    def warning_id(self):
        while np.any(self.assignment == 0):
            self.warning_prop()
            if self.status == 'UNCONVERGED':
                return
            self.wid_localfield()
            if self.sat == 'UNSAT':
                return
            self.wid_assignment()
            self.decimate_graph()
            if(self.verbose):
                print(self.assignment)
                print(self.H)
                print("NODES = ", self.dgraph.number_of_nodes())
                print("EDGES = ", self.dgraph.number_of_edges())
                print(self.dgraph.edges())
        self.dgraph = self.graph.copy()
    
    def wid_localfield(self):
        # Compute local fields and contradiction numbers
        for i in range(self.N):
            if i not in self.dgraph.nodes():
                continue
            self.H[i] = 0
            self.U[i] = 0
            for a in self.dgraph.neighbors(i):
                self.H[i] -=  self.dgraph[i][a]['u'] * self.dgraph[i][a]['J']
                self.U[i, int(np.heaviside(self.dgraph[i][a]['J'], 0))] += self.dgraph[i][a]['u']
        c = self.U[:,0] * self.U[:,1]
        if np.amax(c) > 0:
            self.sat = 'UNSAT'
            return
        self.sat = 'SAT'
    
    def wid_assignment(self):
        # Determine satisfiable assignment
        mask = np.array([i in self.dgraph.nodes() for i in range(self.N)])
        if np.any(self.H[mask] != 0):
            self.assignment[(self.H > 0) & mask] = 1
            self.assignment[(self.H < 0) & mask] = -1
        else:
            p = np.argmax(self.H == 0)
            self.H[p] = 1
            self.assignment[p] = 1
    
    
    ###########################################################################
    # SURVEY PROPAGATION
    ###########################################################################
    
    def belief_prop(self):
        # Initialize deltas on the edges
#        for (i,a) in self.dgraph.edges():
#            self.dgraph[i][a]['delta'] = np.random.rand(1)
        
        #Iteration
        for t in range(self.max_iter):
            d = np.array(list(nx.get_edge_attributes(self.dgraph, 'delta').values()))
            self.sp_update()
            d_ = np.array(list(nx.get_edge_attributes(self.dgraph, 'delta').values()))
            if np.all(np.abs(d - d_) < self.eps):
                self.SPstatus = 'CONVERGED'
                return
        self.SPstatus = 'UNCONVERGED'
        
        
    def sp_update(self):
        L = list(self.dgraph.edges())
        shuffle(L)
        for (i,a) in L:
            # Compute cavity fields
            for j in self.dgraph.neighbors(a):
                if i == j:
                    continue
                prod_tmp = np.ones(2)
                for b in self.dgraph.neighbors(j):
                    if b == a:
                        continue
                    p = int(np.heaviside(self.dgraph[j][a]['J'] * self.dgraph[j][b]['J'], 0))
                    prod_tmp[p] *= (1 - self.dgraph[j][b]['delta'])
                self.dgraph[j][a]['P_u'] = (1 - prod_tmp[0]) * prod_tmp[1]
                self.dgraph[j][a]['P_s'] = (1 - prod_tmp[1]) * prod_tmp[0]
                self.dgraph[j][a]['P_0'] = prod_tmp[0] * prod_tmp[1]
                self.dgraph[j][a]['P_c'] = (1-prod_tmp[0]) * (1-prod_tmp[1])
                        
            # Compute warnings
            self.dgraph[i][a]['delta'] = 1
            for j in self.dgraph.neighbors(a):
                if i == j:
                    continue
                tot = (self.dgraph[j][a]['P_u'] + self.dgraph[j][a]['P_s'] + self.dgraph[j][a]['P_0'] + self.dgraph[j][a]['P_c'])
                if tot == 0:
                    self.dgraph[i][a]['delta'] = 1
                    break
                p = self.dgraph[j][a]['P_u'] / tot
                self.dgraph[i][a]['delta'] *= p
        
        
    def survey_id(self):
        max_it = 0
        while np.any(self.assignment == 0) & (self.dgraph.number_of_edges() > 0) & (max_it< self.max_iter):
            self.belief_prop()
            if self.SPstatus == 'UNCONVERGED':
                print('UNCONVERGED SID')
                return
            if np.amax(np.array(list(nx.get_edge_attributes(self.dgraph, 'delta').values()))) > self.eps:
#                mask = np.array([i in self.dgraph.nodes() for i in range(self.N)])
                self.sid_localfield()
                p = np.argmax(np.abs(self.W[:,0] - self.W[:,1]))
                self.assignment[p] = np.sign(self.W[p,0] - self.W[p,1])
            else:
                p = list(self.dgraph.nodes())[0]
                self.assignment[p] = 1
            self.decimate_graph()
            if(self.verbose):
                print(self.assignment)
                print("NODES = ", self.dgraph.number_of_nodes())
                print("EDGES = ", self.dgraph.number_of_edges())
                print(self.dgraph.edges())
                print('\n')
            max_it += 1
        self.dgraph = self.graph.copy()
        if max_it == self.max_iter:
            self.SPstatus = 'UNCONVERGED'
            print("UNCONVERGED SID")
        
    def sid_localfield(self):
        prod_tmp = np.ones((self.N, 2))
        for i in range(self.N):
            if i not in self.dgraph.nodes():
                continue
            for a in self.dgraph.neighbors(i):
                p = int(np.heaviside(self.dgraph[i][a]['J'], 0))
                prod_tmp[i,p] *= (1 - self.dgraph[i][a]['delta'])
        pi = np.ones((self.N, 4))
        pi[:,0] = (1 - prod_tmp[:,0]) * prod_tmp[:,1] # V plus
        pi[:,1] = (1 - prod_tmp[:,1]) * prod_tmp[:,0]
        pi[:,2] = prod_tmp[:,0] * prod_tmp[:,1]
        pi[:,2] = (1 - prod_tmp[:,0]) * (1 - prod_tmp[:,1])
        tot = (pi[:,0] + pi[:,1] + pi[:,2] + pi[:,3])
        pi[(tot == 0),0] = 0
        pi[(tot == 0),1] = 0
        pi[(tot == 0),2] = 0
        tot[tot == 0] = 1
        self.W[:,0] = pi[:,0] / tot
        self.W[:,1] = pi[:,1] / tot
        self.W[:,2] = pi[:,2] / tot
            
        
prop = randomKSAT(50,200, 3, 1000, verbose=False)
prop.warning_prop()
prop.wid_localfield()
print("WP Status = ", prop.WPstatus)
print("Satisfiability = ", prop.sat)
if(prop.sat == 'SAT'):
    prop.survey_id()
print("SP Status = ", prop.SPstatus)
print(prop.check_truth())

#alpha = np.linspace(3,5,num=11, endpoint=True)
#trial = 50
#res_1 = np.zeros((11, 3))
#for i in (range(len(alpha))):
#    print(i)
#    for j in tqdm(range(trial)):
#        prop = randomKSAT(50,int(alpha[i] * 50), 3, 200, verbose=False)
#        prop.warning_prop()
#        prop.wid_localfield()
#        if(prop.WPstatus == 'UNCONVERGED'):
#            res_1[i, 2] += 1
#            continue
#        if(prop.sat == 'SAT'):
#            res_1[i,0] +=1
#            continue
#        if(prop.sat == 'UNSAT'):
#            res_1[i,1] +=1
#            
#
#            
#print(repr(res_1))
#
#
#plt.plot(alpha, res_1[:,0], label = 'SAT')
#plt.plot(alpha, res_1[:,1], label = 'UNSAT')
#plt.plot(alpha, res_1[:,2], label = 'UNCONVERGED')
#plt.legend()
#plt.savefig('fig50.png', transparent=True)
#plt.show()

#res_2 = np.zeros((11, 3))
#for i in (range(len(alpha))):
#    print(i)
#    for j in tqdm(range(trial)):
#        prop = randomKSAT(100,int(alpha[i] * 100), 3, 20, verbose=False)
#        prop.warning_prop()
#        prop.wid_localfield()
#        if(prop.WPstatus == 'UNCONVERGED'):
#            res_2[i, 2] += 1
#            continue
#        if(prop.sat == 'SAT'):
#            res_2[i,0] +=1
#            continue
#        if(prop.sat == 'UNSAT'):
#            res_2[i,1] +=1
#
#print(repr(res_2))
#
#plt.plot(alpha, res_2[:,0], label = 'SAT')
#plt.plot(alpha, res_2[:,1], label = 'UNSAT')
#plt.plot(alpha, res_2[:,2], label = 'UNCONVERGED')
#plt.legend()
#plt.savefig('fig50.png', transparent=True)
#plt.show()
#
