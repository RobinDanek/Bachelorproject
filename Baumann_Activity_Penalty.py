import numpy as np
import pandas as pd
from numba import njit, prange





#################### ODEs_pen_act ###########################################
# ODEs as above, but with various functions that penalize by activity

@njit(parallel = True)
def ODEs_pen_act(N, T, A, G, K, alpha, Phi, current_opinions, opinions_step, model, model_params):
    # Determine social influence to return the differential. current_opinions and opinions_step
    # are arrays of the form [..., [agent_i_op1, agent_i_op2], [agent_i+1_op1, agent_i+1_op2], ...]
    influence = np.zeros((N,T))
    count = np.zeros((N,T))
    for i in prange (N):
        for o in range (T):
            # Go through connected agents of i
            if model == -1:
                while A[i][int(count[i][o])] != N+1:
                    # Cuts all social influence as test wether everything works.
                    # act is the activity of connected node j
                    act = G[A[i][int(count[i][o])]][T]
                    influence[i][o] += np.tanh(alpha * np.dot(Phi, current_opinions[A[i][int(count[i][o])]])[o]) * (0) # times zero
                    count[i][o] += 1
            if model == 0:
                while A[i][int(count[i][o])] != N+1:
                    # act is the activity of connected node j
                    act = G[A[i][int(count[i][o])]][T]
                    # Penalized term, linear in activity
                    influence[i][o] += np.tanh(alpha * np.dot(Phi, current_opinions[A[i][int(count[i][o])]])[o]) * (1 - act)
                    count[i][o] += 1
            if model == 1:
                while A[i][int(count[i][o])] != N+1:
                    # act is the activity of connected node j
                    act = G[A[i][int(count[i][o])]][T]
                     # Penalized term, 1/2 * tanh(-a*(x-b)) + 1/2
                    influence[i][o] += np.tanh(alpha * np.dot(Phi, current_opinions[A[i][int(count[i][o])]])[o]) * (0.5 * np.tanh(-model_params[0] * (act - model_params[1])) + 0.5)
                    count[i][o] += 1
            if model == 2:
                while A[i][int(count[i][o])] != N+1:
                    # act is the activity of connected node j
                    act = G[A[i][int(count[i][o])]][T]
                    # Penalized term, 1-exp(a * (x-1))
                    influence[i][o] += np.tanh(alpha * np.dot(Phi, current_opinions[A[i][int(count[i][o])]])[o]) * (1 - np.exp( model_params[0] * (act - 1) ))
                    count[i][o] += 1
            if model == 3:
                while A[i][int(count[i][o])] != N+1:
                    # act is the activity of connected node j
                    act = G[A[i][int(count[i][o])]][T]
                    # Penalized term, heaviside
                    if act <= model_params[0]:
                        influence[i][o] += np.tanh(alpha * np.dot(Phi, current_opinions[A[i][int(count[i][o])]])[o])
                    else:
                        influence[i][o] += 0
                    count[i][o] += 1


    #return differential
    dxdt = -opinions_step + K * influence
    return dxdt

#################### RK4_act_pen #############################################
# RK4 with activity penalized ODEs

@njit(parallel = True)
def RK4_act_pen(G, T, A, K, alpha, Phi, dt, model, model_params):
    N = len(G)
    # Save current opinions
    current_opinions = np.zeros((N,T))
    for i in prange (N):
        for j in range (T):
            current_opinions[i][j] += G[i][j]

    # Calculate ks
    k1 = dt * ODEs_pen_act(N, T, A, G, K, alpha, Phi, current_opinions, current_opinions, model, model_params)
    k2 = dt * ODEs_pen_act(N, T, A, G, K, alpha, Phi, current_opinions, current_opinions + 0.5 * k1, model, model_params)
    k3 = dt * ODEs_pen_act(N, T, A, G, K, alpha, Phi, current_opinions, current_opinions + 0.5 * k2, model, model_params)
    k4 = dt * ODEs_pen_act(N, T, A, G, K, alpha, Phi, current_opinions, current_opinions + k3, model, model_params)

    # Calculate total change and update opinions
    k = 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    current_opinions += k
    for i in prange (N):
        for j in range (T):
            G[i][j] = current_opinions[i][j]

##################### ASSING_ACTIVITY ###############################
# Assigns activities following a powerlaw distribution
@njit
def Assign_Activity(eps1, eps2, gamma):
    rand = np.random.uniform(0, 1)
    power_rand = (eps1**(1-gamma) + (eps2**(1-gamma) - eps1**(1-gamma)) * rand)**(1/(1-gamma))
    return power_rand

#################### PROBABILITY_DISTRIBUTION_OPINION ###############
# Calculates the Probabilities of Agent numb_node connecting with others based
# on their opinion distance

@njit(parallel=True)
def Probability_Distribution_Opinion(G, numb_node, T, beta, Phi):
    N = len(G)
    prob = np.zeros(N)

    distance = np.zeros(N)

    # Calculate Norm
    norm = 0
    for i in prange (N):
        if i != numb_node: 
            # numb_node is the number of the selected node
            for u in range (T):
                for v in range (T):
                    distance[i] += (G[i][u] - G[numb_node][u]) * (G[i][v] - G[numb_node][v]) * Phi[u][v]
            norm += np.sqrt(distance[i])**(-beta)
        

    # Calculate Probability of contact
    for i in prange (N): 
        if i != numb_node:
            prob[i] = (np.sqrt(distance[i])**(-beta) / norm)
    return prob

##################### SAVE ###########################################

def Save(save, Adj, N, T, filename, path, save_Adj = False):
    # Create Dataframe and convert it to .csv
    df = pd.DataFrame(save)
    df.to_csv(f"{path}\{filename}.csv", index = False, header = False)

    # Save Adjacency Matrix
    if save_Adj == True:
        pd.DataFrame(Adj).to_csv(f"{path}\{filename}_mat.csv", index = False, header = False)


################# OPINION_DYNAMICS_ACT_PEN #####################
def Opinion_Dynamics_act_pen(N, T, m, K, alpha, beta, gamma, Phi, eps1, eps2, runtime_net, runtime_op, dt, filename, path, model=2, model_params=[3.0]):
    # Parameters N to eps1 are the same as in the paper introducing the model. Eps2 is the upper bound for activity,
    # runtime_net is the number of times a new AD-network is formed, runtime_op is the number of opinion-iterations
    # performed on that network. Default is runtime_op = 1. dt is the integration time-step and filename is a string
    # containing the name of the file in which the opinion dynamcis of the agents are saved (filename.csv).
    # The parameter "model" determines which penalty function is used:
    # -1: zero, as test if everything works
    # 0: linear penality (1-act), 
    # 1: 0.5*tanh(-a*(x-b))+0.5, model_params: [a,b]
    # 2: 1 - exp(a*(x-1)), model_params: [a]
    # 3: heaviside function: 1 for activity below a, 0 for above a. model_params: [a]  

    # Create array to save opinions of all agents after every network-iteration
    #save = np.zeros((runtime_net * T + 1, N))
    # ALTERNATIVE: only save beginning and end opinions for saving a lot of space on your hard-drive.
    save = np.zeros((5, N))

    # Create Array that contains node's activities and opinions
    G = np.zeros((N,T+1))
    # Give each array entry (node) a number, save in array G_num
    G_num = np.arange(0,N)
    # To later retrieve the integrated network over the last 70 iterations (As in the paper) create
    # an adjacency matrix
    Adj = np.zeros((N,N))

    # Initialize Activity and Opinions of Nodes
    for i in range (N):
        G[i][T] = Assign_Activity(eps1, eps2, gamma)
        for j in range (T):
            G[i][j] = np.random.normal(0, np.sqrt(2.5))
    # Save activities of agents in first row of save
    for i in range (N):
        save[0][i] += G[i][T]

    # Save first two opinions
    save[1] = G[:,0]
    save[2] = G[:,1]

    # Perform Iterations until runtime_net is reached
    iteration_net = 0
    while iteration_net < runtime_net:

        # Form adjecency array for timestep. Array entry i contains all adjacent nodes of i. Other values are initialized with N+1
        # as a breaking value so ODEs doesn´t go through the whole array
        A = np.full((N,N), N+1)
        # Create counter array to place adjacent nodes correctly in array
        count_arr = np.zeros(N, dtype=int)

        for i in range (N):
            # Go through nodes and possibly activate them
            rand = np.random.uniform(0,1)
            if rand <= G[i][T]:
                # Pick m other nodes randomly. No exception of i needed, since
                # P_D_O excludes i already
                prob = Probability_Distribution_Opinion(G, i, T, beta, Phi)
                picks = np.random.choice(G_num, m, replace=False, p=prob)
                #Update adjacency list: append nodes j to i´s place and vice versa
                for j in picks:
                    A[i][count_arr[i]] = j
                    A[j][count_arr[j]] = i
                    count_arr[i] += 1
                    count_arr[j] += 1
        
        # Calculate the influence of the nodes on eachother
        iteration_op = 0
        while iteration_op < runtime_op:
            # Update opinions via Runge-Kutta 4
            RK4_act_pen(G, T, A, K, alpha, Phi, dt, model, model_params)
            # increase iteration
            iteration_op += 1

        iteration_net += 1

    # Save last opinions
    save[3] = G[:,0]
    save[4] = G[:,1]

    Save(save, Adj, N, T, filename, path)



# Set parameters
N = 2500 # 1000 nodes should be the minimum, below initial fluctuations take over and polarization becomes instable
T = 2
m = 10 # np.floor(N+50/100)
K = 3
alphas = np.arange(0.0,4.1,0.1)
beta = 5.0 # 5.0 for replicating the phase-space
gamma = 2.1
cosd = np.arange(-0.25,1.05,0.05)
#Phi = np.array([[1.0,0.0],[0.0,1.0]])
eps1 = 0.01
eps2 = 1.0
runtime_net = 10**3 # Stability is reached from about 500 iterations
runtime_op = 1
step = 0.01
model = 3
model_params = np.array([[0.2]])

path = f"D:\Daten mit Änderungen\Physik\Bachelorarbeit\Generated_Data\Test"

# Start simulations

File_Names = []
for i in range (len(alphas)):
    File_Names2 = []
    for j in range (len(cosd)):
        File_Names3 = []
        for k in range (3):
            File_Names3.append(f'Activity_a{alphas[i]:.1f}_b{beta}_cosd{cosd[j]:.2f}_m{model_params[0][0]}_{k+1}')
        File_Names2.append(File_Names3)
    File_Names.append( File_Names2 )

#print(File_Names)

for i in range (len(alphas)):
    for j in range (len(cosd)):
        for k in range(3):
            #print(f"\rsimulation {i*len(File_Names[0]) + j + 1} of {len(File_Names) * len(File_Names[0])}", flush=True)
            Phi = np.array( [[1.0, cosd[j]], [cosd[j], 1.0]] )
            Opinion_Dynamics_act_pen(N, T, m, K, alphas[i], beta, gamma, Phi, eps1, eps2, runtime_net, runtime_op, step, File_Names[i][j][k], path, model, model_params[0])