import numpy as np
from numba import prange, njit
import pandas as pd
from scipy.stats import truncnorm





# Define functions
#################### ASSIGN_ACT_PROBABILITY #########################

def Assign_Act_Probability(N, model, model_params):
    probs = np.zeros(N)
    # Fill array and normalize
    if model == 0:
        norm = 0
        # Powerlaw distribution with exponent model_params[0]
        for i in range(N):
            probs[i] = 1-np.random.power(model_params[0]+1)
            norm += probs[i]
        # Normalize
        for i in range (N):
            probs[i] = probs[i] / norm

    if model == 1:
        norm = 0
        mean = model_params[0]
        std = model_params[1]

        a = (0 - mean) / std
        b = (1 - mean) / std

        # Gaussian mean=model_params[0], std=model_params[1]
        for i in range(N):
            probs[i] = truncnorm.rvs(a, b, mean, std, 1)
            norm += probs[i]
        # Normalize
        for i in range (N):
            probs[i] = probs[i] / norm

    return probs


#################### PROBABILITY_DISTRIBUTION_OPINION ###############

@njit(parallel=True)
def Probability_Distribution_Opinion(G, numb_node, T, beta, Phi):
    # Calculates probability of node numb_node connecting to each other node.
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
    
#################### ODEs ###########################################

@njit(parallel = True)
def ODEs(N, T, A, K, alpha, Phi, current_opinions, opinions_step):
    # Determine social influence to return the differential. current_opinions and opinions_step
    # are arrays of the form [..., [agent_i_op1, agent_i_op2], [agent_i+1_op1, agent_i+1_op2], ...]
    influence = np.zeros((N,T))
    count = np.zeros((N,T))
    for i in prange (N):
        for o in range (T):
            # Go through connected agents of i
            while A[i][int(count[i][o])] != N+1:
                influence[i][o] += np.tanh(alpha * np.dot(Phi, current_opinions[A[i][int(count[i][o])]])[o])
                count[i][o] += 1

    #return differential
    dxdt = -opinions_step + K * influence
    return dxdt

#################### RK4 #############################################

@njit(parallel = True)
def RK4(G, T, A, K, alpha, Phi, dt):
    N = len(G)
    # Save current opinions
    current_opinions = np.zeros((N,T))
    for i in prange (N):
        for j in range (T):
            current_opinions[i][j] += G[i][j]

    # Calculate ks
    k1 = dt * ODEs(N, T, A, K, alpha, Phi, current_opinions, current_opinions)
    k2 = dt * ODEs(N, T, A, K, alpha, Phi, current_opinions, current_opinions + 0.5 * k1)
    k3 = dt * ODEs(N, T, A, K, alpha, Phi, current_opinions, current_opinions + 0.5 * k2)
    k4 = dt * ODEs(N, T, A, K, alpha, Phi, current_opinions, current_opinions + k3)

    # Calculate total change and update opinions
    k = 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    current_opinions += k
    for i in range (N):
        for j in range (T):
            G[i][j] = current_opinions[i][j]

#################### UPDATE_ADJACENCY_MATRIX #########################

@njit(parallel = True)
def Update_Adjacency_Matrix(Adj, A, N):
    # Updates Adjacency Matrix. If a connection is established multiple times in the
    # Network iterations, the connection is still only counted as one connection,
    # so that it isn´t weighted.
    count = np.zeros(N)
    for i in range (N):
        # Again N+1 is the breaking value
        while A[i][int(count[i])] != N+1:
            con = A[i][int(count[i])]
            Adj[i][con] = 1
            Adj[con][i] = 1
            count[i] += 1

##################### CONNECT_DIST #######################################

def Connect_Dist(N, m, G, G_num, T, beta, Phi, probs, num_act):
    # Form adjacency array for timestep. Array entry i contains all adjacent nodes of i. Other values are initialized with N+1
    # as a breaking value so ODEs doesn´t go through the whole array
    A = np.full((N,N), N+1)
    # Create counting array to place adjacent nodes correctly in array
    count_arr = np.zeros(N, dtype=int)

    act_count = 0

    # Pick active nodes
    picks_act = np.random.choice(G_num, num_act, replace=False, p=probs)

    for i in picks_act:
        act_count += 1
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

    return A, act_count

##################### SAVE ###########################################

def Save(save, Adj, N, T, filename, path, save_Adj = False):
    # Create Dataframe and convert it to .csv
    df = pd.DataFrame(save)
    df.to_csv(f"{path}\{filename}.csv", index = False, header = False)

    # Save Adjacency Matrix
    if save_Adj == True:
        pd.DataFrame(Adj).to_csv(f"{path}\{filename}_mat.csv", index = False, header = False)
    

##################### OPINION_DYNAMICS ################################

def Opinion_Dynamics(N, T, m, K, alpha, beta, gamma, Phi, eps1, eps2, runtime_net, runtime_op, dt, filename, path, model, model_params, num_act):
    # Parameters N to eps1 are the same as in the paper introducing the model. Eps2 is the upper bound for activity,
    # runtime_net is the number of times a new AD-network is formed, runtime_op is the number of opinion-iterations
    # performed on that network. Default is runtime_op = 1. dt is the integration time-step and filename is a string
    # containing the name of the file in which the opinion dynamcis of the agents are saved (filename.csv).

    # Create array to save opinions of all agents at the beginning and the end of the simulation
    save = np.zeros((5, N))
    # Create Array that contains node's activities and opinions
    G = np.zeros((N,T))
    # Give each array entry (node) a number, save in array G_num
    G_num = np.arange(0,N)
    # To later retrieve the integrated network over the last 70 iterations (As in the paper) create
    # an adjacency matrix
    Adj = np.zeros((N,N))
    # save # of active nodes per iteration
    acts = np.zeros(runtime_net)

    # Create Activity-Probability Array 
    probs_act = Assign_Act_Probability(N, model, model_params)

    # Save Activity-Probabilites in first array of save
    save[0] = probs_act

    # Initialize Activity and Opinions of Nodes
    for i in range (N):
        for j in range (T):
            G[i][j] = np.random.normal(0, np.sqrt(2.5))

    # Save first opinions
    save[1] = G[:,0]
    save[2] = G[:,1]

    # Perform Iterations until runtime_net is reached
    iteration_net = 0
    while iteration_net < runtime_net:

        # Form connections between Agents
        A, acts[iteration_net] = Connect_Dist(N, m, G, G_num, T, beta, Phi, probs_act, num_act)
        
        # Calculate the influence of the nodes on eachother
        iteration_op = 0
        while iteration_op < runtime_op:
            # Update opinions via Runge-Kutta 4
            RK4(G, T, A, K, alpha, Phi, dt)
            # increase iteration
            iteration_op += 1
        
        # Update adjacency matrix if last 70 iterations are reached
        if iteration_net >= runtime_net - 71:
            Update_Adjacency_Matrix(Adj, A, N)

        iteration_net += 1

    # Save last opinions
    save[3] = G[:,0]
    save[4] = G[:,1]

    Save(save, Adj, N, T, filename, path)




# Perform simulations

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
model = 1
model_params = [0.5, 0.1]

path = f"D:\Daten mit Änderungen\Physik\Bachelorarbeit\Generated_Data\Test"

# Start simulations

File_Names = []
for i in range (len(alphas)):
    File_Names2 = []
    for j in range (len(cosd)):
        File_Names3 = []
        for k in range (3):
            File_Names3.append(f'Gaussian_a{alphas[i]:.1f}_b{beta}_cosd{cosd[j]:.2f}_m{model_params[0]}_std{model_params[1]}_{k+1}')
        File_Names2.append(File_Names3)
    File_Names.append( File_Names2 )

#print(File_Names)

for i in range (len(alphas)):
    for j in range (len(cosd)):
        for k in range(3):
            #print(f"\rsimulation {i*len(File_Names[0]) + j + 1} of {len(File_Names) * len(File_Names[0])}", flush=True)
            Phi = np.array( [[1.0, cosd[j]], [cosd[j], 1.0]] )
            Opinion_Dynamics(N, T, m, K, alphas[i], beta, gamma, Phi, eps1, eps2, runtime_net, runtime_op, step, File_Names[i][j][k], path, model, model_params, 102)