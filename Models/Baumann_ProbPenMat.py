import numpy as np
import pandas as pd
from numba import njit, prange





##################### CONNECT #######################################

def Connect_Prob(N, m, G, G_num, T, beta, Phi, Pen):
    # Form adjacency array for timestep. Array entry i contains all adjacent nodes of i. Other values are initialized with N+1
    # as a breaking value so ODEs doesn´t go through the whole array
    A = np.full((N,N), N+1)
    # Create counting array to place adjacent nodes correctly in array
    count_arr = np.zeros(N, dtype=int)

    for i in range (N):
        # Go through nodes and possibly activate them
        rand = np.random.uniform(0,1)
        if rand <= G[i][T]:
            # Pick m other nodes randomly. No exception of i needed, since
            # P_D_O excludes i already
            prob = Probability_Distribution_Opinion_Pen(G, i, T, beta, Phi, Pen)
            picks = np.random.choice(G_num, m, replace=False, p=prob)
            #Update adjacency list: append nodes j to i´s place and vice versa
            for j in picks:
                A[i][count_arr[i]] = j
                A[j][count_arr[j]] = i
                count_arr[i] += 1
                count_arr[j] += 1

    return A

##################### SAVE ###########################################

def Save(save, Adj, N, T, filename, path, save_Adj = False):
    # Create Dataframe and convert it to .csv
    df = pd.DataFrame(save)
    df.to_csv(f"{path}\{filename}.csv", index = False, header = False)

    # Save Adjacency Matrix
    if save_Adj == True:
        pd.DataFrame(Adj).to_csv(f"{path}\{filename}_mat.csv", index = False, header = False)

##################### ASSING_ACTIVITY ###############################
# Assigns activities following a powerlaw distribution
@njit
def Assign_Activity(eps1, eps2, gamma):
    rand = np.random.uniform(0, 1)
    power_rand = (eps1**(1-gamma) + (eps2**(1-gamma) - eps1**(1-gamma)) * rand)**(1/(1-gamma))
    return power_rand

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
# Perform 4th order Runge-Kutta on the differential equation of the model

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

##################### EXP_DECAY ######################################
# Calculates the exponentially decaying terms of the penalty matrix

@njit
def Exp_Decay( N, Pen, model_params, iteration_net ):
    for i in range (N):
        for j in range (N):
            # Calculate entries
            if Pen[i][j][1] != 0:
                Pen[i][j][0] = np.exp(-model_params * float((iteration_net) - (Pen[i][j][1])))

                
##################### UPDATE_PENALTY_MATRIX ##########################
# Updates the penalty matrix by setting the time of connection of connected pairs ij 
# to the current iteration-step

@njit
def Update_Penalty_Matrix( N, Pen, A, iteration_net, model, model_params ):
    # Go through Adjacency list and save all #iteration where a connection is made
    for i in range (N):
        j = 0
        while A[i][j] != N+1:
            Pen[i][A[i][j]][1] = iteration_net
            Pen[A[i][j]][i][1] = iteration_net
            j += 1
    
    # Differentiate between models
    if model == 0:
        Exp_Decay( N, Pen, model_params, iteration_net )

#################### PROBABILITY_DISTRIBUTION_OPINION_PEN ###############
# Calculates the probabilities of agents to connect including reduced probability for recent connections

@njit(parallel = True)
def Probability_Distribution_Opinion_Pen(G, numb_node, T, beta, Phi, Pen):
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
            norm += np.sqrt(distance[i])**(-beta) * (1-Pen[numb_node][i][0]) # Added penalty

    # Calculate Probability of contact
    for i in prange (N): 
        if i != numb_node:
            prob[i] = (np.sqrt(distance[i])**(-beta) / norm) * (1-Pen[numb_node][i][0]) # Added penalty
    return prob






###################### OPINION_DYNAMICS_PEN_MAT_PROB #################
def Opinion_Dynamics_Pen_Mat_Prob(N, T, m, K, alpha, beta, gamma, Phi, eps1, eps2, runtime_net, runtime_op, dt, filename, path, model=0, model_params=0.1):
    # Parameters N to eps1 are the same as in the paper introducing the model. Eps2 is the upper bound for activity,
    # runtime_net is the number of times a new AD-network is formed, runtime_op is the number of opinion-iterations
    # performed on that network. Default is runtime_op = 1. dt is the integration time-step and filename is a string
    # containing the name of the file in which the opinion dynamcis of the agents are saved (filename.csv).
    # In this variation of the model multiple connections between agents in a certain time period are punished
    # by reducing the probability for the agents to connect to eachother.

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
    # Create Penalty matrix
    Pen = np.zeros((N,N,2), dtype=float)
    #print(Pen[0][0][1])

    # Initialize Activity and Opinions of Nodes
    for i in range (N):
        G[i][T] = Assign_Activity(eps1, eps2, gamma)
        for j in range (T):
            G[i][j] = np.random.normal(0, np.sqrt(2.5))
    # Save activities of agents in first row of save

    for i in range (N):
        save[0][i] += G[i][T]

    save[1] = G[:,0]
    save[2] = G[:,1]

    # Perform Iterations until runtime_net is reached
    iteration_net = 0
    while iteration_net < runtime_net:

        # Form connections between Agents
        A = Connect_Prob(N, m, G, G_num, T, beta, Phi, Pen)

        # Calculate the influence of the nodes on eachother
        iteration_op = 0
        while iteration_op < runtime_op:
            # Update opinions via Runge-Kutta 4
            RK4(G, T, A, K, alpha, Phi, dt)
            # increase iteration
            iteration_op += 1

        # Update penalty matrix. This has to be done after nodes taking influence on eachother, since else connections would be 
        # punished the moment they are created.
        Update_Penalty_Matrix( N, Pen, A, float(iteration_net), model, model_params )

        iteration_net += 1

    save[3] = G[:,0]
    save[4] = G[:,1]

    Save(save, Adj, N, T, filename, path)

    return 0






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
model = 0
model_params = np.array([0.1])

path = f"D:\Daten mit Änderungen\Physik\Bachelorarbeit\Generated_Data\Test"

# Start simulations

File_Names = []
for i in range (len(alphas)):
    File_Names2 = []
    for j in range (len(cosd)):
        File_Names3 = []
        for k in range (3):
            File_Names3.append(f'ProbPenMat_a{alphas[i]:.1f}_b{beta}_cosd{cosd[j]:.2f}_param{model_params[0]}_{k+1}')
        File_Names2.append(File_Names3)
    File_Names.append( File_Names2 )

#print(File_Names)

for i in range (len(alphas)):
    for j in range (len(cosd)):
        for k in range(3):
            #print(f"\rsimulation {i*len(File_Names[0]) + j + 1} of {len(File_Names) * len(File_Names[0])}", flush=True)
            Phi = np.array( [[1.0, cosd[j]], [cosd[j], 1.0]] )
            Opinion_Dynamics_Pen_Mat_Prob(N, T, m, K, alphas[i], beta, gamma, Phi, eps1, eps2, runtime_net, runtime_op, step, File_Names[i][j][k], path, model, model_params[0])
