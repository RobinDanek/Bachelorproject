import numpy as np
from numba import prange, njit
import pandas as pd




# Define functions
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

#################### ODEs ###########################################

@njit(parallel = True)
def ODEs_Infl(N, T, A, K, alpha, Phi, current_opinions, opinions_step, num_infl, infl_strength):
    # Determine social influence to return the differential. current_opinions and opinions_step
    # are arrays of the form [..., [agent_i_op1, agent_i_op2], [agent_i+1_op1, agent_i+1_op2], ...]
    dxdt = np.zeros((N,T))
    influence = np.zeros((N,T))
    count = np.zeros((N,T))
    for i in prange (N):
        for o in range (T):
            # Go through connected agents of i
            while A[i][int(count[i][o])] != N+1:
                # Add negative force to DGL if agent i is connected to a moderator
                if A[i][int(count[i][o])] >= N-num_infl:
                    influence[i][o] += -np.tanh(current_opinions[i][o])*infl_strength
                    count[i][o] += 1
                else:
                    influence[i][o] += np.tanh(alpha * np.dot(Phi, current_opinions[A[i][int(count[i][o])]])[o])
                    count[i][o] += 1

    #return differential
    dxdt += -opinions_step + K * influence
    return dxdt

#################### RK4 #############################################
# Perform 4th order Runge-Kutta on the differential equation of the model

@njit(parallel = True)
def RK4_Infl(G, T, A, K, alpha, Phi, dt, num_infl, infl_strength):
    N = len(G)
    # Save current opinions
    current_opinions = np.zeros((N,T))
    for i in prange (N):
        for j in range (T):
            current_opinions[i][j] += G[i][j]

    # Calculate ks
    k1 = dt * ODEs_Infl(N, T, A, K, alpha, Phi, current_opinions, current_opinions, num_infl, infl_strength)
    k2 = dt * ODEs_Infl(N, T, A, K, alpha, Phi, current_opinions, current_opinions + 0.5 * k1, num_infl, infl_strength)
    k3 = dt * ODEs_Infl(N, T, A, K, alpha, Phi, current_opinions, current_opinions + 0.5 * k2, num_infl, infl_strength)
    k4 = dt * ODEs_Infl(N, T, A, K, alpha, Phi, current_opinions, current_opinions + k3, num_infl, infl_strength)

    # Calculate total change and update opinions
    k = 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    current_opinions += k
    for i in range (N):
        for j in range (T):
            G[i][j] = current_opinions[i][j]

##################### CONNECT #######################################
# Form adjacency array for timestep. Array entry i contains all adjacent nodes of i. Other values are initialized with N+1
# as a breaking value so ODEs doesn´t go through the whole array

def Connect_Infl(N, m, G, G_num, T, beta, Phi, num_infl, m_infl):
    # Fill A with placeholder value of all #nodes+#influencers+1
    A = np.full((N+num_infl, N+num_infl), N+1+num_infl)
    # Create counting array to place adjacent nodes correctly in array
    count_arr = np.zeros(N+num_infl, dtype=int)

    act_count = 0

    for i in range (N+num_infl):
        # Go through nodes and possibly activate them
        rand = np.random.uniform(0,1)
        if rand <= G[i][T]:
            act_count += 1
            # Pick m other nodes randomly. No exception of i needed, since
            # P_D_O excludes i already
            # Exclude moderators
            if i >= N:
                # Pick random people, since moderators sjhouldn´t be strongly influenced by homophily
                picks = np.random.choice(G_num[:N], m_infl, replace=False)
            else:
                prob = Probability_Distribution_Opinion(G, G_num[i], T, beta, Phi)
                picks = np.random.choice(G_num, m, replace=False, p=prob)
            #Update adjacency list: append nodes j to i´s place and vice versa
            for j in picks:
                A[i][count_arr[i]] = j
                A[j][count_arr[j]] = i
                count_arr[i] += 1
                count_arr[j] += 1

    return A, act_count

#################### SET_BACK ########################################
# Sets back agent "numb" isnide a circular boundary around 0. 

@njit
def Set_back(G, numb, boundary):
    # This function sets the radius in the opinion-phase-space of the node back to boundary.

    # Calculate radius
    radius = 0
    for i in range(2):
        radius += G[numb][i]**2
    radius = np.sqrt(radius)

    if radius > boundary:
        # Calculate angle in xy-plain so that only the radius is affected by the setback.
        # A small deviation from the boundary is needed because for large boundaries + Polarization
        # some nodes get the same opinion and thus their probability to interact becomes infinetly large.
        phi = np.arctan(G[numb][1]/G[numb][0])
        rand = np.random.uniform(-1e-12,1e-12)
        G[numb][0] = np.cos(phi) * (boundary + rand)
        G[numb][1] = np.sin(phi) * (boundary + rand)

##################### SAVE ###########################################

def Save(save, Adj, N, T, filename, path, save_Adj = False):
    # Create Dataframe and convert it to .csv
    df = pd.DataFrame(save)
    df.to_csv(f"{path}\{filename}.csv", index = False, header = False)

    # Save Adjacency Matrix
    if save_Adj == True:
        pd.DataFrame(Adj).to_csv(f"{path}\{filename}_mat.csv", index = False, header = False)





##################### OPINION_DYNAMICS ################################

def Opinion_Dynamics_Infl(N, T, m, K, alpha, beta, gamma, Phi, eps1, eps2, runtime_net, runtime_op, dt, filename, path, num_infl, m_infl, infl_strength):
    # Parameters N to eps1 are the same as in the paper introducing the model. Eps2 is the upper bound for activity,
    # runtime_net is the number of times a new AD-network is formed, runtime_op is the number of opinion-iterations
    # performed on that network. Default is runtime_op = 1. dt is the integration time-step and filename is a string
    # containing the name of the file in which the opinion dynamcis of the agents are saved (filename.csv).

    # Create array to save opinions of all agents after every network-iteration
    #save = np.zeros((runtime_net * T + 1, N))
    # ALTERNATIVE: only save beginning and end opinions for saving a lot of space on your hard-drive.
    save = np.zeros((5, N+num_infl))

    # Create Array that contains node's activities and opinions
    G = np.zeros((N+num_infl,T+1))
    # To later retrieve the integrated network over the last 70 iterations (As in the paper) create
    # an adjacency matrix
    Adj = np.zeros((N+num_infl,N+num_infl))
    # Give each array entry (node) a number, save in array G_num
    G_num = np.arange(0,N+num_infl)
    # save # of active nodes per iteration
    acts = np.zeros(runtime_net)

    # Initialize Activity and Opinions of Nodes
    # Activity of moderators (influencers) is set to one
    for i in range (N):
        G[i][T] = Assign_Activity(eps1, eps2, gamma)
        for j in range (T):
            G[i][j] = np.random.normal(0, np.sqrt(2.5))
    for i in range (num_infl):
        G[N+i][T] = 1.0
        for j in range (T):
            G[N+i][j] = np.random.uniform(1e-10, 1e-10)
    # Save activities of agents in first row of save

    for i in range (N+num_infl):
        save[0][i] += G[i][T]

    # Save first two opinions
    save[1] = G[:,0]
    save[2] = G[:,1]

    # Perform Iterations until runtime_net is reached
    iteration_net = 0
    while iteration_net < runtime_net:

        # Set the moderators back to approx. zero.
        for i in range(num_infl):
            Set_back(G, N+i, 1e-10)

        # Form connections between Agents
        A, acts[iteration_net] = Connect_Infl(N, m, G, G_num, T, beta, Phi, num_infl, m_infl)
        
        # Calculate the influence of the nodes on eachother
        iteration_op = 0
        while iteration_op < runtime_op:
            # Update opinions via Runge-Kutta 4
            RK4_Infl(G, T, A, K, alpha, Phi, dt, num_infl=num_infl, infl_strength=infl_strength)
            # increase iteration
            iteration_op += 1

        iteration_net += 1

    # Save last opinions
    save[3] = G[:,0]
    save[4] = G[:,1]

    Save(save, Adj, N, T, filename, path=path, save_Adj=False)


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
num_infl = 25
m_infl = 50
infl_strength = 0.3

path = f"D:\Daten mit Änderungen\Physik\Bachelorarbeit\Generated_Data\Test"

# Start simulations

File_Names = []
for i in range (len(alphas)):
    File_Names2 = []
    for j in range (len(cosd)):
        File_Names3 = []
        for k in range (3):
            File_Names3.append(f'Influencers_a{alphas[i]:.1f}_b{beta}_cosd{cosd[j]:.2f}_m{m_infl}_N{num_infl}_str{infl_strength}_{k+1}')
        File_Names2.append(File_Names3)
    File_Names.append( File_Names2 )

#print(File_Names)

for i in range (len(alphas)):
    for j in range (len(cosd)):
        for k in range(3):
            #print(f"\rsimulation {i*len(File_Names[0]) + j + 1} of {len(File_Names) * len(File_Names[0])}", flush=True)
            Phi = np.array( [[1.0, cosd[j]], [cosd[j], 1.0]] )
            Opinion_Dynamics_Infl(N, T, m, K, alphas[i], beta, gamma, Phi, eps1, eps2, runtime_net, runtime_op, step, File_Names[i][j][k], path, num_infl, m_infl, infl_strength)