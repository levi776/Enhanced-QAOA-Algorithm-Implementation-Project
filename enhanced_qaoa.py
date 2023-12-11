#Package Imports
import networkx as nx

# Plotting packages
import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
#from matplotlib.ticker import MultipleLocator
#from matplotlib.backends.backend_pdf import PdfPages # For saving the graph plots as pdf files

# Other packages
import numpy as np
#import math
import sys
#import os
import datetime

# Pre-defined ansatz circuit, operator class and visualization tools
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
#from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

# Aer for running simolation on your on computer
from qiskit_aer.primitives import Estimator as AerEstimator

# Other packages
from scipy.optimize import minimize

#General Parameters In the code:
#params (ndarray): Array of ansatz parameters
#ansatz (QuantumCircuit): Parameterized ansatz circuit
#hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
#estimator (Estimator): Estimator primitive instance

# For prettier Prints :)
def printWithTimestamp(message):
  currentTime = datetime.datetime.now()
  formattedTime = currentTime.strftime("%H:%M:%S")
  print(f"[{formattedTime}]\t\t {message}")

# Translation of the graph problem in quadratic form into a problem hamiltonian
# Compatible with the conventions of the article on which the project based on
def my_to_ising(quadProg: QuadraticProgram) -> tuple[SparsePauliOp, float]:
  """Return the Ising Hamiltonian of this problem.

  Variables are mapped to qubits in the same order, i.e.,
  i-th variable is mapped to i-th qubit.
  See https://github.com/Qiskit/qiskit-terra/issues/1148 for details.

  Args:
    quadProg: The problem to be translated.

  Returns:
    qubitOp comprising the qubit operator for the problem.

  Aborts:
    1. If an integer variable or a continuous variable exists
        in the problem.
    2. If constraints exist in the problem.
  """
  # if problem has variables that are not binary, raise an error
  if quadProg.get_num_vars() > quadProg.get_num_binary_vars():
    printWithTimestamp("The type of all variables must be binary.")
    printWithTimestamp("Algorithm run Aborted.")
    sys.exit(1)
  # if constraints exist, raise an error
  if quadProg.linear_constraints or quadProg.quadratic_constraints:
    printWithTimestamp("There must be no constraint in the problem. ")
    printWithTimestamp("Algorithm run Aborted.")
    sys.exit(1)
  
  # initialize Hamiltonian.
  numVars = quadProg.get_num_vars()
  pauliList = []
  offset = 0.0
  zero = np.zeros(numVars, dtype=bool)

  # set a sign corresponding to a maximized or minimized problem.
  # sign == 1 is for minimized problem. sign == -1 is for maximized problem.
  sense = quadProg.objective.sense.value

  # convert a constant part of the objective function into Hamiltonian.
  offset += quadProg.objective.constant * sense

  # convert linear parts of the objective function into Hamiltonian.
  for idx, coef in quadProg.objective.linear.to_dict().items():
    z_p = zero.copy()
    weight = coef * sense / 2
    z_p[idx] = True

    pauliList.append(SparsePauliOp(Pauli((z_p, zero)), weight))
    offset += weight

  # create Pauli terms
  for (i, j), coeff in quadProg.objective.quadratic.to_dict().items():
    weight = coeff * sense / 4

    if i == j:
        offset += weight
    else:
      z_p = zero.copy()
      z_p[i] = True
      z_p[j] = True
      pauliList.append(SparsePauliOp(Pauli((z_p, zero)), weight))

    z_p = zero.copy()
    z_p[i] = True
    pauliList.append(SparsePauliOp(Pauli((z_p, zero)), weight))

    z_p = zero.copy()
    z_p[j] = True
    pauliList.append(SparsePauliOp(Pauli((z_p, zero)), weight))
    offset += weight

  if pauliList:
    # Remove paulis whose coefficients are zeros.
    pauliList.append(SparsePauliOp("I" * numVars, offset));
    qubitOp = sum(pauliList).simplify(atol=0)
  else:
    # If there is no variable, we set num_nodes=1 so that qubit_op should be an operator.
    # If num_nodes=0, I^0 = 1 (int).
    numVars = max(1, numVars)
    qubitOp = SparsePauliOp("I" * numVars, 0)

  return qubitOp

# Construct the networx graph struct from the txt file input
def constructGraphFromInput():
  graph = nx.DiGraph()
  if len(sys.argv) <= 1:
    printWithTimestamp("Please start the python script again with a valid txt file path as input.")
    printWithTimestamp("The file need to containe a graph formated as an adjacency matrix")
    printWithTimestamp("Example of the text file inner data: \n 1 2 3 \n 4 5 6 \n 7 8 9")
    printWithTimestamp("S.t. each line is a row in the adjacency matrix")
  
  sCostMatrixFilepath = sys.argv[1]
  printWithTimestamp("Reading the input file: \n")
  printWithTimestamp(sCostMatrixFilepath)
  adjacencyMatrix = np.loadtxt(sCostMatrixFilepath)
  printWithTimestamp("Input adjecency matrix is: ")
  print(adjacencyMatrix)
  graph = nx.from_numpy_array(adjacencyMatrix)
  return graph

# Builds a QAOA circuit without adjusted angle parameters
def getQAOACircuit(hamiltonian,numberOfQAOALayers=1):
  printWithTimestamp("Building the QAOA ansatz")
  ansatz = QAOAAnsatz(hamiltonian, numberOfQAOALayers)
  ansatz.measure_all()
  return ansatz

# Reformulate the graph into a qubo problem and then into a hamiltonian
def produceProblemHamiltonian(graph):  
  printWithTimestamp("Reformulate the graph into a qubo problem")
  adjecency_matrix = nx.adjacency_matrix(graph)
  adjecency_matrix_as_np_array = nx.to_numpy_array(graph)
  numberOfQubits = len(adjecency_matrix_as_np_array[0])
  
  QuboModel  = QuadraticProgram("EnhancedQAOA")
  
  printWithTimestamp("Converting the given graph to Qubo describing the Maximum-Independent-Set problem with respect to the given graph")

  if(numberOfQubits>32):
    # Limiting problem size to ensure the simulator can handle it 
    printWithTimestamp("A Problem which needs "+str(numberOfQubits)+" qubits was specified but the specified Simulator only has 32 qubits.\n Job was aborted!")
    sys.exit(1)

  for qubitIndex in range(numberOfQubits):
      QuboModel.binary_var(name="x_%s"%qubitIndex)
  
  # Compatible to the conversion to CUBO in the article
  quadraticTerm = 4*np.triu(adjecency_matrix_as_np_array)
  linearTerm = -2*np.ones(numberOfQubits)

  QuboModel.minimize(linear=linearTerm, quadratic=quadraticTerm)
  
  printWithTimestamp(QuboModel.prettyprint())
  
  printWithTimestamp("Converting QUBO to Ising Hamiltonian")
  hamiltonian = my_to_ising(QuboModel)
  hamiltonian = hamiltonian
  
  printWithTimestamp("The problem hamiltonian: \n")
  print(hamiltonian)
  
  return hamiltonian, numberOfQubits

# Returns estimate of energy from estimator
def energyFunc(params, ansatz, hamiltonian, estimator):
  # It is possible to add explicitly the number of shots the estimator will take
  # For example: shots = 1e4
  result=estimator.run(ansatz, hamiltonian, parameter_values=params).result() 
  energy = (result.values[0])
  return energy

# Returns a gate needed for the given node assossiated energy 
def buildSigmaHamiltonian(node, numberOfQbits):
  gateString = ""
  zIndex = numberOfQbits - (node+1)
  for QbitIndex in range(numberOfQbits):
    if zIndex == QbitIndex:
      gateString += "Z"
    else:
      gateString += "I"
  return Pauli(gateString)

# Calculated the input node energy
def calculateNodeEnergy(node, numberOfQbits, params, ansatz, estimator):
  sigmaHamiltonian = buildSigmaHamiltonian(node, numberOfQbits)
  return energyFunc(params, ansatz, sigmaHamiltonian, estimator)

# Calculated the input node energy
def getMaxEnergyNode(graph, numberOfQubits, params, ansatz, estimator):
  maxNode = list(graph)[0]
  maxNodeEnergy = calculateNodeEnergy(maxNode, numberOfQubits, params, ansatz, estimator)
  nodeDict = dict()
  for node in graph:
    currentNodeEnergy = calculateNodeEnergy(node, numberOfQubits, params, ansatz, estimator)
    nodeDict[node] = currentNodeEnergy
    if currentNodeEnergy > maxNodeEnergy:
      maxNodeEnergy = currentNodeEnergy
      maxNode = node
  return maxNode

# Removes the given node and all it's neighboors from the graph
def removeNodeAndNeighboringNodes(graph, toDeleteNode):
  toDeleteList = [n for n in graph.neighbors(toDeleteNode)]
  toDeleteList.append(toDeleteNode)
  graph.remove_nodes_from(toDeleteList)

# Opzimizes the angles of a 1-depth QAOA circuit with a grid search in addition to the COBYLA qiskit optimizer
# Might be redundant to add the grid search, implemented to increase the quality of the results
def getOptimizedCircuit(costFunc, ansatz, hamiltonian, estimator): 
  bestParams = [0.1,0.1]
  bestResult = minimize(costFunc, bestParams, args=(ansatz, hamiltonian, estimator), method="COBYLA")
  resolution = 50
  gammaRange=(0, np.pi)
  betaRange=(0, np.pi)
  gammaArray = np.linspace(gammaRange[0], gammaRange[1], resolution)
  betaArray = np.linspace(betaRange[0], betaRange[1], resolution)
  for gamma,beta in zip(gammaArray,betaArray):
    res = minimize(costFunc, [beta,gamma], args=(ansatz, hamiltonian, estimator), method="COBYLA")
    params = res.x
    if(costFunc(params, ansatz, hamiltonian, estimator)<costFunc(bestParams, ansatz, hamiltonian, estimator)):
      bestParams = params
      bestResult = res
  return bestResult.x
  

# the number of layers of the QAOA can be adjusted by changing the value below
numberOfQAOALayers=1

printWithTimestamp("Start of Python Script\n")

printWithTimestamp("setting an estimator")
estimator = AerEstimator()

#reading txt input file
graph = constructGraphFromInput()

# To add the visualization of the input graph remove the start-comment signs below
#nx.draw(graph)
#plt.show()

#This list is 
solutionSetSize = 0

while(graph.number_of_nodes()!=0):
  # This line renames the nodes of the graph from 0 to the number of vertices the graph has minus one 
  graph = nx.from_numpy_array(nx.to_numpy_array(graph))
  
  # To get a visualized graph remove the comment below, errors might occure from the visualization but it doesn't affect the results
  #nx.draw(graph, with_labels = True)
  #plt.show()

  # Produce the problem hamiltonian that contains the graph problem details
  hamiltonian, numberOfQubits = produceProblemHamiltonian(graph)
  
  # Produe the QAOA circuit according to the problem hamiltonian and the number of layer choosen
  ansatz = getQAOACircuit(hamiltonian, numberOfQAOALayers) 
  
  # Uses a grid search to increase the quality of the optimization of the QAOA circuit
  # Used to ensure we get a greedy algorithm behavour for a depth one QAOA circuit as depict in the original article 
  # implementation fits only for a depth one QAOA cicuit 
  
  # Reminder: params: the QAOA optimized angels
  params = getOptimizedCircuit(energyFunc, ansatz, hamiltonian, estimator)
 
  # In case of QAOA depth bigger then one 
  # Initial angles must be sutable for the number of layers
  # We can either set the angles randomly or choose speciefic angles
 
  # Example for predetermined angles for any QAOA depth:
  # params = 0.1 * np.ones(ansatz.num_parameters)
  # Example for random angles generation for any QAOA depth:
  # params = 2 * np.pi * np.random.rand(ansatz.num_parameters)
  
  # Getting the QAOA optimized angels without the grid search for any QAOA depth
  #res = minimize(costFunc, params, args=(ansatz, hamiltonian, estimator), method="COBYLA")
  #params = res.x
   
  maxEnergyNode = getMaxEnergyNode(graph, numberOfQubits, params, ansatz, estimator)
  removeNodeAndNeighboringNodes(graph, maxEnergyNode)
  solutionSetSize += 1
 

# The process of calculation the energies is not exact
# Therefore the output can sometimes change between iterations
printWithTimestamp("Approximate Max Independent Set Size: \n")
printWithTimestamp(solutionSetSize)


