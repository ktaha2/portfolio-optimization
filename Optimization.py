from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, QAOA, SamplingVQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.result import QuasiDistribution
from qiskit_aer.primitives import Sampler
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import yfinance as yf
import numpy as np
from qiskit.utils import algorithm_globals
from qiskit_finance.data_providers import *

import seaborn as sns
import matplotlib.pyplot as plt
     


num_assets = 3
seed = 1234
num_runs = 1
resultList = []

tickers=["MSFT", "AAPL", "GOOG"]

# Define which assets you are going to use here
# Generate expected return and covariance matrix from (random) time-series
stocks = [("TICKER%s" % i) for i in range(num_assets)]
data = YahooDataProvider(
tickers,
start = datetime.datetime(2021, 1, 1),
end = datetime.datetime(2021, 4, 30),
)

data.run()

mu = data.get_period_return_mean_vector()
sigma = data.get_period_return_covariance_matrix()

# plot sigma
plt.imshow(sigma, interpolation="nearest")
plt.show()

q = 0.8  # set risk factor

budget = num_assets/2 # set budget
penalty = 0.1  # set parameter to scale the budget penalty term

portfolio = PortfolioOptimization(
    expected_returns = mu, covariances = sigma, risk_factor = q, budget = budget
)

qp = portfolio.to_quadratic_program()
qp

def print_result(result):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))

    eigenstate = result.min_eigen_solver_result.eigenstate
    probabilities = (
        eigenstate.binary_probabilities()
            if isinstance(eigenstate, QuasiDistribution)
            else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
    )
    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for k, v in probabilities:
        x = np.array([int(i) for i in list(reversed(k))])
        value = portfolio.to_quadratic_program().objective.evaluate(x)
        print("%10s\t%.4f\t\t%.4f" % (x, value, v))
         

from qiskit.utils import algorithm_globals

algorithm_globals.random_seed = 1234

cobyla = COBYLA()
cobyla.set_options(maxiter=500)
##Determimes the parameters for the generated wave function

ry = TwoLocal(num_assets, "ry", "cz", reps=3, entanglement="full")
vqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=cobyla)
vqe = MinimumEigenOptimizer(vqe_mes)

result = vqe.solve(qp)

print_result(result)

##Gets the highest value out of all the selections and its corresponding selection
selection = result.x
value = result.fval
eigenstate = result.min_eigen_solver_result.eigenstate

Scores=[]

for asset in range(num_assets):
     Scores.append(0)

     ##Each score starts at zero

##Creates an array of scores that have a corresponding asset
##Depending on the value of a given selection the stocks in that selection will be given a value that is added to their score
##The Score will be as a percentage of the optimal value
##The final score will determine the weighting of each asset
##Takes into account that lower values are better as they represent a lower energy level and thus a more optimal path for the function

probabilities = (
        eigenstate.binary_probabilities()
            if isinstance(eigenstate, QuasiDistribution)
            else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
    )

probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)


def calculateScore(value):

     Bestval=result.fval
     #Lowest Energy level

     lowestval=-100
     #Gets all the values from the array or results

     for k, v in probabilities:

          x = np.array([int(i) for i in list(reversed(k))])
          values = portfolio.to_quadratic_program().objective.evaluate(x)

          if(values>lowestval): ##Checks to see if the Value is the lowest so we can get the arrangement with the lowest amount of assets
               lowestval=values

     print(lowestval)
     print(Bestval)

     return 1 - ((Bestval - value) / (Bestval - lowestval)) ##Returns the score from a value from 0-1 with the worst (highest energy level) having a score of 1 and the highest having a score of 0


##Calculates the Score for each asset
for k, v in probabilities:
       
        x = np.array([int(i) for i in list(reversed(k))])
        value = portfolio.to_quadratic_program().objective.evaluate(x)
       
        score = calculateScore(value)
        ##Gets the score of the arrangement
 
        count = 0
        
        ##Iterates through set of assets and adds score to total count if assets exists in the set
        for i in list(reversed(k)):
            if (int(i) == 1):
                
                print("hello")
                Scores[count] += score
            
            count += 1

def sum_numbers(numbers):
     
     total = 0
     
     for number in numbers:
         total += number
     return total

##Asset Weighting(how much you should invest in each asset)
AssetWeightings=[]

##Creates array with AssetWeightings
for asset in range(num_assets):
     AssetWeightings.append(Scores[asset]/sum_numbers(Scores))

##prints the different assets and the wieghting associated with them    
for ticks in range(num_assets):
    print(tickers[ticks]+" "+str(round(AssetWeightings[ticks]*100,3))+"%")
    ##s
