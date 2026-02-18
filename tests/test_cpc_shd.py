import sys
import openturns as ot
from algorithms.CPCAdapter import CPCAdapter
from pipeline.Dataset import Dataset
from pipeline.Pipeline import StructureLearningPipeline
from metrics.SHDMetric import SHDMetric
import numpy as np

def generateDataForSpecificInstance(size):
    R = ot.CorrelationMatrix(3)
    R[0, 1] = 0.5
    R[0, 2] = 0.45
    collection = [ot.FrankCopula(3.0), ot.NormalCopula(R), ot.ClaytonCopula(2.0)] # dimension 2+3+2 = 7
    copula = ot.BlockIndependentCopula(collection)
    copula.setDescription("ABCDEFG") # 7 variables
    return copula.getSample(size)

def test_CPC_SHD():
    # Dataset
    size = 1000
    data = generateDataForSpecificInstance(size) # ot.Sample
    np_data = np.array(data) # Convert to numpy array
    dataset = Dataset(np_data)

    # CPC
    alpha = 0.1
    binNumber = 3
    CPC_adapted = CPCAdapter(alpha=alpha, max_conditioning_set_size=binNumber)

    # Pipeline
    pipeline = StructureLearningPipeline(dataset)
    pipeline.add_algorithm(CPC_adapted)
    pipeline.add_metric(SHDMetric())
    results = pipeline.run()
    
    for algo_name, result in results.items():
        print("Algo:", algo_name)
        print("Learned structure:", result)

if __name__ == "__main__":
    # Run with: python -m tests.integration.test_cpc_shd  
    test_CPC_SHD()
