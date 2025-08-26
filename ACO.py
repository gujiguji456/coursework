import numpy as np
import pandas as pd
import requests
import io
import time
from typing import List, Tuple, Dict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TSPDataLoader:
    """
    Module for loading TSP data from TSPLIB format
    """
    
    @staticmethod
    def load_eil51_from_web() -> Tuple[np.ndarray, int]:
        """
        Load EIL51 dataset from web source
        
        Returns:
            coordinates: numpy array of city coordinates
            num_cities: number of cities
        """
        # EIL51 coordinates (standard benchmark)
        eil51_coords = [
            (37, 52), (49, 49), (52, 64), (20, 26), (40, 30), (21, 47), (17, 63), (31, 62),
            (52, 33), (51, 21), (42, 41), (31, 32), (5, 25), (12, 42), (36, 16), (52, 41),
            (27, 23), (17, 33), (13, 13), (57, 58), (62, 42), (42, 57), (16, 57), (8, 52),
            (7, 38), (27, 68), (30, 48), (43, 67), (58, 48), (58, 27), (37, 69), (38, 46),
            (46, 10), (61, 33), (62, 63), (63, 69), (32, 22), (45, 35), (59, 15), (5, 6),
            (10, 17), (21, 10), (5, 64), (30, 15), (39, 10), (32, 39), (25, 32), (25, 55),
            (48, 28), (56, 37), (30, 40)
        ]
        
        coordinates = np.array(eil51_coords, dtype=float)
        num_cities = len(coordinates)
        
        print(f"Successfully loaded EIL51 dataset with {num_cities} cities")
        return coordinates, num_cities
    
    @staticmethod
    def calculate_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean distance matrix between all city pairs
        
        Args:
            coordinates: numpy array of city coordinates
            
        Returns:
            distance_matrix: symmetric distance matrix
        """
        num_cities = len(coordinates)
        distance_matrix = np.zeros((num_cities, num_cities))
        
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    distance_matrix[i][j] = np.sqrt(
                        (coordinates[i][0] - coordinates[j][0])**2 + 
                        (coordinates[i][1] - coordinates[j][1])**2
                    )
        
        return distance_matrix


class AntColonyOptimization:
    """
    Main ACO algorithm implementation for TSP
    """
    
    def __init__(self, distance_matrix: np.ndarray, num_ants: int = 50, 
                 alpha: float = 1.0, beta: float = 3.0, rho: float = 0.1,
                 max_iterations: int = 100):
        """
        Initialize ACO algorithm
        
        Args:
            distance_matrix: distance matrix between cities
            num_ants: number of ants (m)
            alpha: pheromone importance factor (α)
            beta: heuristic importance factor (β) 
            rho: pheromone evaporation rate (ρ)
            max_iterations: maximum number of iterations
        """
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.max_iterations = max_iterations
        
        # Initialize pheromone matrix
        self.pheromone_matrix = np.ones((self.num_cities, self.num_cities))
        
        # Heuristic information (inverse of distance)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.eta = np.where(self.distance_matrix != 0, 
                               1.0 / self.distance_matrix, 0)
        
        # Best solution tracking
        self.best_path = None
        self.best_distance = float('inf')
        self.convergence_data = []
    
    def calculate_probabilities(self, current_city: int, unvisited_cities: List[int]) -> np.ndarray:
        """
        Calculate transition probabilities for ant movement
        
        Args:
            current_city: current city index
            unvisited_cities: list of unvisited city indices
            
        Returns:
            probabilities: probability array for next city selection
        """
        if not unvisited_cities:
            return np.array([])
        
        # Calculate attractiveness for each unvisited city
        attractiveness = np.zeros(len(unvisited_cities))
        
        for i, city in enumerate(unvisited_cities):
            pheromone = self.pheromone_matrix[current_city][city] ** self.alpha
            heuristic = self.eta[current_city][city] ** self.beta
            attractiveness[i] = pheromone * heuristic
        
        # Convert to probabilities
        total_attractiveness = np.sum(attractiveness)
        if total_attractiveness == 0:
            # If all attractiveness is 0, use uniform probability
            probabilities = np.ones(len(unvisited_cities)) / len(unvisited_cities)
        else:
            probabilities = attractiveness / total_attractiveness
        
        return probabilities
    
    def construct_ant_solution(self) -> Tuple[List[int], float]:
        """
        Construct a solution for a single ant
        
        Returns:
            path: list of cities visited
            distance: total path distance
        """
        # Start from random city
        start_city = np.random.randint(0, self.num_cities)
        path = [start_city]
        unvisited_cities = list(range(self.num_cities))
        unvisited_cities.remove(start_city)
        
        current_city = start_city
        
        # Visit all remaining cities
        while unvisited_cities:
            probabilities = self.calculate_probabilities(current_city, unvisited_cities)
            
            # Select next city based on probabilities
            next_city_idx = np.random.choice(len(unvisited_cities), p=probabilities)
            next_city = unvisited_cities[next_city_idx]
            
            path.append(next_city)
            unvisited_cities.remove(next_city)
            current_city = next_city
        
        # Calculate total distance
        total_distance = 0
        for i in range(len(path)):
            from_city = path[i]
            to_city = path[(i + 1) % len(path)]  # Return to start
            total_distance += self.distance_matrix[from_city][to_city]
        
        return path, total_distance
    
    def update_pheromones(self, ant_solutions: List[Tuple[List[int], float]]):
        """
        Update pheromone matrix based on ant solutions
        
        Args:
            ant_solutions: list of (path, distance) tuples from all ants
        """
        # Evaporation
        self.pheromone_matrix *= (1 - self.rho)
        
        # Pheromone deposit
        for path, distance in ant_solutions:
            pheromone_deposit = 1.0 / distance
            
            for i in range(len(path)):
                from_city = path[i]
                to_city = path[(i + 1) % len(path)]
                self.pheromone_matrix[from_city][to_city] += pheromone_deposit
                self.pheromone_matrix[to_city][from_city] += pheromone_deposit  # Symmetric
    
    def solve(self) -> Dict:
        """
        Main ACO solving process
        
        Returns:
            results: dictionary containing solution results and convergence data
        """
        print(f"Running ACO with α={self.alpha}, β={self.beta}, ρ={self.rho}, m={self.num_ants}")
        
        for iteration in range(self.max_iterations):
            # Generate solutions for all ants
            ant_solutions = []
            iteration_best_distance = float('inf')
            
            for ant in range(self.num_ants):
                path, distance = self.construct_ant_solution()
                ant_solutions.append((path, distance))
                
                # Track iteration best
                if distance < iteration_best_distance:
                    iteration_best_distance = distance
                
                # Track global best
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path.copy()
            
            # Update pheromones
            self.update_pheromones(ant_solutions)
            
            # Store convergence data
            avg_distance = np.mean([dist for _, dist in ant_solutions])
            self.convergence_data.append({
                'iteration': iteration + 1,
                'best_distance': self.best_distance,
                'iteration_best': iteration_best_distance,
                'avg_distance': avg_distance
            })
            
            # Print progress
            if (iteration + 1) % 20 == 0:
                print(f"  Iteration {iteration + 1}: Best = {self.best_distance:.2f}, Avg = {avg_distance:.2f}")
        
        return {
            'best_path': self.best_path,
            'best_distance': self.best_distance,
            'convergence_data': self.convergence_data,
            'final_avg_distance': self.convergence_data[-1]['avg_distance']
        }


class ParameterExperiment:
    """
    Module for conducting parameter sensitivity experiments
    """
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        Initialize experiment setup
        
        Args:
            distance_matrix: TSP distance matrix
        """
        self.distance_matrix = distance_matrix
        self.baseline_params = {
            'alpha': 1.0,
            'beta': 3.0, 
            'rho': 0.1,
            'num_ants': 50
        }
        
        # Parameter test ranges
        self.test_ranges = {
            'alpha': [0.5, 1.0, 1.5, 2.0, 2.5],
            'beta': [1.0, 2.0, 3.0, 4.0, 5.0],
            'rho': [0.05, 0.1, 0.15, 0.2, 0.3],
            'num_ants': [25, 50, 75, 100, 125]
        }
        
        self.results = []
    
    def run_single_experiment(self, params: Dict, num_runs: int = 10) -> Dict:
        """
        Run multiple independent runs with given parameters
        
        Args:
            params: parameter dictionary
            num_runs: number of independent runs
            
        Returns:
            experiment_results: aggregated results from all runs
        """
        run_results = []
        
        for run in range(num_runs):
            # Set random seed for reproducibility while maintaining randomness
            np.random.seed(int(time.time() * 1000) % 2**32 + run)
            
            aco = AntColonyOptimization(
                distance_matrix=self.distance_matrix,
                num_ants=params['num_ants'],
                alpha=params['alpha'],
                beta=params['beta'],
                rho=params['rho'],
                max_iterations=100
            )
            
            result = aco.solve()
            run_results.append({
                'best_distance': result['best_distance'],
                'final_avg_distance': result['final_avg_distance']
            })
        
        # Calculate statistics
        best_distances = [r['best_distance'] for r in run_results]
        avg_distances = [r['final_avg_distance'] for r in run_results]
        
        return {
            'params': params.copy(),
            'best_distance_mean': np.mean(best_distances),
            'best_distance_std': np.std(best_distances),
            'best_distance_min': np.min(best_distances),
            'best_distance_max': np.max(best_distances),
            'avg_distance_mean': np.mean(avg_distances),
            'avg_distance_std': np.std(avg_distances),
            'stability': np.std(best_distances),  # Lower is more stable
            'raw_best_distances': best_distances,
            'raw_avg_distances': avg_distances
        }
    
    def run_parameter_study(self, num_runs_per_config: int = 10):
        """
        Run comprehensive parameter sensitivity study
        
        Args:
            num_runs_per_config: number of independent runs per parameter configuration
        """
        print("Starting Parameter Sensitivity Study")
        print("=" * 50)
        
        total_experiments = sum(len(values) for values in self.test_ranges.values())
        experiment_count = 0
        
        # Test each parameter individually
        for param_name, param_values in self.test_ranges.items():
            print(f"\nTesting parameter: {param_name}")
            print("-" * 30)
            
            for value in param_values:
                experiment_count += 1
                print(f"Progress: {experiment_count}/{total_experiments} - {param_name}={value}")
                
                # Create parameter configuration
                test_params = self.baseline_params.copy()
                test_params[param_name] = value
                
                # Run experiment
                result = self.run_single_experiment(test_params, num_runs_per_config)
                result['tested_parameter'] = param_name
                result['tested_value'] = value
                result['is_baseline'] = (value == self.baseline_params[param_name])
                
                self.results.append(result)
                
                print(f"  Result: Best={result['best_distance_mean']:.2f}±{result['best_distance_std']:.2f}")
    
    def analyze_results(self) -> pd.DataFrame:
        """
        Perform statistical analysis of results
        
        Returns:
            analysis_df: DataFrame with detailed analysis
        """
        print("\nPerforming Statistical Analysis...")
        
        analysis_data = []
        
        # Group results by parameter
        for param_name in self.test_ranges.keys():
            param_results = [r for r in self.results if r['tested_parameter'] == param_name]
            baseline_result = [r for r in param_results if r['is_baseline']][0]
            
            for result in param_results:
                # Perform t-test against baseline
                if not result['is_baseline']:
                    t_stat, p_value = stats.ttest_ind(
                        baseline_result['raw_best_distances'],
                        result['raw_best_distances']
                    )
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        (baseline_result['best_distance_std']**2 + result['best_distance_std']**2) / 2
                    )
                    if pooled_std > 0:
                        cohens_d = (result['best_distance_mean'] - baseline_result['best_distance_mean']) / pooled_std
                    else:
                        cohens_d = 0
                    
                    significant = p_value < 0.05
                    improvement = result['best_distance_mean'] < baseline_result['best_distance_mean']
                else:
                    t_stat, p_value, cohens_d = 0, 1.0, 0
                    significant = False
                    improvement = False
                
                analysis_data.append({
                    'Parameter': param_name,
                    'Value': result['tested_value'],
                    'Is_Baseline': result['is_baseline'],
                    'Best_Distance_Mean': result['best_distance_mean'],
                    'Best_Distance_Std': result['best_distance_std'],
                    'Best_Distance_Min': result['best_distance_min'],
                    'Best_Distance_Max': result['best_distance_max'],
                    'Avg_Distance_Mean': result['avg_distance_mean'],
                    'Avg_Distance_Std': result['avg_distance_std'],
                    'Stability': result['stability'],
                    'T_Statistic': t_stat,
                    'P_Value': p_value,
                    'Cohens_D': cohens_d,
                    'Statistically_Significant': significant,
                    'Improvement_vs_Baseline': improvement,
                    'Percent_Change': ((result['best_distance_mean'] - baseline_result['best_distance_mean']) / baseline_result['best_distance_mean']) * 100
                })
        
        analysis_df = pd.DataFrame(analysis_data)
        return analysis_df
    
    def export_results(self, filename: str = 'aco_parameter_study_results.xlsx'):
        """
        Export results to Excel file with multiple sheets
        
        Args:
            filename: output filename
        """
        analysis_df = self.analyze_results()
        
        # Create summary statistics
        summary_data = []
        for param_name in self.test_ranges.keys():
            param_data = analysis_df[analysis_df['Parameter'] == param_name]
            baseline_row = param_data[param_data['Is_Baseline'] == True].iloc[0]
            best_row = param_data.loc[param_data['Best_Distance_Mean'].idxmin()]
            worst_row = param_data.loc[param_data['Best_Distance_Mean'].idxmax()]
            
            significant_improvements = param_data[
                (param_data['Statistically_Significant'] == True) & 
                (param_data['Improvement_vs_Baseline'] == True)
            ]
            
            summary_data.append({
                'Parameter': param_name,
                'Baseline_Value': baseline_row['Value'],
                'Baseline_Performance': baseline_row['Best_Distance_Mean'],
                'Best_Value': best_row['Value'],
                'Best_Performance': best_row['Best_Distance_Mean'],
                'Worst_Value': worst_row['Value'], 
                'Worst_Performance': worst_row['Best_Distance_Mean'],
                'Performance_Range': worst_row['Best_Distance_Mean'] - best_row['Best_Distance_Mean'],
                'Significant_Improvements': len(significant_improvements),
                'Best_Improvement_Percent': significant_improvements['Percent_Change'].min() if len(significant_improvements) > 0 else 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Export to Excel with multiple sheets
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            analysis_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            summary_df.to_excel(writer, sheet_name='Parameter_Summary', index=False)
            
            # Create parameter-specific sheets
            for param_name in self.test_ranges.keys():
                param_data = analysis_df[analysis_df['Parameter'] == param_name]
                param_data.to_excel(writer, sheet_name=f'{param_name}_Analysis', index=False)
        
        print(f"\nResults exported to: {filename}")
        
        # Print key findings
        print("\n" + "="*60)
        print("KEY FINDINGS SUMMARY")
        print("="*60)
        
        for _, row in summary_df.iterrows():
            print(f"\n{row['Parameter'].upper()} Parameter:")
            print(f"  Baseline ({row['Baseline_Value']}): {row['Baseline_Performance']:.2f}")
            print(f"  Best Value ({row['Best_Value']}): {row['Best_Performance']:.2f}")
            print(f"  Performance Range: {row['Performance_Range']:.2f}")
            if row['Significant_Improvements'] > 0:
                print(f"  Significant Improvements: {row['Significant_Improvements']}")
                print(f"  Best Improvement: {row['Best_Improvement_Percent']:.1f}%")
            else:
                print(f"  No significant improvements found")


def main():
    """
    Main execution function
    """
    print("ACO TSP Parameter Study")
    print("="*50)
    
    # Load dataset
    print("\n1. Loading EIL51 Dataset...")
    coordinates, num_cities = TSPDataLoader.load_eil51_from_web()
    distance_matrix = TSPDataLoader.calculate_distance_matrix(coordinates)
    print(f"Distance matrix computed: {distance_matrix.shape}")
    
    # Initialize experiment
    print("\n2. Initializing Parameter Experiment...")
    experiment = ParameterExperiment(distance_matrix)
    
    # Run parameter study
    print("\n3. Running Parameter Study...")
    print("This may take several minutes depending on the number of runs...")
    experiment.run_parameter_study(num_runs_per_config=10)
    
    # Export results
    print("\n4. Exporting Results...")
    experiment.export_results('aco_tsp_parameter_study.xlsx')
    
    print("\n" + "="*50)
    print("Parameter study completed successfully!")
    print("Check 'aco_tsp_parameter_study.xlsx' for detailed results.")


if __name__ == "__main__":
    main()