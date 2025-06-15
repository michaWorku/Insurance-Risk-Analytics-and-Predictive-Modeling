from abc import ABC, abstractmethod
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # For numerical operations
import os # For dummy data creation

# Add current directory to path for import of data_loader and data_summarization
sys.path.append(str(Path(__file__).parent.parent))
from data_loader import load_data
from data_summarization import calculate_loss_ratio # Re-use existing loss ratio calculation


# Abstract Base Class for Temporal Analysis Strategy
# --------------------------------------------------
# This class defines a common interface for temporal analysis strategies.
# Subclasses must implement the analyze method.
class TemporalAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, time_column: str, metrics: list):
        """
        Perform temporal analysis on a dataframe, visualizing trends of specified metrics
        over time.

        Parameters:
        df (pd.DataFrame): The dataframe containing the time-series data.
        time_column (str): The name of the datetime column to use for temporal analysis.
        metrics (list): A list of numerical columns to aggregate and plot as trends.

        Returns:
        None: This method visualizes temporal trends.
        """
        pass


# Concrete Strategy for Monthly Trend Analysis
# ---------------------------------------------
# This strategy aggregates data monthly and plots trends for specified numerical metrics.
class MonthlyTrendAnalysis(TemporalAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, time_column: str, metrics: list):
        """
        Aggregates data by month and plots trends for specified numerical metrics.
        Also calculates and plots monthly Loss Ratio, Policy Count, and Claim Frequency.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        time_column (str): The name of the datetime column (e.g., 'TransactionMonth').
        metrics (list): A list of numerical columns (e.g., 'TotalPremium', 'TotalClaims')
                        to plot as individual trends.

        Returns:
        None: Displays a series of line plots showing monthly trends.
        """
        if time_column not in df.columns:
            print(f"Error: Time column '{time_column}' not found in the DataFrame.")
            return
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            print(f"Error: Time column '{time_column}' must be of datetime type for temporal analysis.")
            return

        # Ensure all metrics are present and numerical
        valid_metrics = [col for col in metrics if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if not valid_metrics:
            print("Warning: No valid numerical metrics provided for temporal analysis.")
            valid_metrics = [] # Ensure it's an empty list if no valid metrics
        
        # Ensure essential columns for Loss Ratio/Claim Frequency are present
        required_cols = ['TotalPremium', 'TotalClaims', 'PolicyID']
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            print(f"Warning: Missing required columns for full temporal analysis ({', '.join(missing_required)}). "
                  f"Loss Ratio and Claim Frequency might not be calculated.")

        # Set time_column as index and sort for time series operations
        df_time = df.set_index(time_column).sort_index().copy()

        # Aggregate by month
        aggregation_dict = {
            'TotalPremium': 'sum',
            'TotalClaims': 'sum',
            'PolicyID': 'nunique' # Count unique policies per month
        }
        # Add specified metrics to aggregation if not already covered
        for metric in valid_metrics:
            if metric not in aggregation_dict:
                aggregation_dict[metric] = 'sum' # Default to sum for numerical metrics

        monthly_summary = df_time.resample('M').agg(aggregation_dict).fillna(0) # Fill months with no data with 0

        # Calculate additional derived metrics if possible
        if 'TotalPremium' in monthly_summary.columns and 'TotalClaims' in monthly_summary.columns:
            monthly_summary['LossRatio'] = monthly_summary['TotalClaims'] / (monthly_summary['TotalPremium'] + np.finfo(float).eps)
            monthly_summary['LossRatio'].replace([np.inf, -np.inf], np.nan, inplace=True)
            monthly_summary['LossRatio'].fillna(0, inplace=True)
        else:
            monthly_summary['LossRatio'] = np.nan # Placeholder if columns missing

        if 'PolicyID' in monthly_summary.columns and 'TotalClaims' in monthly_summary.columns:
            monthly_summary['ClaimsOccurred'] = df_time.resample('M')['TotalClaims'].apply(lambda x: (x > 0).sum())
            monthly_summary['ClaimFrequency'] = monthly_summary['ClaimsOccurred'] / (monthly_summary['PolicyID'] + np.finfo(float).eps)
            monthly_summary['ClaimFrequency'].replace([np.inf, -np.inf], np.nan, inplace=True)
            monthly_summary['ClaimFrequency'].fillna(0, inplace=True)
        else:
            monthly_summary['ClaimsOccurred'] = np.nan
            monthly_summary['ClaimFrequency'] = np.nan

        print("\nMonthly Aggregated Summary:")
        print(monthly_summary)

        # Plotting trends for each metric
        num_plots = len(valid_metrics) + 3 # For LossRatio, ClaimsOccurred, ClaimFrequency
        fig, axes = plt.subplots(num_plots, 1, figsize=(14, 5 * num_plots), sharex=True)
        if num_plots == 1: # If only one plot, axes is not an array
            axes = [axes] 

        plot_idx = 0
        # Plot specified metrics
        for metric in valid_metrics:
            sns.lineplot(ax=axes[plot_idx], data=monthly_summary, x=monthly_summary.index, y=metric, marker='o', color='darkblue')
            axes[plot_idx].set_title(f'Monthly {metric} Trend')
            axes[plot_idx].set_ylabel(metric)
            axes[plot_idx].grid(True, linestyle='--', alpha=0.7)
            plot_idx += 1
        
        # Plot Loss Ratio
        if 'LossRatio' in monthly_summary.columns:
            sns.lineplot(ax=axes[plot_idx], data=monthly_summary, x=monthly_summary.index, y='LossRatio', marker='o', color='green')
            axes[plot_idx].set_title('Monthly Loss Ratio Trend')
            axes[plot_idx].set_ylabel('Loss Ratio')
            axes[plot_idx].grid(True, linestyle='--', alpha=0.7)
            plot_idx += 1

        # Plot Policy Count
        if 'PolicyID' in monthly_summary.columns:
            sns.lineplot(ax=axes[plot_idx], data=monthly_summary, x=monthly_summary.index, y='PolicyID', marker='o', color='purple')
            axes[plot_idx].set_title('Monthly Unique Policy Count Trend')
            axes[plot_idx].set_ylabel('Unique Policies')
            axes[plot_idx].grid(True, linestyle='--', alpha=0.7)
            plot_idx += 1

        # Plot Claim Frequency
        if 'ClaimFrequency' in monthly_summary.columns:
            sns.lineplot(ax=axes[plot_idx], data=monthly_summary, x=monthly_summary.index, y='ClaimFrequency', marker='o', color='orange')
            axes[plot_idx].set_title('Monthly Claim Frequency Trend')
            axes[plot_idx].set_ylabel('Claim Frequency (Claims / Policy)')
            axes[plot_idx].set_xlabel('Transaction Month')
            axes[plot_idx].grid(True, linestyle='--', alpha=0.7)
            plot_idx += 1

        plt.tight_layout()
        plt.show()
    
    
# Context Class that uses a TemporalAnalysisStrategy
# ---------------------------------------------------
# This class allows you to switch between different temporal analysis strategies.
class TemporalAnalyzer:
    def __init__(self, strategy: TemporalAnalysisStrategy):
        """
        Initializes the TemporalAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (TemporalAnalysisStrategy): The strategy to be used for temporal analysis.
        """
        if not isinstance(strategy, TemporalAnalysisStrategy):
            raise TypeError("Provided strategy must be an instance of TemporalAnalysisStrategy.")
        self._strategy = strategy

    def set_strategy(self, strategy: TemporalAnalysisStrategy):
        """
        Sets a new strategy for the TemporalAnalyzer.

        Parameters:
        strategy (TemporalAnalysisStrategy): The new strategy to be used for temporal analysis.
        """
        if not isinstance(strategy, TemporalAnalysisStrategy):
            raise TypeError("Provided strategy must be an instance of TemporalAnalysisStrategy.")
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, time_column: str, metrics: list):
        """
        Executes the temporal analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        time_column (str): The name of the datetime column.
        metrics (list): A list of numerical columns to aggregate and plot.
        """
        if df.empty:
            print("DataFrame is empty. Cannot perform temporal analysis.")
            return
        print(f"\n--- Executing Temporal Analysis for '{time_column}' ---")
        self._strategy.analyze(df, time_column, metrics)


# Example usage
if __name__ == "__main__":
    # Example usage of the TemporalAnalyzer with different strategies.
    
    # Define paths for raw and processed data files
    current_script_dir = Path(__file__).parent
    project_root = current_script_dir.parent.parent # Assuming src is one level down from project root
    
    # Define the path to your processed data file for loading
    processed_data_path = project_root / "data" / "processed" / "processed_insurance_data.csv"
    
    # Create dummy data if processed data not found (for standalone testing)
    if not processed_data_path.is_file():
        print(f"Processed data not found at: {processed_data_path}. Creating dummy data for testing.")
        os.makedirs(processed_data_path.parent, exist_ok=True)
        dummy_content = """UnderwrittenCoverID|PolicyID|TransactionMonth|IsVATRegistered|Citizenship|LegalType|Title|Language|Bank|AccountType|MaritalStatus|Gender|Country|Province|PostalCode|MainCrestaZone|SubCrestaZone|ItemType|Mmcode|VehicleType|RegistrationYear|Make|Model|Cylinders|Cubiccapacity|Kilowatts|Bodytype|NumberOfDoors|VehicleIntroDate|CustomValueEstimate|AlarmImmobiliser|TrackingDevice|CapitalOutstanding|NewVehicle|WrittenOff|Rebuilt|Converted|CrossBorder|NumberOfVehiclesInFleet|SumInsured|TermFrequency|CalculatedPremiumPerTerm|ExcessSelected|CoverCategory|CoverType|CoverGroup|Section|Product|StatutoryClass|StatutoryRiskType|TotalPremium|TotalClaims
1|1001|202301|True|SA|Individual|Mr|EN|FNB|Savings|Married|Male|ZA|Gauteng|1234|CrestA|SubCrest1|Car|12345|Sedan|2015|Toyota|Corolla|4|1600|80|Sedan|4|2015-01-01|150000|True|True|100000|False|False|False|False|False|1|200000|12|1000|500|Comprehensive|Full|Motor|A|ProductA|Class1|RiskA|12000|5000
2|1002|202301|False|SA|Company|Pty Ltd|EN|Standard Bank|Cheque|Single|Female|ZA|KZN|5678|CrestB|SubCrest2|Car|67890|SUV|2018|BMW|X5|6|3000|180|SUV|5|2018-03-15|500000|True|True|350000|True|False|False|False|False|3|700000|6|2500|1000|Third Party|Basic|Motor|B|ProductB|Class2|RiskB|15000|0
3|1003|202302|True|SA|Individual|Mrs|EN|Absa|Savings|Married|Female|ZA|Western Cape|9101|CrestC|SubCrest3|Car|11223|Hatchback|2010|VW|Polo|4|1400|70|Hatchback|3|2010-06-01|80000|False|False|50000|False|False|False|False|False|1|100000|12|800|1500
4|1004|202302|False|SA|Individual|Mr|AF|Nedbank|Cheque|Divorced|Male|ZA|Gauteng|1212|CrestA|SubCrest1|Car|44556|Bakkie|2022|Ford|Ranger|5|3200|150|Bakkie|4|2022-02-20|450000|True|False|300000|True|False|False|False|False|2|600000|12|2000|100
5|1005|202303|True|SA|Individual|Ms|EN|Capitec|Savings|Single|Female|ZA|Limpopo|1313|CrestD|SubCrest4|Car|77889|Sedan|2019|Honda|Civic|4|1800|90|Sedan|4|2019-05-10|200000|False|True|150000|False|False|False|False|False|1|250000|12|1100|100
6|1006|202303|False|SA|Company|CC|EN|FNB|Cheque|Married|Male|ZA|Eastern Cape|1414|CrestE|SubCrest5|Truck|99001|Truck|2017|Mercedes-Benz|Actros|6|12000|300|Truck|2|2017-08-01|1000000|True|True|700000|False|False|False|False|False|5|1500000|3|5000|20000
7|1007|202304|True|SA|Individual|Mr|EN|FNB|Savings|Married|Male|ZA|Gauteng|1234|CrestA|SubCrest1|Car|12345|Sedan|2015|Toyota|Corolla|4|1600|80|Sedan|4|2015-01-01|150000|True|True|100000|False|False|False|False|False|1|200000|12|1000|80000 
8|1008|202304|False|SA|Company|Pty Ltd|EN|Standard Bank|Cheque|Single|Female|ZA|KZN|5678|CrestB|SubCrest2|Car|67890|SUV|2018|BMW|X5|6|3000|180|SUV|5|2018-03-15|500000|True|True|350000|True|False|False|False|False|3|700000|6|2500|5000000 
"""
        with open(processed_data_path, 'w') as f:
            f.write(dummy_content)
        print("Dummy processed data created for testing temporal_analysis.py.")
    
    # Load the data using our data_loader module
    df = load_data(processed_data_path, delimiter='|')

    if not df.empty:
        # Convert 'TransactionMonth' to datetime, essential for temporal analysis
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], format='%Y%m', errors='coerce')
        else:
            print("Warning: 'TransactionMonth' column not found, temporal analysis will likely fail.")
            
        # Ensure numerical columns are correctly typed
        numerical_cols_to_convert = [
            'TotalPremium', 'TotalClaims', 'CustomValueEstimate', 'PolicyID' # PolicyID for nunique count
        ]
        for col in numerical_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print("\n--- Temporal Analysis Examples ---")

        # Example 1: Analyze monthly trends for Total Premium and Total Claims
        # Expect multiple line plots: TotalPremium, TotalClaims, LossRatio, PolicyCount, ClaimFrequency
        temporal_analyzer = TemporalAnalyzer(MonthlyTrendAnalysis())
        temporal_analyzer.execute_analysis(df, 'TransactionMonth', ['TotalPremium', 'TotalClaims'])

        # Example 2: Test with a non-existent time column
        print("\n--- Example 2: Testing with Non-Existent Time Column ---")
        temporal_analyzer.execute_analysis(df, 'NonExistentMonth', ['TotalPremium'])

        # Example 3: Test with a non-numerical metric (should be skipped or warned)
        print("\n--- Example 3: Testing with Non-Numerical Metric ---")
        temporal_analyzer.execute_analysis(df, 'TransactionMonth', ['Gender'])

        print("\nTemporal analysis examples complete.")
    else:
        print("DataFrame is empty. Cannot run temporal analysis examples.")
    
    # Clean up dummy file after demonstration
    if processed_data_path.is_file():
        processed_data_path.unlink()
        print(f"Cleaned up dummy processed file: {processed_data_path.name}")
        if not os.listdir(processed_data_path.parent):
            processed_data_path.parent.rmdir()
            print(f"Removed empty dummy processed directory: {processed_data_path.parent}")
