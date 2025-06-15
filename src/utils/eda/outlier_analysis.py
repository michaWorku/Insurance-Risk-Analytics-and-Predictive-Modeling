from abc import ABC, abstractmethod
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # For numerical operations
import os # For dummy data creation

# Add current directory to path for import of data_loader
sys.path.append(str(Path(__file__).parent.parent))
from data_loader import load_data


# Abstract Base Class for Outlier Analysis Strategy
# -------------------------------------------------
# This class defines a common interface for outlier detection strategies.
# Subclasses must implement the analyze method.
class OutlierAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform outlier analysis on a specific numerical feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed for outliers.

        Returns:
        None: This method visualizes and/or prints outlier information.
        """
        pass


# Concrete Strategy for IQR-based Outlier Analysis
# -------------------------------------------------
# This strategy detects outliers in a numerical feature using the Interquartile Range (IQR) method.
class IQRBasedOutlierAnalysis(OutlierAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Detects and visualizes outliers in a numerical feature using the IQR method.
        Prints the count and a sample of detected outliers.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a box plot and prints outlier statistics.
        """
        if feature not in df.columns:
            print(f"Error: Feature '{feature}' not found in the DataFrame.")
            return
        if not pd.api.types.is_numeric_dtype(df[feature]):
            print(f"Error: Feature '{feature}' must be numerical for IQR-based outlier analysis.")
            return

        # Drop NaN values for accurate quartile calculation
        data = df[feature].dropna()

        if data.empty:
            print(f"Warning: Feature '{feature}' is empty or contains only NaN values. Skipping outlier analysis.")
            return

        # Calculate Q1, Q3, and IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR

        # Identify outliers
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)].copy()

        print(f"\n--- Outlier Analysis for '{feature}' (IQR Method) ---")
        print(f"  Q1: {Q1:.2f}")
        print(f"  Q3: {Q3:.2f}")
        print(f"  IQR: {IQR:.2f}")
        print(f"  Lower Bound: {lower_bound:.2f}")
        print(f"  Upper Bound: {upper_bound:.2f}")
        print(f"  Number of outliers detected: {len(outliers)}")

        if not outliers.empty:
            print(f"\nSample of detected outliers for '{feature}':")
            # Display top 5 highest and top 5 lowest if present
            print(outliers.nlargest(min(5, len(outliers)), feature))
            if len(outliers) > 5 and len(outliers[outliers[feature] < lower_bound]) > 0:
                print(outliers.nsmallest(min(5, len(outliers)), feature))
        else:
            print("  No significant outliers detected.")

        # Visualize with a box plot
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=df[feature], color='lightcoral')
        plt.title(f'Box Plot of {feature} with IQR Outliers Indicated')
        plt.ylabel(feature)
        plt.show()


# Context Class that uses an OutlierAnalysisStrategy
# ----------------------------------------------------
# This class allows you to switch between different outlier analysis strategies.
class OutlierAnalyzer:
    def __init__(self, strategy: OutlierAnalysisStrategy):
        """
        Initializes the OutlierAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (OutlierAnalysisStrategy): The strategy to be used for outlier analysis.
        """
        if not isinstance(strategy, OutlierAnalysisStrategy):
            raise TypeError("Provided strategy must be an instance of OutlierAnalysisStrategy.")
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierAnalysisStrategy):
        """
        Sets a new strategy for the OutlierAnalyzer.

        Parameters:
        strategy (OutlierAnalysisStrategy): The new strategy to be used for outlier analysis.
        """
        if not isinstance(strategy, OutlierAnalysisStrategy):
            raise TypeError("Provided strategy must be an instance of OutlierAnalysisStrategy.")
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Executes the outlier analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed for outliers.
        """
        if df.empty:
            print("DataFrame is empty. Cannot perform outlier analysis.")
            return
        self._strategy.analyze(df, feature)


# Example usage
if __name__ == "__main__":
    # Example usage of the OutlierAnalyzer with different strategies.
    
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
5|1005|202303|True|SA|Individual|Ms|EN|Capitec|Savings|Single|Female|ZA|Limpopo|1313|CrestD|SubCrest4|Car|77889|Sedan|2019|Honda|Civic|4|1800|90|Sedan|4|2019-05-10|200000|False|True|150000|False|False|False|False|False|1|250000|12|1100|100 # Added a small claim here
6|1006|202303|False|SA|Company|CC|EN|FNB|Cheque|Married|Male|ZA|Eastern Cape|1414|CrestE|SubCrest5|Truck|99001|Truck|2017|Mercedes-Benz|Actros|6|12000|300|Truck|2|2017-08-01|1000000|True|True|700000|False|False|False|False|False|5|1500000|3|5000|20000
7|1007|202304|True|SA|Individual|Mr|EN|FNB|Savings|Married|Male|ZA|Gauteng|1234|CrestA|SubCrest1|Car|12345|Sedan|2015|Toyota|Corolla|4|1600|80|Sedan|4|2015-01-01|150000|True|True|100000|False|False|False|False|False|1|200000|12|1000|80000 # High outlier for TotalClaims
8|1008|202304|False|SA|Company|Pty Ltd|EN|Standard Bank|Cheque|Single|Female|ZA|KZN|5678|CrestB|SubCrest2|Car|67890|SUV|2018|BMW|X5|6|3000|180|SUV|5|2018-03-15|500000|True|True|350000|True|False|False|False|False|3|700000|6|2500|5000000 # Extreme outlier
"""
        with open(processed_data_path, 'w') as f:
            f.write(dummy_content)
        print("Dummy processed data created for testing outlier_analysis.py.")
    
    # Load the data using our data_loader module
    df = load_data(processed_data_path, delimiter='|')

    if not df.empty:
        # Convert relevant columns to their intended types for proper analysis
        # This mimics the preprocessing step that would have happened earlier
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
        
        numerical_cols_to_convert = [
            'TotalPremium', 'TotalClaims', 'CustomValueEstimate', 
            'CalculatedPremiumPerTerm', 'SumInsured', 'Kilowatts'
        ]
        for col in numerical_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print("\n--- Outlier Analysis Examples ---")

        # Example 1: Analyze outliers in 'TotalClaims'
        outlier_analyzer = OutlierAnalyzer(IQRBasedOutlierAnalysis())
        outlier_analyzer.execute_analysis(df, 'TotalClaims')

        # Example 2: Analyze outliers in 'CustomValueEstimate'
        outlier_analyzer.execute_analysis(df, 'CustomValueEstimate')
        
        # Example 3: Analyze outliers in 'TotalPremium'
        outlier_analyzer.execute_analysis(df, 'TotalPremium')

        # Example 4: Test with a non-existent column
        outlier_analyzer.execute_analysis(df, 'NonExistentColumn')

        # Example 5: Test with a non-numerical column
        outlier_analyzer.execute_analysis(df, 'Gender')

        print("\nOutlier analysis examples complete.")
    else:
        print("DataFrame is empty. Cannot run outlier analysis examples.")
    
    # Clean up dummy file after demonstration
    # if processed_data_path.is_file():
    #     processed_data_path.unlink()
    #     print(f"Cleaned up dummy processed file: {processed_data_path.name}")
    #     # Only remove parent directory if it's the temp_extracted_data and is empty
    #     if not os.listdir(processed_data_path.parent) and "temp_extracted_data" in str(processed_data_path.parent):
    #         processed_data_path.parent.rmdir()
    #         print(f"Removed empty dummy processed directory: {processed_data_path.parent}")
