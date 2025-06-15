from abc import ABC, abstractmethod
from pathlib import Path
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # For numerical operations like np.finfo
import os # For dummy data creation

# Add current directory to path for import of data_loader
sys.path.append(str(Path(__file__).parent.parent)) 
from data_loader import load_data


# Abstract Base Class for Bivariate Analysis Strategy
# ----------------------------------------------------
# This class defines a common interface for bivariate analysis strategies.
# Subclasses must implement the analyze method.
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on two features of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.

        Returns:
        None: This method visualizes the relationship between the two features.
        """
        pass


# Concrete Strategy for Numerical vs Numerical Analysis
# ------------------------------------------------------
# This strategy analyzes the relationship between two numerical features using scatter plots.
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two numerical features using a scatter plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first numerical feature/column to be analyzed.
        feature2 (str): The name of the second numerical feature/column to be analyzed.

        Returns:
        None: Displays a scatter plot showing the relationship between the two features.
        """
        # Input validation: check if features exist and are numerical
        if feature1 not in df.columns or feature2 not in df.columns:
            print(f"Error: One or both features ('{feature1}', '{feature2}') not found in the DataFrame.")
            return
        if not pd.api.types.is_numeric_dtype(df[feature1]) or not pd.api.types.is_numeric_dtype(df[feature2]):
            print(f"Error: Both features '{feature1}' and '{feature2}' must be numerical for scatter plot analysis.")
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df, alpha=0.6)
        plt.title(f"Scatter Plot: {feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()


# Concrete Strategy for Categorical vs Numerical Analysis
# --------------------------------------------------------
# This strategy analyzes the relationship between a categorical feature and a numerical feature using box plots.
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between a categorical feature and a numerical feature using a box plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the categorical feature/column to be analyzed.
        feature2 (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a box plot showing the relationship between the categorical and numerical features.
        """
        # Input validation: check if features exist and are of correct types
        if feature1 not in df.columns or feature2 not in df.columns:
            print(f"Error: One or both features ('{feature1}', '{feature2}') not found in the DataFrame.")
            return
        # `is_categorical_dtype` covers pandas Category Dtype, `is_object_dtype` covers generic strings
        if not pd.api.types.is_categorical_dtype(df[feature1]) and not pd.api.types.is_object_dtype(df[feature1]):
            print(f"Error: Feature '{feature1}' must be categorical or object type for box plot analysis.")
            return
        if not pd.api.types.is_numeric_dtype(df[feature2]):
            print(f"Error: Feature '{feature2}' must be numerical for box plot analysis.")
            return

        plt.figure(figsize=(12, 7))
        # Use `order` to sort categories by median of the numerical feature for better insights
        order = df.groupby(feature1)[feature2].median().sort_values(ascending=False).index
        sns.boxplot(x=feature1, y=feature2, data=df, palette="viridis", order=order)
        plt.title(f"Box Plot: {feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
        plt.tight_layout() # Adjust layout to prevent overlapping labels
        plt.show()

# Concrete Strategy for Categorical vs Categorical Analysis
# --------------------------------------------------------
# This strategy analyzes the relationship between two categorical features using a countplot or heatmap of counts.
class CategoricalVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two categorical features using a countplot (stacked bar)
        or a heatmap of counts, depending on the number of unique combinations.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first categorical feature/column to be analyzed.
        feature2 (str): The name of the second categorical feature/column to be analyzed.

        Returns:
        None: Displays a plot showing the relationship between the two categorical features.
        """
        # Input validation
        if feature1 not in df.columns or feature2 not in df.columns:
            print(f"Error: One or both features ('{feature1}', '{feature2}') not found in the DataFrame.")
            return
        if (not pd.api.types.is_categorical_dtype(df[feature1]) and not pd.api.types.is_object_dtype(df[feature1])) or \
           (not pd.api.types.is_categorical_dtype(df[feature2]) and not pd.api.types.is_object_dtype(df[feature2])):
            print(f"Error: Both features '{feature1}' and '{feature2}' must be categorical or object types for this analysis.")
            return
        
        # Heuristic to decide between countplot and heatmap:
        # If the product of unique categories is large, a heatmap is usually more readable.
        num_unique_f1 = df[feature1].nunique()
        num_unique_f2 = df[feature2].nunique()
        
        # Limit for countplot to avoid excessive bars (adjust as needed)
        if num_unique_f1 * num_unique_f2 > 50 or num_unique_f1 > 15 or num_unique_f2 > 15: 
            print(f"Too many unique combinations ({num_unique_f1}x{num_unique_f2}) for a readable stacked countplot. Generating a heatmap of counts instead.")
            cross_tab = pd.crosstab(df[feature1], df[feature2])
            plt.figure(figsize=(min(15, num_unique_f2*1.2 + 2), min(12, num_unique_f1*0.8 + 2))) # Dynamic sizing
            sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', linewidths=.5, linecolor='black', cbar_kws={'label': 'Count'})
            plt.title(f'Count Heatmap: {feature1} vs {feature2}')
            plt.xlabel(feature2)
            plt.ylabel(feature1)
            plt.tight_layout()
            plt.show()
        else:
            plt.figure(figsize=(12, 7))
            sns.countplot(x=feature1, hue=feature2, data=df, palette="tab10", 
                          order=df[feature1].value_counts().index) # Order by frequency of feature1
            plt.title(f"Stacked Bar Chart: Distribution of {feature1} by {feature2}")
            plt.xlabel(feature1)
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plt.legend(title=feature2)
            plt.tight_layout()
            plt.show()


# Context Class that uses a BivariateAnalysisStrategy
# ---------------------------------------------------
# This class allows you to switch between different bivariate analysis strategies.
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the BivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): The strategy to be used for bivariate analysis.
        """
        if not isinstance(strategy, BivariateAnalysisStrategy):
            raise TypeError("Provided strategy must be an instance of BivariateAnalysisStrategy.")
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the BivariateAnalyzer.

        Parameters:
        strategy (BivariateAnalysisStrategy): The new strategy to be used for bivariate analysis.
        """
        if not isinstance(strategy, BivariateAnalysisStrategy):
            raise TypeError("Provided strategy must be an instance of BivariateAnalysisStrategy.")
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Executes the bivariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.
        """
        if df.empty:
            print("DataFrame is empty. Cannot perform bivariate analysis.")
            return
        
        print(f"\n--- Executing Bivariate Analysis: {feature1} vs {feature2} ---")
        self._strategy.analyze(df, feature1, feature2)


# Example usage
if __name__ == "__main__":
    # Example usage of the BivariateAnalyzer with different strategies.
    
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
5|1005|202303|True|SA|Individual|Ms|EN|Capitec|Savings|Single|Female|ZA|Limpopo|1313|CrestD|SubCrest4|Car|77889|Sedan|2019|Honda|Civic|4|1800|90|Sedan|4|2019-05-10|200000|False|True|150000|False|False|False|False|False|1|250000|12|1100|NaN
6|1006|202303|False|SA|Company|CC|EN|FNB|Cheque|Married|Male|ZA|Eastern Cape|1414|CrestE|SubCrest5|Truck|99001|Truck|2017|Mercedes-Benz|Actros|6|12000|300|Truck|2|2017-08-01|1000000|True|True|700000|False|False|False|False|False|5|1500000|3|5000|20000
"""
        with open(processed_data_path, 'w') as f:
            f.write(dummy_content)
        print("Dummy processed data created for testing bivariate_analysis.py.")
    
    # Load the data using our data_loader module
    df = load_data(processed_data_path, delimiter='|')

    if not df.empty:
        # Convert relevant columns to their intended types for proper analysis
        # This mimics the preprocessing step that would have happened earlier
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
        
        # Ensure categorical columns are actual 'category' dtype for seaborn
        categorical_cols_to_convert = [
            'IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 'Language', 
            'Bank', 'AccountType', 'MaritalStatus', 'Gender', 'Country', 
            'Province', 'PostalCode', 'MainCrestaZone', 'SubCrestaZone', 
            'ItemType', 'VehicleType', 'Make', 'Model', 'Bodytype', 
            'AlarmImmobiliser', 'TrackingDevice', 'NewVehicle', 
            'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder', 
            'CoverCategory', 'CoverType', 'CoverGroup', 'Section', 'Product', 
            'StatutoryClass', 'StatutoryRiskType'
        ]
        for col in categorical_cols_to_convert:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        numerical_cols_to_convert = [
            'Mmcode', 'RegistrationYear', 'Cylinders', 'Cubiccapacity', 
            'Kilowatts', 'NumberOfDoors', 'CustomValueEstimate', 
            'CapitalOutstanding', 'NumberOfVehiclesInFleet', 
            'SumInsured', 'TermFrequency', 'CalculatedPremiumPerTerm', 
            'ExcessSelected', 'TotalPremium', 'TotalClaims'
        ]
        for col in numerical_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print("\n--- Bivariate Analysis Examples ---")

        # Example 1: Numerical vs. Numerical Analysis (TotalPremium vs TotalClaims)
        # Expect a scatter plot
        print("\n--- Example 1: TotalPremium vs TotalClaims (Numerical vs Numerical) ---")
        analyzer = BivariateAnalyzer(NumericalVsNumericalAnalysis())
        analyzer.execute_analysis(df, 'TotalPremium', 'TotalClaims')

        # Example 2: Categorical vs. Numerical Analysis (Province vs TotalPremium)
        # Expect a box plot showing premium distribution across provinces
        print("\n--- Example 2: Province vs TotalPremium (Categorical vs Numerical) ---")
        analyzer.set_strategy(CategoricalVsNumericalAnalysis())
        analyzer.execute_analysis(df, 'Province', 'TotalPremium')

        # Example 3: Categorical vs. Numerical Analysis (Gender vs TotalClaims)
        # Expect a box plot showing claim distribution by gender
        print("\n--- Example 3: Gender vs TotalClaims (Categorical vs Numerical) ---")
        analyzer.execute_analysis(df, 'Gender', 'TotalClaims')

        # Example 4: Categorical vs. Categorical Analysis (VehicleType vs Gender)
        # Expect a stacked bar chart or heatmap depending on unique values
        print("\n--- Example 4: VehicleType vs Gender (Categorical vs Categorical) ---")
        analyzer.set_strategy(CategoricalVsCategoricalAnalysis())
        analyzer.execute_analysis(df, 'VehicleType', 'Gender')

        # Example 5: Categorical vs. Categorical Analysis (Make vs Bodytype)
        # This example will likely generate a heatmap due to more combinations
        print("\n--- Example 5: Make vs Bodytype (Categorical vs Categorical - likely heatmap) ---")
        if 'Make' in df.columns and 'Bodytype' in df.columns:
            analyzer.execute_analysis(df, 'Make', 'Bodytype')
        else:
            print("Skipping 'Make' vs 'Bodytype' example as one or both columns are not present in the DataFrame.")
            
        # Example 6: Test with non-existent columns
        print("\n--- Example 6: Testing with Non-Existent Columns ---")
        analyzer.execute_analysis(df, 'NonExistentFeature1', 'TotalClaims')

        # Example 7: Test with incorrect data types (e.g., passing categorical to numerical strategy)
        print("\n--- Example 7: Testing with Incorrect Data Types ---")
        analyzer.set_strategy(NumericalVsNumericalAnalysis())
        analyzer.execute_analysis(df, 'Gender', 'TotalPremium') # Gender is categorical, should error

        print("\nBivariate analysis examples complete.")
    else:
        print("DataFrame is empty. Cannot run bivariate analysis examples.")
    
    # Clean up dummy file after demonstration
    if processed_data_path.is_file():
        processed_data_path.unlink()
        print(f"Cleaned up dummy processed file: {processed_data_path.name}")
        # Only remove parent directory if it's the temp_extracted_data and is empty
        if not os.listdir(processed_data_path.parent) and "temp_extracted_data" in str(processed_data_path.parent):
            processed_data_path.parent.rmdir()
            print(f"Removed empty dummy processed directory: {processed_data_path.parent}")
