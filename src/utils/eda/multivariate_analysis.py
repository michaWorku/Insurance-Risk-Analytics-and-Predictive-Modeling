from abc import ABC, abstractmethod
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os # For dummy data creation

# Add current directory to path for import of data_loader
sys.path.append(str(Path(__file__).parent.parent)) 
from data_loader import load_data


# Abstract Base Class for Multivariate Analysis
# ----------------------------------------------
# This class defines a template for performing multivariate analysis.
# Subclasses can override specific steps like correlation heatmap and pair plot generation.
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame, features: list = None):
        """
        Perform a comprehensive multivariate analysis by generating a correlation heatmap and pair plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.
        features (list, optional): A list of features to include in the pair plot.
                                   If None, all numerical features will be used.

        Returns:
        None: This method orchestrates the multivariate analysis process.
        """
        if df.empty:
            print("DataFrame is empty. Cannot perform multivariate analysis.")
            return
        
        print("\n--- Performing Multivariate Analysis ---")
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df, features) # Pass features to pairplot


    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate and display a heatmap of the correlations between numerical features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method should generate and display a correlation heatmap.
        """
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame, features: list = None):
        """
        Generate and display a pair plot of the selected features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.
        features (list, optional): A list of features to include in the pair plot.
                                   If None, all numerical features will be used.

        Returns:
        None: This method should generate and display a pair plot.
        """
        pass


# Concrete Class for Multivariate Analysis with Correlation Heatmap and Pair Plot
# -------------------------------------------------------------------------------
# This class implements the methods to generate a correlation heatmap and a pair plot.
class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generates and displays a correlation heatmap for the numerical features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays a heatmap showing correlations between numerical features.
        """
        numerical_df = df.select_dtypes(include=[np.number])
        if numerical_df.empty:
            print("No numerical columns found for correlation heatmap.")
            return

        # Exclude common ID columns from correlation calculation if they exist
        cols_to_exclude = ['PolicyID', 'UnderwrittenCoverID', 'Mmcode']
        numerical_df = numerical_df.drop(columns=[col for col in cols_to_exclude if col in numerical_df.columns], errors='ignore')

        if numerical_df.shape[1] < 2:
            print("Not enough numerical columns (after excluding IDs) to compute a meaningful correlation heatmap.")
            return

        plt.figure(figsize=(12, 10))
        # Ensure only non-null values are used for correlation, though df.corr() handles NaNs by default
        sns.heatmap(numerical_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, linecolor='black')
        plt.title("Correlation Heatmap of Numerical Features")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame, features: list = None):
        """
        Generates and displays a pair plot for the selected features in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.
        features (list, optional): A list of features to include in the pair plot.
                                   If None, all numerical features will be used.

        Returns:
        None: Displays a pair plot for the selected features.
        """
        if features:
            # Filter features to only include those present in the DataFrame and are numerical
            cols_for_pairplot = [col for col in features if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            if not cols_for_pairplot:
                print(f"No numerical features found from the provided list for pair plot: {features}")
                return
            plot_df = df[cols_for_pairplot]
        else:
            # If no features specified, use all numerical columns (excluding common IDs)
            numerical_df = df.select_dtypes(include=[np.number])
            cols_to_exclude = ['PolicyID', 'UnderwrittenCoverID', 'Mmcode']
            plot_df = numerical_df.drop(columns=[col for col in cols_to_exclude if col in numerical_df.columns], errors='ignore')
            
            if plot_df.shape[1] == 0:
                print("No numerical columns available for pair plot.")
                return

        if plot_df.shape[1] > 7: # Limit the number of features for readability of pairplot
            print(f"Warning: Too many numerical features ({plot_df.shape[1]}) for a readable pair plot. "
                  f"Consider providing a 'features' list with fewer columns (e.g., up to 7-8).")
            # Optionally, you might select top N correlated features or specific important ones
            # For now, we'll proceed, but the plot might be crowded.

        sns.pairplot(plot_df.dropna()) # Dropna for pairplot as it can't handle NaNs by default
        plt.suptitle("Pair Plot of Selected Numerical Features", y=1.02) # Adjust suptitle position
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example usage of the SimpleMultivariateAnalysis class.
    import os # For dummy data creation

    # Define paths for raw and processed data files
    current_script_dir = Path(__file__).parent
    project_root = current_script_dir.parent.parent # Assuming src is one level down from project root
    
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
        print("Dummy processed data created for testing multivariate_analysis.py.")

    # Load the data using our data_loader module
    df = load_data(processed_data_path, delimiter='|')

    if not df.empty:
        # Convert relevant columns to their intended types for proper analysis
        # This mimics the preprocessing step that would have happened earlier
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
        
        categorical_cols = ['Gender', 'Province', 'VehicleType', 'MaritalStatus', 'Make', 'Model']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        numerical_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate', 'RegistrationYear', 'Cylinders', 'Cubiccapacity', 'Kilowatts', 'NumberOfDoors', 'CapitalOutstanding', 'NumberOfVehiclesInFleet', 'SumInsured', 'TermFrequency', 'CalculatedPremiumPerTerm', 'ExcessSelected']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print("\n--- Multivariate Analysis Examples ---")

        # Example 1: Correlation Heatmap (using all relevant numerical columns)
        multivariate_analyzer = SimpleMultivariateAnalysis()
        multivariate_analyzer.analyze(df)

        # Example 2: Pair Plot of selected important numerical features
        selected_features_for_pairplot = [
            'TotalPremium', 'TotalClaims', 'CustomValueEstimate', 
            'RegistrationYear', 'Cubiccapacity', 'Kilowatts'
        ]
        multivariate_analyzer.analyze(df, features=selected_features_for_pairplot)

        print("\nMultivariate analysis examples complete.")
    else:
        print("DataFrame is empty. Cannot run multivariate analysis examples.")
    
    # Clean up dummy file after demonstration
    # if processed_data_path.is_file():
    #     processed_data_path.unlink()
    #     print(f"Cleaned up dummy processed file: {processed_data_path.name}")
    #     if not os.listdir(processed_data_path.parent):
    #         processed_data_path.parent.rmdir()
    #         print(f"Removed empty dummy processed directory: {processed_data_path.parent}")
