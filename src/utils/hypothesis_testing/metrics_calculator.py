import pandas as pd
import numpy as np

def calculate_claim_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Claim Frequency for each policy (1 if TotalClaims > 0, else 0).
    Adds a 'HasClaim' binary column (or similar, indicating claim occurrence).

    Args:
        df (pd.DataFrame): DataFrame with 'TotalClaims' and 'PolicyID' (for unique policy count).

    Returns:
        pd.DataFrame: DataFrame with 'HasClaim' column.
    """
    if 'TotalClaims' not in df.columns or 'PolicyID' not in df.columns:
        print("Error: 'TotalClaims' or 'PolicyID' column missing for Claim Frequency calculation.")
        return df.copy()
    
    df_copy = df.copy()
    # If TotalClaims is NaN, assume no claim for frequency calculation
    df_copy['HasClaim'] = (df_copy['TotalClaims'].fillna(0) > 0).astype(int)
    print("Calculated 'HasClaim' (Claim Frequency indicator) column.")
    return df_copy


def calculate_claim_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Claim Severity (average claim amount WHEN a claim occurred).
    This function *doesn't* add a column per row. It's meant for aggregate calculation.
    For A/B testing, you'll filter for policies with claims and then pass that subset's
    'TotalClaims' to the T-test strategy.

    This helper function is more for clarifying the definition.
    """
    if 'TotalClaims' not in df.columns:
        print("Error: 'TotalClaims' column missing for Claim Severity calculation.")
        return df.copy()

    # Filter for policies with actual claims
    claims_occurred_df = df[df['TotalClaims'] > 0].copy()

    if claims_occurred_df.empty:
        print("No claims occurred in the DataFrame to calculate severity.")
        return pd.DataFrame() # Return empty if no claims

    # The actual calculation for a group's severity is mean of 'TotalClaims' for those with claims
    # e.g., claims_occurred_df['TotalClaims'].mean()
    print("Claim Severity defined as average 'TotalClaims' for policies with claims.")
    return claims_occurred_df # Return the filtered DataFrame, then its 'TotalClaims' can be used

def calculate_margin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Margin (TotalPremium - TotalClaims) for each policy.

    Args:
        df (pd.DataFrame): DataFrame with 'TotalPremium' and 'TotalClaims' columns.

    Returns:
        pd.DataFrame: DataFrame with 'Margin' column.
    """
    if 'TotalPremium' not in df.columns or 'TotalClaims' not in df.columns:
        print("Error: 'TotalPremium' or 'TotalClaims' column missing for Margin calculation.")
        return df.copy()

    df_copy = df.copy()
    df_copy['Margin'] = df_copy['TotalPremium'].fillna(0) - df_copy['TotalClaims'].fillna(0)
    print("Calculated 'Margin' column.")
    return df_copy

# if __name__ == "__main__":
#     print("--- Testing metrics_calculator.py ---")

#     # Create a dummy DataFrame for demonstration
#     dummy_data = {
#         'PolicyID': [1, 2, 3, 4, 5, 6],
#         'TotalPremium': [1000, 1200, 800, 1500, 900, 1100],
#         'TotalClaims': [500, 0, 1500, 100, np.nan, 2000] # NaN for no claim, 0 for no claim
#     }
#     df_dummy = pd.DataFrame(dummy_data)
#     print("\nOriginal Dummy DataFrame:")
#     print(df_dummy)

#     # Test calculate_claim_frequency
#     df_freq = calculate_claim_frequency(df_dummy)
#     print("\nDataFrame after calculating Claim Frequency:")
#     print(df_freq)
#     print(f"Overall Claim Frequency: {df_freq['HasClaim'].mean():.2f}")

#     # Test calculate_margin
#     df_margin = calculate_margin(df_dummy)
#     print("\nDataFrame after calculating Margin:")
#     print(df_margin)
#     print(f"Overall Average Margin: {df_margin['Margin'].mean():.2f}")

#     # Test calculate_claim_severity (conceptual usage)
#     # To get overall claim severity, first filter for claims > 0, then take mean
#     claims_only_df = df_dummy[df_dummy['TotalClaims'].fillna(0) > 0]
#     if not claims_only_df.empty:
#         overall_severity = claims_only_df['TotalClaims'].mean()
#         print(f"\nOverall Claim Severity (for policies with claims): {overall_severity:.2f}")
#     else:
#         print("\nNo claims in dummy data for severity calculation.")

