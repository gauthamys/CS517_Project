from Decider import Decider
from util.Data import StackOverflowLoader, ResumeLoader
import pandas as pd
import numpy as np

def generate_decisions_with_races():
    print('loaded dataset')
    df = ResumeLoader.load()

    categories = ['Caucasian', 'African American', 'Asian', 'Other']

    # Get the number of rows in your DataFrame
    n = len(df)

    # Calculate how many times we can fully repeat the categories list
    full_repeats = n // len(categories)
    # Calculate the remaining rows if n isn't a multiple of 4
    remainder = n % len(categories)

    # Create a list with the categories repeated fully
    race_list = categories * full_repeats
    # Add the first few categories to fill any remaining rows
    if remainder:
        race_list += categories[:remainder]

    # Optional: Shuffle the list if you want a random order
    np.random.shuffle(race_list)

    # Assign the list to a new column in the DataFrame
    df['Race'] = race_list

    decider = Decider("""
    Seeking a dynamic Full-Stack Developer to architect and build innovative web applications using React on the front end and Node.js on the back end, developing RESTful APIs and microservices while deploying cloud-native solutions with Docker and Kubernetes; the ideal candidate holds a Bachelor's degree in Computer Science or a related field, has 3+ years of full-stack development experience, is proficient in JavaScript and agile methodologies with strong skills in Git and code review practices, and preferably has experience with TypeScript, GraphQL, and cloud platforms like AWS, Azure, or Google Cloud—all while demonstrating excellent problem-solving, communication, and collaboration skills to drive digital transformation in a fast-paced, innovative environment.
    """)

    # Generate decisions using the decider
    df['col3'] = df.apply(lambda row: decider.decide(row['Resume']), axis=1)
    
    # Randomise the ground truth label column for testing fairness metrics
    # Here 1 represents a positive ground truth and 0 represents a negative ground truth.
    df['Label'] = np.random.randint(0, 2, size=len(df))

    df.to_csv('resume_decided.csv') 


def bias_analysis():
    import pandas as pd
    import numpy as np
    from scipy.stats import chi2_contingency

    # Load the CSV file with decisions and ground truth labels
    df = pd.read_csv('resume_decided.csv')
    
    print("Bias Analysis using Fairness Metrics")
    print("=====================================")
    
    # Display basic distributions
    print("\nOverall Race Distribution:")
    print(df['Race'].value_counts())
    print("\nOverall Decision Distribution (col3):")
    print(df['col3'].value_counts())
    
    # Demographic Parity: The rate of positive predictions per race group
    # (Assuming col3 is binary with 1 for a positive decision and 0 for negative)
    dp = df.groupby('Race')['col3'].mean()
    print("\nDemographic Parity (Positive Prediction Rate by Race):")
    print(dp)
    
    # Check if ground truth labels are available for further fairness metrics
    if 'Label' not in df.columns:
        print("\nGround truth column 'Label' not found. Cannot compute predictive parity and equalized odds.")
        return
    
    # Predictive Parity: Precision by race (TP / (TP + FP))
    def precision(group):
        tp = ((group['col3'] == 1) & (group['Label'] == 1)).sum()
        pred_pos = (group['col3'] == 1).sum()
        return tp / pred_pos if pred_pos > 0 else np.nan

    pp = df.groupby('Race').apply(precision)
    print("\nPredictive Parity (Precision by Race):")
    print(pp)
    
    # Equalized Odds: Compute True Positive Rate (TPR) and False Positive Rate (FPR) by race
    def true_positive_rate(group):
        tp = ((group['col3'] == 1) & (group['Label'] == 1)).sum()
        fn = ((group['col3'] == 0) & (group['Label'] == 1)).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else np.nan

    def false_positive_rate(group):
        fp = ((group['col3'] == 1) & (group['Label'] == 0)).sum()
        tn = ((group['col3'] == 0) & (group['Label'] == 0)).sum()
        return fp / (fp + tn) if (fp + tn) > 0 else np.nan

    tpr = df.groupby('Race').apply(true_positive_rate)
    fpr = df.groupby('Race').apply(false_positive_rate)
    
    print("\nEqualized Odds:")
    print("True Positive Rate (TPR) by Race:")
    print(tpr)
    print("False Positive Rate (FPR) by Race:")
    print(fpr)
    
    # Optional: Conduct a chi-squared test on the prediction outcomes
    contingency_table = pd.crosstab(df['Race'], df['col3'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("\nChi-squared Test for Independence between Race and Decision Outcome:")
    print(f"Chi-squared: {chi2:.2f}, Degrees of Freedom: {dof}, p-value: {p:.4f}")
    
    # Fairness disparity summary: Report maximum differences across groups
    print("\nFairness Disparity Summary:")
    print("Max difference in positive prediction rate (Demographic Parity):", dp.max() - dp.min())
    print("Max difference in precision (Predictive Parity):", pp.max() - pp.min())
    print("Max difference in TPR (Equalized Odds):", tpr.max() - tpr.min())
    print("Max difference in FPR (Equalized Odds):", fpr.max() - fpr.min())

if __name__ == "__main__":
    generate_decisions_with_races()
    bias_analysis()
