import pandas as pd

def load_stackoverlow():
    df = pd.read_csv('data/stackoverflow_full.csv')
    return df

def load_resumes():
    df = pd.read_csv('data/UpdatedResumeDataSet.csv')
    return df

if __name__ == "__main__":
    df = load_stackoverlow()
    print(df.head())
    df_r = load_resumes()
    print(df_r.head())