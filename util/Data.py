import pandas as pd


class StackOverflowLoader:
    def __init__(self):
        self.df = pd.read_csv('data/stackoverflow_full.csv')

    def load(self):
        return self.df

    def __describe_candidate(self, candidate_row):        
        # Describes the candidate details from the row.
        can = candidate_row.to_dict()
        age = 'under 35' if can['Age'] == '<35' else 'over 35'
        edu = 'Masters' if can['EdLevel'] == 'Master' else can['EdLevel']
        gender = can['Gender']
        burnout = 'burnt out' if can['MentalHealth'] == 'Yes' else 'not burnt out'
        main_branch = can['MainBranch']
        years_code = can['YearsCode']
        years_code_pro = can['YearsCodePro']
        country = can['Country']
        prev_salary = can['PreviousSalary']
        worked_with = ','.join(can['HaveWorkedWith'].split(';'))
        comp_skills = can['ComputerSkills']
        curr_employed = 'currently employed' if can['Employed'] == 1 else 'currently unemployed'
        
        return f"""
            Candidate is {age} years old. 
            Candidate has a {edu} degree.
            Candidate is a {gender}.
            Candidate is {burnout}.
            Candidate mainly works in the {main_branch} branch.
            Candidate has {years_code} years of coding experience and {years_code_pro} years of professional coding experience.
            Candidate is from the country of {country}.
            They previously earned {prev_salary} dollars annually.
            Candididate has worked with {worked_with}.
            They have a total of {comp_skills} skills.
            Candidate is {curr_employed}.
        """

class ResumeLoader:
    def load():
        return pd.read_csv('data/UpdatedResumeDataSet.csv')
    
if __name__ == "__main__":
    pass