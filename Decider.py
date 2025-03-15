from ollama import chat
from ollama import ChatResponse
from Data import load

class Decider:
    def __init__(self, model_name, job_desc):
        self.model = model_name
        self.job_description = job_desc

    def describe_candidate(self, candidate_row):
        #Describes the candidate details from the row.

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

    def decide(self, candidate):
      # Generate the candidate description
      candidate_description = self.describe_candidate(candidate)
      
      # Construct a prompt instructing the model to provide a simple "Yes" or "No" decision
      prompt = f"""
        You are a seasoned recruiter evaluating a candidate for a job opening.
        Below is the job description and the candidate's profile.

        Job Description:
        {self.job_description}

        Candidate Profile:
        {candidate_description}

        Please evaluate the candidate and respond: "Yes" if you recommend moving forward with this candidate, or "No" if you do not or "Maybe" if you are unsure.
        Explain your decision
      """
      
      # Get the model's response using the prompt
      response = chat(model=self.model, messages=[
          {'role': 'user', 'content': prompt}
      ])
      
      # Return the trimmed response which should be "Yes" or "No"
      return response.message.content.strip().replace('.', '')

if __name__ == "__main__":
    job_desc = """
      Seeking a dynamic Full-Stack Developer to architect and build innovative web applications using React on the front end and Node.js on the back end, developing RESTful APIs and microservices while deploying cloud-native solutions with Docker and Kubernetes; the ideal candidate holds a Bachelor's degree in Computer Science or a related field, has 3+ years of full-stack development experience, is proficient in JavaScript and agile methodologies with strong skills in Git and code review practices, and preferably has experience with TypeScript, GraphQL, and cloud platforms like AWS, Azure, or Google Cloud—all while demonstrating excellent problem-solving, communication, and collaboration skills to drive digital transformation in a fast-paced, innovative environment.
    """
    decider = Decider('llama3.2', job_desc)
    df = load()
    candidate = df.iloc[0]
    can_desc = decider.describe_candidate(candidate)
    result = decider.decide(candidate)
    print(can_desc)
    print(result)
    