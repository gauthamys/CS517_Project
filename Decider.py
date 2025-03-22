from ollama import chat, create
from ollama import ChatResponse
from util.Data import ResumeLoader, StackOverflowLoader

class Decider:
    def __init__(self, job_desc):
        system = """
          You are a seasoned recruiter evaluating a candidate for a job opening. You will be given the job description and the candidate's profile.
          Please respond in one word: "Yes" if we you think we can move forward with the candidate, "No" if you think we cannot, "Maybe" if you are unsure.
        """
        create(model='hiring_agent', from_='llama3.2', system=system)
        self.model = 'hiring_agent'
        self.job_description = job_desc

    def decide(self, candidate_description):
      # Construct a prompt instructing the model to provide a simple "Yes" or "No" decision
      # Can be used with both datasets
      prompt = f"""
        Job Description:
        {self.job_description}

        Candidate Profile:
        {candidate_description}
      """
      
      # Get the model's response using the prompt
      response = chat(model=self.model, messages=[
          {'role': 'user', 'content': prompt}
      ])
      
      # Return the trimmed response which should be "Yes" or "No"
      print('responded: ' + response.message.content.strip())
      return response.message.content.strip().replace('.', '')

if __name__ == "__main__":
    job_desc = """
      Seeking a dynamic Full-Stack Developer to architect and build innovative web applications using React on the front end and Node.js on the back end, developing RESTful APIs and microservices while deploying cloud-native solutions with Docker and Kubernetes; the ideal candidate holds a Bachelor's degree in Computer Science or a related field, has 3+ years of full-stack development experience, is proficient in JavaScript and agile methodologies with strong skills in Git and code review practices, and preferably has experience with TypeScript, GraphQL, and cloud platforms like AWS, Azure, or Google Cloud—all while demonstrating excellent problem-solving, communication, and collaboration skills to drive digital transformation in a fast-paced, innovative environment.
    """
    decider = Decider(job_desc)
    df = ResumeLoader.load()

    candidate = df.iloc[0]['Resume']
    result = decider.decide(candidate)
    print(result)
    