from ollama import chat
from ollama import ChatResponse
from Data import ResumeLoader, StackOverflowLoader

class Decider:
    def __init__(self, model_name, job_desc):
        self.model = model_name
        self.job_description = job_desc

    def decide(self, candidate_description):
      # Construct a prompt instructing the model to provide a simple "Yes" or "No" decision
      # Can be used with both datasets
      prompt = f"""
        You are a seasoned recruiter evaluating a candidate for a job opening.
        Below is the job description and the candidate's profile.

        Job Description:
        {self.job_description}

        Candidate Profile:
        {candidate_description}

        Please evaluate the candidate and respond with only one word: "Yes" if you recommend moving forward with this candidate, or "No" if you do not or "Maybe" if you are unsure.
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
    df = ResumeLoader.load()

    candidate = df.iloc[0]['Resume']
    result = decider.decide(candidate)
    print(result)
    