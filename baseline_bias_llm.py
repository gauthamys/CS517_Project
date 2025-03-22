from Decider import Decider
from util.Data import StackOverflowLoader, ResumeLoader
import pandas as pd

print('loaded dataset')
df = ResumeLoader.load()

decider = Decider("""
Seeking a dynamic Full-Stack Developer to architect and build innovative web applications using React on the front end and Node.js on the back end, developing RESTful APIs and microservices while deploying cloud-native solutions with Docker and Kubernetes; the ideal candidate holds a Bachelor's degree in Computer Science or a related field, has 3+ years of full-stack development experience, is proficient in JavaScript and agile methodologies with strong skills in Git and code review practices, and preferably has experience with TypeScript, GraphQL, and cloud platforms like AWS, Azure, or Google Cloud—all while demonstrating excellent problem-solving, communication, and collaboration skills to drive digital transformation in a fast-paced, innovative environment.
""")

df['col3'] = df.apply(lambda row: decider.decide(row['Resume']), axis=1)

df.to_csv('resume_decided.csv')