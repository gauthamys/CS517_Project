from Decider import Decider
from util.Data import StackOverflowLoader, ResumeLoader
import pandas as pd
import numpy as np

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

df['col3'] = df.apply(lambda row: decider.decide(row['Resume']), axis=1)

df.to_csv('resume_decided.csv') 