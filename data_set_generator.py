#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy as np
import pandas as pd

parser = ArgumentParser(description="Generate a random students' performance dataset with N students and M marks for L Bloom's knowledge levels.")
parser.add_argument('N', help="the number of students", type=int)
parser.add_argument('M', help="the number of questions", type=int)
parser.add_argument('-b', '--bloom', help="the number of bloom levels", type=int, default=6)
parser.add_argument('-m', '--min', help="the minimum marks for a question", type=int, default=2)
parser.add_argument('-n', '--max', help="the maximum marks for a question", type=int, default=8)
parser.add_argument('file', help="the output file name without extension. Saves student data to <file>_student.csv and question details to <file>_questions.csv")
args = parser.parse_args()

args.bloom = min(args.bloom, args.M)

percent = np.random.rand(args.N) * 0.8 + 0.2

per_question = np.repeat(percent[:, np.newaxis], args.M, 1)

noise = np.random.rand(args.N, args.M) * 0.5 - 0.25

random_data = per_question + noise

filtered_data = np.clip(random_data, 0, 1)

max_marks = np.random.randint(args.min, args.max + 1, args.M)

bloom_level = np.hstack((np.random.randint(1, args.bloom + 1, args.M - args.bloom), np.arange(1, args.bloom + 1)))
np.random.shuffle(bloom_level)

data = np.around(filtered_data * max_marks * 2) / 2.0

question_data = np.vstack((max_marks, bloom_level)).transpose()

b = np.max(question_data[:, 1])
a = np.zeros((data.shape[0], b))
m = np.zeros(b)
for i, j in enumerate(question_data):
    a[:, j[1] - 1] += data[:,i]
    m[j[1] - 1] += j[0]
percent = np.divide(a, m, out=np.zeros_like(a), where=m != 0)

target = np.argmax(percent, axis=1) + 1

final_data = np.hstack((data, target.reshape((target.shape[0], 1))))

df = pd.DataFrame(final_data)
df.columns = [str(i + 1) for i in range(question_data.shape[0])] + ["Target"]
df.index += 1
df["Target"] = df["Target"].astype('int32')
df.to_csv(args.file + "_student.csv", index_label="Roll No")
df = pd.DataFrame(question_data)
df.columns = ["Max Marks", "Bloom Level"]
df.index += 1
df.to_csv(args.file + "_questions.csv", index_label="Q#")
