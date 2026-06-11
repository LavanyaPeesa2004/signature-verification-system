import os
import random
import pickle
from itertools import combinations

# ==================================
# PATH
# ==================================

DATASET_PATH = r"E:\Models\signature\Reference_Images"

# ==================================
# LOAD PERSONS
# ==================================

persons = {}

for person in os.listdir(DATASET_PATH):

    person_path = os.path.join(
        DATASET_PATH,
        person
    )

    if not os.path.isdir(person_path):
        continue

    images = []

    for img in os.listdir(person_path):

        img_path = os.path.join(
            person_path,
            img
        )

        images.append(img_path)

    persons[person] = images

# ==================================
# POSITIVE PAIRS
# ==================================

positive_pairs = []

for person, images in persons.items():

    for img1, img2 in combinations(images, 2):

        positive_pairs.append(
            (img1, img2, 1)
        )

# ==================================
# NEGATIVE PAIRS
# ==================================

negative_pairs = []

person_names = list(persons.keys())

for i in range(len(person_names)):

    for j in range(i + 1, len(person_names)):

        personA = persons[
            person_names[i]
        ]

        personB = persons[
            person_names[j]
        ]

        samples = min(
            len(personA),
            len(personB),
            20
        )

        for _ in range(samples):

            img1 = random.choice(
                personA
            )

            img2 = random.choice(
                personB
            )

            negative_pairs.append(
                (img1, img2, 0)
            )

# ==================================
# COMBINE
# ==================================

all_pairs = (
    positive_pairs
    + negative_pairs
)

random.shuffle(all_pairs)

# ==================================
# SAVE
# ==================================

with open(
    "custom_pairs.pkl",
    "wb"
) as f:

    pickle.dump(
        all_pairs,
        f
    )

print(
    "Positive:",
    len(positive_pairs)
)

print(
    "Negative:",
    len(negative_pairs)
)

print(
    "Total:",
    len(all_pairs)
)

print(
    "Saved: custom_pairs.pkl"
)