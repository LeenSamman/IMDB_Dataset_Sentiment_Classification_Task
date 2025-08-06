"""
4. GPT-2 Classification via Few‐Shot Prompting

In this section we will use the pre‐processed reviews saved on Hugging Face to construct our few-shot prompts. Rather than hard-coding examples, we’ll:

1. **Load** the processed dataset from Hugging Face.
2. **Sample** three representative review–sentiment pairs (with a fixed random seed for reproducibility).
3. **Display** the samples.
4. **Use** them as 1-shot, 2-shot, and 3-shot prompts for GPT-2 to classify new reviews.
"""

# Standard imports
import re
import pandas as pd
import torch

# Hugging Face Hub download helper
from huggingface_hub import hf_hub_download

# Transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Note: ensure you have installed the required packages:
# pip install transformers torch

# Download the preprocessed reviews CSV from Hugging Face
# Processed data link:
# https://drive.google.com/file/d/1PLoYC8owKyqwUpO9Td9mJC23JZ9zU1Cr/view?usp=sharing
csv_path = hf_hub_download(
    repo_id="LeenSMN/IMDB-50k-reviews",
    repo_type="dataset",
    filename="preprocessed_reviews.csv"
)

# Load processed reviews
reviews_df = pd.read_csv(csv_path)

# Sample three examples reproducibly
sample_examples = reviews_df.sample(n=3, random_state=42)[['review', 'sentiment']]

# Print each example clearly, one after another
print("Few-Shot Examples:")
for i, row in enumerate(sample_examples.itertuples(index=False), start=1):
    print(f"\nExample {i}:")
    print(f"Review   : {row.review}")
    print(f"Sentiment: {row.sentiment}")

# Convert to list of tuples for prompting
few_shot_examples = list(sample_examples.itertuples(index=False, name=None))

# Define the 3-shot examples for GPT-2 prompting
# Each sampled review and its sentiment label were printed so that
# the chosen examples could be inspected and verified before being used.
few_shot_examples = [
    (
        "really like summerslam look arena curtain just look overall interesting reason anyways good summerslam wwf do not lex luger main event yokozuna time ok huge fat man vs strong man I m glad time change terrible main event just like match luger terrible match card razor ramon vs ted dibiase steiner brother vs heavenly body shawn michael vs curt hene event shawn name big monster body guard diesel irs vs 123 kid bret hart take doink take jerry lawler stuff hart lawler interesting ludvig borga destroy marty jannetty undertaker take giant gonzalez terrible match smoking gunn tatanka take bam bam bigelow headshrinker yokozuna defend world title lex luger match boring terrible ending deserve 810",
        "positive"
    ),
    (
        "television show appeal quite different kind fan like farscape doesi know youngster 3040 year oldfan male female different country think just adore tv miniserie element tv character drive drama australian soap opera episode science fact fiction hardy trekkie run money brainbender stake wormhole theory time travel true equational formmagnificent embrace culture map possibility endless have multiple star thousand planet choose fromwith broad scope expect able illusion long farscape really come elementit succeed fail especially like star trek universe practically zero kaos element run idea pretty quickly keep rehash course 4 season manage audience attention use good continuity constant character evolution multiple thread episode unique personal touch camera specific certain character group structure allow extremely large area subject matter loyalty forge broken way issue happen pilot premiere pass just tune crichton girl see television delight available dvd admit thing keep sane whilst 12 hour night shift develop chronic insomniafarscape thing extremely long nightsdo favour watch pilot meanfarscape comet",
        "positive"
    ),
    (
        "film quickly get major chase scene increase destruction really bad thing guy hijacking steven seagal beat pulp seagal drive probably end premise movieit like decide make kind change movie plot just plan enjoy action expect coherent plot turn sense logic reduce chance get headachei do hope steven seagal try type character portray popular movie",
        "negative"
    )
]

# Load GPT-2 and set up the prompt/classification functions
MODEL_ID = "openai-community/gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
model     = GPT2LMHeadModel.from_pretrained(MODEL_ID)
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

def build_prompt(review: str, k: int) -> str:
    """
    Constructs a few-shot prompt by concatenating the first k examples,
    then appending the new review to classify.
    """
    prompt = ""
    for ex_review, ex_label in few_shot_examples[:k]:
        prompt += f"Review: \"{ex_review}\"\nSentiment: {ex_label}\n\n"
    prompt += f"Review: \"{review}\"\nSentiment:"
    return prompt

def classify_review(review: str, shots: int = 1) -> str:
    """
    Generates a classification from GPT-2 by appending up to `shots` examples,
    then selecting 'Positive' or 'Negative' from the model output.
    """
    prompt   = build_prompt(review, shots)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    output = model.generate(
        input_ids,
        max_new_tokens=3,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    gen = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    m = re.search(r"(Positive|Negative)", gen, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    # fallback
    return "Positive" if gen.lower().count("positive") >= gen.lower().count("negative") else "Negative"

# Print six additional random samples for manual review
sample_six = reviews_df.sample(n=12, random_state=42)[['review','sentiment']].reset_index(drop=True)
for i, row in sample_six.iterrows():
    print(f"\nExample {i+1}:")
    print(f"Review   : {row['review']}")
    print(f"Sentiment: {row['sentiment']}")

# Definitions 💎
"""
One-shot prompting provided GPT-2 with one example review–label pair before asking it to classify a new review.
Two-shot prompting included two example pairs in the prompt.
Three-shot prompting supplied three example pairs.
"""

# 1-Shot Prompting Example
test_review = '''jane austen definitely approve onegwyneth paltrow do awesome job capture attitude emma funny excessively silly elegant put convince british accent british maybe I m good judge fool meshe excellent slide doorsi forget she s american brilliant jeremy northam sophie thompson phyllida law emma thompson sister mother bate woman nearly steal showand ms law do not lineshighly recommend'''
print("\n--- 1-Shot Prompting ---")
print("The example used for 1-shot prompting:")
prompt_1 = build_prompt(test_review, k=1)
print("New review to classify:")
print(prompt_1)
print("Prediction:", classify_review(test_review, shots=1))

# 2-Shot Prompting Example
test_review = '''story hope highlight tragic reality youth face favela rise draw scary unsafe unfair world show beautiful color move music man dedicated friend choose accept world change action art entertain interesting emotional aesthetically beautiful film show film numerous high school student live neighborhood poverty gun violence enamor anderson protagonist recommend film age 13 subtitle image death background'''
print("\n--- 2-Shot Prompting ---")
print("The example used for 2-shot prompting:")
prompt_2 = build_prompt(test_review, k=2)
print("New review to classify:")
print(prompt_2)
print("Prediction:", classify_review(test_review, shots=2))

# 3-Shot Prompting Example
test_review = '''jeez immensely boring lead man christian schoyen get bad actor see thing character movie move america live 20 year speak lot well english pull say language skikkelig gebrokkent cool norwegian dude movie hollywood just damn shame talentless hack storyline mediocre suspicion christian schoyen do movie just live dream clearly do film hump beautiful babe'''
print("\n--- 3-Shot Prompting ---")
print("The example used for 3-shot prompting:")
prompt_3 = build_prompt(test_review, k=3)
print("New review to classify:")
print(prompt_3)
print("Prediction:", classify_review(test_review, shots=3))

# Observation:
# While giving the model more examples should outperform the model,
# the model seems to be  biased toward the classes shown in the few-shot examples.
