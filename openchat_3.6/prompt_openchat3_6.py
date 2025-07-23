import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm 

# Load OpenChat model
model_id = "openchat/openchat-3.6-8b-20240522"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Fix: Set pad_token_id if undefined
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# CSV files
PROFILES_CSV = "donor_persona_all.csv"
OPTIONS_CSV = "donation_options_10000.csv"
profiles = pd.read_csv(PROFILES_CSV)
options = pd.read_csv(OPTIONS_CSV)

# Country to region mapping
country_to_region = {
    'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
    'Costa Rica': 'Central America', 'Panama': 'Central America', 'Guatemala': 'Central America',
    'El Salvador': 'Central America', 'Honduras': 'Central America', 'Brazil': 'South America',
    'Argentina': 'South America', 'Chile': 'South America', 'Colombia': 'South America',
    'Uruguay': 'South America', 'Venezuela': 'South America', 'Ecuador': 'South America',
    'Peru': 'South America', 'United Kingdom': 'Western Europe', 'France': 'Western Europe',
    'Germany': 'Western Europe', 'Spain': 'Western Europe', 'Italy': 'Western Europe',
    'Belgium': 'Western Europe', 'Netherlands': 'Western Europe', 'Switzerland': 'Western Europe',
    'Austria': 'Western Europe', 'Ireland': 'Western Europe', 'Portugal': 'Western Europe',
    'Luxembourg': 'Western Europe', 'Poland': 'Eastern Europe', 'Hungary': 'Eastern Europe',
    'Czech Republic': 'Eastern Europe', 'Slovakia': 'Eastern Europe', 'Romania': 'Eastern Europe',
    'Ukraine': 'Eastern Europe', 'Russia': 'Eastern Europe', 'Bulgaria': 'Eastern Europe',
    'Serbia': 'Eastern Europe', 'Croatia': 'Eastern Europe', 'Lithuania': 'Eastern Europe',
    'Latvia': 'Eastern Europe', 'Estonia': 'Eastern Europe', 'Slovenia': 'Eastern Europe',
    'Belarus': 'Eastern Europe', 'Bosnia and Herzegovina': 'Eastern Europe', 'Georgia': 'Eastern Europe',
    'Moldova': 'Eastern Europe', 'Egypt': 'North Africa', 'Morocco': 'North Africa',
    'Algeria': 'North Africa', 'Tunisia': 'North Africa', 'South Africa': 'South Africa',
    'Democratic Republic of the Congo': 'Central Africa', 'Cameroon': 'Central Africa',
    'Angola': 'Central Africa', 'China': 'East Asia', 'Japan': 'East Asia', 'South Korea': 'East Asia',
    'Taiwan': 'East Asia', 'Hong Kong': 'East Asia', 'Mongolia': 'East Asia', 'Singapore': 'SouthEast Asia',
    'Vietnam': 'SouthEast Asia', 'Thailand': 'SouthEast Asia', 'Indonesia': 'SouthEast Asia',
    'Philippines': 'SouthEast Asia', 'Malaysia': 'SouthEast Asia', 'Myanmar': 'SouthEast Asia',
    'India': 'SouthEast Asia', 'Israel': 'Western Europe', 'Turkey': 'Eastern Europe',
    'Iran': 'SouthEast Asia', 'Pakistan': 'SouthEast Asia', 'Bangladesh': 'SouthEast Asia',
    'Saudi Arabia': 'Central Africa', 'United Arab Emirates': 'Central Africa', 'Qatar': 'Central Africa',
    'Kazakhstan': 'Central Asia', 'Lebanon': 'Central Africa', 'Iceland': 'Western Europe',
    'Norway': 'Western Europe', 'Finland': 'Western Europe', 'Greece': 'Western Europe',
    'Macau': 'East Asia'
}

# Human-readable mappings
freq_map = {
    "MoreThanOnceMonth": "More than once a month",
    "LessThanOnceMonth": "Less than once a month",
    "LessThanOnceYear": "Less than once a year",
    "": "Unknown"
}
edu_map = {
    "LessThanHS": "Less than high school",
    "HighSchool": "High school diploma",
    "VocTraining": "Vocational training",
    "SomeCollege": "Some college education",
    "Bachelors": "Bachelor’s degree",
    "Graduate": "Graduate degree",
    "": "Unknown"
}

# Describe a donor profile
def describe_profile(p):
    return (
        f"- Age: {p['age']}\n"
        f"- Gender: {p['gender']}\n"
        f"- Income: {p['income']}\n"
        f"- Education: {edu_map.get(p['edu'], 'Unknown')}\n"
        f"- Political leaning (0 = Conservative, 10 = Progressive): {round(p['pol'] * 10)}\n"
        f"- Religious level (0 = Not religious, 10 = Very religious): {round(p['rel'] * 10)}\n"
        f"- Donated before: {'Yes' if p['donated'] == 1 else 'No'}\n"
        f"- Donation frequency: {freq_map.get(p.get('freq', ''), 'Unknown')}\n"
        f"- Agreement with 'I trust charitable organizations': {round(p['trust'], 2)}\n"
        f"- Agreement with 'Charities can be rated by effectiveness': {round(p['bel_rate'], 2)}\n"
        f"- Agreement with 'Objective measures help choose charities': {round(p['bel_obj'], 2)}\n"
        f"- Agreement with 'I choose charities matching my values': {round(p['bel_view'], 2)}\n"
        f"- Country: {p['country']}, Province: {p['province']}"
    )

# Build prompt for the LLM
def build_prompt(profile, optA, optB):
    return (
        # f"<|user|>\n"
        f"You are a donor with the following profile:\n{describe_profile(profile)}\n\n"
        f"You must choose between two donation options:\n\n"
        f"Option A: {optA.strip()}\n\n"
        f"Option B: {optB.strip()}\n\n"
        f"Which option would you choose? Answer with only 'Option A' or 'Option B'. Nothing is allowed to answer except 'Option A' or 'Option B'\n"
        # f"<|assistant|>\n"
    )

# Call LLM and return its choice
def get_llm_decision(prompt_str, max_attempts=3):
    for attempt in range(max_attempts):
        encoded = tokenizer(prompt_str, return_tensors="pt").to(model.device)
        attention_mask = encoded['input_ids'].ne(tokenizer.pad_token_id).long()

        outputs = model.generate(
            input_ids=encoded['input_ids'],
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=100,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.decode(outputs[0][encoded['input_ids'].shape[-1]:], skip_special_tokens=True).strip().lower()

        if "option a" in decoded and "option b" in decoded:
            continue
        elif "option a" in decoded:
            return decoded, "Option A" 
        elif "option b" in decoded:
            return decoded, "Option B" 

    return decoded, "Ambiguous/Failed"

# Run the simulation
results = []
progress_bar = tqdm(profiles.iloc[:50].iterrows(), total=50, desc="Processing Profiles")
for idx, profile in progress_bar:  # Adjust number of profiles as needed
    region = country_to_region.get(profile['country'], None)
    near_opts = options[options['prompt'].str.contains(region, case=False, na=False)] if region else pd.DataFrame()
    far_opts = options[~options['prompt'].str.contains(region, case=False, na=False)] if region else options.copy()

    # for i in range(2):  # Number of choices per profile
    #     optA = near_opts.sample(1)['prompt'].values[0] if not near_opts.empty else far_opts.sample(1)['prompt'].values[0]
    #     while True:
    #         optB = far_opts.sample(1)['prompt'].values[0]
    #         if optB != optA:
    #             break

    for i in range(10):  # Try 10 combinations per profile
        if not near_opts.empty:
            optA_row = near_opts.sample(1).iloc[0]
            optA = optA_row['prompt']
            optA_region = "near"
        else:
            optA_row = far_opts.sample(1).iloc[0]
            optA = optA_row['prompt']
            optA_region = "far"

        progress_bar.set_description(f"Profile {idx} | A_region: {optA_region}")

        while True:
            optB_row = far_opts.sample(1).iloc[0]
            optB = optB_row['prompt']
            if optB.strip() != optA.strip():
                break
        optB_region = "far"

        prompt_str = build_prompt(profile, optA, optB)
        answer_text, choice, = get_llm_decision(prompt_str)

        results.append({
            "profile_id": idx,
            # "profile_description": describe_profile(profile),
            "prompt": prompt_str,
            "optA": optA.strip(),
            "optB": optB.strip(),
            "optA_region": optA_region,
            "optB_region": optB_region,
            "llm_answer": answer_text,
            "chosen_option": choice
        })

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("openchat_donation_choices_all.csv", index=False)
print("Saved decisions to openchat_donation_choices.csv")
