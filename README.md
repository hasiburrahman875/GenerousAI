# 🧠 Charitable Donation Choice Parser & Analysis

This repository contains code to parse structured profiles and donation options from LLM-generated outputs, converting them into a tabular format suitable for downstream analysis and modeling. This project was developed by Md Hasibur Rahman, Ganesh Sapkota, and Raja Sunkara, with the guidance of Dr. Suman Maity as a course project in the "Introduction to NLP" course.

## 📂 Project Structure

```
.
├── llama3/
│   ├── persona.py         # Script to generate user profiles
│   ├── donation.py        # Script to generate donation options
├── analysis/
│   └── analysis.ipynb     # Notebook to analyze structured data
├── donation_choices_llama3.csv  # Raw LLM output data
├── structured_donation_choices_llama3_full.csv  # Final structured dataset
├── parse_and_structure.py        # Main parsing script
└── README.md
```

## 📌 Overview

This project:

* Parses user persona data from LLM-generated prompts.
* Extracts attributes of two competing donation options (Option A and Option B).
* Combines all structured data into a final DataFrame and saves it as a CSV.
* Prepares the data for preference modeling or causal inference studies.

---

## 🚀 How to Run

1. **Generate Personas**
   Navigate to the folder for your model (e.g., `llama3/`) and run:

   ```bash
   python persona.py
   ```

2. **Generate Donation Options**
   Inside the same folder, run:

   ```bash
   python donation.py
   ```

3. **Parse & Structure the Data**
   Update the CSV path if needed in `parse_and_structure.py`, then run:

   ```bash
   python parse_and_structure.py
   ```

   This will save the cleaned dataset as:

   ```
   structured_donation_choices_llama3_full.csv
   ```

4. **Analyze Results**
   Go into the `analysis/` folder and open:

   ```bash
   analysis.ipynb
   ```
## 📊 Columns in Final Structured Data

* **User Profile Fields:**

  * `age`, `gender`, `income`, `education`, `political_leaning`, `religious_level`, etc.

* **Donation Option A and B:**

  * `A_gender`, `A_age`, `A_identifiability`, `A_relatedness`, `A_num_recipients`, `A_cause`, `A_brand`, `A_location`
  * `B_gender`, `B_age`, ... (same as above)

* **Other Fields:**

  * `option_A_region`, `option_B_region`, `chosen_option`

## 📎 Notes

* All LLM-specific generation scripts (e.g., for LLaMA-3 or others) are organized by folder name.
* The `analysis.ipynb` helps you visualize and evaluate patterns in donation preferences.

## 🖼️ For More Details

See the accompanying **presentation slide** for a full walkthrough of the dataset schema, methodology, and findings.
