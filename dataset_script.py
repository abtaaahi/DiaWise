import random
import csv
import math
import pandas as pd
from collections import defaultdict
import itertools
import uuid

random.seed(42)

# === CONFIG ===
DISEASES = [
"Dengue Fever","Malaria","Typhoid Fever","Pneumonia","Tuberculosis","Urinary Tract Infection",
"Diabetes Mellitus","Hypertension","Coronary Heart Disease","Stroke","Asthma","COPD",
"Influenza","Common Cold","Gastroenteritis","Food Poisoning","Hepatitis","Peptic Ulcer",
"GERD","Migraine","Anemia","Arthritis","Skin Allergy","Fungal Skin Infection","Tonsillitis",
"Sinusitis","Kidney Stones","Chronic Kidney Disease","Thyroid Disorders","Depression","Anxiety Disorder",
"Heart Attack","Gallstones","PCOS","Meningitis","Jaundice","Eczema","Gout","Conjunctivitis",
"Otitis Media","Appendicitis","Chikungunya","Lupus","Bronchitis","IBS","Schistosomiasis","Dengue+Chik",
"Hepatitis B","Diabetic Ketoacidosis"
]

# target per disease
PER_DISEASE = 200

# token-overlap threshold to consider as near-duplicate for same disease (0..1)
MAX_OVERLAP = 0.8

# max attempts to create non-duplicate sample per disease instance
MAX_TRIES = 50

# output file
OUT_CSV = "synthetic_medical_dataset_bangla.csv"

# === helper resources ===

# symptom seeds & plausible confounders & reasoning_keywords & recommendations per disease (brief)
# For brevity we add key seeds. You can expand these lists later.
SEED = {
"Dengue Fever": {
    "symptoms":["fever","severe headache","body pain","rash","nausea","joint pain"],
    "confound":["Chikungunya","Typhoid Fever","Viral Fever"],
    "reason":"fever; severe headache; body pain; rash",
    "rec":"Get dengue test; Drink fluids; Rest and monitor fever"
},
"Malaria": {
    "symptoms":["fever","chills","sweating","headache","body pain"],
    "confound":["Dengue Fever","Typhoid Fever","Viral Fever"],
    "reason":"fever; chills; sweating",
    "rec":"Get malaria test; Use mosquito net; Visit clinic"
},
"Typhoid Fever": {
    "symptoms":["prolonged fever","abdominal pain","weakness","headache"],
    "confound":["Gastroenteritis","Dengue Fever","Urinary Tract Infection"],
    "reason":"prolonged fever; abdominal pain",
    "rec":"Get Widal/blood test; Start antibiotics if prescribed; Stay hydrated"
},
"Pneumonia": {
    "symptoms":["high fever","cough with mucus","chest pain","shortness of breath"],
    "confound":["Bronchitis","Tuberculosis","COVID-19"],
    "reason":"cough; mucus; chest pain",
    "rec":"Consult doctor; Chest X-ray; Complete antibiotics"
},
"Tuberculosis": {
    "symptoms":["persistent cough","weight loss","night sweats","fever"],
    "confound":["Pneumonia","Lung Cancer","Bronchitis"],
    "reason":"persistent cough; night sweats; weight loss",
    "rec":"Get TB test; Take full medication course; Nutrition support"
},
"Urinary Tract Infection": {
    "symptoms":["burning urination","lower abdominal pain","frequent urination","cloudy urine"],
    "confound":["Kidney Infection","Dehydration","Kidney Stones"],
    "reason":"burning urination; abdominal pain",
    "rec":"Drink water; Get urine test; Take antibiotics if prescribed"
},
"Diabetes Mellitus": {
    "symptoms":["increased thirst","frequent urination","fatigue","weight loss"],
    "confound":["Thyroid Disorders","Dehydration","Anemia"],
    "reason":"thirst; urination; fatigue",
    "rec":"Get fasting blood sugar test; Stay hydrated; Follow diabetic diet"
},
"Hypertension": {
    "symptoms":["headache","nosebleed","dizziness","blurred vision"],
    "confound":["Anxiety","Migraine","Stroke"],
    "reason":"headache; nosebleed; high blood pressure signs",
    "rec":"Check blood pressure; Reduce salt; Consult doctor"
},
"Coronary Heart Disease": {
    "symptoms":["chest pain","shortness of breath","sweating","nausea"],
    "confound":["Heart Attack","Angina","Anxiety"],
    "reason":"chest pain; sweating; breathlessness",
    "rec":"Seek emergency care if severe; ECG; Follow cardiology advice"
},
"Stroke": {
    "symptoms":["sudden weakness","facial droop","slurred speech","confusion"],
    "confound":["Migraine","Seizure","Hypoglycemia"],
    "reason":"sudden neurological deficit; speech problem",
    "rec":"Call emergency; Get CT/MRI; Immediate medical attention"
},
"Asthma": {
    "symptoms":["wheezing","shortness of breath","chest tightness","cough"],
    "confound":["Bronchitis","COPD","Allergic Reaction"],
    "reason":"wheezing; breathlessness; cough",
    "rec":"Use inhaler; Avoid triggers; See doctor if severe"
},
"COPD": {
    "symptoms":["chronic cough","shortness of breath","sputum production","wheezing"],
    "confound":["Asthma","Bronchitis","Pneumonia"],
    "reason":"chronic cough; sputum; breathlessness",
    "rec":"Avoid smoke; Use inhaler as prescribed; Pulmonary check-up"
},
"Influenza": {
    "symptoms":["fever","cough","sore throat","body ache"],
    "confound":["Common Cold","Viral Fever","COVID-19"],
    "reason":"fever; cough; body ache",
    "rec":"Rest; Drink fluids; Paracetamol if needed"
},
"Common Cold": {
    "symptoms":["sneezing","runny nose","sore throat","mild cough"],
    "confound":["Allergic Rhinitis","Influenza","Sinusitis"],
    "reason":"sneezing; runny nose; sore throat",
    "rec":"Rest; Warm fluids; Saline gargle"
},
"Gastroenteritis": {
    "symptoms":["diarrhea","vomiting","abdominal cramps","fever"],
    "confound":["Food Poisoning","Cholera","IBS"],
    "reason":"diarrhea; vomiting; abdominal cramps",
    "rec":"Drink ORS; Rest; Avoid outside food"
},
"Food Poisoning": {
    "symptoms":["vomiting","diarrhea","abdominal pain","nausea"],
    "confound":["Gastroenteritis","Cholera","Typhoid"],
    "reason":"vomiting; diarrhea; abdominal pain",
    "rec":"Drink ORS; Avoid solid food; Visit doctor if persists"
},
"Hepatitis": {
    "symptoms":["jaundice","dark urine","abdominal pain","nausea"],
    "confound":["Jaundice from other causes","Gallstones","Liver Infection"],
    "reason":"jaundice; dark urine; abdominal pain",
    "rec":"Get liver function test; Avoid alcohol; Rest"
},
"Peptic Ulcer": {
    "symptoms":["upper abdominal pain","black stool","vomiting blood","heartburn"],
    "confound":["Gastritis","GI bleed","Gastric cancer"],
    "reason":"upper abdominal pain; black stool; vomiting blood",
    "rec":"Consult gastroenterologist; Avoid NSAIDs; Get endoscopy if advised"
},
"GERD": {
    "symptoms":["heartburn","acid reflux","sour taste","bloating"],
    "confound":["Peptic Ulcer","Gastritis","Acid Reflux"],
    "reason":"heartburn; sour taste; reflux",
    "rec":"Avoid spicy food; Eat smaller meals; Take antacids"
},
"Migraine": {
    "symptoms":["severe headache","sensitivity to light","nausea","visual aura"],
    "confound":["Tension Headache","Stroke","Sinusitis"],
    "reason":"severe headache; light sensitivity; nausea",
    "rec":"Rest in dark room; Prescribed migraine meds; Hydrate"
},
"Anemia": {
    "symptoms":["fatigue","pale skin","dizziness","shortness of breath"],
    "confound":["Thyroid Disorders","Vitamin Deficiency","Chronic Disease"],
    "reason":"fatigue; pale skin; dizziness",
    "rec":"Eat iron-rich foods; Take iron supplements; Get blood test"
},
"Arthritis": {
    "symptoms":["joint pain","stiffness","swelling","reduced mobility"],
    "confound":["Gout","Rheumatoid Arthritis","Injury"],
    "reason":"joint pain; stiffness; swelling",
    "rec":"Consult doctor; Light exercise; Pain relief as advised"
},
"Skin Allergy": {
    "symptoms":["rash","itching","redness","hives"],
    "confound":["Eczema","Fungal Infection","Contact Dermatitis"],
    "reason":"rash; itching; redness",
    "rec":"Apply prescribed cream; Avoid allergens; Keep area clean"
},
"Fungal Skin Infection": {
    "symptoms":["ring-shaped rash","itching","scaly skin","spreading lesion"],
    "confound":["Eczema","Bacterial Infection","Skin Allergy"],
    "reason":"ring lesion; itching; scaly",
    "rec":"Use antifungal cream; Keep area dry; See dermatologist"
},
"Tonsillitis": {
    "symptoms":["sore throat","swollen glands","fever","difficulty swallowing"],
    "confound":["Pharyngitis","Strep Throat","Common Cold"],
    "reason":"sore throat; swollen glands; fever",
    "rec":"Gargle with salt water; Take antibiotics if bacterial; Rest"
},
"Sinusitis": {
    "symptoms":["facial pain","blocked nose","headache","thick nasal discharge"],
    "confound":["Allergic Rhinitis","Common Cold","Migraine"],
    "reason":"facial pain; blocked nose; nasal discharge",
    "rec":"Steam inhalation; Nasal spray; Consult ENT if chronic"
},
"Kidney Stones": {
    "symptoms":["severe flank pain","blood in urine","nausea","sweating"],
    "confound":["Kidney Infection","Appendicitis","Gallstones"],
    "reason":"flank pain; hematuria; severe pain",
    "rec":"Hydrate; Pain control; Ultrasound/CT as advised"
},
"Chronic Kidney Disease": {
    "symptoms":["fatigue","reduced urine output","swelling","nausea"],
    "confound":["Dehydration","Heart Failure","Kidney Infection"],
    "reason":"reduced kidney function; swelling; fatigue",
    "rec":"Consult nephrologist; Check renal function tests; Diet modification"
},
"Thyroid Disorders": {
    "symptoms":["weight change","fatigue","hair loss","palpitations"],
    "confound":["Anemia","Mental health issues","Hormonal imbalances"],
    "reason":"weight change; fatigue; hair loss",
    "rec":"Get thyroid function test; Start medication if needed; Follow up"
},
"Depression": {
    "symptoms":["sad mood","loss of interest","sleep changes","fatigue"],
    "confound":["Anxiety","Thyroid Disorder","Sleep Disorder"],
    "reason":"sadness; loss of interest; fatigue",
    "rec":"Talk to counselor; Consider therapy; Maintain routine"
},
"Anxiety Disorder": {
    "symptoms":["racing heart","sweating","worry","insomnia"],
    "confound":["Panic Attack","Hyperthyroidism","Cardiac issues"],
    "reason":"anxiety; palpitations; sweating",
    "rec":"Practice breathing exercises; Seek counseling; Medication if advised"
},
"Heart Attack": {
    "symptoms":["severe chest pain","sweating","nausea","shortness of breath"],
    "confound":["Angina","Gastroesophageal reflux","Panic Attack"],
    "reason":"severe chest pain; sweating; nausea",
    "rec":"Call emergency; Chew aspirin if advised; Urgent hospital visit"
},
"Gallstones": {
    "symptoms":["right upper quadrant pain","nausea after fatty meals","fever"],
    "confound":["Gastritis","Hepatitis","Pancreatitis"],
    "reason":"RUQ pain; fatty food trigger; nausea",
    "rec":"Avoid fatty food; Ultrasound; Consult surgeon/gastroenterologist"
},
"PCOS": {
    "symptoms":["irregular periods","acne","weight gain","hirsutism"],
    "confound":["Endometriosis","Thyroid Disorder","Hormonal imbalance"],
    "reason":"irregular periods; acne; weight changes",
    "rec":"Consult gynecologist; Lifestyle changes; Hormonal evaluation"
},
"Meningitis": {
    "symptoms":["severe headache","neck stiffness","fever","vomiting"],
    "confound":["Migraine","Encephalitis","Sepsis"],
    "reason":"neck stiffness; severe headache; fever",
    "rec":"Emergency care; Hospitalization; Lumbar puncture if advised"
},
"Jaundice": {
    "symptoms":["yellowing skin/eyes","dark urine","fatigue","abdominal pain"],
    "confound":["Hepatitis","Gallstones","Hemolytic anemia"],
    "reason":"jaundice; dark urine; fatigue",
    "rec":"Get liver tests; Avoid alcohol; Medical evaluation"
},
"Eczema": {
    "symptoms":["dry itchy skin","red patches","scaly lesions","flare-ups"],
    "confound":["Skin Allergy","Psoriasis","Fungal Infection"],
    "reason":"dry itchy skin; red patches; scaly lesions",
    "rec":"Moisturize; Avoid triggers; Use prescribed topical meds"
},
"Gout": {
    "symptoms":["sudden joint pain","redness","swelling","big toe pain"],
    "confound":["Arthritis","Septic arthritis","Trauma"],
    "reason":"sudden joint pain; swelling; redness",
    "rec":"Avoid purine-rich food; Take anti-inflammatory meds; See physician"
},
"Conjunctivitis": {
    "symptoms":["red eyes","itching","discharge","tearing"],
    "confound":["Allergic conjunctivitis","Dry eyes","Uveitis"],
    "reason":"red eyes; discharge; itching",
    "rec":"Avoid touching eyes; Use eye drops; Consult ophthalmologist if worsening"
},
"Otitis Media": {
    "symptoms":["ear pain","fever","reduced hearing","discharge"],
    "confound":["Ear wax impaction","Sinusitis","Mastoiditis"],
    "reason":"ear pain; fever; hearing reduced",
    "rec":"Ear drops as prescribed; Avoid water in ear; See ENT if severe"
},
"Appendicitis": {
    "symptoms":["right lower abdominal pain","nausea","fever","loss of appetite"],
    "confound":["Gastroenteritis","Kidney Stones","Ovarian torsion"],
    "reason":"RLQ pain; fever; nausea",
    "rec":"Emergency surgery evaluation; Do not eat; Seek immediate care"
},
"Chikungunya": {
    "symptoms":["fever","severe joint pain","rash","fatigue"],
    "confound":["Dengue Fever","Viral Fever","Arthritis"],
    "reason":"fever; joint pain; rash",
    "rec":"Rest; Hydrate; Pain control"
},
"Lupus": {
    "symptoms":["joint pain","rash","fatigue","photosensitivity"],
    "confound":["Rheumatoid Arthritis","Dermatomyositis","Viral Infection"],
    "reason":"autoimmune signs; rash; joint pain",
    "rec":"Refer to rheumatology; Immunological tests; Manage symptoms"
},
"Bronchitis": {
    "symptoms":["cough with/without sputum","chest discomfort","fatigue","mild fever"],
    "confound":["Pneumonia","Asthma","COPD"],
    "reason":"cough; sputum; chest discomfort",
    "rec":"Rest; Hydrate; Seek doctor if fever or worsening"
},
"IBS": {
    "symptoms":["abdominal pain","bloating","constipation/diarrhea","relief after defecation"],
    "confound":["Gastritis","Lactose Intolerance","Inflammatory Bowel Disease"],
    "reason":"abdominal pain; bloating; bowel irregularity",
    "rec":"Diet modification; Fiber intake; See gastroenterologist if severe"
},
"Schistosomiasis": {
    "symptoms":["abdominal pain","blood in urine/stool","fever","rash"],
    "confound":["UTI","Parasitic infection","Schistosoma-related disease"],
    "reason":"blood in urine/stool; exposure to freshwater; abdominal pain",
    "rec":"Get stool/urine test; Anti-helminthic treatment; Public health advice"
},
"Dengue+Chik": {
    "symptoms":["fever","body pain","rash","severe joint pain"],
    "confound":["Dengue Fever","Chikungunya","Viral Fever"],
    "reason":"fever; joint pain; rash",
    "rec":"Get blood tests; Hydrate; Rest; Monitor warning signs"
},
"Hepatitis B": {
    "symptoms":["jaundice","fatigue","abdominal pain","dark urine"],
    "confound":["Hepatitis A","Hepatitis C","Alcoholic liver disease"],
    "reason":"jaundice; abdominal pain; dark urine",
    "rec":"Get hepatitis panel; Avoid alcohol; Clinical follow-up"
},
"Diabetic Ketoacidosis": {
    "symptoms":["excessive thirst","frequent urination","abdominal pain","confusion","fruity breath"],
    "confound":["Diabetes Mellitus","Dehydration","Sepsis"],
    "reason":"high sugar; dehydration; fruity breath",
    "rec":"Seek emergency care; Check blood glucose and ketones; IV fluids and insulin"
}
}

# generic templates to express symptoms in natural language
EN_TEMPLATES = [
    "I have been experiencing {symptoms} for {duration}.",
    "Since {duration} I've had {symptoms}.",
    "For the past {duration} I'm noticing {symptoms}.",
    "{symptoms} started {duration} ago and it's getting {severity}.",
    "I feel {symptoms} and it's been {duration}.",
    "I've been feeling {severity} with {symptoms} for {duration}.",
    "My problem: {symptoms}. It's been {duration}."
]

BN_TEMPLATES = [
    "আমি {symptoms} {duration} থেকে অনুভব করছি।",
    "গত {duration} ধরে আমার সমস্যা হচ্ছে {symptoms}।",
    "আমার কাছে {duration} ধরে {symptoms} আছে।",
    "{symptoms} {duration} থেকে হচ্ছে, দয়া করে দেখুন।",
    "ডাক্তার, আমি {duration} ধরে {symptoms} অনুভব করছি।",
    "আমার সমস্যা: {symptoms}, এটি {duration} ধরে চলছে।"
]

# Expanded English templates
EN_TEMPLATES += [
    "Lately, I’ve noticed {symptoms} over {duration}.",
    "For the last {duration}, I’m troubled by {symptoms}.",
    "I’m experiencing {symptoms} since {duration}, it seems {severity}.",
    "Could you help? I have {symptoms} for {duration}.",
    "It started {duration} ago: {symptoms}, getting {severity}.",
    "I feel {symptoms} for {duration}, can this be serious?"
]

BN_SMALL_MAP = {
    "I have been experiencing":"আমি অনুভব করছি",
    "Since":"হতে",
    "For the past":"গত",
    "it's been":"এটা হয়েছে",
    "I feel":"আমি অনুভব করছি",
    "My problem":"আমার সমস্যা",
    "fever":"জ্বর",
    "severe headache":"চরম মাথা ব্যথা",
    "body pain":"শরীর ব্যথা",
    "rash":"চামড়ার র‍্যাশ",
    "nausea":"বমি ভাব",
    "joint pain":"জয়েন্ট ব্যথা",
    "chills":"কাঁপুনি",
    "sweating":"ঘাম",
    "cough":"কাশি",
    "mucus":"কফ",
    "abdominal pain":"পেট ব্যথা",
    "vomiting":"বমি",
    "diarrhea":"ডায়রিয়া",
    "dark urine":"গাঢ় প্রস্রাব",
    "yellowing skin/eyes":"চামড়া/চোখ হলুদ হওয়া",
    "shortness of breath":"শ্বাসকষ্ট",
    "burning urination":"প্রস্রাবে জ্বালা",
    "frequent urination":"ঘন ঘন প্রস্রাব",
    "increased thirst":"বেশি তৃষ্ণা",
    "fatigue":"ক্লান্তি",
    "weight loss":"ওজন কমা",
    "headache":"মাথা ব্যথা",
    "dizziness":"মাথা ঘোরা",
    "pale skin":"ফ্যাকাসে ত্বক",
    "sore throat":"গলা ব্যথা",
    "runny nose":"নাক দিয়ে পানি পড়া",
    "sneezing":"হাঁচি",
    "itching":"চুলকানি",
    "redness":"লালচে ভাব",
    "hives":"যোন",
    "blurred vision":"দৃষ্টি ঝাপসা",
    "nausea after fatty meals":"তেলযুক্ত খাবারের পর বমি",
    "neck stiffness":"ঘাড় শক্ত",
    "sensitivity to light":"আলোর প্রতি সংবেদনশীলতা",
}

# transliteration map for Banglish (very approximate)
BN_TO_BANGLISH = {
    "আমি":"ami","অনুভব করছি":"onubhob korchi","জ্বর":"jôr","চরম":"charom","মাথা":"matha","ব্যথা":"byatha",
    "শরীর":"shorir","বমি":"bomi","র‍্যাশ":"rash","কাঁপুনি":"kapuni","ঘাম":"gham","কাশি":"kashi","কফ":"kof",
    "পেট":"pet","বমি ভাব":"bomi vhab","ডায়রিয়া":"diarrhea","প্রস্রাবে":"prosrab e","জ্বালা":"jwala","তৃষ্ণা":"trishna",
    "ক্লান্তি":"klanti","ওজন":"ozon","কমা":"koma","গলা":"gola","নাক":"nak","হাঁচি":"hanchi","চুলকানি":"chulkani",
    "লালচে":"lalche","দৃষ্টি":"drishti","ঝাপসা":"jhapsa","ঘাড়":"ghar","শক্ত":"shokto","আলো":"alo","সংবেদনশীলতা":"songbedonshilota"
}

DURATIONS = ["2 hours", "6 hours", "1 day", "2 days", "3 days", "1 week", "2 weeks", "since yesterday", "since morning"]
DURATIONS_BN = ["২ ঘন্টা","৬ ঘন্টা","১ দিন","২ দিন","৩ দিন","১ সপ্তাহ","২ সপ্তাহ","গতকাল থেকে","আজ সকাল থেকে"]

SEVERITIES = ["mild","moderate","severe","getting worse","worsening","a little better"]
SEVERITIES_BN = ["হালকা","মধ্যম","তীব্র","খারাপ হয়ে যাচ্ছে","বেশি খারাপ","আংশিক উন্নতি"]

LANG_CHOICES = ["bn"]

# create candidate confounders for diseases missing above by picking other diseases
for d in DISEASES:
    if d not in SEED:
        # fallback generic seeds
        SEED[d] = {
            "symptoms":["symptom1","symptom2","symptom3"],
            "confound":["Common Cold","Viral Infection","Gastroenteritis"],
            "reason":"symptom1; symptom2",
            "rec":"Consult doctor; Rest; Symptomatic care"
        }

# helper functions
def choose_symptoms(disease, k=3):
    pool = SEED[disease]["symptoms"]
    k = min(k, len(pool))
    chosen = random.sample(pool, k)
    return chosen

def symptom_phrase_from_list(symptoms, language="en"):
    # join with natural connectors
    if language=="en":
        if len(symptoms)==1:
            return symptoms[0]
        if len(symptoms)==2:
            return f"{symptoms[0]} and {symptoms[1]}"
        return ", ".join(symptoms[:-1]) + ", and " + symptoms[-1]
    elif language=="bn":
        # translate tokens present in BN_SMALL_MAP if possible
        translated = []
        for w in symptoms:
            translated.append(BN_SMALL_MAP.get(w,w))
        if len(translated)==1:
            return translated[0]
        if len(translated)==2:
            return f"{translated[0]} এবং {translated[1]}"
        return " , ".join(translated[:-1]) + " এবং " + translated[-1]
    elif language=="banglish":
        # transliterate via BN_TO_BANGLISH where possible, else use original
        transl = []
        for w in symptoms:
            b = BN_SMALL_MAP.get(w,w)
            transl_words = []
            for tok in b.split():
                transl_words.append(BN_TO_BANGLISH.get(tok,tok))
            transl.append(" ".join(transl_words))
        if len(transl)==1: return transl[0]
        if len(transl)==2: return f"{transl[0]} and {transl[1]}"
        return ", ".join(transl[:-1]) + ", and " + transl[-1]
    else:
        # mixed: combine english & bangla tokens randomly
        mixed=[]
        for w in symptoms:
            if random.random()<0.5:
                mixed.append(BN_SMALL_MAP.get(w,w))
            else:
                mixed.append(w)
        if len(mixed)==1: return mixed[0]
        if len(mixed)==2: return f"{mixed[0]} and {mixed[1]}"
        return ", ".join(mixed[:-1]) + ", and " + mixed[-1]

def make_input_text(disease, language):
    # choose number of symptom tokens 2-4
    n = random.choices([2,3,4],weights=[0.3,0.5,0.2])[0]
    symptoms = choose_symptoms(disease,k=n)
    symptom_text = symptom_phrase_from_list(symptoms,language=language)
    # duration and severity
    if language=="en":
        dur = random.choice(DURATIONS)
        sev = random.choice(SEVERITIES)
        template = random.choice(EN_TEMPLATES)
        s = template.format(symptoms=symptom_text, duration=dur, severity=sev)
        # small prefixes/suffixes
        if random.random()<0.15:
            s = random.choice(["Hi doc, ","Doc, ","Hello, "]) + s
        if random.random()<0.12:
            s += " Could this be serious?"
        return s

    elif language=="bn":
        dur = random.choice(DURATIONS_BN)
        template = random.choice(BN_TEMPLATES)
        s = template.format(symptoms=symptom_text, duration=dur)
        # optional chatty prefix
        if random.random()<0.15:
            s = "ডাক্তার, " + s
        return s

    elif language=="banglish":
        dur = random.choice(DURATIONS)
        sev = random.choice(SEVERITIES)
        symptom_text_b = symptom_phrase_from_list(symptoms, language="banglish")
        template = random.choice(EN_TEMPLATES)
        s = template.format(symptoms=symptom_text_b, duration=dur, severity=sev)
        if random.random()<0.12:
            s += " doctor please"
        return s

    else:
        # mixed: English sentence with Bangla suffix
        dur = random.choice(DURATIONS)
        sev = random.choice(SEVERITIES)
        en_tpl = random.choice(EN_TEMPLATES)
        en_part = en_tpl.format(symptoms=symptom_text, duration=dur, severity=sev)
        bn_suffix = random.choice([" দয়া করে সাহায্য করুন।"," একটু দেখবেন?"," ধন্যবাদ।"])
        return en_part + bn_suffix

def pick_confounders(disease):
    confs = SEED[disease]["confound"]
    # ensure primary disease not repeated
    cands = [c for c in confs if c!=disease]
    if len(cands) < 2:
        # fill with random other diseases
        extras = random.sample([d for d in DISEASES if d!=disease], 2-len(cands))
        cands += extras
    # choose two distinct confounders
    chosen = random.sample(cands, 2)
    return [disease] + chosen

def sample_probabilities(primary_weight_base=0.6):
    # primary gets base between 0.5 and 0.85, others split remainder
    p1 = random.uniform(0.55,0.85)
    r = 1.0 - p1
    p2 = random.uniform(0.05, r-0.02) if r>0.02 else r/2
    p3 = r - p2
    # ensure numerical stability
    probs = [p1, max(0.0,p2), max(0.0,p3)]
    # normalize
    s = sum(probs)
    probs = [round(p/s,2) for p in probs]
    # if rounding causes sum !=1, adjust the largest
    diff = 1.0 - sum(probs)
    if abs(diff) > 1e-6:
        idx = probs.index(max(probs))
        probs[idx] = round(probs[idx] + diff, 2)
    return probs

def lime_explainability_from_symptoms(symptoms, language="en"):
    # choose top contributing tokens (3)
    toks = []
    for s in symptoms:
        # pick significant words
        toks += [w for w in s.split() if len(w)>2]
    random.shuffle(toks)
    toks = list(dict.fromkeys(toks))
    top = toks[:3]
    # clean up for language if needed
    return ", ".join(top)

def reasoning_keywords_from_seed(disease):
    return SEED[disease]["reason"]

def recommendations_from_seed(disease):
    return SEED[disease]["rec"]

def uncertainty_from_probs(probs):
    # measure uncertainty as normalized entropy (0..1)
    ent = -sum([p*math.log(p+1e-12) for p in probs])
    # max entropy for 3-class uniform is ln(3)
    max_ent = math.log(3)
    unc = round(ent / max_ent, 2)
    return unc

# function to get token set for overlap checking
def token_set(s):
    toks = [w.strip().lower() for w in s.replace(","," ").replace(";"," ").split() if w.strip()]
    return set(toks)

# generate dataset
rows = []
existing_texts_per_disease = defaultdict(list)
global_id = 16914

for disease in DISEASES:
    tries_total = 0
    generated = 0
    target = PER_DISEASE
    while generated < target:
        tries_total += 1
        if tries_total > target * MAX_TRIES:
            # give up if too many failures (shouldn't happen normally)
            print(f"Warning: too many tries for {disease}, generated {generated}/{target}")
            break
        lang = "bn"
        input_text = make_input_text(disease, language=lang)
        # duplicate/near-duplicate check within same disease
        ts = token_set(input_text)
        too_similar = False
        for prev in existing_texts_per_disease[disease]:
            prev_ts = token_set(prev)
            if len(prev_ts)==0: continue
            overlap = len(ts & prev_ts) / max(len(prev_ts),1)
            if overlap >= MAX_OVERLAP:
                too_similar = True
                break
        if too_similar:
            continue
        # build predicted labels & probs
        preds = pick_confounders(disease)
        probs = sample_probabilities()
        # LIME explanation tokens (approx)
        # Choose symptoms used in the input by scanning seed symptom words present
        used_symptoms = [s for s in SEED[disease]["symptoms"] if any(s.lower() in input_text.lower() for _ in [0] )]
        # fallback: pick first 3 seed symptoms
        if not used_symptoms:
            used_symptoms = SEED[disease]["symptoms"][:3]
        lime = lime_explainability_from_symptoms(used_symptoms, language=lang)
        reasoning = reasoning_keywords_from_seed(disease)
        recs = recommendations_from_seed(disease)
        unc = uncertainty_from_probs(probs)
        # build predicted_diseases string and probability string
        pred_str = ",".join(preds)
        prob_str = ",".join([f"{p:.2f}" for p in probs])
        # append row
        rows.append({
            "id": global_id,
            "input_text": input_text,
            "predicted_diseases": pred_str,
            "probabilities": prob_str,
            "lime_explainability": lime,
            "reasoning_keywords": reasoning,
            "recommendations": recs,
            "uncertainty_score": unc,
            "language": lang
        })
        existing_texts_per_disease[disease].append(input_text)
        global_id += 1
        generated += 1

# convert to DataFrame and save
df = pd.DataFrame(rows, columns=["id","input_text","predicted_diseases","probabilities","lime_explainability","reasoning_keywords","recommendations","uncertainty_score","language"])
df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_ALL)
print(f"Saved {len(df)} rows to {OUT_CSV}")
