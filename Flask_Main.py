from flask import Flask, request, jsonify, render_template
from unsloth import FastLanguageModel
from peft import PeftModel
import torch
import os
from huggingface_hub import login
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
import sqlite3
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import random
from langchain_community.embeddings import HuggingFaceEmbeddings  # updated import
from langchain_community.vectorstores import FAISS  # updated import
import re

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "gizli_anahtar")

login(token="")

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-2b-it-bnb-4bit",
    max_seq_length=4096,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    load_in_4bit=True,
    device_map={"": torch.cuda.current_device()} if torch.cuda.is_available() else {"": "cpu"}
)

lora_path = "/home/ertua/lora_model_Best_For_All_Metrics"
model = PeftModel.from_pretrained(base_model, lora_path)
model = FastLanguageModel.for_inference(model)
model.eval()

conn = sqlite3.connect("/home/ertua/food_data.db")
recipes_df = pd.read_sql_query("SELECT * FROM recipes", conn)
nutrient_cols = ['RecipeId', 'Name', 'CookTime', 'PrepTime', 'TotalTime', 'RecipeIngredientParts',
                 'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                 'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent',
                 'ProteinContent', 'RecipeInstructions','Images']
recipes_df = recipes_df[nutrient_cols]
conn.close()

CHAT_HISTORY_FILE = "chat_history.json"
# Reset chat history on app startup
with open(CHAT_HISTORY_FILE, "w") as f:
    json.dump([], f)

#Embedding Model Part
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_path = "/home/ertua/faiss_index_openfoodfact"  # This folder must contain index.faiss and index.pkl
index_path_plan = "/home/ertua/faiss_index_Profile_Diets"
vector_db = FAISS.load_local(index_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
vector_db_plan = FAISS.load_local(index_path_plan, embeddings=embedding_model, allow_dangerous_deserialization=True)


def format_diet_plan(plan_text):
    lines = []
    # First, the summary in the first sentence (goal/condition/preference etc.) and demographics
    if "." in plan_text:
        header, body = plan_text.split(".", 1)
        lines.append(f"🟢 {header.strip()}.")
    else:
        body = plan_text

    # Divide into days and meals
    day_blocks = [block.strip() for block in body.split("|") if block.strip()]
    for block in day_blocks:
        if ":" in block:
            meal_title, meal_content = block.split(":", 1)
            meal_title = meal_title.replace("_", " ").capitalize()
            lines.append(f"    • {meal_title.strip()}: {meal_content.strip()}")
        else:
            lines.append(f"    • {block}")
    return "\n".join(lines)


def load_chat_history():
    try:
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as f:
       json.dump(history, f, indent=2, ensure_ascii=False)



# Çmultiple KNN recommendation function
def knn_recommendation(user_vector, top_k=5, tolerance_ratio=1.25, included_ingredients=None):
    nutrient_columns = ["Calories", "FatContent", "SaturatedFatContent", "CholesterolContent",
                        "SodiumContent", "CarbohydrateContent", "FiberContent", "SugarContent",
                        "ProteinContent"]

    filtered_df = recipes_df.copy()
    for col, val in zip(nutrient_columns, user_vector):
        threshold = val * tolerance_ratio
        filtered_df = filtered_df[filtered_df[col] <= threshold]

    if included_ingredients:
        included_ingredients = [i.strip().lower() for i in included_ingredients if i.strip()]
        filtered_df = filtered_df[
            filtered_df['RecipeIngredientParts'].str.lower().apply(
                lambda x: all(ing in x for ing in included_ingredients)
            )
        ]

    if filtered_df.empty:
        filtered_df = recipes_df.copy()

    X = filtered_df[nutrient_columns].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    user_scaled = scaler.transform([user_vector])

    knn = NearestNeighbors(n_neighbors=min(top_k, len(filtered_df)), metric='cosine', algorithm='brute')
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors(user_scaled)

    return filtered_df.iloc[indices[0]]


from collections import Counter

def intent_detection(chat_history, tokenizer=tokenizer, model=model):
    """
    Analyzes chat history with context windows and multi-prompts to determine intent. If the initial intent is 'recipe_request', summarizes the chat and checks if the user is asking for a specific recipe ('recipe_lookup') or a general suggestion.
    Returns: one of INTENTS (see below).
    """
    
    MULTI_PROMPTS = [
        # 1. Comprehensive prompt with examples
        "<|im_start|>system\n"
        "You are given a RECORD of user and nutrition assistant chat. Classify ONLY the user's intent for this chat. "
        "Options: recipe_request, diet_plan, nutrition_value_of_product, general_info, smalltalk. "
        "NO explanations, NO greetings, NO extra text. If unsure, answer: smalltalk.\n"
        "Examples:\n"
        "User: Can you give me nutrition values of Coca Cola?\nAssistant: Coca Cola contains X calories.\nAnswer: nutrition_value_of_product\n"
        "User: Can you suggest a breakfast recipe?\nAssistant: Here’s a healthy omelet recipe.\nAnswer: recipe_request\n"
        "User: Give me a diet plan according to my profile\nAssistant: Here’s a diet plan based on your profile.\nAnswer: diet_plan\n"
        "User: Can you prepare a meal plan according to my profile?\nAssistant: Sure, here’s a meal plan tailored to your profile.\nAnswer: diet_plan\n"
        "User: I need a personalized diet plan according to my profile\nAssistant: Here’s a personalized diet plan based on your profile.\nAnswer: diet_plan\n"
        "User: What should I eat according to my profile?\nAssistant: Here’s a recommended meal plan based on your profile.\nAnswer: diet_plan\n"
        "User: Suggest a weekly meal plan according to my profile\nAssistant: Here’s a weekly plan tailored to your profile.\nAnswer: diet_plan\n"
        "User: What are the benefits of vitamin D?\nAssistant: Vitamin D supports bone health.\nAnswer: general_info\n"
        "User: Hello!\nAssistant: Hi! How can I help?\nAnswer: smalltalk\n"
        "User: How are you?\nAssistant: I am all good, thanks.\nAnswer: smalltalk\n"
        "RECORD:\n"
        "{history}\nAnswer:",
        # 2. Direct, short prompt
        "<|im_start|>system\n"
        "Classify the user's intent in this nutrition chat. Options: recipe_request, diet_plan, nutrition_value_of_product, general_info, smalltalk. "
        "Only output the label. If unsure, say: smalltalk.\nChat:\n{history}\nIntent:",
        # 3. Explanatory and in natural language
        "<|im_start|>system\n"
        "Read the following chat and decide if the user wants a recipe, a diet plan, nutrition values for a product, general nutrition info, or is just making small talk. "
        "Pick one: recipe_request, diet_plan, nutrition_value_of_product, general_info, smalltalk.\n"
        "Output only the correct label.\nConversation:\n{history}\nIntent:",
        # 4. Clear question in label format
        "<|im_start|>system\n"
        "Which intent best describes the user's request? Valid labels: recipe_request, diet_plan, nutrition_value_of_product, general_info, smalltalk.\n"
        "Say ONLY the label.\nHistory:\n{history}\nLabel:"
    ]

    INTENTS = [
        "recipe_request",              # Suggest a recipe based on user preference (not by name)
        "recipe_lookup",               # Specific recipe by name (e.g. "Shish kebab recipe")
        "profile_based_diet_plan",     # Personalized diet plan for user profile
        "diet_plan",                   # General info about a named diet (not user-specific)
        "nutrition_value_of_product",  # Nutrition facts for a product/ingredient
        "general_info",                # Other general questions (e.g. health, food science)
        "smalltalk"
    ]

    # Only user/assistant turns
    total_turns = len([m for m in chat_history if m["role"] in ("user", "assistant")])
    window_sizes = [total_turns, 8, 5, 3, 2, 1]
    window_sizes = sorted(set([w for w in window_sizes if w > 0 and w <= total_turns]), reverse=True)

    results = []
    for window in window_sizes:
        recent_history = [turn for turn in chat_history if turn["role"] in ("user", "assistant")][-window:]
        history_text = ""
        for turn in recent_history:
            who = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{who}: {turn['content']}\n"
        for prompt_template in MULTI_PROMPTS:
            prompt = prompt_template.replace("{history}", history_text.strip())
            prompt_input = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                prompt_output = model.generate(
                    **prompt_input,
                    max_new_tokens=8,
                    temperature=0.1,
                    top_p=0.95,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_out_text = tokenizer.decode(prompt_output[0], skip_special_tokens=True).strip()
            # Extract label
            if "Answer:" in prompt_out_text:
                intent_raw = prompt_out_text.rsplit("Answer:", 1)[-1].strip()
            elif "Intent:" in prompt_out_text:
                intent_raw = prompt_out_text.rsplit("Intent:", 1)[-1].strip()
            elif "Label:" in prompt_out_text:
                intent_raw = prompt_out_text.rsplit("Label:", 1)[-1].strip()
            else:
                intent_raw = prompt_out_text.strip()


            intent_clean = intent_raw.strip().split()[0].lower() if intent_raw.strip() else "smalltalk"
            if intent_clean in INTENTS:
                results.append(intent_clean)
            else:
                results.append("smalltalk")


    from collections import Counter
    intent_general = Counter(results).most_common(1)[0][0]

    # Multi-stage: refine recipe_request/diet_plan
    if intent_general == "recipe_request":
        # Chat summary for second-stage check
        turns = [f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}" for t in chat_history if t["role"] in ("user", "assistant")]
        history_text = "\n".join(turns)
        detail_prompt = (
            "You are an intent classifier for a nutrition assistant chatbot.\n"
            "Decide the user's intent for the following conversation.\n"
            "There are ONLY TWO possible labels:\n"
            " - recipe_lookup: The user is asking for a specific recipe by name, mentioning a particular dish, or how to make a well-known meal. (e.g., 'How do I make tiramisu?', 'Give me a recipe for shish kebab', 'I was searching for recipe of meatballs, do you know?', 'How can I cook menemen?')\n"
            " - recipe_request: The user wants any general suggestion or idea, such as a type of meal, diet-friendly recipes, or meal for a specific context. (e.g., 'Suggest a healthy lunch recipe', 'What can I eat for breakfast?', 'I need a low calorie dinner idea')\n"
            "EXAMPLES:\n"
            "User: Can you give me a recipe for Shish kebab?\nLabel: recipe_lookup\n"
            "User: How do I make tiramisu?\nLabel: recipe_lookup\n"
            "User: Suggest a healthy lunch recipe.\nLabel: recipe_request\n"
            "User: Can you recommend something low calorie for dinner?\nLabel: recipe_request\n"
            "User: I was searching for recipe of meatballs, do you know?\nLabel: recipe_lookup\n"
            "INSTRUCTIONS: Just answer with one of these EXACT labels: recipe_lookup OR recipe_request. No explanation, no extra words, no punctuation.\n"
            "Now decide for the following conversation:\n"
            f"{history_text}\nLabel:"
        )
        prompt_input = tokenizer(detail_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            prompt_output = model.generate(
                **prompt_input,
                max_new_tokens=8,
                temperature=0.1,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
            )
        sub_intent = tokenizer.decode(prompt_output[0], skip_special_tokens=True).strip()
        # Take what comes after the expression 'Label:' at the end
        if "Label:" in sub_intent:
            label = sub_intent.rsplit("Label:", 1)[-1].strip()
            # Extra cleaning: if multiple words come, take the first one
            label = label.split()[0].lower()
        else:
            # fallback, if all are a single word or if it returns otherwise
            label = sub_intent.strip().split()[0].lower()

        if label in ["recipe_lookup", "recipe_request"]:
            return label
        else:
            return "recipe_request"

    elif intent_general == "diet_plan":
        # Similar logic for diet_plan
        turns = [f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}" for t in chat_history if t["role"] in ("user", "assistant")]
        history_text = "\n".join(turns)
        detail_prompt = (
            "<|im_start|>system\n"
            "You are given a RECORD of user and nutrition assistant chat. Classify ONLY the user's intent. "
            "Options: personal, general. "
            "Choose 'personal' if the user asks for a diet plan for themselves. "
            "Choose 'general' if the user asks about general diets, or for someone else.\n"
            "NO explanations, NO greetings, NO extra text. If unsure, answer: general.\n"
            "Examples:\n"
            "User: I need a personal diet plan\n"
            "Assistant: Sure! Here’s one based on your profile.\n"
            "Answer: personal\n"
            "User: Give me a meal plan according to my profile\n"
            "Assistant: Here’s a diet plan based on your body.\n"
            "Answer: personal\n"
            "User: Can you give a personal diet plan for me?\n"
            "Assistant: Here's a personalized meal plan.\n"
            "Answer: personal\n"
            "User: What is the vegan diet?\n"
            "Assistant: It excludes all animal products.\n"
            "Answer: general\n"
            "User: What should a person eat to lose weight?\n"
            "Assistant: Usually a calorie deficit helps.\n"
            "Answer: general\n"
            "User: Give me a diet for a 70kg man\n"
            "Assistant: Here's a general plan.\n"
            "Answer: general\n"
            "RECORD:\n"
            "{history}\nAnswer:"
        )

        detail_prompt = detail_prompt.replace("{history}", history_text.strip())
        prompt_input = tokenizer(detail_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            prompt_output = model.generate(
                **prompt_input,
                max_new_tokens=100,
                temperature=0.1,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
            )

        sub_intent = tokenizer.decode(prompt_output[0], skip_special_tokens=True).strip()
        label = sub_intent.rsplit("Answer:", 1)[-1].strip()
        label = label.splitlines()[0].strip().split()[0].lower()

        # simple label → technical label transformation
        if label == "personal":
            return "diet_plan"
        elif label == "general":
            return "smalltalk"
        else:
            return "smalltalk"  # fallback
    else:
        return intent_general
    


def recommend_diet_plan_with_allergy(user_profile_dict, top_k=3):
    goal = user_profile_dict.get("goal", "")
    condition = user_profile_dict.get("condition", "") or user_profile_dict.get("conditions", "")
    preference = user_profile_dict.get("preference", "")
    combined_text = f"{goal} {condition} {preference}".strip()
    allergies = user_profile_dict.get("allergies", "").lower().replace(", ", ",").split(",")
    allergies = [a.strip() for a in allergies if a.strip()]

    user_embedding = embedding_model.embed_query(combined_text)
    # Diet plan search with FAISS index
    docs_plan = vector_db_plan.similarity_search_by_vector(user_embedding, k=10)  # Over-fetch and filter

    filtered_docs = []
    for doc in docs_plan:
        # If the diet plan’s menu or the whole plan is in page_content (or .text/.content):
        plan_text = doc.page_content.lower() if hasattr(doc, "page_content") else str(doc)
        # Alternatively, it may also be in another text field.
        if not any(allergen in plan_text for allergen in allergies):
            filtered_docs.append(doc)
        if len(filtered_docs) == top_k:
            break
    return filtered_docs


def profile_update_multi_prompt_en(chat_history, tokenizer, model, window_sizes=[8, 5, 3, 1]):
    """
    Extracts nutrition profile update (field, value) with multi-prompt and context window.
    Returns the majority answer, or {"field": None, "value": None} if nothing is found.
    """
    MULTI_PROMPTS = [
                # 1. Detailed prompt with example
                """
        <|im_start|>system
        You are a nutritionist assistant. Read the following recent chat history.
        If the user has said anything that should change or update their nutrition profile, return ONLY this as JSON in this format:
        {"field": "...", "value": "..."}
        If there is no new or changed profile information, return:
        {"field": null, "value": null}
        Example:
        User: I started a vegan diet.
        Assistant: {"field": "diet_type", "value": "vegan"}
        User: Okay, thanks!
        Assistant: {"field": null, "value": null}
        <|im_end|>
        {history}<|im_start|>assistant
        """,
                # 2. Short and direct
                """
        <|im_start|>system
        Review the chat history below. If you detect a new or changed nutrition profile detail (such as allergy, diet type, goal, dislike, health condition, supplement, exercise), return it as JSON: {"field": "...", "value": "..."}. If not, return: {"field": null, "value": null}
        {history}<|im_start|>assistant
        """,
                # 3. Instructional
                """
        <|im_start|>system
        Given this chat, output ONLY a JSON with a field and value for nutrition profile update (if any). Otherwise, return both as null. Format: {"field": "...", "value": "..."}
        {history}<|im_start|>assistant
        """
            ]

    results = []
    for window in window_sizes:
        recent_history = [turn for turn in chat_history if turn["role"] in ("user", "assistant")][-window:]
        history_text = ""
        for turn in recent_history:
            who = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{who}: {turn['content']}\n"
        for prompt_template in MULTI_PROMPTS:
            prompt = prompt_template.replace("{history}", history_text.strip())
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.1,
                    top_p=0.95,
                    eos_token_id=tokenizer.eos_token_id,
                )
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # Only keep the assistant's output
            if "<|im_start|>assistant" in output_text:
                output_text = output_text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
            try:
                json_out = json.loads(output_text)
                if json_out.get("field"):
                    results.append((json_out["field"], json_out["value"]))
            except Exception:
                continue
    # Majority vote
    if results:
        count = Counter(results)
        best = count.most_common(1)[0][0]  # (field, value)
        return {"field": best[0], "value": best[1]}
    else:
        return {"field": None, "value": None}



@app.route("/")
def home():
    return render_template("bot.html")


PROFILE_PATH = r"/home/ertua/FLASK_APP/static/data/user_profile.json"

@app.route('/submit-profile', methods=['POST'])
def submit_profile():
    data = request.get_json()
    print("Gelen veri:", data)  # For checking

    # Save to file as JSON
    profile_save_path = "/home/ertua/FLASK_APP/static/data/user_profile.json"
    with open(profile_save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # Analyze user data
    try:
        height_cm = float(data.get("height", 0))
        weight_kg = float(data.get("weight", 0))
        age = int(data.get("age", 0))
        gender = data.get("gender", "unspecified").lower()
        activity_level = data.get("activity", "low").lower()
        goal_user = data.get("goal").lower()
        allergies=data.get("allergies").lower()
        conditions=data.get("conditions").lower()


        height_m = height_cm / 100
        bmi = round(weight_kg / (height_m ** 2), 2) if height_m > 0 else 0
        # Add the BMI value to the data dict
        data["bmi"] = bmi
        # Now write the updated data to the file
        profile_save_path = "/home/ertua/FLASK_APP/static/data/user_profile.json"
        with open(profile_save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        # diet goal setting (simple recommendation logic)
        if bmi < 18.5:
            goal = "gain weight"
            message = "Your BMI is {:.2f}, which is considered underweight. A calorie surplus plan is recommended.".format(bmi)
        elif 18.5 <= bmi < 25:
            goal = "maintain weight"
            message = "Your BMI is {:.2f}, which is considered healthy. A balanced nutrition plan will help you maintain it.".format(bmi)
        elif 25 <= bmi < 30:
            goal = "lose weight"
            message = "Your BMI is {:.2f}, which is considered overweight. A calorie deficit diet may be helpful.".format(bmi)
        else:
            goal = "lose weight"
            message = "Your BMI is {:.2f}, which is considered obese. A structured weight loss plan is recommended.".format(bmi)

        system_message = {
            "id": generate_message_id(),
            "role": "system",
            "content": "You are a friendly nutrition assistant—ready to chat food, health, and feel-good habits whenever you are!"
        }

        user_summary = (
            "Here is a summary of the information you provided:\n\n"
            f"• Height: {height_cm} cm\n"
            f"• Weight: {weight_kg} kg\n"
            f"• Age: {age}\n"
            f"• Gender: {gender.capitalize()}\n"
            f"• Activity level: {activity_level.capitalize()}\n"
            f"• Goal: {goal_user}\n"
            f"• Health conditions: {conditions}\n"
            f"• Allergies: {allergies}\n"
        )

        # 2. PROMPT FOR MODEL ANALYSIS
        chat_history = [
            system_message,
            {"role": "user", "content": (
                f"My height is {height_cm} cm, my weight is {weight_kg} kg, "
                f"I am {age} years old, gender: {gender}, "
                f"and my activity level is {activity_level}. "
                f"My goal is {goal}. "
                f"I have these health conditions: {conditions}. "
                f"My allergies: {allergies}. "
                f"Please summarize my information, tell me my BMI value and what it means, and then provide your suggestions for my goal."
            )}
        ]

        prompt = ""
        for turn in chat_history:
            prompt += f"<|im_start|>{turn['role']}\n{turn['content']}\n<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        reply = output_text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip() if "<|im_start|>assistant" in output_text else output_text.strip()

        # 3. ADD TWO MESSAGES TO CHAT HISTORY (first summary, then model)
        chat_history.append({"id": generate_message_id(), "role": "assistant", "content": user_summary})
        msg_id = generate_message_id()
        chat_history.append({"id": msg_id, "role": "assistant", "content": reply})
        save_chat_history(chat_history)


        # 4. RETURN TWO MESSAGES TO FRONTEND AT ONCE
        return jsonify({
            "status": "success",
            "message_id": msg_id,
            "message": "Profile saved and analysed.",
            "bmi": bmi,
            "goal": goal,
            "chat_message": user_summary + "\n\n" + reply
        })
    

    except Exception as e:
        print("Profil analiz hatası:", e)
        return jsonify({"status": "error", "message": "Profile analysis failed."}), 500

def generate_message_id():
    # Generates a unique message ID
    return f"msg-{int(datetime.now().timestamp() * 1000)}-{random.randint(100,999)}"


@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json()
        message_id = data.get("message_id")
        feedback_type = data.get("feedback_type")

        # Find the commented message
        chat_history = load_chat_history()
        matched = next((msg for msg in chat_history if msg.get("id") == message_id), None)

        if not matched:
            return jsonify({"status": "error", "message": "Message ID not found"}), 404

        # 1. Write the FEEDBACK inside the message in chat_history.json
        matched["feedback_type"] = feedback_type
        save_chat_history(chat_history)   


        return jsonify({"status": "success"})

    except Exception as e:
        print("Feedback error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500




@app.route("/ask", methods=["POST"])
def ask():
    with open("/home/ertua/FLASK_APP/static/data/user_profile.json") as f:
        user_profile = json.load(f)
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided."}), 400

    chat_history = load_chat_history()

    # Prepare profile_text in any case
    if user_profile:
        profile_text = (
            f"User profile:\n"
            f"Name: {user_profile.get('name', '')}\n"
            f"Age: {user_profile.get('age', '')}\n"
            f"Gender: {user_profile.get('gender', '')}\n"
            f"Height: {user_profile.get('height', '')} cm\n"
            f"Weight: {user_profile.get('weight', '')} kg\n"
            f"Goal: {user_profile.get('goal', '')}\n"
            f"Activity level: {user_profile.get('activity', '')}\n"
            f"Allergies: {user_profile.get('allergies', '')}\n"
            f"Health conditions: {user_profile.get('conditions', '')}\n"
        )
    else:
        profile_text = "No user profile available."

    # If chat_history is empty, add a profile system message
    if not chat_history or chat_history[0]["role"] != "system":
        chat_history.insert(0, {"id": generate_message_id(), "role": "system", "content": profile_text})
        chat_history.append({"id": generate_message_id(), "role": "system", "content": "You are a friendly nutrition assistant—ready to chat food, health, and feel-good habits whenever you are!"})

    chat_history.append({"id": generate_message_id(), "role": "user", "content": user_input})



    # --- PROFILE EXTRACTION --- 
    result = profile_update_multi_prompt_en(chat_history, tokenizer, model)
    if result and result["field"]:
        
        with open("/home/ertua/FLASK_APP/static/data/user_profile.json", "r", encoding="utf-8") as f:
            profile = json.load(f)
        field = result["field"]
        value = result["value"]
        # Multi-value fields: append if not present
        if field in ["allergies", "dislikes", "conditions", "supplements", "exercise"]:
            profile.setdefault(field, [])
            if value not in profile[field]:
                profile[field].append(value)
        else:
            profile[field] = value
     
        with open("/home/ertua/FLASK_APP/static/data/user_profile.json", "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
       
        chat_history.append({"id": generate_message_id(), "role": "system", "content": "Profile updated: ..."})


    intent = intent_detection(chat_history)

    if intent=="recipe_request":
        profile_text = profile_text if user_profile else "No user profile available."
        prompt = (
            "<|im_start|>system\n"
            f"{profile_text}\n"
            "The user wants a recipe suggestion. Use the user's profile information to suggest the optimal combination of the following parameters in grams:\n"
            "- Calories\n"
            "- Fat\n"
            "- SaturatedFat\n"
            "- Cholesterol\n"
            "- Sodium\n"
            "- Carbohydrate\n"
            "- Fiber\n"
            "- Sugar\n"
            "- Protein\n"
            "First, provide only the most suitable values in grams for each, based on the user's profile, in this format:\n"
            "Calories: [value]g, Fat: [value]g, SaturatedFat: [value]g, Cholesterol: [value]g, Sodium: [value]g, Carbohydrate: [value]g, Fiber: [value]g, Sugar: [value]g, Protein: [value]g\n"
            "Then, briefly explain why you chose these values for this user profile."
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "According to your profile information,"
        )
        prompt_input = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            prompt_output = model.generate(
                **prompt_input,
                max_new_tokens=300,
                temperature=0.1,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_out_text = tokenizer.decode(prompt_output[0], skip_special_tokens=True).strip()
        if "<|im_start|>assistant" in prompt_out_text:
            prompt_out_text = prompt_out_text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
        msg_id = generate_message_id()
        chat_history.append({"id": msg_id, "role": "assistant", "content": prompt_out_text})
        save_chat_history(chat_history)

        return jsonify({
            "ai_response": prompt_out_text,
            "message_id": msg_id,
            "trigger": "nutrition_form"
        })
    #---------------------------------------------------------------------------------------------******
    elif intent=="nutrition_value_of_product":
        query = user_input
        query_embedding = embedding_model.embed_query(query)
        docs = vector_db.similarity_search_by_vector(query_embedding, k=1)
        reply = docs[0].page_content if docs else "No info found."
        msg_id = generate_message_id()
        chat_history.append({"id": msg_id, "role": "assistant", "content": reply})
        save_chat_history(chat_history)
        return jsonify({"ai_response": reply, "message_id": msg_id})
    #---------------------------------------------------------------------------------
    elif intent == "diet_plan":
        recommendations = recommend_diet_plan_with_allergy(user_profile, top_k=1)
        if not recommendations:
            reply = "Sorry, I couldn't find a suitable diet plan for your profile and allergies."
        else:
            reply = "Here are your personalized diet plans:\n\n"
            for idx, doc in enumerate(recommendations, 1):
                formatted_plan = format_diet_plan(doc.page_content)
                reply += f"--- Diet Plan {idx} ---\n{formatted_plan}\n\n"
        msg_id = generate_message_id()
        chat_history.append({"id": msg_id, "role": "assistant", "content": reply})
        save_chat_history(chat_history)
        return jsonify({"ai_response": reply, "message_id": msg_id})
    #---------------------------------------------------------------------------------
    else:
        # 1. Make the expert role clear with a system message
        system_prompt = (
            "<|im_start|>system\n"
            "You are a highly knowledgeable and friendly nutritionist. "
            "When a user asks about recipes, diets, nutrition facts, or healthy eating, always answer in detail, provide evidence-based explanations, "
            "include tips, and educate the user. "
            "If the user asks for a recipe by name, give the full recipe step by step. "
            "If the user asks about a diet (e.g., 'What is the vegan diet?'), explain what it is, its pros and cons, typical foods, and give practical tips. "
            "Always use clear, concise, and helpful language. "
            "NEVER answer with only 'yes' or 'no'. Always elaborate."
            "<|im_end|>\n"
        )
        # 2. Add the chat history
        prompt = system_prompt
        for turn in chat_history:
            prompt += f"<|im_start|>{turn['role']}\n{turn['content']}\n<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
            )
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        reply = output_text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip() if "<|im_start|>assistant" in output_text else output_text.strip()
        msg_id = generate_message_id()
        chat_history.append({"id": msg_id, "role": "assistant", "content": reply})
        save_chat_history(chat_history)
        return jsonify({"ai_response": reply, "message_id": msg_id})

@app.route('/chat-history', methods=['GET'])
def get_chat_history():
    history = load_chat_history()  
    return jsonify({"chat_history": history})



@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json

    user_vector = [
        data.get("Calories", 0),
        data.get("Fat", 0),
        data.get("SaturatedFat", 0),
        data.get("Cholesterol", 0),
        data.get("Sodium", 0),
        data.get("Carbohydrate", 0),
        data.get("Fiber", 0),
        data.get("Sugar", 0),
        data.get("Protein", 0)
    ]

    try:
        count = int(data.get("RecommendationCount", 5))
    except ValueError:
        count = 5

    included_ingredients = [
        i.strip().lower() for i in data.get("IncludedIngredients", "").split(";") if i.strip()
    ]

    results = knn_recommendation(user_vector, top_k=count, included_ingredients=included_ingredients)
    records = results.to_dict(orient="records")

    for recipe in records:
        if isinstance(recipe["RecipeIngredientParts"], str) and recipe["RecipeIngredientParts"].startswith("["):
            try:
                recipe["RecipeIngredientParts"] = "; ".join(eval(recipe["RecipeIngredientParts"]))
            except Exception:
                pass
        if isinstance(recipe["RecipeInstructions"], str) and recipe["RecipeInstructions"].startswith("["):
            try:
                recipe["RecipeInstructions"] = "; ".join(eval(recipe["RecipeInstructions"]))
            except Exception:
                pass
        if isinstance(recipe.get("Images"), str):
            try:
                if recipe["Images"].startswith("["):
                    urls = eval(recipe["Images"])
                    if isinstance(urls, list) and urls:
                        recipe["Images"] = urls[0]
                else:
                   
                    recipe["Images"] = recipe["Images"].strip('"')
            except Exception:
                pass

    return jsonify({"recommendations": records})


@app.route('/log-selection', methods=['POST'])
def log_selection():
    try:
        selected_recipe = request.get_json()

       
        log_file = "selected_recipes.json"
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []

        selected_recipe["timestamp"] = datetime.now().isoformat()
        history.append(selected_recipe)

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

       
        chat_history = load_chat_history()

        recipe_message = f"""
                            Here is the recipe you selected:

                            **{selected_recipe.get('Name', 'Unknown Recipe')}**\n
                            Ingredients:
                            {selected_recipe.get('RecipeIngredientParts', 'N/A')}\n
                            Instructions:
                            {selected_recipe.get('RecipeInstructions', 'N/A')}\nCalories: {selected_recipe.get('Calories', '?')} kcal\nProtein: {selected_recipe.get('ProteinContent', '?')} g\nFat: {selected_recipe.get('FatContent', '?')} g\nCarbs: {selected_recipe.get('CarbohydrateContent', '?')} g 
                        """.strip()

        chat_history.append({
            "id": generate_message_id(),
            "role": "assistant",
            "content": recipe_message
        })
        save_chat_history(chat_history)

        return jsonify({
            "status": "success",
            "message": "Recipe saved and added to conversation.",
            "chat_message": recipe_message  
        })
    
    except Exception as e:
        print("Error in /log-selection:", e)
        return jsonify({"status": "error", "message": "Failed to save recipe"}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
