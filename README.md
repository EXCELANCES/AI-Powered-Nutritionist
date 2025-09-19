# AI-Powered Nutritionist Chatbot

This project was developed as part of my Master’s thesis.  
The goal is to leverage **Artificial Intelligence** to provide **personalized, reliable, and scalable nutrition advice**.  

!!This is only proof of project. Repository includes fundamental code scripts to use AI system which is created by developer. Libraries and train data are not included.!!

## 🎯 Features
- Generates **personalized diet plans** based on user profiles (age, weight, goals, allergies)  
- Provides **recipe suggestions** with detailed nutritional values  
- Retrieves nutrition facts from **millions of data points**, independent of brand or country  
- Trained with **real anonymized patient data from a dietitian clinic in Turkey**  
- Built-in **feedback system** (thumbs-up / thumbs-down) to improve responses  
- Full-stack application with **Flask backend** and **interactive frontend**  

## 🛠 Technologies Used
- **Fine-tuning (Gemma 2B)**  
- **LoRA / PEFT** for efficient model training  
- **RAG (Retrieval-Augmented Generation)**  
- **FAISS** for fast similarity search  
- **KNN** for recipe recommendation filtering  
- **Flask & SQLite** for system implementation  

## 📊 Data Sources
- **USDA** and **OpenFoodFacts** nutritional databases  
- **MenuWithNutrition** structured recipe dataset  
- **20 anonymized clinical patient cases** from a dietitian clinic in Turkey  

## 🚀 Installation
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/nutritionist-chatbot.git
   cd nutritionist-chatbot
