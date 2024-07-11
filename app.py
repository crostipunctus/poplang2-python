from flask import Flask, request, jsonify
import openai
from google.cloud import translate_v2 as translate
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
import os
import redis

app = Flask(__name__)

load_dotenv()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Google Translate
translate_client = translate.Client()

# Initialize Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')




def generate_scenario(prompt, difficulty='intermediate', theme='general'):
    system_message = f"""
    You are a language learning assistant. Generate a {difficulty}-level
    scenario about {theme}. The scenario should be engaging, culturally 
    relevant, and appropriate for language learners.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Generate a scenario based on this prompt: {prompt}"}
        ],
        max_tokens=200,
        n=1,
        temperature=0.7,
    )
    return response.choices[0].message['content'].strip()


def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    keywords = [word for word in word_tokens if word.lower() not in stop_words and word.isalnum()]
    return list(set(keywords))  # Remove duplicates

def translate_keywords(scenario, keywords, target_language):
    # Translate the entire scenario
    translated_scenario = translate_client.translate(scenario, target_language=target_language)['translatedText']
    
    # Translate each keyword
    translated_keywords = {}
    for keyword in keywords:
        translation = translate_client.translate(keyword, target_language=target_language)['translatedText']
        translated_keywords[keyword] = translation
    
    return translated_scenario, translated_keywords

def generate_audio(text):
    # This is a placeholder. You would implement your audio generation logic here.
    # For example, you could use Google Text-to-Speech API
    return "http://example.com/audio.mp3"

@app.route('/generate', methods=['POST'])
def generate_lesson():
    data = request.json
    prompt = data['prompt']
    target_language = data['target_language']
    
    scenario = generate_scenario(prompt)
    keywords = extract_keywords(scenario)
    translated_scenario, translated_keywords = translate_keywords(scenario, keywords, target_language)
    audio_url = generate_audio(translated_scenario)  # This is a placeholder
    
    return jsonify({
        'scenario': scenario,
        'translated_scenario': translated_scenario,
        'keywords': translated_keywords,
        'audio_url': audio_url
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)