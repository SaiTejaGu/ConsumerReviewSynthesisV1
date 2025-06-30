import streamlit as st
import pandas as pd
import re
import os
import numpy as np
import pandas as pd
from openai import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import warnings
warnings.filterwarnings("ignore")


# Load the dataset
df = pd.read_csv(r'FinalData_Prediction_1806_25.csv', encoding='ISO-8859-1')
columns_to_group = ['Cleaned_DeviceFamilyName','PFT', 'Processor Generation', 'GPU_Cleaned', 'NPU_TOPS','RAM', 'HardDrive', 'ScreenSize', 'Display Type', 'Pixels', 'PriceInDollars']
df = df.groupby(columns_to_group, as_index=False).filter(lambda x: x['Review_Count'].sum() >= 1000)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

# client = AzureOpenAI(
    # api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    # api_version="2024-02-01",
    # azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
# )

# deployment_name = 'Surface_Analytics'

# start_phrase = """

# You are an AI Assistant, your task is to extract relevant sub-aspects from below mentioned list of Aspect and Sub-Aspects and their corresponding sentiment (Positive or Negative) from the user's prompt. This information will be used to forecast user sentiment accurately.

# Please carefully analyze the user's prompt to identify the appropriate sub-aspects from the predefined list provided below.
 

## Instructions:
# 1. Identify only Relevant Sub-Aspects: Based on the user's prompt, select the most appropriate sub-aspects from the predefined list.
# 2. Assign Sentiments: Classify each sub-aspect as either Positive or Negative. Make sure:
   # - Sub-aspects are unique within each sentiment category.
# 3. IMPORTANT: ENSURE THAT NO SUB-ASPECT IS REPEATED IN BOTH THE POSITIVE AND NEGATIVE CATEGORIES UNDER ANY CIRCUMSTANCES.
# 4. Handle Irrelevant Prompts: If the prompt is unrelated to any predefined sub-aspects, mark both Positive and Negative categories as "None."
# 5. If the RAM increases and the price also increases accordingly, that is expected behavior. In such cases, sub-aspects related to price will not be highlighted, as the price change is justified. However, if the price increase is disproportionately high and does not align with the value of the upgrade, it should be flagged as a negative aspect of price.

# *Note: Thoroughly examine each prompt for accurate extraction like for example if in user prompt he mentioned about any aspect which will affect the net sentiment less then you should mention those related keywords in Negative, as these sub-aspects are essential for reliable sentiment forecasting.*

## Example Format:
# - User Prompt: "I have increased the performance for the Surface Laptop Pro 11 and decreased the display size, so can you generate hypothetical reviews?"
  
# - Expected Answer Format:
  # - Positive: Processor/CPU Related, Thermal/Temperature,Efficiency, Optimization, Improvement, Upgrades, Comparison, Benchmarking, Productivity, Work, Testing, Development, Thermal Management, Hardware, Connectivity,Ports, Storage/Memory.
  # - Negative: Display, Size, Price, Size and Form Factor, Brightness and Visibility.


## Predefined List of Aspects and Sub-Aspects:
# - Performance: Performance, Processor/CPU Related, Thermal/Temperature, Power, Energy, Gaming, Graphics, Efficiency, Optimization, Issues, Failures, Improvement, Upgrades, Comparison, Benchmarking, Productivity, Work, Testing, Development, Thermal Management, Memory, Storage.
# - Design: Aesthetics, Appearance, Size, Weight, Build Quality, Durability, Materials, Finishes, Portability, Flexibility, Design, Form Factor, Customization, Personalization, Branding, Logos, Interaction, Innovation, Technology, Usability, Comfort, Environmental, Sustainability, Accessories, Add-ons, Ventilation, Cooling.
# - Software: Operating Systems, Development, Programming, Software Applications, System Management Tools, Software Functionality, Features, Multimedia, Creative Tools, Web & Cloud Services, Database, Data Management, Security, Privacy, Automation, AI, Productivity Tools, Performance, Optimization, Design, User Experience, Learning, Development, Integration, Interoperability, Hardware, Devices, Miscellaneous.
# - AI Capabilities: Core AI Technologies and Concepts, AI Applications and Capabilities, AI Hardware and Processing, AI Platforms and Tools, AI Functions and Performance, AI in Development and Engineering, AI Use Cases and Applications, AI in Business and Marketing, AI Ethics and Risks, AI Hype and Buzzwords, Miscellaneous Technologies, Common AI Keywords.
# - Display: Display Technology Types, Resolution and Image Quality, Size and Form Factor, Color and Visual Quality, Touch and Interaction, Brightness and Visibility, Refresh Rate and Performance, Display Features and Enhancements, Screen Design and Build, Display Modes and Settings, Usage and Viewing Experience, Issues and Drawbacks, Advanced Features and Specs, General and Descriptive Terms.
# - Co-Creator: Collaboration & Communication, Creativity & Innovation, Tools & Technology, Support & Guidance, Productivity & Efficiency, Content Creation & Sharing, Business & Partnerships, Innovation & Future, Design & Development.
# - Battery: Battery Life & Longevity, Power Consumption & Efficiency, Charging & Power Supply, Performance & Usage, Size & Capacity, Additional Features & Characteristics.
# - Hardware: Processor & CPU, Memory & Storage, Hardware Components & Build, Brands & Models, Performance & Specifications, Components & Accessories, Form Factors & Design, Manufacturing & Technology, Miscellaneous.
# - Security: Authentication and Access Control, Data Protection and Privacy, Cybersecurity and Threats, System Security and Integrity, Management and Monitoring, Encryption and Data Storage, Security Practices and Tools, Security Vulnerabilities and Risks, Regulatory and Compliance, Encryption and Authentication Technologies.
# - Price: Pricing and Cost Overview, Discounts, Deals, and Savings, Expense and Investment, Price Changes and Adjustments, Comparison and Valuation, Budgeting and Affordability, Sales and Revenue, Special Terms and Conditions, Financial Management and Tools.
# - Connectivity: Connection Types and Technologies, Connection Methods and Devices, Connectivity and Integration, Data Transfer and Communication, Network and Signal, Compatibility and Access, Connectivity Issues and Support, Smart and Advanced Features, Miscellaneous and Additional Terms.
# - Gaming: Gaming Platforms and Devices, Performance and Hardware, Gaming Genres and Types, Gaming Experiences and Features, Game Titles and Franchises, Game Development and Customization, Gaming Accessories and Equipment, Gaming Terminology and Metrics, Community and Social Aspects, Game Modes and Play Styles, Miscellaneous.
# - Audio: Audio Devices and Equipment, Audio Quality and Characteristics, Audio Effects and Features, Audio Playback and Usage, Audio Technologies and Brands, Audio Characteristics and Performance, Audio Settings and Adjustments, Audio Descriptors and Qualifiers, Audio Interactions and Community, Miscellaneous.
# - Keyboard: Keyboard Types and Layouts, Keyboard Features and Functionality, Comfort and Usability, Design and Aesthetics, Performance and Durability, Size and Form Factor, Adjustment and Customization, Additional Features and Accessories, Feedback and User Experience, Miscellaneous.
# - Ports: USB Types and Variants, Connections and Compatibility, Ports and Slots, Design and Placement, Performance and Usability, Types of Ports, Miscellaneous.
# - Graphics: Graphics Hardware and Models, Graphics Performance and Quality, Graphics Processing and Technology, Graphics Cards and Features, Visual Effects and Quality, Testing and Benchmarking, Drivers and Updates, Configuration and Settings, Miscellaneous.
# - Recall: Recall and Memory Processes, Search and Retrieval Techniques, Information Handling and Management, Accuracy and Verification, Communication and Documentation, Issue and Problem Handling, Search Tools and Technology, Data Analysis and Processing, Recovery and Restoration, Memory and Cognitive Processes, Miscellaneous Concepts, Security and Privacy.
# - Touchpad: Touchpad Functionality and Performance, Touchpad Size and Shape, Touchpad Features and Gestures, Touchpad Materials and Design, Touchpad Issues and Problems, Touchpad Interaction and Feedback, Touchpad Usability and Customization, Touchpad Aesthetics and Design Considerations, Touchpad Enhancements and Adjustments, Touchpad Interaction with Other Devices, Touchpad Quality and Feedback, Touchpad Adjustments and Controls.
# - Automatic Super Resolution: Resolution and Quality, Generation and Enhancement, Features and Tools, Visual Effects and Processing, Detection and Analysis, Technical Aspects, Miscellaneous, Quality Improvement and Features.
# - Camera: Camera Types and Features, Image Quality and Resolution, Camera Functionality and Controls, Technical and Image Processing, Visual Effects and Improvements, Camera Design and Build, Miscellaneous.
# - Live Captions: Live Captions and Subtitles, Translation and Transcription, Video Streaming, Functionality and Features, Communication and Interaction, Accessibility and Usability, Editing and Quality, Technical and System Aspects, Issues and Problems, Additional Context and Information.
# - Account: Account Access and Authentication, Account MaCnagement and Configuration, Communication and Support, Privacy and Security, Billing and Financials, Account Information and Data, Integration and Synchronization, Account Features and Customization, Miscellaneous.
# - Storage/Memory: Types of Storage, Capacity and Size, Upgrade and Expansion, Performance and Speed, Storage Management, Brands and Models, Technical Specifications, Requirements and Cost, Miscellaneous.

# Here is the user prompt: 
# """
# Define sub_aspect_extraction function
# def sub_aspect_extraction(user_prompt):
    # try:
        # response = client.completions.create(
            # model=deployment_name,
            # prompt=start_phrase + user_prompt,
            # max_tokens=10000,
            # temperature=0
        # )
        # return response.choices[0].text.strip()
    # except Exception as e:
        # print(f"Error processing review: {e}")
        # return "Generic"
        
endpoint = os.environ.get("o4_ENDPOINT")
deployment = "R9T"
api_key = os.environ.get("o4_API_KEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint,
)

# Define a function for sentiment classification
def sub_aspect_extraction(sentence: str) -> str:
    try:        
        response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": """
        Purpose
        To accurately extract relevant sub-aspects and assign sentiment (Positive or Negative) from a user's prompt, based on a predefined mapping of Aspect → Sub-Aspects. This structured extraction enables precise sentiment prediction for hypothetical or simulated real-world changes in device specifications.


        Role
        You are an AI Assistant. Your task is to meticulously analyse the user’s prompt to extract meaningful sub-aspects and correctly classify their sentiment (Positive or Negative) based on context and semantics. The sub-aspects must align strictly with the predefined mapping provided in the {Aspect : Sub-Aspect} list. The resulting output will be used to model sentiment changes triggered by device feature modifications.

        Your role is to identify which predefined sub-aspects (from a given Aspect → Sub-Aspect list) are logically affected by those changes and classify the sentiment associated with each — from the perspective of consumer experience and perception.
        Think through the user's query as if you're a consumer writing or reading product reviews. What would they likely complain about if a feature is downgraded? What would they likely celebrate or appreciate when a feature is upgraded? This consumer-first reasoning must guide your sub-aspect and sentiment identification process.

        Output
        A structured response consisting of two distinct lists:
            • Positive: Unique sub-aspects from the predefined list that the user has implicitly or explicitly expressed favorable sentiment toward.
            • Negative: Unique sub-aspects the user has shown dissatisfaction with.
            • If no relevant sub-aspects apply to the prompt, both categories should be marked as "None."

        Marker
        Please adhere to the following instructions precisely—these are mandatory rules that ensure the response is both accurate and logically grounded:
            1. Understand the Task
                ○ Only extract sub-aspects from the provided {Aspect : Sub-Aspect} mapping.
                ○ No sub-aspect may appear in both the Positive and Negative categories—this is non-negotiable.
                
            2. Identify Relevant Sub-Aspects
                ○ Closely read the user’s prompt and deconstruct it semantically and logically. The objective is to uncover every relevant sub-aspect impacted by the change—not just those explicitly mentioned, but also those implicitly affected through valid chains of correlation.
                    ○ Begin by identifying the primary sub-aspects—those directly referenced or clearly implied by the user. Then, reason out secondary and tertiary sub-aspects—those that are affected indirectly as a consequence of changes to primary aspects
                    ○ Follow causal chains of correlation to extract not just directly mentioned sub-aspects, but also those that are logically impacted—through primary, secondary, or tertiary relationships.
                    ○ These sub-aspects should flow through valid correlations, and you must cease extraction of positively or negatively correlated sub-aspects once logical or contextual correlation no longer exists. Do not force extrapolation beyond what can be reasoned contextually
                    ○ When a feature is upgraded or downgraded, you must extract a minimum of ten (10) relevant sub-aspects. This helps capture the broader sentiment impact and ensures holistic evaluation.
                        - However, do not artificially stretch the reasoning to reach five. Only include sub-aspects that have a clear technical or experiential correlation to the change.
                        - If the number of valid sub-aspects is still fewer than ten after examining all logical correlations, mention only the valid ones and note explicitly: “Fewer than ten relevant sub-aspects found based on meaningful correlations; no forced extrapolation applied.”
                    ○ For example, changing the display type from LCD to OLED doesn't only affect the display technology; it also has downstream effects on power efficiency, brightness levels, and potentially heat management. Follow this correlated chain fully, but with discipline.
                
        Example Prompt:
        "What is the sentiment of the device when the display type changes from LCD to OLED while keeping all other specifications the same for the hypothetical device compared to the actual device?"
        Reasoning:
            • The primary sub-aspect affected is Display Technology Types.
            • Secondary effects may include Color and Visual Quality, Brightness and Visibility (due to OLED), and Power Consumption & Efficiency (as OLED may drain more battery).
        Sub-Aspect Extraction:
            • Positive: Display Technology Types, Color and Visual Quality, Brightness and Visibility
            • Negative: Power Consumption & Efficiency

            3. Assign Sentiments via Causal Deduction
                • For each extracted sub-aspect, infer whether the user's tone, context, or logical implication is Positive or Negative.
                • Use contextual reasoning to evaluate implicit consequences (e.g., battery drain from better display, cost implications of minor upgrades)
                • Do not mention any price-related details or cost implications unless there is a significant price difference or the price information is specifically required due to specification changes.
                
            IMPORTANT: The model must simulate realistic device-level change modeling. In business decisions, over-tagging minor or low-impact negative aspects distorts sentiment modeling
            Correlation Logic Example:
            Example 1: If RAM increases and price also rises modestly, price may not be flagged negatively. But if the price jumps disproportionately (e.g., +$500 for a 4GB or 8GB RAM difference), assign a Negative sentiment for the Price sub-aspect.
            Example 2: If a specification upgrades, the first thing the business will do is increase the price for the particular product(in this case device).In such cases, flagging price as a sub-aspect will not make sense. What will make sense is flagging price aspect if the user explicitly mentions that he is increasing/decreasing price for the particular feature change. 
            Example 3: Usually specification changes such as processor changes will impact performance and battery significantly, therefore there is a primary correlation. Additionally, the sub aspects related to Thermal will change if there is a major processor upgrade in the same chassis. A minor processor change will not affect thermals as much.
            Example Issue for Example 3:
                RAM increase from 4GB to 8GB should not trigger:
                
                Negative: Thermal/Temperature or Price
                unless the change is explicitly mentioned as affecting those areas or is unusually large (e.g., +16GB RAM).
        4.  Handle Irrelevant Prompts
            • If the prompt doesn’t map to any sub-aspects, return:
                ○ Positive: None
                ○ Negative: None

        Pattern
        Example (Final Output Format)
            • User Prompt:
        "I have increased the performance for the Surface Laptop Pro 11 and decreased the display size, so can you generate hypothetical reviews?"
            • Expected Output:
                ○ Positive: Processor/CPU Related, Thermal/Temperature, Efficiency, Optimization, Improvement, Upgrades, Comparison, Benchmarking, Productivity, Work, Testing, Development, Thermal Management, Hardware, Connectivity, Ports, Storage/Memory
                ○ Negative: Display, Size, Price, Size and Form Factor, Brightness and Visibility

        Tone
            • Professional, logical, and analytical
            • Prioritize clarity, precision, and structured reasoning
            • Follow a consistent and formal linguistic style while maintaining accessibility
            • Interpret user inputs thoughtfully and systematically, drawing from both explicit cues and logical implications. Avoid assumptions not grounded in the provided sub-aspect taxonomy




        ### Predefined List of Aspects and Sub-Aspects:
        - Performance: Performance, Processor/CPU Related, Thermal/Temperature, Power, Energy, Gaming, Graphics, Efficiency, Optimization, Issues, Failures, Improvement, Upgrades, Comparison, Benchmarking, Productivity, Work, Testing, Development, Thermal Management, Memory, Storage.
        - Design: Aesthetics, Appearance, Size, Weight, Build Quality, Durability, Materials, Finishes, Portability, Flexibility, Design, Form Factor, Customization, Personalization, Branding, Logos, Interaction, Innovation, Technology, Usability, Comfort, Environmental, Sustainability, Accessories, Add-ons, Ventilation, Cooling.
        - Software: Operating Systems, Development, Programming, Software Applications, System Management Tools, Software Functionality, Features, Multimedia, Creative Tools, Web & Cloud Services, Database, Data Management, Security, Privacy, Automation, AI, Productivity Tools, Performance, Optimization, Design, User Experience, Learning, Development, Integration, Interoperability, Hardware, Devices, Miscellaneous.
        - AI Capabilities: Core AI Technologies and Concepts, AI Applications and Capabilities, AI Hardware and Processing, AI Platforms and Tools, AI Functions and Performance, AI in Development and Engineering, AI Use Cases and Applications, AI in Business and Marketing, AI Ethics and Risks, AI Hype and Buzzwords, Miscellaneous Technologies, Common AI Keywords.
        - Display: Display Technology Types, Resolution and Image Quality, Size and Form Factor, Color and Visual Quality, Touch and Interaction, Brightness and Visibility, Refresh Rate and Performance, Display Features and Enhancements, Screen Design and Build, Display Modes and Settings, Usage and Viewing Experience, Issues and Drawbacks, Advanced Features and Specs, General and Descriptive Terms.
        - Co-Creator: Collaboration & Communication, Creativity & Innovation, Tools & Technology, Support & Guidance, Productivity & Efficiency, Content Creation & Sharing, Business & Partnerships, Innovation & Future, Design & Development.
        - Battery: Battery Life & Longevity, Power Consumption & Efficiency, Charging & Power Supply, Performance & Usage, Size & Capacity, Additional Features & Characteristics.
        - Hardware: Processor & CPU, Memory & Storage, Hardware Components & Build, Brands & Models, Performance & Specifications, Components & Accessories, Form Factors & Design, Manufacturing & Technology, Miscellaneous.
        - Security: Authentication and Access Control, Data Protection and Privacy, Cybersecurity and Threats, System Security and Integrity, Management and Monitoring, Encryption and Data Storage, Security Practices and Tools, Security Vulnerabilities and Risks, Regulatory and Compliance, Encryption and Authentication Technologies.
        - Price: Pricing and Cost Overview, Discounts, Deals, and Savings, Expense and Investment, Price Changes and Adjustments, Comparison and Valuation, Budgeting and Affordability, Sales and Revenue, Special Terms and Conditions, Financial Management and Tools.
        - Connectivity: Connection Types and Technologies, Connection Methods and Devices, Connectivity and Integration, Data Transfer and Communication, Network and Signal, Compatibility and Access, Connectivity Issues and Support, Smart and Advanced Features, Miscellaneous and Additional Terms.
        - Gaming: Gaming Platforms and Devices, Performance and Hardware, Gaming Genres and Types, Gaming Experiences and Features, Game Titles and Franchises, Game Development and Customization, Gaming Accessories and Equipment, Gaming Terminology and Metrics, Community and Social Aspects, Game Modes and Play Styles, Miscellaneous.
        - Audio: Audio Devices and Equipment, Audio Quality and Characteristics, Audio Effects and Features, Audio Playback and Usage, Audio Technologies and Brands, Audio Characteristics and Performance, Audio Settings and Adjustments, Audio Descriptors and Qualifiers, Audio Interactions and Community, Miscellaneous.
        - Keyboard: Keyboard Types and Layouts, Keyboard Features and Functionality, Comfort and Usability, Design and Aesthetics, Performance and Durability, Size and Form Factor, Adjustment and Customization, Additional Features and Accessories, Feedback and User Experience, Miscellaneous.
        - Ports: USB Types and Variants, Connections and Compatibility, Ports and Slots, Design and Placement, Performance and Usability, Types of Ports, Miscellaneous.
        - Graphics: Graphics Hardware and Models, Graphics Performance and Quality, Graphics Processing and Technology, Graphics Cards and Features, Visual Effects and Quality, Testing and Benchmarking, Drivers and Updates, Configuration and Settings, Miscellaneous.
        - Recall: Recall and Memory Processes, Search and Retrieval Techniques, Information Handling and Management, Accuracy and Verification, Communication and Documentation, Issue and Problem Handling, Search Tools and Technology, Data Analysis and Processing, Recovery and Restoration, Memory and Cognitive Processes, Miscellaneous Concepts, Security and Privacy.
        - Touchpad: Touchpad Functionality and Performance, Touchpad Size and Shape, Touchpad Features and Gestures, Touchpad Materials and Design, Touchpad Issues and Problems, Touchpad Interaction and Feedback, Touchpad Usability and Customization, Touchpad Aesthetics and Design Considerations, Touchpad Enhancements and Adjustments, Touchpad Interaction with Other Devices, Touchpad Quality and Feedback, Touchpad Adjustments and Controls.
        - Automatic Super Resolution: Resolution and Quality, Generation and Enhancement, Features and Tools, Visual Effects and Processing, Detection and Analysis, Technical Aspects, Miscellaneous, Quality Improvement and Features.
        - Camera: Camera Types and Features, Image Quality and Resolution, Camera Functionality and Controls, Technical and Image Processing, Visual Effects and Improvements, Camera Design and Build, Miscellaneous.
        - Live Captions: Live Captions and Subtitles, Translation and Transcription, Video Streaming, Functionality and Features, Communication and Interaction, Accessibility and Usability, Editing and Quality, Technical and System Aspects, Issues and Problems, Additional Context and Information.
        - Account: Account Access and Authentication, Account MaCnagement and Configuration, Communication and Support, Privacy and Security, Billing and Financials, Account Information and Data, Integration and Synchronization, Account Features and Customization, Miscellaneous.
        - Storage/Memory: Types of Storage, Capacity and Size, Upgrade and Expansion, Performance and Speed, Storage Management, Brands and Models, Technical Specifications, Requirements and Cost, Miscellaneous.

        Here is the user prompt: 
        """},
                {"role": "user", "content": sentence}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return "Generic"
    
def calculate_net_sentiment(group):
    total_sentiment_score = group['Sentiment_Score'].sum()
    total_review_count = group['Review_Count'].sum()

    if total_review_count == 0:
        return 0
    else:
        return (total_sentiment_score / total_review_count) * 100
        
        
def calculate_negative_review_percentage(group):
    total_reviews = group['Review_Count'].sum()
    negative_reviews = group[group['Sentiment_Score'] == -1]['Review_Count'].sum()
    
    if total_reviews == 0:
        return 0
    else:
        return (negative_reviews / total_reviews) * 100


def calculate_review_percentages(group):
    total_reviews = group['Review_Count'].sum()
    
    positive_reviews = group[group['Sentiment_Score'] == 1]['Review_Count'].sum()
    neutral_reviews = group[group['Sentiment_Score'] == 0]['Review_Count'].sum()
    negative_reviews = group[group['Sentiment_Score'] == -1]['Review_Count'].sum()
    
    if total_reviews == 0:
        return {'Positive_Percentage': 0, 'Neutral_Percentage': 0, 'Negative_Percentage': 0}
    
    return {
        'Positive_Percentage': (positive_reviews / total_reviews) * 100,
        'Neutral_Percentage': (neutral_reviews / total_reviews) * 100,
        'Negative_Percentage': (negative_reviews / total_reviews) * 100
    }

        
        
# def get_conversational_chain_hypothetical_summary():
    # global model
    # global history
    # try:
        # prompt_template = """  
        # Analyze the sentiment changes between the **actual device (Before%)** and the **hypothetical device (After%)** for the top 5 aspects with the highest review counts (excluding 'Generic' aspect).  
         
        ## **Expected Output Format (STRICTLY follow this format):**  
         
        # 1. **Aspect Name** (**Actual Device: Before% → Hypothetical Device: After%**): **Brief summary of user dissatisfaction**  
        # 2. **Aspect Name** (**Actual Device: Before% → Hypothetical Device: After%**): **Brief summary of user dissatisfaction**  
        # 3. **Aspect Name** (**Actual Device: Before% → Hypothetical Device: After%**): **Brief summary of user dissatisfaction**  
        # 4. **Aspect Name** (**Actual Device: Before% → Hypothetical Device: After%**): **Brief summary of user dissatisfaction**  
        # 5. **Aspect Name** (**Actual Device: Before% → Hypothetical Device: After%**): **Brief summary of user dissatisfaction**  
         
        ## **Conclusion (1-2 Sentences):**  
        # Summarize which aspects improved and which still have issues. **Do NOT include generic statements about the device. Focus only on sentiment changes!**  
         
        ### **Data Provided:**  
        # - **Actual Device (Before%) vs. Hypothetical Device (After%) Sentiment Data:**  
        # {context}  
         
        ### **Question:**  
        # {question}  
         
        ### **Answer:**  
        # """

        # prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        # model = AzureChatOpenAI(azure_deployment="Thruxton_R",api_version='2024-03-01-preview',temperature = 0.0)
        # chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        # return chain
    # except Exception as e:
        # err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        # return err
        
# Function to handle user queries using the existing vector store
# def hypothetical_summary(user_question, vector_store_path="Prediction_Indexes_1306_25"):
    # try:
        # embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        # vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        # chain = get_conversational_chain_hypothetical_summary()
        # docs = vector_store.similarity_search(user_question)
        # response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        # return response["output_text"]
    # except Exception as e:
        # err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        # return err
        
def get_conversational_chain_hypothetical_summary():
    global model
    global history
    try:
        prompt_template = """
        You are a data analytics expert with 10 years of experience. Provide rich, insightful English and always adhere exactly to the specified format.

        Step-by-step reasoning:
        1. Identify the top 5 aspects by review count, excluding “Generic.”
        2. For each aspect, extract Actual Device sentiment (Before%) and Hypothetical Device sentiment (After%).
        3. Compare Before% → After%, noting “improved” if After% > Before% or “declined” otherwise.
        4. Summarize each aspect’s user dissatisfaction in 1–2 sentences, focusing strictly on the sentiment change.
        5. Present your results in this exact format (use Markdown bold, colons, and arrows as shown):

           1. **Aspect Name** (**Actual Device: Before% → Hypothetical Device: After%**): Brief summary of change-focused user dissatisfaction  
           2. **Aspect Name** (**Actual Device: Before% → Hypothetical Device: After%**): Brief summary of change-focused user dissatisfaction  
           3. **Aspect Name** (**Actual Device: Before% → Hypothetical Device: After%**): Brief summary of change-focused user dissatisfaction  
           4. **Aspect Name** (**Actual Device: Before% → Hypothetical Device: After%**): Brief summary of change-focused user dissatisfaction  
           5. **Aspect Name** (**Actual Device: Before% → Hypothetical Device: After%**): Brief summary of change-focused user dissatisfaction  

        ### Conclusion (1–2 sentences):
        Summarize which aspects improved and which still have issues. Do NOT include any generic device comments—focus solely on sentiment changes.

        #### Data Provided:
        {context}

        #### Question:
        {question}

        #### Answer:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(azure_deployment="Thruxton_R",api_version='2024-03-01-preview',temperature = 0.0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err
        
# Function to handle user queries using the existing vector store
def hypothetical_summary(user_question, vector_store_path="Prediction_Indexes_1306_25"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_hypothetical_summary()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review hypothetical summary summarization: {e}"
        return err
    
def assign_sentiment(row,a,b):
    global positive_aspects, negative_aspects
    
    subaspect = row['Sub_Aspect']
    current_sentiment = row.get('Sentiment_Score', 0)

    if pd.isna(subaspect):
        return current_sentiment

    subaspect = subaspect.lower()

    if any(pos.lower() in subaspect for pos in a):
        return 1
    elif any(neg.lower() in subaspect for neg in b):
        return -1
    else:
        return current_sentiment

# def assign_sentiment(row, positive_aspects, negative_aspects):
    # subaspect = row['SubAspect']  # Replace with the correct column name
    # current_sentiment = row.get('Sentiment_Score', 0)  # Get the current sentiment, default to 0 if not available

    # if pd.isna(subaspect):
        # return current_sentiment  # Return the existing sentiment if subaspect is NaN

    # subaspect = subaspect.lower()

    # if any(pos.lower() in subaspect for pos in positive_aspects):
        # return 1  # Update to positive sentiment
    # elif any(neg.lower() in subaspect for neg in negative_aspects):
        # return -1  # Update to negative sentiment
    # else:
        # return current_sentiment  # Keep the existing sentiment

        
def Specs_Summary():
    global model
    global history
    try:
        prompt_template = """  

        You are a tool designed to frame a question based to specification inputs: 
        you will get actual device specifications and hypothetical device specifications so based on the specs you need to identify changes and give the question
        
        Identify the differences in specifications. For each difference, frame a question about the sentiment or implications of these changes.
        For example, if the processor changes from Intel Core i9 to Intel Core Ultra 7, or if the hard drive changes from 1TB to 512GB, frame a question that reflects on the significance of that change. Ignore any specifications that are the same.
        Output: What is the sentiment of the device when the processor changes from Intel Core i9 to Intel Core Ultra 7, the hard drive changes from 1TB to 512GB, and the screen size changes from 16.0 to 17.3 inches?"
        
        Important:**DO NOT INCLUDE SPECIFICATIONS WHICH REMAINS SAME IN BOTH ACTUAL AND HYPOTHETICAL DEVICES**
                  **ONLY GIVE SPECIFICATIONS WHICH ARE CHANGING - DO NOT TALK ABOUT SPECIFICATIONS WHICH REMAINS SAME**
                  
        These are the Specifications that actual device and hypothetical device has:Processor, RAM, Hard Drive, Screen Size, Price Band, GPU, Resolution, Processor Generation,display Type, NPU etc...
        
        Common Comparisons to consider when framing questions:
        1. Processor: 
            - Intel Core i7 is more powerful than Intel Core i5.
            - Intel Core i9 is more powerful than Intel Core i7.
            - Intel Core Ultra 7 is more powerful than Intel Core i9.

        2. RAM: 
            - More RAM (e.g., 16GB vs 8GB) improves multitasking and performance.
            - DDR5 is faster and more efficient than DDR4.

        3. Hard Drive:
            - SSD is faster and more reliable than HDD.
            - 1TB is better than 512GB for more storage space.

        4. Screen Size:
            - A larger screen (e.g., 17.3 inches vs 15.6 inches) offers better visibility and multitasking.

        5. Price Band:
            - A higher price band suggests better performance, features, and build quality.

        6. GPU:
            - NVIDIA RTX 3080 is better than NVIDIA GTX 1660 Ti for better gaming and performance.
            - AMD Radeon often provides good value compared to NVIDIA GPUs.

        7. Resolution:
            - 4K is better than Full HD for sharper, clearer visuals.
            - 1440p offers a balanced option between 1080p and 4K.

        8. Processor Generation:
            - A newer 12th generation Intel processor is better than a 10th generation for better power efficiency and performance.

        9. Display Type:
            - OLED is better than LED for better contrast and color accuracy.
            - IPS offers wider viewing angles and better color accuracy compared to TN panels.
            
        10. NPU:
            - NPU is the processor performance wich can be quantified by TOPS(Trillion Operations Per Second)
            - 40 TOPS > 30 TOPS> 20 TOPS> 10 TOPS
            - iF we increase the no.of TOPS for a processor , then the processor power and performance gets increased.

        Specifications to Compare:
        Processor, RAM, Hard Drive, Screen Size, Price Band, GPU, Resolution, Processor Generation, Display Type, etc.
        
        NOTE: **DO NOT INCLUDE SPECIFICATIONS WHICH REMAINS SAME IN BOTH ACTUAL AND HYPOTHETICAL DEVICES** 
        
        Example Question: For the selected device 
        Context: {context}

        Question: {question}

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(azure_deployment="Thruxton_R",api_version='2024-03-01-preview',temperature = 0.0)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def spec_sentence(user_question, vector_store_path="Prediction_Indexes_1306_25"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = Specs_Summary()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review spec_sentence: {e}"
        return err
        
def calculate_sentiment_percentage(df):
    df = df[df["Display Type"] == "OLED"]
    sentiment_summary = df.groupby(["PriceBand", "Sentiment_Score"])["Review_Count"].sum().unstack(fill_value=0)
    total_reviews = sentiment_summary.sum(axis=1)
    sentiment_percentage = (sentiment_summary.div(total_reviews, axis=0) * 100).reset_index()
    sentiment_percentage = sentiment_percentage.rename(columns={-1: "Negative (%)", 0: "Neutral (%)", 1: "Positive (%)"})
    sentiment_percentage = sentiment_percentage.reset_index().rename_axis(None, axis=1)
    return sentiment_percentage
    
def adjust_display_sentiment(filtered_df, target_percentages):
    aspect_counts = filtered_df["Aspect"].value_counts()
    highest_aspect_count = aspect_counts.max()
    threshold_count = round(0.25 * highest_aspect_count)
    display_df = filtered_df[filtered_df["Aspect"] == "Display"].copy()
    display_count = len(display_df)
    if display_count >= threshold_count:
        increase_factor = 0
    else:
        required_count = threshold_count - display_count 
        increase_factor = required_count / display_count
    if increase_factor > 0:
        num_extra_rows = round(display_count * increase_factor)
        extra_display_df = display_df.sample(n=num_extra_rows, replace=True, random_state=42)
        display_df = pd.concat([display_df, extra_display_df], ignore_index=True)
    total_reviews = len(display_df)
    num_positive = round((target_percentages["positive"] / 100) * total_reviews)
    num_neutral = round((target_percentages["neutral"] / 100) * total_reviews)
    num_negative = total_reviews - (num_positive + num_neutral)
    new_sentiments = (
        [1] * num_positive + 
        [0] * num_neutral + 
        [-1] * num_negative)
    np.random.shuffle(new_sentiments)
    display_df["Sentiment_Score"] = new_sentiments
    updated_df = pd.concat([filtered_df[filtered_df["Aspect"] != "Display"], display_df], ignore_index=True)
    return updated_df

def find_changed_aspects(actual_df, hypothetical_df):
    actual_df = actual_df.loc[:, ~actual_df.columns.str.contains("^Unnamed")]
    hypothetical_df = hypothetical_df.loc[:, ~hypothetical_df.columns.str.contains("^Unnamed")]
    actual_df = actual_df.rename(columns=lambda x: x.strip())
    hypothetical_df = hypothetical_df.rename(columns=lambda x: x.strip())
    merged_df = actual_df.merge(hypothetical_df, on="Specification")
    actual_col = "Actual Value"
    hypothetical_col = "Hypothetical Value"
    if actual_col not in merged_df.columns or hypothetical_col not in merged_df.columns:
        raise KeyError(f"Expected columns '{actual_col}' and '{hypothetical_col}' not found. Check column names.")
    changed_aspects = merged_df[merged_df[actual_col] != merged_df[hypothetical_col]]
    changes = dict(zip(changed_aspects["Specification"], changed_aspects[hypothetical_col]))
    if "Price Band" in hypothetical_df["Specification"].values:
        price_band_value = hypothetical_df[hypothetical_df["Specification"] == "Price Band"][hypothetical_col].values[0]
        changes["Price Band"] = price_band_value
    return changes

def extract_price_band(value):
    numbers = re.findall(r'\d+', str(value)) 
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    elif len(numbers) == 1:
        return int(numbers[0]), None
    return None, None

def find_best_price_band(lower_band, upper_band, available_bands):    
    for band in available_bands:
        band_lower, band_upper = extract_price_band(band)        
        if band_lower is not None and band_upper is not None:
            if lower_band >= band_lower and upper_band <= band_upper:
                print(f"🎯 Matched with: {band}")
                return band
    nearest_band = min(available_bands, key=lambda b: abs(extract_price_band(b)[0] - lower_band))
    print(f"⚠ No exact match. Selecting closest band: {nearest_band}")
    return nearest_band

def get_target_percentages(specs_df, oled_percentage_df):
    price_band_value = specs_df.loc[specs_df["Specification"] == "Price Band", "Hypothetical Value"].values[0]
    lower_band, upper_band = extract_price_band(price_band_value)    
    if lower_band is None or upper_band is None:
        print("❌ Invalid Price Band format. Using default values.")
        return None
    available_bands = oled_percentage_df["Price Band"].unique()
    best_band = find_best_price_band(lower_band, upper_band, available_bands)
    print(f"🔹 Selected Price Band: {best_band}")
    row = oled_percentage_df[oled_percentage_df["Price Band"] == best_band]
    if not row.empty:
        return {
            "positive": int(round(row["Positive (%)"].values[0])),
            "neutral": int(round(row["Neutral (%)"].values[0])),
            "negative": int(round(row["Negative (%)"].values[0]))
        }
    else:
        print(f"⚠ No suitable price band found for {best_band}. Using default values.")
        return None

def map_aspects(column_name):
    aspect_mapping = {
        "Processor Generation": "Performance",
        "GPU": "Graphics",
        "NPU": "AI Capabilities",
        "RAM": "Storage/Memory",
        "Hard Drive": "Storage",
        # "Pixels": "Display",
        # "Display Type": "Display",
        "Screen Size": "Design"
    }
    return aspect_mapping.get(column_name.strip(), "Unknown")
  

def calculate_sentiment_percentage_per_column_oem(df, filter_dict, selected_oem):
    sentiment_results = {}
    filter_dict = filter_dict.copy()

    # Extract Price Band from filter_dict
    price_band = filter_dict.pop("Price Band", None)
    available_bands = df['Price Band'].unique()

    if not price_band:
        raise ValueError("Price Band must be included in the filter dictionary.")

    price_band_values = extract_price_band(price_band)

    if isinstance(price_band_values, (tuple, list)) and len(price_band_values) == 2:
        upper_band, lower_band = price_band_values
    else:
        raise ValueError(f"extract_price_band(price_band) did not return exactly two values: {price_band_values}")

    price_band_new = find_best_price_band(upper_band, lower_band, available_bands)

    # First, filter by selected OEM in the OEM column
    df_filtered = df[df["OEM"] == selected_oem]
    
    # Special case handling for 'Display' aspect
    display_columns = ["Display Type", "Resolution"]
    # Handle display-related filters
    display_filters = {}
    if "Display Type" in filter_dict:
        display_filters["Display Type"] = filter_dict.pop("Display Type")
    if "Resolution" in filter_dict:
        display_filters["Pixels"] = filter_dict.pop("Resolution")  # Map Resolution to Pixels column

    if display_filters:
        aspect = "Display"

        # Start with full combination of filters
        display_filtered_df = df_filtered.copy()
        for col, val in display_filters.items():
            display_filtered_df = display_filtered_df[display_filtered_df[col] == val]

        oem_review_count = display_filtered_df[display_filtered_df["Price Band"] == price_band_new]["Review_Count"].sum()

        # If combination doesn't meet threshold, check each filter individually
        if display_filtered_df.empty or oem_review_count < 400:
            print("Warning: OEM Display data insufficient for combined filters. Trying individual filters.")

            individual_pass = False
            for single_col in display_filters:
                temp_df = df_filtered[df_filtered[single_col] == display_filters[single_col]]
                temp_count = temp_df[temp_df["Price Band"] == price_band_new]["Review_Count"].sum()

                if temp_count >= 400:
                    display_filtered_df = temp_df.copy()
                    individual_pass = True
                    print(f"Using Display filter based on: {single_col}")
                    break

            # Final fallback: use full dataset
            if not individual_pass:
                print("Fallback: Using full dataset for Display.")
                display_filtered_df = df.copy()
                for col, val in display_filters.items():
                    display_filtered_df = display_filtered_df[display_filtered_df[col] == val]

        # Proceed to sentiment calculation
        if not display_filtered_df.empty:
            sentiment_summary = (
                display_filtered_df[display_filtered_df["Aspect"] == aspect]
                .groupby(["Price Band", "Sentiment_Score"])["Review_Count"]
                .sum()
                .unstack(fill_value=0)
            )

            if not sentiment_summary.empty:
                total_reviews = sentiment_summary.sum(axis=1).replace(0, np.nan)
                sentiment_percentage = (sentiment_summary.div(total_reviews, axis=0) * 100).reset_index()
                sentiment_percentage = sentiment_percentage.rename(columns={-1: "Negative (%)", 0: "Neutral (%)", 1: "Positive (%)"})

                if price_band_new in sentiment_percentage["Price Band"].values:
                    sentiment_percentage = sentiment_percentage[sentiment_percentage["Price Band"] == price_band_new]
                    sentiment_results[aspect] = {
                        "Aspect": aspect,
                        "Sentiment DataFrame": sentiment_percentage
                    }

    for column, value in filter_dict.items():
        aspect = map_aspects(column)

        # Skip if there is no valid aspect mapping
        if not aspect:
            print(f"Warning: No aspect mapping found for '{column}'. Skipping.")
            continue

        if column not in df.columns:
            print(f"Error: Column '{column}' not found in DataFrame. Skipping.")
            continue

        # Check if data is available in OEM filtered dataset
        filtered_df = df_filtered[df_filtered[column] == value]

        # Compute total review count at the Price Band level for OEM data
        oem_review_count = filtered_df[filtered_df["Price Band"] == price_band_new]["Review_Count"].sum()

        # If OEM data is empty or review count < 400, switch to full dataset
        if filtered_df.empty or oem_review_count < 400:
            print(f"Warning: OEM data for {column} has less than 400 reviews. Using full dataset.")
            filtered_df = df[df[column] == value]

        if filtered_df.empty:
            print(f"Warning: No data found for filter {column} = {value}. Skipping.")
            continue

        # Group and sum review counts ONLY for the mapped aspect
        sentiment_summary = (
            filtered_df[filtered_df["Aspect"] == aspect]  # Filter for only the mapped aspect
            .groupby(["Price Band", "Sentiment_Score"])["Review_Count"]
            .sum()
            .unstack(fill_value=0)
        )

        if sentiment_summary.empty:
            print(f"Warning: No sentiment data available for {aspect}. Skipping.")
            continue

        total_reviews = sentiment_summary.sum(axis=1)

        # Prevent division errors
        total_reviews = total_reviews.replace(0, np.nan)  # Avoid division by zero
        sentiment_percentage = (sentiment_summary.div(total_reviews, axis=0) * 100).reset_index()
        sentiment_percentage = sentiment_percentage.rename(columns={-1: "Negative (%)", 0: "Neutral (%)", 1: "Positive (%)"})

        # Ensure price band exists
        if price_band_new in sentiment_percentage["Price Band"].values:
            sentiment_percentage = sentiment_percentage[sentiment_percentage["Price Band"] == price_band_new]
        else:
            continue  

        # Store results only for the mapped aspect
        sentiment_results[aspect] = {
            "Aspect": aspect,
            "Sentiment DataFrame": sentiment_percentage
        }

    return sentiment_results
    
def calculate_sentiment_percentage_per_column_oem_chassis(df, filter_dict, selected_oem, selected_chassis):
    sentiment_results = {}
    filter_dict = filter_dict.copy()

    # Extract Price Band from filter_dict
    price_band = filter_dict.pop("Price Band", None)
    available_bands = df['Price Band'].unique()

    if not price_band:
        raise ValueError("Price Band must be included in the filter dictionary.")

    price_band_values = extract_price_band(price_band)

    if isinstance(price_band_values, (tuple, list)) and len(price_band_values) == 2:
        upper_band, lower_band = price_band_values
    else:
        raise ValueError(f"extract_price_band(price_band) did not return exactly two values: {price_band_values}")

    price_band_new = find_best_price_band(upper_band, lower_band, available_bands)

    # First, filter by selected OEM in the OEM column
    df_filtered = df[df["OEM"] == selected_oem]
    
    # Special case handling for 'Display' aspect
    # display_columns = ["Display Type", "Resolution"]
    # Handle display-related filters
    # display_filters = {}
    # if "Display Type" in filter_dict:
        # display_filters["Display Type"] = filter_dict.pop("Display Type")
    # if "Resolution" in filter_dict:
        # display_filters["Pixels"] = filter_dict.pop("Resolution")  # Map Resolution to Pixels column


    # if display_filters:
        # aspect = "Display"
        
        # Filter data
        # display_filtered_df = df_filtered.copy()
        # for col, val in display_filters.items():
            # display_filtered_df = display_filtered_df[display_filtered_df[col] == val]

        # oem_review_count = display_filtered_df[display_filtered_df["Price Band"] == price_band_new]["Review_Count"].sum()

        # if display_filtered_df.empty or oem_review_count < 400:
            # print("Warning: Chassis Display data insufficient. Using full dataset.")
            # display_filtered_df = df.copy()
            # for col, val in display_filters.items():
                # display_filtered_df = display_filtered_df[display_filtered_df[col] == val]

        # if not display_filtered_df.empty:
            # sentiment_summary = (
                # display_filtered_df[display_filtered_df["Aspect"] == aspect]
                # .groupby(["Price Band", "Sentiment_Score"])["Review_Count"]
                # .sum()
                # .unstack(fill_value=0)
            # )

            # if not sentiment_summary.empty:
                # total_reviews = sentiment_summary.sum(axis=1).replace(0, np.nan)
                # sentiment_percentage = (sentiment_summary.div(total_reviews, axis=0) * 100).reset_index()
                # sentiment_percentage = sentiment_percentage.rename(columns={-1: "Negative (%)", 0: "Neutral (%)", 1: "Positive (%)"})

                # if price_band_new in sentiment_percentage["Price Band"].values:
                    # sentiment_percentage = sentiment_percentage[sentiment_percentage["Price Band"] == price_band_new]
                    # sentiment_results[aspect] = {
                        # "Aspect": aspect,
                        # "Sentiment DataFrame": sentiment_percentage
                    # }
                    
    display_filters_raw = {}
    if "Display Type" in filter_dict:
        display_filters_raw["Display Type"] = filter_dict.pop("Display Type")
    if "Resolution" in filter_dict:
        display_filters_raw["Pixels"] = filter_dict.pop("Resolution")  # Map Resolution to Pixels column

    if display_filters_raw:
        aspect = "Display"

        def apply_filters(data, filters_to_use):
            temp_df = data.copy()
            for col, val in filters_to_use.items():
                temp_df = temp_df[temp_df[col] == val]
            return temp_df

        # Attempt 1: Both filters together
        df_combined = apply_filters(df_filtered, display_filters_raw)
        review_count_combined = df_combined[df_combined["Price Band"] == price_band_new]["Review_Count"].sum()
        print(f"Trying combined filters: {display_filters_raw} -> Review count: {review_count_combined}")

        # Attempt 2: Each individual filter
        individual_dfs = {}
        for col in display_filters_raw:
            temp_df = apply_filters(df_filtered, {col: display_filters_raw[col]})
            temp_review_count = temp_df[temp_df["Price Band"] == price_band_new]["Review_Count"].sum()
            print(df_filtered["Pixels"].unique())
            individual_dfs[col] = (temp_df, temp_review_count)
            print(f"Trying individual filter: {col} = {display_filters_raw[col]} -> Review count: {temp_review_count}")

        # Decision logic
        if review_count_combined >= 400:
            print("Using combined filters (Display Type and Resolution)")
            display_filtered_df = df_combined
        else:
            valid_options = {col: (df, count) for col, (df, count) in individual_dfs.items() if count >= 400}
            if valid_options:
                # Choose the one with highest review count
                best_col = max(valid_options, key=lambda x: valid_options[x][1])
                display_filtered_df = valid_options[best_col][0]
                print(f"Combined filters insufficient. Using individual filter: {best_col}")
            else:
                print("All individual display filters had insufficient reviews. Checking at Chassis level.")
                
                if selected_chassis:
                    chassis_df = df[
                        (df["Chassis Segment_New"] == selected_chassis) &
                        (df["Price Band"] == price_band_new)
                    ]
                    display_filtered_df = apply_filters(chassis_df, display_filters_raw)
                    chassis_review_count = display_filtered_df["Review_Count"].sum()

                    print(f"Chassis level filtered review count: {chassis_review_count}")
                    if chassis_review_count < 300:
                        print("Chassis Segment data is also insufficient. Using full dataset.")
                        display_filtered_df = apply_filters(df.copy(), display_filters_raw)
                    else:
                        print("Using Chassis Segment level data for Display.")
                else:
                    print("No Chassis Segment provided. Using full dataset.")
                    display_filtered_df = apply_filters(df.copy(), display_filters_raw)



        # Proceed with sentiment analysis
        if not display_filtered_df.empty:
            sentiment_summary = (
                display_filtered_df[display_filtered_df["Aspect"] == aspect]
                .groupby(["Price Band", "Sentiment_Score"])["Review_Count"]
                .sum()
                .unstack(fill_value=0)
            )

            if not sentiment_summary.empty:
                total_reviews = sentiment_summary.sum(axis=1).replace(0, np.nan)
                sentiment_percentage = (sentiment_summary.div(total_reviews, axis=0) * 100).reset_index()
                sentiment_percentage = sentiment_percentage.rename(columns={-1: "Negative (%)", 0: "Neutral (%)", 1: "Positive (%)"})

                if price_band_new in sentiment_percentage["Price Band"].values:
                    sentiment_percentage = sentiment_percentage[sentiment_percentage["Price Band"] == price_band_new]
                    sentiment_results[aspect] = {
                        "Aspect": aspect,
                        "Sentiment DataFrame": sentiment_percentage
                    }

    for column, value in filter_dict.items():
        aspect = map_aspects(column)

        if not aspect:
            print(f"Warning: No aspect mapping found for '{column}'. Skipping.")
            continue

        if column not in df.columns:
            print(f"Error: Column '{column}' not found in DataFrame. Skipping.")
            continue

        # Filter data for the selected column and value within the selected OEM
        filtered_df = df_filtered[df_filtered[column] == value]

        # Compute total review count at the Price Band level for OEM data
        oem_review_count = filtered_df[filtered_df["Price Band"] == price_band_new]["Review_Count"].sum()

        # If OEM data is insufficient, check within Chassis Segment_New + Price Band
        if filtered_df.empty or oem_review_count < 200:
            print(f"OEM data for {column} has less than 200 reviews. Checking within Chassis Segment.")
            
            if selected_chassis:
                filtered_df = df[
                    (df["Chassis Segment_New"] == selected_chassis) & 
                    (df["Price Band"] == price_band_new) & 
                    (df[column] == value)
                ]

                chassis_review_count = filtered_df["Review_Count"].sum()

                # If still insufficient, use full dataset
                if filtered_df.empty or chassis_review_count < 300:
                    print(f"Chassis Segment data is also insufficient. Using full dataset.")
                    filtered_df = df[df[column] == value]

        if filtered_df.empty:
            print(f"Warning: No data found for filter {column} = {value}. Skipping.")
            continue

        # Group and sum review counts ONLY for the mapped aspect
        sentiment_summary = (
            filtered_df[filtered_df["Aspect"] == aspect]
            .groupby(["Price Band", "Sentiment_Score"])["Review_Count"]
            .sum()
            .unstack(fill_value=0)
        )

        if sentiment_summary.empty:
            print(f"Warning: No sentiment data available for {aspect}. Skipping.")
            continue

        total_reviews = sentiment_summary.sum(axis=1)
        total_reviews = total_reviews.replace(0, np.nan)  # Avoid division by zero
        sentiment_percentage = (sentiment_summary.div(total_reviews, axis=0) * 100).reset_index()
        sentiment_percentage = sentiment_percentage.rename(columns={-1: "Negative (%)", 0: "Neutral (%)", 1: "Positive (%)"})

        if price_band_new in sentiment_percentage["Price Band"].values:
            sentiment_percentage = sentiment_percentage[sentiment_percentage["Price Band"] == price_band_new]
        else:
            continue  

        sentiment_results[aspect] = {
            "Aspect": aspect,
            "Sentiment DataFrame": sentiment_percentage
        }

    return sentiment_results

    
def calculate_target_percentages(sentiment_dfs):
    target_percentages_dict = {}

    for column, data in sentiment_dfs.items():
        aspect = data["Aspect"]
        sentiment_df = data["Sentiment DataFrame"]

        if not sentiment_df.empty:
            target_percentages_dict[aspect] = {
                "negative": sentiment_df["Negative (%)"].values[0],
                "neutral": sentiment_df["Neutral (%)"].values[0],
                "positive": sentiment_df["Positive (%)"].values[0]
            }

    return target_percentages_dict

def adjust_sentiment_distribution(df, aspect, target_percentages):
    # Select rows for the given aspect
    display_df = df[df["Aspect"] == aspect].copy()

    # If no matching rows, return original df
    if display_df.empty:
        return df

    total_reviews = len(display_df)

    # Compute the count for each sentiment
    num_positive = round((target_percentages["positive"] / 100) * total_reviews)
    num_neutral = round((target_percentages["neutral"] / 100) * total_reviews)
    
    # Ensure the sum equals total_reviews
    num_negative = total_reviews - (num_positive + num_neutral)

    # Adjust in case of rounding errors
    if num_negative < 0:
        num_negative = 0
        num_neutral = total_reviews - num_positive
    elif num_positive < 0:
        num_positive = 0
        num_negative = total_reviews - num_neutral
    elif num_neutral < 0:
        num_neutral = 0
        num_negative = total_reviews - num_positive

    # Create the new sentiment scores
    new_sentiments = np.concatenate([
        np.full(num_positive, 1),
        np.full(num_neutral, 0),
        np.full(num_negative, -1)
    ])

    # Shuffle to randomize sentiment distribution
    np.random.shuffle(new_sentiments)

    # Assign new sentiments
    display_df["Sentiment_Score"] = new_sentiments

    # Update the original dataframe
    df.update(display_df)

    return df


def apply_target_percentages(filtered_df, sentiment_dfs):
    target_percentages_dict = calculate_target_percentages(sentiment_dfs)

    for aspect, target_percentages in target_percentages_dict.items():
        filtered_df = adjust_sentiment_distribution(filtered_df, aspect, target_percentages)

    return filtered_df
    
def apply_target_percentages(sentiment_dfs, filtered_df):
    target_percentages_dict = {}

    for column, data in sentiment_dfs.items():
        aspect = data["Aspect"]
        sentiment_df = data["Sentiment DataFrame"]

        if not sentiment_df.empty:
            target_percentages_dict[aspect] = {
                "negative": sentiment_df["Negative (%)"].values[0],
                "neutral": sentiment_df["Neutral (%)"].values[0],
                "positive": sentiment_df["Positive (%)"].values[0]
            }

    # print(target_percentages_dict)

    for aspect, target_percentages in target_percentages_dict.items():
        filtered_df = adjust_sentiment_distribution(filtered_df, aspect, target_percentages)

    return filtered_df

def clean_aspects(raw_string):
    items = re.split(r'[-•]\s*', raw_string)
    cleaned = [item.strip(" \n'") for item in items if item.strip()]
    return cleaned
    
# def extract_aspects(raw_text):
    # if not raw_text:
        # return []
    # Split on -, •, ; or newlines
    # parts = re.split(r'[-•;\n]\s*', raw_text)
    # cleaned = [p.strip(" \n'") for p in parts if p.strip()]
    # return cleaned
    
    
def extract_aspects_section(text, label):
    pattern = rf'{label}\s*:?(.*?)(?=(?:\n[A-Z][a-z]+:|$))'  # Match until next label or end of string
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if not match:
        return []

    raw_text = match.group(1).strip()

    if raw_text.lower() == 'none':
        return ['None']

    lines = raw_text.splitlines()
    aspects = []

    for line in lines:
        # Remove bullets like -, •, *
        cleaned = re.sub(r'^[-•*]\s*', '', line).strip()
        # Split on semicolon or comma if present
        if ';' in cleaned or ',' in cleaned:
            aspects.extend([p.strip(" \n'") for p in re.split(r'[;,]', cleaned) if p.strip()])
        elif cleaned:
            aspects.append(cleaned)

    return aspects
