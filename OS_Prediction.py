import streamlit as st
import pandas as pd
import re
import os
import pandas as pd
from openai import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import warnings
warnings.filterwarnings("ignore")

df_os = pd.read_csv('Cleaned_OS_Prediction_Data.csv')

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment_name = 'Surface_Analytics'


start_phrase = """

You are an AI Assistant, your task is to extract relevant sub-aspects from below mentioned list of Aspect and Sub-Aspects and their corresponding sentiment (Positive or Negative) from the user's prompt. This information will be used to forecast user sentiment accurately.

Please carefully analyze the user's prompt to identify the appropriate sub-aspects from the predefined list provided below.
 

### Instructions:
1. Identify only Relevant Sub-Aspects: Based on the user's prompt, select the most appropriate sub-aspects from the predefined list.
2. Assign Sentiments: Classify each sub-aspect as either Positive or Negative. Make sure:
   - Sub-aspects are unique within each sentiment category.
3. IMPORTANT: ENSURE THAT NO SUB-ASPECT IS REPEATED IN BOTH THE POSITIVE AND NEGATIVE CATEGORIES UNDER ANY CIRCUMSTANCES.
4. Handle Irrelevant Prompts: If the prompt is unrelated to any predefined sub-aspects, mark both Positive and Negative categories as "None."
5. If the RAM increases and the price also increases accordingly, that is expected behavior. In such cases, sub-aspects related to price will not be highlighted, as the price change is justified. However, if the price increase is disproportionately high and does not align with the value of the upgrade, it should be flagged as a negative aspect of price.

*Note: Thoroughly examine each prompt for accurate extraction like for example if in user prompt he mentioned about any aspect which will affect the net sentiment less then you should mention those related keywords in Negative, as these sub-aspects are essential for reliable sentiment forecasting.*
IMPORTANT: If the prompt is not making sense return None for both Positive and Negative.
### Example Format:
User Prompt: "We have introduced an improved security patch cycle and enhanced the compatibility of the OS with older devices for the latest update. Could you generate hypothetical reviews?"

Expected Answer Format:
Positive: Security, Patch Quality, Virus Protection, Firewall, Update Frequency, Older Software Compatibility, Peripheral Support, Multi-device Sync, App Compatibility, Stability, Reliability, System Integrity, Support Duration, Privacy Controls, Bug Fixes.
Negative: Disk Space, Update Size

### Predefined List of Aspects and Sub-Aspects:
- Updates & Support: Update frequency, Bug fixes, Support duration, Patch quality, Installation ease, Feedback response, Update size, Device compatibility, Security patches, Update transparency.
- Compatibility: App compatibility, Peripheral support, Older software, File formats, Multi-device sync, Accessory support, Display resolutions, Language support, External drives, Mobile compatibility.
- Price: Affordability, Value for money, Transparent pricing, Renewal costs, Discounts, Add-on costs, Subscription model, Competitive pricing, Refund policy, Hidden fees.
- Connectivity: Wi-Fi stability, Bluetooth pairing, External displays, Hotspot support, Connection speed, Network drivers, VPN support, Seamless switching, Port compatibility, Ethernet stability
- Security: Virus protection, Firewall, Data encryption, Privacy controls, Biometric login, Security patches, Parental controls, Anti-phishing, Security alerts, Vulnerability protection
- Installation: Install ease, Install speed, Disk space, Install instructions, Hardware compatibility, Custom options, Reinstall process, Recovery options, Install support, User-friendly setup.
- User Interface (UI): Design intuitiveness, UI customization, Font clarity, Layout consistency, Navigation ease, Accessibility, Touchscreen optimization, Dark mode, Theme options, Multi-language UI
- Licensing: License cost, Terms clarity, Renewal ease, Multi-device use, License transfer, Student license, Regional license, Trial period, License management, Subscription options.
- Customization & Personalization: Theme options, Widget choices, Taskbar settings, Backgrounds, Shortcuts, Accessibility settings, Layout adjustments, Notifications, Icon visibility, Folder organization.
- System Resources: Memory usage, CPU load, Disk space, Power efficiency, Low-spec compatibility, GPU use, System load balance, Resource monitoring, Startup speed, Background processes.
- Performance: Boot speed, Responsiveness, Animation smoothness, Multitasking, Background apps, File transfer speed, Settings load time, Update speed, App launch speed, Stability.
- App Support: Productivity apps, Entertainment apps, Developer tools, App store, Native apps, Regular updates, Third-party apps, Load speed, System integration, Enterprise apps.
- Gaming: Graphics performance, Frame rate, VR support, Game mode, Network latency, Resource allocation, Controller support, Graphics drivers, Low latency, Anti-cheat tools.
- Virtualization & Cross-OS Compatibility: Dual-boot support, VM compatibility, Cross-platform apps, Remote desktop, Emulation, Linux support, Virtual memory, Network bridging, File sharing, SDK compatibility.
- Ecosystem: Device sync, Mobile OS integration, Wearable support, Cloud storage, Family sharing, App ecosystem, Home device integration, Ecosystem apps, Multi-device use, Cross-platform.
- Productivity: Productivity tools, App integration, Cloud storage, Task management, Focus mode, Document editing, Time-tracking, Screen sharing, Shortcuts, Workflow support.
- Privacy: Data tracking, Privacy controls, Data transparency, Privacy tools, Secure browsing, Ad blocking, Alerts, Encryption, Customizable settings, Privacy notifications.
- Battery: Power-saving, Battery monitoring, Fast charging, Battery health, Low-power apps, Life under load, Battery notifications, Dark mode efficiency, Battery diagnostics, Energy optimization.
- Voice & Gesture Control: Voice assistant, Voice accuracy, Gesture recognition, Command range, Custom commands, Accessibility, Device integration, Language support, Voice privacy, Response speed.
- Stability: Crash frequency, Error recovery, System uptime, Multitasking stability, Driver compatibility, Shutdown handling, System diagnostics, Restore reliability, Resilience, Update stability.
- Reliability: Consistent performance, Error handling, Support reliability, Troubleshooting, Recovery ease, System durability, Predictable updates, App consistency, System integrity, Dependability.

Here is the user prompt: 
"""


def sub_aspect_extraction_os(user_prompt):
    try:
        response = client.completions.create(
            model=deployment_name,
            prompt=start_phrase + user_prompt,
            max_tokens=2000,
            temperature=0
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error processing review: {e}")
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
        
# def assign_sentiment(row):
    # subaspect = row['Sub_Aspect']
    # current_sentiment = row.get('Sentiment_Score', 0)

    # if pd.isna(subaspect):
        # return current_sentiment

    # subaspect = subaspect.lower()

    # if any(pos.lower() in subaspect for pos in positive_aspects):
        # return 1
    # elif any(neg.lower() in subaspect for neg in negative_aspects):
        # return -1
    # else:
        # return current_sentiment

def get_conversational_chain_hypothetical_summary_os():
    global model
    global history
    try:
        prompt_template = """  
        
        As an AI assistant, your task is to provide a detailed summary based on the aspect-wise sentiment, focusing on the top contributing aspects in terms of review count. Ensure that the analysis explains the percentage of negative reviews for each aspect, highlighting the reasons behind them.
        ## Instructions:
        1.Summarize the user feedback for the top 5 aspects based on the highest review counts, excluding the Generic aspect.

        2.The summary should include both positive and negative insights, highlighting the improvements or ongoing issues that users have noted in each aspect.

        3.Ensure the summaries are clear, concise, and reflect specific user sentiments (positive or negative) around changes or enhancements in the OS.
        
        4.Do not include Generic aspect in the top 5, and do not mention any sentiment percentages.
        
        5.End with a brief conclusion summarizing the key themes around Windows OS improvements and areas that still require attention based on user reviews. 
        
        6.When delivering the output, be confident and avoid using future tense expressions like "could be" or "may be."
         
        Context:The DataFrame contains aspect-wise review data, including sentiment scores, review counts, and calculated negative review percentages. The summary should reflect these metrics, emphasizing the negative sentiment and its causes in aspects that contribute the most to the overall review count.
        Note:**Mention the negative review percentages for top 5 aspects based on highest review count with negative summary** except Generic aspect
 
        Context:\n {context}?\n
        Question: \n{question}\n
 
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
def hypothetical_summary_os(user_question, vector_store_path="OS_Indexes"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_hypothetical_summary_os()
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err
        
def process_aspect_sentiment(actual_sentiment_df, min_review_count=200):
    aspect_wise_net_sentiment_hypothetical_os = actual_sentiment_df.groupby('Aspect').apply(lambda group: pd.Series({
        'Net_Sentiment': calculate_net_sentiment(group),
        'Review_Count': group['Review_Count'].sum(),
        'Negative_Percentage': calculate_negative_review_percentage(group)
    })).reset_index()

    aspect_wise_net_sentiment_hypothetical_os = aspect_wise_net_sentiment_hypothetical_os.sort_values(by='Review_Count', ascending=False)
    aspect_wise_net_sentiment_hypothetical_os['Net_Sentiment'] = aspect_wise_net_sentiment_hypothetical_os['Net_Sentiment'].apply(lambda x: f"{x:.2f}%")
    aspect_wise_net_sentiment_hypothetical_os['Negative_Percentage'] = aspect_wise_net_sentiment_hypothetical_os['Negative_Percentage'].apply(lambda x: f"{x:.2f}%")

    aspect_wise_net_sentiment_hypothetical_os = aspect_wise_net_sentiment_hypothetical_os[aspect_wise_net_sentiment_hypothetical_os['Review_Count'] >= min_review_count]
    
    aspect_sentiment_forecast_os = aspect_wise_net_sentiment_hypothetical_os[['Aspect', 'Net_Sentiment']].reset_index(drop=True)

    return aspect_sentiment_forecast_os
