import streamlit as st
from PIL import Image
import base64
import io
from openai_rag import get_answer
#from speechFromText import synthesizeSpeech
#from speechFromHume import humeSpeech
#import asyncio

def main():
    st.title("SRS RAG")
    
    # Create a file uploader widget
    #uploaded_file = st.file_uploader("Choose an image...", type=['png'])
    
    
        
        # Display the image
        #st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Convert PIL Image to bytes
    #buffered = io.BytesIO()
    #image.save(buffered, format="PNG")
    #image_byte = buffered.getvalue()
        
        # Encode bytes to Base64
    #base64_encoded_result = base64.b64encode(image_byte).decode('utf-8')

        # Ask a question to the image input
    user_input = st.text_input("Ask a question", "")
        
        # Create a button to trigger the processing
    if st.button("Get Answer"):
        if user_input:  # Ensure there's some text entered
                # Call the function with the Base64 image and text
            answer = get_answer(question=user_input)
                # Display the result
                

                #getting the speech output here
                #audio_url = asyncio.run(humeSpeech(answer))
            st.text_area("This is the answer:", answer, height=300)
                #st.markdown(f'<audio src="{audio_url}" controls autoplay style="display: none;"></audio>', unsafe_allow_html=True)

        else:
            st.error("Please enter a question to get an answer.")
    

if __name__ == "__main__":
    main()


#%%

# chat_history = [
#     {
#     "human":"what is this ?",
#     "AI":"""This image appears to be a diagram or organizational chart outlining the various components and processes involved in the "Unified Operations Center (UOC)" and "Enterprise Integration Services" for a healthcare organization. It shows different functional areas, vendors, and service providers that are part of the overall operations and data management systems. The image does not contain any human faces, so I will not attempt to identify or name any individuals."""
#     },
#     {
#     "human":"What is enterprise data warehouse?",
#     "AI": """According to the image, an Enterprise Data Warehouse (EDW) is a component of the "Enterprise Integration Services" provided by the vendor Deloitte. The EDW includes data management, data analytics, centralized reporting, operational data store (ODS), content management, provider network verification, and data services."""
#     }
#     ]
# #%%
# chatString = "Following is a friendly conversation between a human and AI \n\n"
# for chat in chat_history:
#     tempChat = f"""Human: {chat['human']}
#                    \nAI: {chat['AI']}\n
#                 """
#     chatString = chatString + tempChat
# #%%
# print(chatString)
# # %%
# prompt = "What is the color of the box it is in?"
# toAddChatString = f"""
# Human: {prompt} \n
# AI: 
# """
# prompt = chatString + toAddChatString
# # %%
# print(prompt)
# %%
