# import torch
# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# from PIL import Image
# from transformers import AutoModel, AutoTokenizer
# import io
# from pyngrok import ngrok  # Import ngrok from pyngrok
# import asyncio
# import uvicorn
# import nest_asyncio

# nest_asyncio.apply()  # This allows nested event loops

# app = FastAPI()

# # Load the model and tokenizer
# model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)

# question = '''Extract the text from the provided image and return only plain text content. 
#               Ensure that no additional formatting, metadata, or fields like title, subtitles, or table headers are included in the response. 
#               Provide only the actual text from the image without explaining about the image or text in the response. 
#               Do not autocorrect the text and do not insert extra characters to the words and do not apply contraction to the words. 
#               If the text is in a language other than English, translate it to English.  
#               Return the extracted text exactly as it appears, without any additional explanation.  
#               If there is no text in the image, simply return '0' but do not miss any word in the image.'''

# # Example images and answers for pre-trained data
# image1 = Image.open('train1.jpg').convert('RGB')
# answer1 = '''In many cases , their homes know little of 
#             their place of work and their associates at the office or works wonder what they they are possibly like in the surroundings of their homes . It is easy , desperately easy , to lead a 'double' life without ever deliberately 
#             planning to do so or in fact being conscious of what is going on . It is easier to live life in 
#             compartments but over the years it builds up, and to do inevitably builds up tensions which need to be handled correctly. 
#             Name:'''
# image2 = Image.open('train2.jpg').convert('RGB')
# answer2 = '''A MOVE to stop Mr. Gaitskell from nominating any more Labour life Peers is to be made at a meeting of Labour MPs tomorrow.  
#             Mr. Michael Foot has put down a resolution on the subject and he is to be backed by Mr. Will Griffiths, MP for Manchester Exchange.'''
# # image_test = Image.open('test.jpg').convert('RGB')

# # Message list for chat inference
# msgs = [
#     {'role': 'user', 'content': [image1, question]}, 
#     {'role': 'assistant', 'content': [answer1]},
#     {'role': 'user', 'content': [image2, question]}, 
#     {'role': 'assistant', 'content': [answer2]},
#     # {'role': 'user', 'content': [image_test, question]}
# ]

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     # Load the image from the uploaded file
#     image_test = Image.open(io.BytesIO(await file.read())).convert('RGB')

#     # Prepare messages for inference
#     msgs.append({'role': 'user', 'content': [image_test, question]})

#     # Perform inference
#     try:
#         answer = model.chat(
#             image=None,
#             msgs=msgs,
#             tokenizer=tokenizer
#         )
#         return JSONResponse(content={"response": answer})
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# async def main():
#     # Set your ngrok auth token
#     # ngrok.set_auth_token("2mb3PuoIvNGKvh7juZpHYR3DHeT_bA9tca7Hc75CcBHAdUW8")  # Replace with your auth token

#     # # Start Ngrok tunnel
#     # public_url = ngrok.connect(8000)
#     # print(f" * Ngrok tunnel available at: {public_url}")

#     # Run the FastAPI app with uvicorn
#     config = uvicorn.Config(app, host="0.0.0.0", port=8000)
#     server = uvicorn.Server(config)
#     await server.serve()

# if __name__ == "__main__":
#     asyncio.run(main())  # Use asyncio.run to start the main function

###############################################################################################



import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import io
from pyngrok import ngrok
import asyncio
import uvicorn
import nest_asyncio

nest_asyncio.apply()

app = FastAPI()

# Load the model and tokenizer
model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)

# Set the model to evaluation mode
model.eval()

# Store learning data
learning_data = []

# Predefined question and answers for learning
question = '''Extract the text from the provided image and return only plain text content. 
              Ensure that no additional formatting, metadata, or fields like title, subtitles, or table headers are included in the response. 
              Provide only the actual text from the image without explaining about the image or text in the response. 
              Do not autocorrect the text and do not insert extra characters to the words and do not apply contraction to the words. 
              If the text is in a language other than English, translate it to English.  
              Return the extracted text exactly as it appears, without any additional explanation.  
              If there is no text in the image, simply return '0' but do not miss any word in the image.'''
image1 = Image.open('train1.jpg').convert('RGB')
answer1 = '''In many cases , their homes know little of 
            their place of work and their associates at the office or works wonder what they they are possibly like in the surroundings of their homes . It is easy , desperately easy , to lead a 'double' life without ever deliberately 
            planning to do so or in fact being conscious of what is going on . It is easier to live life in 
            compartments but over the years it builds up, and to do inevitably builds up tensions which need to be handled correctly. 
            Name:'''
image2 = Image.open('train2.jpg').convert('RGB')
answer2 = '''A MOVE to stop Mr. Gaitskell from nominating any more Labour life Peers is to be made at a meeting of Labour MPs tomorrow.  
            Mr. Michael Foot has put down a resolution on the subject and he is to be backed by Mr. Will Griffiths, MP for Manchester Exchange.'''

# Append predefined learning messages to learning_data
learning_data.append({'role': 'user', 'content': [image1, question]})
learning_data.append({'role': 'assistant', 'content': [answer1]})
learning_data.append({'role': 'user', 'content': [image2, question]})
learning_data.append({'role': 'assistant', 'content': [answer2]})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load the image from the uploaded file
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')

    # Prepare messages including learning data
    msgs = learning_data + [{'role': 'user', 'content': [image, question]}]

    # Perform inference
    try:
        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            temperature=0.1
        )
        return JSONResponse(content={"response": res})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

async def main():
    # Set your ngrok auth token
    ngrok.set_auth_token("2mb3PuoIvNGKvh7juZpHYR3DHeT_bA9tca7Hc75CcBHAdUW8")  # Replace with your auth token

    # Start Ngrok tunnel
    public_url = ngrok.connect(8000)
    print(f" * Ngrok tunnel available at: {public_url}")

    # Run the FastAPI app with uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    # Run the main function within the event loop
    asyncio.run(main())

