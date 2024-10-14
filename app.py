import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import ImageOps
import uvicorn

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("CUDA cache cleared")
 
# Initialize FastAPI app
app = FastAPI()
 
torch.set_grad_enabled(False)
# Load model and tokenizer
model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)


# model = model.to(device='cuda')
model.eval()  # Set the model to evaluation mode
 
# Disable gradient computation for model parameters
for param in model.parameters():
    param.requires_grad = False
    
    
    
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
 
# Sample question
question = '''Extract the text from the provided image and return only plain text content. 
              Ensure that no additional formatting, metadata, or fields like title, subtitles, or table headers are included in the response. 
              Provide only the actual text from the image without explaining about the image or text in the response. 
              Do not autocorrect the text and do not insert extra characters to the words and do not apply contraction to the words.   
              Return the extracted text exactly as it appears, without any additional explanation.  
              If there is no text in the image, simply return '0' but do not miss any word in the image.'''
 
# Initialize msgs for context learning
msgs = [
    {'role': 'user', 'content': [Image.open('train1.jpeg').convert('RGB'), question]},
    {'role': 'assistant', 'content': '''Hi , How are you you ? /nI am fine fine , what /nabout you ? /nThis is a test image for /nOCR whcih is opticall /ncharacterrr recognition . /nIt looks cool to get the /ndigitalized and it is a /ngood thing that can be /ndone . /nNotes:'''},
    # {'role': 'user', 'content': [Image.open('train2.jpeg').convert('RGB'), question]},
    # {'role': 'assistant', 'content': '''Title : Donut OCR /nDonut (Document understanding /ntransformer) is one of the ways /nwe can exxtract into form /ndocs and we use them in /nvarious ways. /nIt is a newest method for /nprocesing & extracting information /nfrom documents. Unlike OCR engines, /nDonut utilizes an end-to-end /ntransformer model. /nIt comprises a vision encoder & /na text - decoder (BART) . /nHi, How you are doing ? /nIt is true ?'''},
    {'role': 'user', 'content': [Image.open('train3.jpg').convert('RGB'), question]},
    {'role': 'assistant', 'content': '''Date: /nsmrt resuable Noetbok /nImagine a notebook that evloves /nwith your thouhgts , a smart /nreusable noetbook that harms /nthe powder of technologi to /nrevolutonze your writing /nxperience. Thi s remarkalbe tool /ncaptures the /ncaptures the esense of your /ncreativity , technology. /ntechnology , effortlessely.'''},
    # {'role': 'user', 'content': [Image.open('train4.jpg').convert('RGB'), question]},
    # {'role': 'assistant', 'content': '''Munday , Fraday , Tusedai , /nwednsedae , satuday /nGood Mrning Pencel /nKatlon studio is gve fre. /ntral for one manth. /nI wil tkae live Today /nbecase I am nat Feling wel'''}

]
 
def preprocess_image(image: Image.Image, target_size=(1344, 1344)):
    """
    Preprocess the image by resizing and padding it to the target size (1344x1344).
    The aspect ratio of the original image is preserved and padding is added to make
    it exactly 1344x1344 without cropping.
    
    Args:
        image (PIL.Image.Image): The input image.
        target_size (tuple): The target size (width, height) for the model.
    
    Returns:
        PIL.Image.Image: The preprocessed image with padding.
    """
    # Resize the image while maintaining aspect ratio
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Calculate padding to match the target size
    delta_width = target_size[0] - image.size[0]
    delta_height = target_size[1] - image.size[1]
    padding = (delta_width // 2, delta_height // 2, delta_width - delta_width // 2, delta_height - delta_height // 2)
    
    # Add padding to the image (fill with white color)
    padded_image = ImageOps.expand(image, padding, fill=(255, 255, 255))  # Using white padding
    
    return padded_image

@app.post("/OCR")
async def extract_text(image: UploadFile = File(...)):
    # Check file type
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        clear_cuda_cache()

        # Load image from the uploaded file
        image_bytes = await image.read()
        img = Image.open(BytesIO(image_bytes)).convert('RGB')

        # Preprocess the image to the required size (1344x1344) with padding
        processed_img = preprocess_image(img, target_size=(1344, 1344))

        # Add the new image and question to the msgs for context learning
        msgs.append({'role': 'user', 'content': [processed_img, question]})

        # Get the model's response using the updated context
        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            temperature= 0.1
        )

        # Append the assistant's response to the context (msgs)
        msgs.append({'role': 'assistant', 'content': answer})
        clear_cuda_cache()

        # Return the result as JSON
        return JSONResponse(content={"text": answer})

    except Exception as e:
        clear_cuda_cache()
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")
