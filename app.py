import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
import uvicorn
import gc  
import re

def clear_cuda_cache():
    """
    Clears the CUDA cache to prevent memory leaks.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()  
        print("CUDA cache cleared")
        
def clean_text(text):
    # List of terms to be removed
    removable_terms = [
        '**Title:**', '**Body Text:**', 'Title:', 'Body Text:',  # Add more terms as needed
    ]
    
    # Iterate over each term and replace it with an empty string
    for term in removable_terms:
        # Regex pattern to match the exact term possibly surrounded by whitespace
        pattern = r'\s*' + re.escape(term) + r'\s*'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text




app = FastAPI()


torch.set_grad_enabled(False)

# Load model and tokenizer
model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)

# Move model to GPU if available
# if torch.cuda.is_available():
#     model = model.to('cuda')

model.eval() 


question = '''Extract the text from the provided image and return only plain text content. 
              Ensure that no additional formatting, metadata, or fields like title, subtitles, or table headers are included in the response. 
              Provide only the actual text from the image without explaining about the image or text in the response. 
              Do not autocorrect the text and do not insert extra characters to the words and do not apply contraction to the words. 
              Return the extracted text exactly as it appears, without any additional explanation. 
              If there is no text in the image, simply return '0' but do not miss any word in the image.
              '''
              
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
    Ensures consistent padding and size for all images.
    """
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    delta_width = target_size[0] - image.size[0]
    delta_height = target_size[1] - image.size[1]
    padding = (delta_width // 2, delta_height // 2, delta_width - delta_width // 2, delta_height - delta_height // 2)
    padded_image = ImageOps.expand(image, padding, fill=(255, 255, 255)) 
    
    return padded_image


@app.post("/OCR")
async def extract_text(image: UploadFile = File(...)):
    
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        clear_cuda_cache()

        # Load image from the uploaded file
        image_bytes = await image.read()
        img = Image.open(BytesIO(image_bytes)).convert('RGB')


        processed_img = preprocess_image(img, target_size=(1344, 1344))


        msgs.append({'role': 'user', 'content': [processed_img, question]})
        # msgs = [{'role': 'user', 'content': [processed_img, question]}]
        clear_cuda_cache()
        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            temperature=0.1
        )


        clear_cuda_cache()
        answer = clean_text(answer)

        return JSONResponse(content={"text": answer})

    except Exception as e:
        clear_cuda_cache()
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)