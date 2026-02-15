from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = Image.open(r"D:\sample\test2.png")
text = pytesseract.image_to_string(img)

print("Extracted Text:")
print(text)
