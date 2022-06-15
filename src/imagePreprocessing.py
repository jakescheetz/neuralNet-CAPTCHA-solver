"""
Author: 
    Jake Scheetz
Date: 
    June 2022
Description: 
    Program looks at a Really Simple CAPTCHA (rsCAPTCHA) and then splits each character
    and places it into a supvised classifier placeholder for training. Classification is
    performed once for each character split from the CAPTCHA. Inspiration for the program
    taken from the "Machine Learning Cookbook for Cyber Security" publication.

    --> This particular script is used to preprocess the extracted images/text from CAPTCHA's
        so that it can be passed to a neural network to be solved

    --> Each script's main logic is placed at the bottom of the code for clarity

Dependencies: 
    - opencv-python
    - imutils
    - numpy
"""

# imports
import os
import cv2
import numpy as np
# --------------

# global vars
captcha_images_folder = "captcha-images"

captchas = [
    os.path.join(captcha_images_folder, f) for f in os.listdir(captcha_images_folder)
]

captcha_processing_output_folder = "extracted-letter-images"

character_counts = {}
# --------------

# ~~~~~~~~~~~~~~~~~~ functions ~~~~~~~~~~~~~~~~~~
def preprocess(img):
    """Takes and image and converts it to its threshold (B/W)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayWithBorder = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    preprocessedImg = cv2.threshold(grayWithBorder, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return grayWithBorder, preprocessedImg

def getCAPTCHALabel(filePath):
    """Gets the text in the CAPTCHA from the file name"""
    filename = os.path.basename(filePath)
    label = filename.split(".")[0]
    return label

def findCharacterBounds(edges):
    """Gets the bounds (surrounding rectangles) of the individual characters
     of CAPTCHA text"""
    letterBoundingRectangles = []
    for edge in edges:
        (x, y, w, h) = cv2.boundingRect(edge)
        if w / h > 1.25:
            halfWidth = int(w / 2)
            letterBoundingRectangles.append((x, y, halfWidth, h))
            letterBoundingRectangles.append((x + halfWidth, y, halfWidth, h))
        else: 
            letterBoundingRectangles.append((x, y, w, h))
    return letterBoundingRectangles

def captchaToGrayscale(captchaImg):
    """Takes a CAPTCHA and converts it to grayscale along
    with the cropped letter bounds"""
    img = cv2.imread(captchaImg)
    gray, preprocessedImg = preprocess(img)
    edges = cv2.findContours(
        preprocessedImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    edges = edges[0]
    letterBoundingRectangles = findCharacterBounds(edges)
    letterBoundingRectangles = sorted(letterBoundingRectangles, key=lambda x: x[0])
    return gray, letterBoundingRectangles

def createCharImages(letterBoundingRect, grayscale):
    """produces an image of the character from the CAPTCHA"""
    x, y, w, h = letterBoundingRect
    letterImg = grayscale[y - 2 : y + h + 2, x - 2 : x + w + 2]
    return letterImg

def cropCaptcha(letterBoundingRect, grayImg, captchaText):
    """Performs the cropping of the CAPTCHA and saves it to output directory"""
    for letterBorders, currentLetter in zip(letterBoundingRect, captchaText):
        letterImage = createCharImages(letterBorders, grayImg)
        savePath = os.path.join(captcha_processing_output_folder, currentLetter)
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        
        charCount = character_counts.get(currentLetter, 1)
        p = os.path.join(savePath, str(charCount) + ".png")
        cv2.imwrite(p, letterImage)
        character_counts[currentLetter] = charCount + 1
# ~~~~~~~~~~~~~~~~~~ end functions ~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~ Main Logic ~~~~~~~~~~~~~~~~~~
for captchaImageFile in captchas:
    captchaText = getCAPTCHALabel(captchaImageFile)
    gray, letterBoundingRect = captchaToGrayscale(captchaImageFile)
    
    """
    this if block skips any character that has been incorrectly saved
    hopefully preventing any bad character extractions from
    making their way into the training set
    """
    if len(letterBoundingRect) != 4:
        continue

    cropCaptcha(letterBoundingRect, gray, captchaText)














