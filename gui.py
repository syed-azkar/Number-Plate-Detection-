import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2
import pytesseract as tess

# 🔴 IMPORTANT: Set Tesseract path
tess.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------------- IMAGE PROCESSING FUNCTIONS ---------------- #

def clean2_plate(plate):
    gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        contour_area = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(contour_area)

        max_cnt = contours[max_index]
        max_area = contour_area[max_index]
        x, y, w, h = cv2.boundingRect(max_cnt)

        if not ratioCheck(max_area, w, h):
            return plate, None

        return thresh[y:y+h, x:x+w], [x, y, w, h]

    return plate, None


def ratioCheck(area, width, height):
    ratio = width / float(height)
    if ratio < 1:
        ratio = 1 / ratio

    if area < 1000 or area > 75000 or ratio < 3 or ratio > 6:
        return False
    return True


def isMaxWhite(plate):
    return np.mean(plate) >= 115


def ratio_and_rotation(rect):
    (x, y), (width, height), angle = rect

    if width > height:
        angle = -angle
    else:
        angle = 90 + angle

    if angle > 15 or width == 0 or height == 0:
        return False

    return ratioCheck(width * height, width, height)


# ---------------- GUI ---------------- #

top = tk.Tk()
top.geometry('900x700')
top.title('Number Plate Recognition')
top.configure(background='#CDCDCD')

label = tk.Label(top, background='#CDCDCD', font=('arial', 30, 'bold'))
label.place(x=500, y=220)

sign_image = tk.Label(top)
sign_image.place(x=70, y=200)

plate_image = tk.Label(top)


# ---------------- MAIN FUNCTION ---------------- #

def classify(file_path):
    img = cv2.imread(file_path)

    if img is None:
        label.config(text="Invalid Image")
        return

    img2 = cv2.GaussianBlur(img, (3, 3), 0)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img2 = cv2.Sobel(img2, cv2.CV_8U, 1, 0, ksize=3)
    _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, element)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    result_text = ""

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)

        if ratio_and_rotation(rect):
            x, y, w, h = cv2.boundingRect(cnt)
            plate = img[y:y+h, x:x+w]

            cv2.imwrite("result.png", plate)

            if isMaxWhite(plate):
                clean_plate, rect = clean2_plate(plate)

                if rect:
                    plate_img = Image.fromarray(clean_plate)
                    text = tess.image_to_string(plate_img, lang='eng')

                    if text.strip():
                        result_text = text.strip()
                        break

    if result_text == "":
        result_text = "No plate detected"

    label.config(text=result_text)

    try:
        uploaded = Image.open("result.png")
        im = ImageTk.PhotoImage(uploaded)

        plate_image.configure(image=im)
        plate_image.image = im
        plate_image.place(x=560, y=320)
    except:
        pass


# ---------------- BUTTONS ---------------- #

def show_classify_button(file_path):
    btn = tk.Button(top, text="Classify Image",
                    command=lambda: classify(file_path),
                    padx=10, pady=5,
                    bg='#364156', fg='white',
                    font=('arial', 15, 'bold'))
    btn.place(x=490, y=550)


def upload_image():
    file_path = filedialog.askopenfilename()

    if not file_path:
        return

    uploaded = Image.open(file_path)
    uploaded.thumbnail((350, 350))
    im = ImageTk.PhotoImage(uploaded)

    sign_image.configure(image=im)
    sign_image.image = im

    label.config(text="")
    show_classify_button(file_path)


upload_btn = tk.Button(top, text="Upload Image",
                       command=upload_image,
                       padx=10, pady=5,
                       bg='#364156', fg='white',
                       font=('arial', 15, 'bold'))

upload_btn.place(x=210, y=550)

top.mainloop()