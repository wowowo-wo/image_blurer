import cv2

def padding_image(image,padding):

    padded_image = cv2.copyMakeBorder(
        image, padding, padding, padding, padding, 
        borderType=cv2.BORDER_REFLECT_101
    )
    return padded_image