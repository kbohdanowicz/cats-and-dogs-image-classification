import cv2


def image_to_vector(image, size=(200, 200)):
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    img_rgb_cubic = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    return img_rgb_cubic


def normalize(image):
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    return image


def image_to_pixels(image, size=(200, 200)):
    pixels = image_to_vector(image, size=size)
    pixels = normalize(pixels)
    return pixels.flatten()


def image_to_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist)
    return hist.flatten()


