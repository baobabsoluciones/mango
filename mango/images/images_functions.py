from imutils import face_utils
import dlib
import cvlib as cv
from math import sqrt
import cv2
import numpy as np


def add_rectangle(img: np.array, face: tuple, color: tuple = (0, 255, 0)):
    """
    The add_rectangle function draws a rectangle in a picture with the given coordinates

    :param :class:`np.array` img: pass the image to be processed
    :param tuple face: pass the coordinates of the rectangle, a tuple with the form (x, y, w, h)
    :param tuple color: set the color of the rectangle (RGB tuple)
    :return: The image with the rectangle plotted
    :doc-author: baobab-soluciones
    """
    img_c = img.copy()
    cv2.rectangle(img_c, (face[0], face[1]), (face[2], face[3]), color, 2)
    cv2.imshow("Output", img_c)
    cv2.waitKey(0)
    return img_c


def add_circles(img, shape, radius: int = 2, color: tuple = (0, 0, 255)):
    """
    The add_circles function adds circles to the image in the coordinates given by shape.
    The function takes three arguments: img, shape and color.
    The img argument is a numpy array representing an image, shape is a list of tuples containing
    the coordinates for each circle in the form (x, y), radius is the radius for the circles, by default 2 pixels,
    and color is a tuple representing BGR values.

    :param img: Pass the image to be processed
    :param shape: the coordinates for the circles
    :param int radius: the radius of the circles
    :param color: Change the color of the circles that are plotted in add_circles
    :return: The image with the eyes plotted
    :doc-author: baobab-soluciones
    """
    img_c = img.copy()
    for (x, y) in shape:
        cv2.circle(img_c, (x, y), radius, color, -1)
    cv2.imshow("Output", img_c)
    cv2.waitKey(0)
    return img_c


def detect_face_cv(img, thres=0.9, size=0.66):
    """
    The detect_face_cv function takes an image and returns a face if it finds one.
    It uses the OpenCV library to detect faces in the image, and then returns the
    bounding box of that face.

    :param img: pass the image to be processed
    :param thres: set the threshold for the confidence of a face detection
    :param size: reduce the size of the face detected
    :return: a face in a given image
    :doc-author: baobab-soluciones
    """
    faces, confidences = cv.detect_face(img)
    faces_ok = []
    confi_ok = []
    for index, face in enumerate(faces):
        if face[0] <= img.shape[1] * size and face[1] <= img.shape[0] * size:
            face[2] = min(face[2], img.shape[1])
            face[3] = min(face[3], img.shape[0])
            faces_ok.append(face)
            confi_ok.append(confidences[index])
    for i, d in enumerate(faces_ok):
        if (confi_ok[i] >= max(confi_ok)) and (round(confi_ok[i], 2) >= thres):
            return d
    return None


def detect_face_dlib(img, detector=None, thres=0):
    """
    The detect_face_dlib function takes as input an image and a face detector.
    It returns the coordinates of the bounding box for each detected face in the image, along with its confidence score.
    It uses the facial recognition library in dlib

    :param img: Pass the image to be processed
    :param detector: Load the default detector
    :param thres: Detect faces with a confidence higher than 0
    :return: the coordinates of the rectangle with the face
    :doc-author: baobab-soluciones
    """
    if detector is None:
        detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets, scores, idx = detector.run(gray, 1, thres)
    faces_ok = None
    confi_ok = thres

    for index, det in enumerate(dets):
        (x, y, w, h) = face_utils.rect_to_bb(det)
        # the size is enough?
        if x <= img.shape[1] * 0.66 and y <= img.shape[0] * 0.66:
            # keeps the best score
            if scores[index] > confi_ok:
                faces_ok = [x, y, min(x + w, img.shape[1]), min(y + h, img.shape[0])]
                confi_ok = scores[index]
    return faces_ok


def detect_eyes(img, face, predictor):
    """
    The detect_eyes function takes an image and a face box (x, y, w, h) as input
    and returns the coordinates of the two eyes.

    :param img: Get the image
    :param face: Get the coordinates of the face
    :param predictor: Detect the landmarks on a face
    :return: A numpy array with the coordinates of the 5 points that represent
    :doc-author: baobab-soluciones
    """
    if predictor is None:
        predictor = dlib.shape_predictor("./data/shape_predictor_5_face_landmarks.dat")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rect = dlib.rectangle(face[0], face[1], face[2], face[3])
    shape = predictor(gray, rect)
    output = face_utils.shape_to_np(shape)
    return output


def resize_img(img, size=(512, 512)):
    """
    The resize_img function modifies the size of an image while maintaining its aspect ratio.
    It takes as input a numpy array representing the image and returns a resized version of it.

    :param img: Pass the image to be resized
    :param size: Define the size of the image that will be used in the training
    :return: The image resized
    :doc-author: baobab-soluciones
    """
    # Keeps the proportion
    proportion = sqrt(img.shape[0] * img.shape[1] / (size[0] * size[1]))
    # shape returns (height, width)
    new_size = (round(img.shape[1] / proportion), round(img.shape[0] / proportion))
    image_rs = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return image_rs


def resize_inv_crop(img, box, size=(512, 512)):
    """
    The resize_inv_crop function crops an image to a given size.

    :param img: Pass the image to be cropped
    :param box: Crop the image
    :param size: Resize the image to 512x512 pixels
    :return: The cropped image
    :doc-author: baobab-soluciones
    """
    proportion = sqrt(img.shape[0] * img.shape[1] / (size[0] * size[1]))
    # shape returns (height, width)
    cropped_image = img[
        round(box[1] * proportion) : round(box[3] * proportion),
        round(box[0] * proportion) : round(box[2] * proportion),
    ]
    return cropped_image


def resize_img_max_size(img, size=(512, 512)):
    """
    The resize_img_max_size function takes an image and a tuple of size (width, height) as input.
    It returns the image with the larger side resized to match the width or height specified in the tuple.
    The other side is resized proportionally.

    :param img: Pass the image to be resized
    :param size: Specify the size of the image that we want to return
    :return: The image with the larger side resized to the size specified (by default 512), and the other side is adjusted proportionally
    :doc-author: baobab-soluciones
    """
    p1 = size[0] / img.shape[0]
    p2 = size[1] / img.shape[1]
    proportion = min(p1, p2)

    # shape returns (height, width)
    new_size = (round(img.shape[1] * proportion), round(img.shape[0] * proportion))
    image_rs = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return image_rs


def fix_size_img(img, size=(512, 512)):
    """
    The fix_size_img function takes an image as input with a side equal to the final size and returns a square
    black border around the image.
    If the original image is already square, it simply returns that same image.

    :param img: Get the image to be resized
    :param size: Resize the image to a 512x512 square
    :return: A black image of the same size as the input or None ifthe teh size is not correct.
    :doc-author: baobab-soluciones
    """
    if img.shape[1] != size[0] and img.shape[0] != size[1]:
        return None
    if len(img.shape) == 3:
        negro = np.zeros([size[0], size[1], 3], dtype=np.uint8)
    else:
        negro = np.zeros([size[0], size[1]], dtype=np.uint8)
    if img.shape[1] == size[0]:
        arriba = negro.shape[0] - img.shape[0]
        negro[arriba:, :] = img
    else:
        izq = round((negro.shape[1] - img.shape[1]) / 2)
        negro[:, izq : (izq + img.shape[1])] = img

    return negro


def get_circle(edges, gval):
    """
    The get_circle function finds the circle in an image.
    It uses a Hough transform to find circles in the image, and then filters out all circles that are inferior to a
    certain radius.
    The function returns None if no circle is found.

    :param edges: Image after edge detection
    :param gval: Determine the threshold for the gradient
    :return: A list of the coordinates of the center and radius of a circle
    :doc-author: baobab-soluciones
    """
    max_radius = round(max(edges.shape[0], edges.shape[1]) / 2)

    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, gval, 10, maxRadius=max_radius
    )
    if circles is None:
        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, gval + 10, 10, maxRadius=max_radius
        )
    if circles is None:
        return None
    c1 = [
        [i[0], i[1], i[2]]
        for i in circles[0]
        if i[2] >= max_radius * 0.4 and i[0] < edges.shape[1] and i[1] < edges.shape[0]
    ]
    if len(c1) == 0:
        return None
    c2 = np.array(c1)
    mc = c2.reshape(1, c2.shape[0], c2.shape[1]).mean(axis=1)[0]
    f_b = [
        max(mc[0] - 0.9 * mc[2], 0),
        max(mc[1] - mc[2], 0),
        min(mc[0] + 0.9 * mc[2], edges.shape[1]),
        min(mc[1] + mc[2], edges.shape[0]),
    ]
    f_b = [round(num) for num in f_b]

    return f_b


def flood_fill_alg(img):
    """
    The flood_fill_alg function takes an image and fills all the connected regions with white.
    The function is called by the flood_fill_alg(img) function which calls it twice, once for each half of the image.
    This is done to ensure that both sides of a coin are filled in case there is a break in one side.

    :param img: A gray image to apply the algorithm
    :return: A binary image with the shapes filled in
    :doc-author: baobab-soluciones
    """

    kernel = np.ones((2, 2), np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    im_th = gradient
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    im_floodfill_2 = im_th.copy()

    # Copy the thresholded image.
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    control = 0
    for i in range(1, round(w / 2)):
        if im_floodfill[5, i] == 0 and control != 1:
            cv2.floodFill(im_floodfill, mask, (i, 5), 255)
            control += 1

        if im_floodfill_2[5, w - i] == 0 and control != 2:
            cv2.floodFill(im_floodfill_2, mask, (w - i, 5), 255)
            control += 2

        if control == 3:
            break
    im_floodfill_fin = (255 * np.logical_or(im_floodfill, im_floodfill_2)).astype(
        np.uint8
    )
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill_fin)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out


def overlay_two_image_v2(image, overlay, prob=0.65, ignore_color=None):
    """
    The overlay_two_image_v2 function takes two images as input and overlays the second image onto the first.
    The overlay_two_image_v2 function has three parameters: image, overlay, and prob.
    The image parameter is a numpy array representing an RGB or grayscale image.
    The overlay parameter is a numpy array representing an RGB or grayscale image that will be overlaid on top of the
    first one (the &quot;base&quot; image).
    The prob parameter is a float between 0 and 1
    and represents how much of each pixel in the base should be replaced by its corresponding pixel in the overlay.

    :param image: Specify the image to be processed
    :param overlay: Overlay the image with the mask
    :param prob: Set the opacity of the overlay being applied to a pixel
    :param ignore_color: Ignore the  color in the overlay image
    :return: The new image
    :doc-author: baobab-soluciones
    """
    if ignore_color is None:
        ignore_color = [0, 0, 0]
    ignore_color = np.asarray(ignore_color)
    mask = (overlay == ignore_color).all(-1, keepdims=True)
    out = np.where(
        mask, image, (image * prob + overlay * (1 - prob)).astype(image.dtype)
    )
    return out


def write_text(url, text, load_func=cv2.imread):
    """
    The write_text function writes text to an image.
    The function takes three arguments:
        url - the path to the image file that will be edited.
        text - a list of strings, each string representing a line of text.
        The strings are written one below another on the bottom of the image file specified by url.
        Each string is also assigned a font and font size, as well as color and thickness (in pixels).
        The default values for these parameters are listed in parentheses after each argument description:

    :param url: Specify the path of the image that will be used for text insertion
    :param text: Specify the text to be written on the image
    :param load_func: Specify the function that is used to load an image
    :return: True if the image has been correctly generated
    :doc-author: baobab-soluciones
    """
    try:
        img = load_func(url)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.6
        font_thickness = 1
        y = img.shape[0] - 5
        x = 20
        for line in text:
            textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
            y = y - textsize[1] - 5

            cv2.putText(
                img,
                line,
                (x, y),
                font,
                font_size,
                (255, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        return cv2.imwrite(url, img)
    except Exception as e:
        print(e)
        return False
