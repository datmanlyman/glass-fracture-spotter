import numpy as np
from PIL import Image
import cv2


def cropImage(image, x1, y1, x2, y2):
    cropped = image[int(y1):int(y2), int(x1):int(x2)]
    return cropped.copy()


def orderPoints(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def fourPointTransform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = orderPoints(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def extractEdges(image1, image2):
    '''Convert to (544, 544) for AI purpose'''
    dimensions = (544, 544)
    fittedImage = cv2.resize(image1, dimensions, interpolation=cv2.INTER_AREA)

    # Gaussian Blur to blur the images
    blurred = cv2.GaussianBlur(fittedImage, (7, 7), 0)

    # Canny edge to spot the cracks
    lowerBoundary = np.percentile(blurred, 25)
    upperBoundary = np.percentile(blurred, 75)
    edges = cv2.Canny(blurred, lowerBoundary, upperBoundary)

    # Convert grayscale to RGB
    potentialCrack = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    '''For masking purpose'''
    # Gaussian Blur to blur the images
    blurredMask = cv2.GaussianBlur(image2, (7, 7), 0)

    # Canny edge to spot the cracks
    lowerBoundaryMask = np.percentile(blurredMask, 25)
    upperBoundaryMask = np.percentile(blurredMask, 75)
    edgesMask = cv2.Canny(blurredMask, lowerBoundaryMask, upperBoundaryMask)

    # Convert grayscale to RGB
    potentialMask = cv2.cvtColor(edgesMask, cv2.COLOR_GRAY2RGB)

    return (potentialCrack, potentialMask)


def ensembleLearning(ensemble, image):
    # Make the predictions using the ensemble of models. Returns the prediction.
    image_batch = np.expand_dims(image, axis=0)
    predictions = [model.predict(image_batch) for model in ensemble]
    crackLabels = ['crack', 'no crack']
    predictions = np.array(predictions)
    probabilities = np.sum(predictions, axis=0)
    result = np.argmax(probabilities, axis=1)
    class_index = result[0]
    prediction = crackLabels[class_index]

    return prediction


def masking(mask, image):
    for i in range(len(image)):
        for j in range(len(image[0])):
            # From white in the mask
            if all(mask[i, j] == [255, 255, 255]):
                # To pink in the image
                image[i, j] = (255, 0, 255)

    return image


def detect(image, model, ensemble):
    # 1st AI to use
    results = model.predict(image)  # image should already be in array form

    # Create a numpy array for the boxes
    boxes = results[0].boxes.boxes.numpy()

    # If array of boxes is empty, return to telegram a message
    if len(boxes) == 0:
        return "No glass!"

    for box in boxes:
        # x1, y1, x2, y2, confidence, class of a box
        x1, y1, x2, y2, confidence, _ = box

        # Crop image
        cropped = cropImage(image, x1, y1, x2, y2)

        # Undo planar perspective of the cropped images
        adjusted = fourPointTransform(image, np.array(
            [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))

        # To extract the edges of the cropped images
        potentialCrack, potentialMask = extractEdges(adjusted, cropped)

        # To check with the ensemble whether there is a crack
        crack = ensembleLearning(ensemble, potentialCrack)

        # If there is a crack, use it as a mask
        if crack == "crack":
            # Change the colour of the actual image through going through the edges
            imageMasked = masking(potentialMask, cropped)

            # Paste it over the original image
            image1 = Image.fromarray(image)
            image2 = Image.fromarray(imageMasked)
            Image.Image.paste(image1, image2, (int(x1), int(y1)))

            # As image1 is a Pillow.Image type, need to convert it back to an array
            image = np.asarray(image1)

    # Returns it as an array form
    cv2.imwrite('results.jpg', image)
    return "Done!"


'''
def getData(input_path=''):
    images = []
    for img in os.listdir('./' + input_path):
        if img.endswith('.png'):
            png = Image.open(input_path + img).convert('RGB')
            jpg = img[:-4] + '.jpg'
            png.save(input_path + jpg)
            images.append(jpg)
            os.remove('./' + input_path + img)
        elif os.path.isfile('/Users/tchenjw/Desktop/FYP/ultralytics/' + input_path + img) and img.endswith(('.jpeg', '.jpg', '.webp')):
            images.append(img)
    return images


def changeSize():
    input_path = 'cracks/'
    output_path = 'processed cracks/'
    dimensions = (544, 544)
    for image in os.listdir('./' + input_path):
        imgWithoutExtension = "".join(image.split('.')[:-1])
        img = cv2.imread(input_path + image)
        try:
            fitted_img = cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)
        except:
            continue
        fitted_img_type = Image.fromarray(fitted_img)
        fitted_img_type.save(output_path + imgWithoutExtension + '.jpg')


def splitImages(image):
    image1 = image[0:272, 0:544]
    image1 = cv2.resize(image1, (544, 544), interpolation=cv2.INTER_AREA)
    image2 = image[272:544, 0:544]
    image2 = cv2.resize(image2, (544, 544), interpolation=cv2.INTER_AREA)
    image3 = image[0:544, 0:272]
    image3 = cv2.resize(image3, (544, 544), interpolation=cv2.INTER_AREA)
    image4 = image[0:544, 272:544]
    image4 = cv2.resize(image4, (544, 544), interpolation=cv2.INTER_AREA)
    image5 = image[135:408, 135:408]
    image5 = cv2.resize(image5, (544, 544), interpolation=cv2.INTER_AREA)

    return [image1, image2, image3, image4, image5]
'''


# For pre-processing windows / no windows
'''
ultralytics = 'ultralytics/'
input_path = 'dataset/'
output_path = 'cracks/'
images = getData(input_path=input_path)

for image in images:
    imageType = Image.open(input_path + image)
    imgArr = np.asarray(imageType)
    try:
        results = model.predict(imgArr)
    except:
        continue
    img = cv2.imread(input_path + image)
    imgWithoutExtension = "".join(image.split(".")[:-1])
    boxes = (results[0].boxes.boxes).numpy()
    if len(boxes) == 0:
        continue
    i = 0
    for box in boxes:
        x1, y1, x2, y2, confidence, _ = box
        croppedImage = img[int(y1):int(y2), int(x1):int(x2)]
        cv2.imwrite(output_path + imgWithoutExtension + str(i) + '.jpg', croppedImage)
        i += 1

changeSize()
'''

# For pre-processing crack / no crack
'''
input_path = 'processed windows/'
output_path = 'processed cracks/'
crack = 'crack/'
no_crack = 'no crack/'


for img in os.listdir('./' + input_path + crack):
    image = cv2.imread(input_path + crack + img)
    images = splitImages(image)
    imgWithoutExtension = "".join(img.split("."))
    for idx in range(len(images)):
        newImage = Image.fromarray(images[idx])
        newImage.save(input_path + crack + imgWithoutExtension + str(idx + 1) + ".jpg")


for img in os.listdir('./' + input_path + crack):
    if not img.endswith(('.jpg')):
        continue
    image = cv2.imread(input_path + crack + img, 0)
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    lowerBoundary = np.percentile(blurred, 25)
    upperBoundary = np.percentile(blurred, 75)
    edges = cv2.Canny(blurred, lowerBoundary, upperBoundary)
    newImage = Image.fromarray(edges)
    newImage.save(output_path + crack + img)


for img in os.listdir('./' + input_path + no_crack):
    if not img.endswith(('.jpg')):
        continue
    image = cv2.imread(input_path + no_crack + img, 0)
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    lowerBoundary = np.percentile(blurred, 25)
    upperBoundary = np.percentile(blurred, 75)
    edges = cv2.Canny(blurred, lowerBoundary, upperBoundary)
    newImage = Image.fromarray(edges)
    newImage.save(output_path + no_crack + img)
'''
