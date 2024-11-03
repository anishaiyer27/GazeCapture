import cv2
import numpy as np
import scipy.io
import sys
#import caffe

"""
    LOAD PRETRAINED MODEL AND RUN INFERENCE ON 1 PRELOADED IMAGE
    

"""
global net


def preprocess_img(filename):
    if not filename:
        filename = "anisha.png"
    
    img = cv2.imread(filename)
    img_left = img
    img_right = img
    img_face = img

    img_resized = cv2.resize(img, (224, 224))
    img_left_resized = cv2.resize(img_left, (224, 224))
    img_right_resized = cv2.resize(img_right, (224, 224))
    img_face_resized = cv2.resize(img_face, (224, 224))

    mean = get_means()
    img_left_blob = cv2.dnn.blobFromImage(img_left_resized, 1.0, (224, 224), mean)
    img_right_blob = cv2.dnn.blobFromImage(img_right_resized, 1.0, (224, 224), mean)
    img_face_blob = cv2.dnn.blobFromImage(img_face_resized, 1.0, (224, 224), mean)

    return img_left_blob, img_right_blob, img_face_blob

def batchify(img_left_blob, img_right_blob, img_face_blob, batch_size=256):
    facegrid_data = get_facegrid_data()

    # TODO: adapt to collect batch of data with persistent memory (RT)
    # test example from existing static image, moved to inside directory
    #img_batch = np.repeat(img_blob[np.newaxis, ...], 256, axis=0)
    img_left_batch = np.repeat(img_left_blob, batch_size, axis=0)  # Shape (256, 3, 224, 224)
    img_right_batch = np.repeat(img_right_blob, batch_size, axis=0)  # Shape (256, 3, 224, 224)
    img_face_batch = np.repeat(img_face_blob, batch_size, axis=0)  # Shape (256, 3, 224, 224)
    facegrid_batch = np.repeat(facegrid_data, batch_size, axis=0)  # Shape (256, 625, 1, 1)

    return img_left_batch, img_right_batch, img_face_batch, facegrid_batch

def get_facegrid_data():
    # TODO: implement accurately
    #np.random.randn(256, 625, 1, 1).astype(np.float32)  # dummy data, adjust as needed
    # for one image
    return np.zeros((625, 1, 1)).astype(np.float32)

def get_means():
    # TODO: get means from mean_images
    return (104.0, 117.0, 123.0) # synthetic means, get mean values from mat files or binaryprot files


def mainAI():
    # for one image
    img_left_blob, img_right_blob, img_face_blob = preprocess_img("anisha.png")

    img_left_batch, img_right_batch, img_face_batch, facegrid_batch = batchify(img_left_blob, img_right_blob, img_face_blob)

    print(img_left_batch.shape)     # should be (256, 3, 224, 224)
    print(img_right_batch.shape)    # should be (256, 3, 224, 224)
    print(img_face_batch.shape)     # should be (256, 3, 224, 224)
    print(facegrid_batch.shape)     # should be (256, 625, 1, 1)

    # model inference
    proto_path="models/itracker_deploy.prototxt"
    caffe_path="models/snapshots/itracker25x_iter_92000.caffemodel"
    model = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=caffe_path)

    model.setInput(img_left_batch, "image_left")
    model.setInput(img_right_batch, "image_right")
    model.setInput(img_face_batch, "image_face")
    model.setInput(facegrid_batch, "facegrid")

    output = model.forward()
    print(output)
    print(output.shape)

def get_mean_image(file_name):
    image_mean = np.array(scipy.io.loadmat(model_root + file_name)['image_mean'])
    image_mean = image_mean.reshape(3, 224, 224)
    
    return image_mean.mean(1).mean(1)

# create transformer for the input called 'data'
def create_image_transformer(layer_name, mean_image=None):  
    transformer = caffe.io.Transformer({layer_name: net.blobs[layer_name].data.shape})

    transformer.set_transpose(layer_name, (2,0,1))  # move image channels to outermost dimension
    if mean_image is not None:
        transformer.set_mean(layer_name, mean_image)            # subtract the dataset-mean value in each channel
    return transformer

def crop_image(img, crop):
    return img[crop[1]:crop[1]+crop[3],crop[0]:crop[0]+crop[2],:] 

def test_face(img, face, face_feature):
    eyes, face_grid = face_feature

    if len(eyes) < 2:
        return None

    start_ms = current_time()
    transformed_right_eye = right_eye_transformer.preprocess('image_right', crop_image(img, eyes[0]))
    #print(eyes[0].shape, transformed_right_eye[0].shape)
    transformed_left_eye = left_eye_transformer.preprocess('image_left', crop_image(img, eyes[1]))
    transformed_face = face_transformer.preprocess('image_face', crop_image(img, face))
    transformed_face_grid = face_grid.reshape(1, 625, 1, 1)
    
    net.blobs['image_left'].data[...] = transformed_left_eye
    net.blobs['image_right'].data[...] = transformed_right_eye
    net.blobs['image_face'].data[...] = transformed_face
    net.blobs['facegrid'].data[...] = transformed_face_grid
    
    output = net.forward()
    net.forward()
    print("Feeding through the network took " + str((current_time() - start_ms) * 1. / 1000) + "s")
    
    return np.copy(output['fc3'][0])

def test_faces(img, faces, face_features):
    outputs = []
    for i, face in enumerate(faces):
        output = test_face(img, face, face_features[i])

        outputs.append(output)

    return outputs

def main():

    img_left_blob, img_right_blob, img_face_blob = preprocess_img("anisha.png")

    img_left_batch, img_right_batch, img_face_batch, facegrid_batch = batchify(img_left_blob, img_right_blob, img_face_blob)

    print(img_left_batch.shape)     # should be (256, 3, 224, 224)
    print(img_right_batch.shape)    # should be (256, 3, 224, 224)
    print(img_face_batch.shape)     # should be (256, 3, 224, 224)
    print(facegrid_batch.shape)     # should be (256, 625, 1, 1)

    # model inference
    proto_path="models/itracker_deploy.prototxt"
    caffe_path="models/snapshots/itracker25x_iter_92000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=caffe_path)

    net.setInput(img_left_batch, "image_left")
    net.setInput(img_right_batch, "image_right")
    net.setInput(img_face_batch, "image_face")
    net.setInput(facegrid_batch, "facegrid")

    output = net.forward()
    print(output)
    print(output.shape)


def caffe_main():
    caffe.set_mode_gpu()

    proto_path="models/itracker_deploy.prototxt"
    caffe_path="models/snapshots/itracker25x_iter_92000.caffemodel"
    net = caffe.Net(proto_path, caffe_path, caffe.TEST)

    def set_batch_size(batch_size):
        net.blobs['image_left'].reshape(batch_size, 3, 224, 224)
        net.blobs['image_right'].reshape(batch_size, 3, 224, 224)
        net.blobs['image_face'].reshape(batch_size, 3, 224, 224)
        net.blobs['facegrid'].reshape(batch_size, 625, 1, 1)   

    # set the batch size to 1
    set_batch_size(1)

    for layer_name, blob in net.blobs.items():
        print(layer_name + '\t' + str(blob.data.shape))
    
    mu_face = get_mean_image('mean_face_224.mat')
    mu_left_eye = get_mean_image('mean_left_224.mat')
    mu_right_eye = get_mean_image('mean_right_224.mat')

    print(mu_face)
    print(mu_left_eye)
    print(mu_right_eye)

    left_eye_transformer = create_image_transformer('image_left', mu_left_eye)
    right_eye_transformer = create_image_transformer('image_right', mu_right_eye)
    face_transformer = create_image_transformer('image_face', mu_face)

    # face grid transformer just passes through the data
    face_grid_transformer = caffe.io.Transformer({'facegrid': net.blobs['facegrid'].data.shape})

    outputs = test_faces(img, faces, face_features)
    print("The outputs:", outputs)

if __name__ == "__main__":
    main()
    print('DONE')