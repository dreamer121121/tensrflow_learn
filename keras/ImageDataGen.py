from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    './data/train',
    batch_size=32,
    save_to_dir='./data/save/')

