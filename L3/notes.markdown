Lesson 1

  
Content and fast.ai commands
 

untar_data()
1. untar_data(url:str, fname:Union[pathlib.Path, str]=None, dest:Union[pathlib.Path, str]=None, data=True, force_download=False)
2. returns pathlib.Path object
3. Download data from the url to the file fname if it doesn't exist, and decompresses it to folder dest
4. Only url is compulsory. fname and dest automatically recognized from URL. 
.
get_image_files()
1. get_image_files(c:Union[pathlib.Path, str], check_ext:bool=True, recurse=False)
2. Return list of files in the provided path that are images.
3. 'check_ext' if true will check for valid image extensions .
pathlib - '/' operator
1. path_anno = path/'annotations'
2. ‘/‘ (slash) operator is python 3 operator which is part of the pathlib library.
3. This is an advantage of using Path class instead of String.
 
Extract lables from images
1. In the filenames, the lables are present, which can be extracted using regex re module of python.
 
ImageDataBunch
1. In fast.ai, everything we model with is going to be a DataBunch object.
2. Basically DataBunch object contains 2 or 3 datasets - it contains training data, validation data, and optionally test data.
3. For each of those, it contains data and labels.
4. All sits there in this one place, which is returned after using the function to get ImageDataBunch.
5. data.classes returns all the labels.
 
ImageDataBunch.from_name_re
1. This function returns the InageDataBunch object, when we need to extract the lables using regular expression.
2. data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224)
3. The data returned has to normalized. This makes all of our data of same size, which is a basic requirement in all machine learning models. This basically makes all of them of same mean and standard deviation.
 
ImageDataBunch.show_batch
1. This function is called upon the ImageDataBunch object, to show the data of the data bunch.
2. data.show_batch(rows=3, figsize=(7,6))
3. Combination of resizing and cropping is used.
 
Learner and convolutional neural network learner. (Training of Model)
1.Learner is a general concept of things which can learn, consisting of various subclasses, one of which is convolutional neural network learner.
2. learn = create_cnn(data, models.resnet34, metrics=error_rate)
3. This learn is a Learner object. The function requires only 2 necessary fields, data and architecture.
4. For now, we know that resnet is an architecture which works pretty well. It has resnet34 and resnet50 as its 2 types.
5. Metrics are used to print something. (We print error rate here)
6. learn.fit_one_cycle(4). This starts the training, and here we train for 4 ephocs (i.e. 4 cycles throughout our data)
.7. learn.save('stage-1'). this function saves the gradients of a particular model in same path as data.
 
Transfer Learning
Transfer learning is a concept, where we use an already existing trained model and fit it so as to predict our required results.
This significantly reduces training time.
 
Overfitting
The phenomenon where the model can't recognize a general image, but can predict only on the training dataset. We need to avoid overfitting, and thus we use Validation data set and print the metrices in Learner creation. The InageDataBunch creates the validation dataset, which is invisible to the model during training.
 
Interpretetion
interp = ClassificationInterpretation.from_learner(learn)
1. This in the Interpretation class object, which basically gives us the results of the model on the validation set.
2. plot_top_losses(n)
    This function gives us the n top losses in our model. By plotting the top losses, we are going to find out what were the things that      we were the most wrong on, or the most confident about what we got wrong.
3. interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
   This plots a matrix which shows us the label vs how many times it was predicted as a particular label.
4. interp.most_confused(min_val=2)
    This shows us the most unmatched labels.
 
learn.unfreeze()
When we train the model, it basically just adds some layers on top of an already trained model. We left the already existing layers untouched. This results in a fast but a less accurate model. Thus, by using unfreeze(), we tell the model to train all the layers, and then we run the training cycle again.
Note - Right now, the loss will increase, because we have trained all the layers at same speed, which is not logical as each layer consists of features of different complexity.
 
Learning rate and Fixing the increased loss
learn.lr_find()
learn.recorder.plot()
This code snippet will tell us what maximun learning rate we can use. The default value was very high for the model, and thus caused an increased loss.
So, we run the cycle by passing the learning rate into it. It takes a slice, and distributes that range of speed throughout the layers while training.
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
Tips, Tricks and General Notes
All the Dataset URLs defined under URLs class.
Use docs() instead of help(), if you need to checkout the documentation.
A good rule of thumb is after we unfreeze (i.e. train the whole thing), pass a max learning rate parameter, pass it a slice, make the second part of that slice about 10 times smaller than your first stage.
ImageDataBunch contains a lot of functions to vreate the data. Choose it as per the presence of your labels. (ie. extracted using regex, present n csv, etc.)
Create your own dataset - https://forums.fast.ai/t/tips-for-building-large-image-datasets/26688
 
QnA - Things unclear and possible answers
 
The image size in ImageDataBunch.from_name_re
We need to pass a parameter "size" in this function. It is a shortcoming of current deep learning technology which is that a GPU has to apply the exact same instruction to a whole bunch of things at the same time in order to be fast. If the images are different shapes and sizes, we can't do that. So we actually have to make all of the images the same shape and size.
 
What does it mean to normalize an image data set
An image consists of 3 channels, of red, blue and green colour. When the image data is normalized, it makes mean = 0 and standard deviation = 1 for all three channels. This removes the case of bright and dark patches, and thus helps the model.
It is always recommended to normalize the data.
 
Regex Used In the Lecture
This website allows debugging RegEx: http://regexr.com/

we can use a regular expression by importing regular expression ‘re’ pa*85ckage in python, to do this.
Regular expressions are a way to search a string in text using pattern matching methods.

pat = r'/([^/]+)_\d+.jpg$'
Let’s deconstruct this regex pattern, /([^/]+)_\d+.jpg$ by reading it backward:

Expression	Explanation
$	end of search
.jpg	last chars to be found in the search string, also right file format checking
\d	numerical digits, ‘+’ sign denotes can be one or more of them
_	should come before the start of digits
()	denotes a group of characters
[]	denotes another subgroup if characters in the previous group
^/	‘^’ denotes negation, so ‘+’ says all other chars except ‘/’
( [ ^/ ] + )	searches all characters except ‘/’
/	first ‘/’ in regex says, end of search
r	The string should be a raw string. Otherwise, \d would have to be written as \\d so that Python doesn’t interpret it to be a special character.
So, this regex pattern will give us a string

/Abyssinian_1.jpg
considering search string was

PosixPath('images/Abyssinian_1.jpg')
Further, by using the fact that the actual name of the breed is in the first parenthesized group of the regular expression, the actual breed name

Abyssinian
 
