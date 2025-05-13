## Results
The table shows that the performance of our implementation and original paper. 

|  | All | Yes/No | Number | Other |
| ------ | ------ | ------ | ------ | ------ |
| Implement |  46.23% | 67.5% | 30.83% | 34.12% |
| Original Paper | 54.22% | 73.46% | 35.18% | 41.38% |

## Dataset
VQA v2.0 release
- Real 
	- 82,783 MS COCO training images, 40,504 MS COCO validation images and 81,434 MS COCO testing images 
	- 443,757 questions for training, 214,354 questions for validation and 447,793 questions for testing
	- 4,437,570 answers for training and 2,143,540 answers for validation (10 per question)

There is only one type of task
- Open-ended task
## Usage
```
#### 1. Download the VQA v2.0 daatset 百度网盘链接: https://pan.baidu.com/s/1nsv-IBv99YLlvyL-Ry6C-g?pwd=nczp 提取码: nczp 
#### 2. Preprocessing input data (images, questions, answers)
```
python preprocess/resize_images.py
python preprocess/make_vocab.py
python preprocess/preprocessing.py
```
#### 3. Train the model
```
python model/train.py 
```
#### 4. Test model and build the result json file
```
python model/test.py
``` 
#### 5. Get evaluation results for open-ended task
```
python vqaEvalDemo.py
```
