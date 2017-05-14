name: inverse
class: center, middle, inverse
layout: true

---
class: titlepage, no-number

# TensorFlow Basic
## .gray.author[Seil Na]
### .gray.small[May 19, 2017]
### .x-small[https://naseil.github.io/tensorflow-basic]
.sklogobg[ ![Sklogo](images/sk-logo.png) ]

---
layout: false

## About

- TensorFlow Basic - Op, Graph, Session, Feed 등
- Rogistic Regression using TensorFlow

---

template: inverse

# TensorFlow Basic

---
## Install configuration
```python
import tensorflow as tf
a = tf.constant("Hello!")
with tf.Session() as sess:
  print sess.run(a)
```

---

## Tensor

데이터 저장의 기본 단위
```python
import tensorflow as tf
a = tf.constant(1.0, dtype=tf.float32) # 1.0 의 값을 갖는 1차원 Tensor 생성
b = tf.constant(1.0, shape=[3,4]) # 1.0 의 값을 갖는 3x4 2차원 Tensor 생성
c = tf.constant(1.0, shape=[3,4,5]) # 1.0 의 값을 갖는 3x4x5 3차원 Tensor 생성
d = tf.random_normal(shape=[3,4,5]) # Gaussian Distribution 에서 3x4x5 Tensor를 Sampling

print c
```

`<tf.Tensor 'Const_24:0' shape=(3, 4, 5) dtype=float32>`

---

## TensorFlow Basic
TensorFlow Programming의 개념
1. `tf.Placeholder` 또는 Input Tensor 를 정의하여 Input **Node**를 구성한다
2. Input Node에서 부터 Output Node까지 이어지는 관계를 정의하여 **Graph**를 그린다
3. **Session**을 이용하여 Input Node(`tf.Placeholder`)에 값을 주입(feeding) 하고, **Graph**를 **Run** 시킨다

---

## TensorFlow Basic
1과 2를 더하여 3을 출력하는 프로그램을 작성

**Tensor**들로 **Input Node**를 구성한다
```python
import tensorflow as tf
a = tf.constant(1) # 1의 값을 갖는 Tensor a 생성
b = tf.constant(2) # 2의 값을 갖는 Tensor b 생성
```

---

## TensorFlow Basic
1과 2를 더하여 3을 출력하는 프로그램을 작성

Output Node까지 이어지는 **Graph**를 그린다

```python
import tensorflow as tf
a = tf.constant(1) # 1의 값을 갖는 Tensor a 생성
b = tf.constant(2) # 2의 값을 갖는 Tensor b 생성

c = tf.add(a,b) # a + b의 값을 갖는 Tensor c 생성
```

---

## TensorFlow Basic
1과 2를 더하여 3을 출력하는 프로그램을 작성

**Session**을 이용하여 **Graph**를 **Run** 시킨다

```python
import tensorflow as tf
a = tf.constant(1) # 1의 값을 갖는 Tensor a 생성
b = tf.constant(2) # 2의 값을 갖는 Tensor b 생성

c = tf.add(a,b) # a + b의 값을 갖는 Tensor c 생성

sess = tf.Session() # Session 생성

# Session을 이용하여 구하고자 하는 Tensor c를 run
print sess.run(c) # 3
```

Tip. native operation op `+,-,*,/` 는 TensorFlow Op 처럼 사용가능
```python
c = tf.add(a,b) <-> c = a + b
c = tf.subtract(a,b) <-> c = a - b
c = tf.mul(a,b) <-> c = a * b
c = tf.div(a,b) <-> c = a / b
```


---

## Exploring in Tensor: Tensor name
```python
import tensorflow as tf
a = tf.constant(1) 
b = tf.constant(2) 
c = tf.add(a,b) 
sess = tf.Session() 

print a, b, c, sess
print sess.run(c) # 3
```


Tensor(".blue[Const:0]", shape=(), dtype=int32 , Tensor(.blue["Const_1:0"], shape=(), dtype=int32), Tensor(.blue["Add:0"], shape=(), dtype=int32) <tensorflow.python.client.session.Session object at 0x7f77ca9dfe50>

3

모든 텐서는 op **name**으로 구분 및 접근되어서, 이후 원하는 텐서를 가져오는 경우나, 저장(Save)/복원(Restore) 또는 재사용(reuse) 할 때에도 name으로 접근하기 때문에 텐서 name 다루는 것에 익숙해지는 것이 좋습니다


---

## Placeholder: Session runtime에 동적으로 Tensor의 값을 주입하기
Placeholder: 선언 당시에 값은 비어있고, 형태(shape)와 타입(dtype)만 정의되어 있어 Session runtime에 지정한 값으로 텐서를 채울 수 있음

Feed: Placeholder에 원하는 값을 주입하는 것
```python
a = tf.placeholder(dtype=tf.float32, shape=[1]) # 1차원 실수형 Placeholder 생성
b = tf.placeholder(dtype=tf.float32, shape=[1]) # 1차원 실수형 Placeholder 생성
c = a + b
with tf.Session() as sess:
  feed = {a:1, b:2} # python dictionary
  print sess.run(c, feed_dict=feed) # 3
  
  feed = {a:2, b:4.5}
  print sess.run(c, feed_dict=feed) # 6.5
```

---

## Variable: 학습하고자 하는 모델의 Parameter
Parameter `W, b` 를 `1.0` 으로 **초기화** 한 후 linear model의 출력 구하기
```python
W = tf.Variable(1.0, dtype=tf.float32)
b = tf.Variable(1.0, dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32, shape=[1])

linear_model_output = W * x + b

# Important!!
*init_op = tf.global_variables_initializer()
with tf.Session() as sess:
* sess.run(init_op)
  
  feed = {x:5.0}
  sess.run(linear_model_output, feed_dict=feed) # 6
```

만약 `sess.run` 하는 op의 그래프에 변수(`tf.Variable`)이 하나라도 포함되어 있다면, 반드시 해당 변수를 초기화 `tf.global_variables_initializer()` 를 먼저 실행해야 합니다

---

## Variable: 학습하고자 하는 모델의 Parameter
Parameter `W, b` 를 **랜덤** 으로 **초기화** 한 후 linear model의 출력 구하기
```python
W = tf.Variable(tf.random_normal(shape=[1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[1]), dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32, shape=[1])

linear_model_output = W * x + b

# Important!!
*init_op = tf.global_variables_initializer()
with tf.Session() as sess:
* sess.run(init_op)
  
  feed = {x:5.0}
  sess.run(linear_model_output, feed_dict=feed) # 6
```

---
template: inverse
# MNIST using Logistic Regression
IPython notebook(https://www.naver.com)

Code(https://www.naver.com)

---

## Example. MNIST Using Logistic Regression
1.  모델의 입력 및 출력 정의
2.  모델 구성하기(Logistic Regression model)
3.  Training

---

##  모델의 입력 및 출력 정의
Input: 28*28 이미지 = 784차원 벡터 
`model_input = [0, 255, 214, ...]` 

각각에 해당하는 정답 `labels = [0.0, 1.0, 0.0, 0.0, ...]`

Output: 이미지가 각 클래스에 속할 확률 예측값을 나타내는 10차원 벡터 `predictions = [0.12, 0.311, ...]`

하고싶은 것은?

 **모델의 예측값이 정답 데이터(Label 또는 Ground-truth)와 최대한 비슷해지도록 모델 Parameter를 학습시키고 싶다** 
<-> `label` 과 `predictions` 의 **오차를 최소화 하고 싶다** 

---
## 모델의 입력 및 출력 정의
데이터 준비
```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data", one_hot=True)
```

```python
for _ in range(10000):
  batch_images, batch_labels = mnist.train.next_batch(100)
  batch_images_val, batch_labels_val = mnist.val.next_batch(100)
  batch_image.shape # [100, 784]
  batch_labels.shape # [100, 10]
```

---

## 모델 구성하기
모델의 입력을 Placeholder로 구성

Batch 단위로 학습할 것이기 때문에 `None`을 이용하여 임의의 batch size를 핸들링할 수 있도록 합니다

```python
# defien model input: image and ground-truth label
model_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 784])
labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])
```

Logistic Regression Model의 Parameter 정의
```python
# define parameters for Logistic Regression model
w = tf.Variable(tf.random_normal(shape=[784, 10]))
b = tf.Variable(tf.random_normal(shape=[10]))
```

---

## 모델 구성하기
그래프 그리기
```python
logits = tf.matmul(model_inputs, w) + b
predictions = tf.nn.softmax(logits)

# define cross entropy loss term
loss = tf.losses.softmax_cross_entropy(
         onehot_labels=labels,
         logits=predictions)
```
---

## 모델 구성하기
Optimizer 정의 -> 모델이 .red[loss](predictions 와 labels 사이의 차이)를 .red[최소화] 하는 방향으로 파라미터 업데이트를 했으면 좋겠다 
```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)
```

---
## Training

Session을 이용하여 

`Variable`들을 초기화시켜준 후에

각 iteration마다 이미지와 라벨 데이터를 batch단위로 가져오고

가져온 데이터를 이용하여 feed를 구성

`train_op`(가져온 데이터에 대한 loss를 최소화 하도록 파라미터 업데이트를 하는 Op)을 실행
```python
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for step in range(10000):
    batch_images, batch_labels = mnist.train.next_batch(100)
    feed = {model_inputs: batch_images, labels: batch_labels}
    _, loss_val = sess.run([train_op, loss], feed_dict=feed)
    print "step {}| loss : {}".format(step, loss_val)
```

---

## Result

.center.img-66[![](images/train_result.png)]

---
## Minor Tips - `tensorflow.flags`

TensorFlow에서 FLAGS를 통한 argparsing 기능도 제공하고 있습니다. HyperParamter(batch size, learning rate, max_step 등) 세팅에 유용!

```python
from tensorflow import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 128, "number of batch size. default 128.")
flags.DEFINE_float("learning_rate", 0.01, "initial learning rate.")
flags.DEFINE_integer("max_steps", 10000, "max steps to train.")
```


```pyhton
# train.py
batch_size = FLAGS.batch_size
learning_rate = FLAGS.learning_rate
max_step = FLAGS.max_steps
```

`$ python train.py --batch_size=256 --learning_rate=0.001 --max_steps=100000`

---

## Minor Tips - Tensorboard
학습 진행 상황을 Visualize 하고 싶다면? -> Tensoroboard 사용

.center.img-100[![](images/tenb_examples.png)]

---
## Minor Tips - Tensorboard

###scalar summary와 histogram summary

loss, learning rate 등 scalar 값을 가지는 텐서들은 scalar summary로,
parameter 등 n차원 텐서들은 histogram summary로 선언한다
```python
tf.summary.scalar("loss", loss)
tf.summary.histogram("W", w)
tf.summary.histogram("b", b)
```
merge_all() 로 summary 모으기
```python
merge_op = tf.summary.merge_all()
```
---
## Minor Tips - Tensorboard

이 후, `tf.summary.FileWriter` 객체를 선언하고, Session으로 `merge_op`을 실행하여 Summary를 얻고, `FileWriter`에 추가
```python
summary_writer = tf.summary.FileWriter("./logs", sess.graph)
for step in range(10000):
  # some training code...
  sess.run(train_op, feed=...)
  if step % 10 == 0:
    # session으로 merge_op을 실행시켜 summary를 얻고
    summary = sess.run(merge_op, feed_dict=feed)
    
    # summary_writer 에 얻은 summary값을 추가
    summary_writer.add_summary(summary, step)
```
---
## Minor Tips - Tensorboard

`$ tensorboard --logdir="./logs" --port=9000` 입력 

& `localhost:9000` 접속

### scalar summary
.center.img-100[![](images/scalar_summary.png)]
---
## Minor Tips - Tensorboard

### histogram summary
.center.img-100[![](images/histogram_summary.png)]
---
## Minor Tips - Tensorboard

summary 폴더 여러 개를 두고 서로 다른 실험 결과를 실시간으로 비교할 수도 있습니다
(여러 실험 결과값을 비교해볼 때 편리)
.center.img-100[![](images/summary_duplicate.png)]

---
## Quiz 1.
[`tf.argmax`](https://www.tensorflow.org/api_docs/python/tf/argmax) [`tf.equal`](https://www.tensorflow.org/api_docs/python/tf/equal) [`tf.cast`](https://www.tensorflow.org/api_docs/python/tf/cast) [`tf.reduce_mean`](https://www.tensorflow.org/api_docs/python/tf/reduce_mean) 을 사용하여 Accuracy Tensor를 정의하고, 이를 Tensorboard에 나타내기
.center.img-66[![](images/quiz1.png)]


---

## Quiz 2.
모델을 트레이닝 할 때, `tf.summary.FileWriter` 를 train, validation 용으로 각각 1개씩 만들어서 Tensorboard로 Training/Validation performance 를 함께 모니터링 할 수 있도록 해보기
.center.img-66[![](images/quiz2.png)]

---

name: last-page
class: center, middle, no-number
## Thank You!


<div style="position:absolute; left:0; bottom:20px; padding: 25px;">
  <p class="left" style="margin:0; font-size: 13pt;">
  <b>Special Thanks to</b>: Jongwook Choi, Byungchang Kim</p>
</div>

.footnote[Slideshow created using [remark](http://github.com/gnab/remark).]




<!-- vim: set ft=markdown: -->
