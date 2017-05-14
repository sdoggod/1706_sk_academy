name: inverse
class: center, middle, inverse
layout: true

---
class: titlepage, no-number

# Deep Neural Network using TensorFlow
## .gray.author[Seil Na]
### .gray.small[May 18, 2017]
### .x-small[https://naseil.github.io/dnn-tensorflow]
.sklogobg[ ![Sklogo](images/sk-logo.png) ]

---
layout: false

## About

* TensorFlow를 이용한 DNN 설계
* TensorFlow 상위모듈 `tf.contrib.layers` 를 이용한 DNN  설계
* Variable: saving, restoring, debugging tips
---

template: inverse

# Deep Neural Network using TensorFlow 

---

## 시작하기 전에...
데이터 준비
```python
mnist = input_data.read_data_sets( # data loading...)
```

그래프 그리기
```python
x = tf.placeholder(dtype=tf.float32, shape=[None, 784]
# ...
logits = tf.matmul() # ...
# ...
predictions = # ...
```

* 모델 부분만 빼면 `day1/train.py` 코드와 대부분 중복된다
* 모델 코드와 트레이닝 코드를 분리하면 각 컴포넌트를 수정하기 매우 편리해짐
* 코드를 `models.py` 와 `train.py` 로 분리해보자!
---
## Code structure

```bash
./
├── train.py
└── models.py
```

- `train.py` : 모델 코드를 제외하고 Loss 계산, Optimizer 정의 및 학습 코드를 포함한다
- `models.py` : class 형태의 모델 코드를 포함한다.


---

## DNN
Input 텐서들을 입력으로 받아, predictions 를 출력으로 하는 구조의 모델
```python
# models.py
class DNN(object):
* def create_model(self, model_inputs):
    # model architectures here!
    # ...

    return predictions
```

---
## DNN

필요한 파라미터 선언
```python
# models.py
def create_model(model_inputs):
  initializer = tf.random_normal
  w1 = tf.Variable(initializer(shape=[784, 128]))
  b1 = tf.Variable(initializer(shape=[128]))

  w2 = tf.Variable(initializer(shape=[128, 10]))
  b2 = tf.Variable(initializer(shape=[10]))
```

---

## DNN
그래프 그리기
```python
# models.py
h1 = tf.nn.relu(tf.matmul(model_inputs, w1) + b1) # 1st hidden layer
    
logits = tf.matmul(h1,  w2) + b2
predictions = tf.nn.softmax(logits)

return predictions
```

`models.py` 안에 있는 모델 class가 input tensor를 argument를 받아 그에 대한 output(`predictions`)를 리턴하도록 합니다

---

## DNN
Trainer - data reader, 모델 불러오기, train_op 정의, Session run 등

```python
# train.py
mnist = input_data.read_data_sets("./data", one_hot=True)
  
# define model input: image and ground-truth label
model_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 784])
labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])
```

---

## DNN
모델 불러오기 `getattr` 함수 사용
```python
# train.py
import models
models = getattr(models, "DNN", None)
predictions = models.create_model(model_inputs)
```

모델이 여러개인 경우에, 다음과 같이 `tf.flags` 모듈을 이용하여 argparse로 사용할 모델을 선택하면 편리합니다. (대신 코드의 일관성을 위해 반드시 .red[모든 모델의 입출력 포맷이 같아야 함])
```python
# train.py
import models
models = getattr(models, flags.model, None)
predictions = models.create_model(model_inputs)
```
`$ python train.py --model=DNN`

`$ python train.py --model=LogisticRegression`


---

## DNN

loss & train op 정의
```python
# train.py
# define cross entropy loss term
loss = tf.losses.softmax_cross_entropy(
    onehot_labels=labels,
    logits=predictions)

# train.py 안에서 정의되는 텐서들에 대하여 summary 생성
tf.summary.scalar("loss", loss)
merge_op = tf.summary.merge_all()

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train_op = optimizer.minimize(loss)
```
---

## DNN
Session으로 Training 실행
```python
with tf.Session() as sess:
  summary_writer_train = tf.summary.FileWriter("./logs/train", sess.graph)

  sess.run(tf.global_variables_initializer())
  for step in range(10000):
    batch_images, batch_labels = mnist.train.next_batch(100)
    feed = {model_inputs: batch_images, labels: batch_labels}
    _, loss_val = sess.run([train_op, loss], feed_dict=feed)
    print "step {} | loss {}".format(step, loss_val)

    if step % 10 == 0:
      summary_train = sess.run(merge_op, feed_dict=feed)
      summary_writer_train.add_summary(summary_train, step)
```
---

## DNN
### Hmm...
.center.img-50[![](images/dnn_results.png)]

---
## DNN
더 잘할 수 없을까?

모델의 구조를 이것저것 바꿔봅시다

- Hidden Layer 의 개수
- 각 Hidden Layer 의 차원(Dimension)
- Learning rate
- Optimizer 종류 (`tf.train.GradientDescentOptimizer`, `tf.train.AdamOptimizer`, ...)
- Batch size

등등... 

Do it!
---
## DNN
더 잘할 수 없을까?

모델의 구조를 이것저것 바꿔봅시다

- Hidden Layer 의 개수
- 각 Hidden Layer 의 차원(Dimension)
- Learning rate
- Optimizer 종류 (`tf.train.GradientDescentOptimizer`, `tf.train.AdamOptimizer`, ...)
- Batch size

등등... 

그런데, 코드에서 일일이 바꿔가면서 수정하려니 실험 결과 관리도 어렵고 귀찮다...

-> 이전 시간에 배운 `tf.flags` 모듈을 적극 활용해봅시다
---
## `tf.flags` 를 이용한 Hyperparameter tuning

다음과 같이 쉘에서 argument로 받을 요소들을 지정한 후
```python
# train.py
from tensorflow import flags
FLAGS = flags.FLAGS
flags.DEFINE_string("log_dir", "./logs/default", "default summary/checkpoint directory")
flags.DEFINE_float("learning_rate", 0.01, "base learning rate")
flags.DEFINE_string("model", "DNN", "model name")
flags.DEFINE_string("optimizer", "GradientDescentOptimizer", "kind of optimizer to use.")
flags.DEFINE_integer("batch_size", 1024, "default batch size.")
flags.DEFINE_integer("max_steps", 10000, "number of max iteration to train.")

```

---
## `tf.flags` 를 이용한 Hyperparameter tuning

코드에선 이렇게
```python
# train.py

model = getattr(models, FLAGS.model, None)()
...
optimizer = getattr(tf.train, FLAGS.optimizer, None)(
FLAGS.learning_rate)
...

for step in range(FLAGS.max_steps):
  batch_images, batch_labels = mnist.train.next_batch(FLAGS.batch_size)
  images_val, labels_val = mnist.validation.next_batch(FLAGS.batch_size)
```

쉘에서 인자들에 값을 동적으로 할당할 수 있습니다.

`$ python train.py --batch_size=128 --learning_rate=0.001 --log_dir=./logs/dnn-10-20-0.1 --model=DNN --optimizer=AdamOptimizer`

---
template: inverse
# Variable Saving & Restoring
---
## Variable Saving, Restoring
학습은 어찌저찌 잘 했는데...

우리의 목적은 모델 학습 그 자체가 아님!

학습한 모델을 이용하여 새로운 입력 X 에 대하여 그에 알맞은 출력을 내는 것이 원래 목표였습니다.

그렇다면, Training Phase에서 모델이 학습한 Parameter들의 값을 디스크에 저장해놓고, 나중에 불러올 수 있어야겠다.

[`tf.train.Saver`](https://www.tensorflow.org/api_docs/python/tf/train/Saver) 모듈을 통해서 이와 같은 기능을 수행할 수 있습니다.
---
## Variable name
시작하기 전에...
첫째 날 배웠던 Tensor name에 대해서 자세히 알아야 합니다.

모든 텐서는 선언하는 시점에 .red[이름이 자동으로 부여]되며, .red[중복되지 않습니다.]
```python
import tensorflow as tf
a = tf.Variable(tf.random_normal(shape=[10]))
b = tf.Variable(tf.zeros(shape=[10]))
print a.name
print b.name
```

Variable:0

Variable_1:0

---
## Variable name
`name` 을 통해서 이름을 명시적으로 지정할 수도 있지만, 같은 이름으로 지정된 경우 중복을 피하기위해 자동으로 인덱스가 붙습니다.

```python
c = tf.Variable(tf.ones(shape=[10]), name="my_variable")
d = tf.Variable(tf.zeros(shape=[1]), name="my_variable")

print c.name
print d.name
```

my_variable:0

my_variable_1:0

---
## Variable name
`name` 을 통해서 이름을 명시적으로 지정할 수도 있지만, 같은 이름으로 지정된 경우 중복을 피하기위해 자동으로 인덱스가 붙습니다.

```python
c = tf.Variable(tf.ones(shape=[10]), name="my_variable")
d = tf.Variable(tf.zeros(shape=[1]), name="my_variable")

print c.name
print d.name
```

한줄 요약: .red[모든 텐서에는 중복되지 않게 이름이 부여된다.]

---
## Variable Saving, Restoring
그럼 이제, `tf.train.Saver` 객체를 이용해 변수 저장을 해봅시다.
```python
import tensorflow as tf
*a = tf.Variable(tf.random_normal(shape=[10])) #a.name="Variable_0:0"
*b = tf.Variable(tf.random_normal(shape=[5])) # b.name="Variable_1:0"
saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # some training code...
  save_path = saver.save(sess, "./logs/model.ckpt")
```

변수 a와 b를 선언합니다. 이름을 따로 지정해주지 않았으므로 `Variable_0:0` 과 같이 자동으로 지정됩니다.

---
## Variable Saving, Restoring
그럼 이제, `tf.train.Saver` 객체를 이용해 변수 저장을 해봅시다.
```python
import tensorflow as tf
a = tf.Variable(tf.random_normal(shape=[10])) #a.name="Variable_0:0"
b = tf.Variable(tf.random_normal(shape=[5])) # b.name="Variable_1:0"
*saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # some training code...
  save_path = saver.save(sess, "./logs/model.ckpt")
```

Saver 객체를 생성합니다. Saver 객체 안에 아무런 파라미터가 없다면, 기본값으로 Saver 객체는 `{key="Variable name", value=Variable Tensor}` 쌍의 dictionary를 내부적으로 가지게 됩니다.

즉, 이 경우에 Saver 객체가 가지고 있는 dictionary는 `{"Variable_0:0":a, "Variable_1:0":b}` 가 됩니다.
---

## Variable Saving, Restoring
그럼 이제, `tf.train.Saver` 객체를 이용해 변수 저장을 해봅시다.
```python
import tensorflow as tf
a = tf.Variable(tf.random_normal(shape=[10])) #a.name="Variable_0:0"
b = tf.Variable(tf.random_normal(shape=[5])) # b.name="Variable_1:0"
saver = tf.train.Saver()
with tf.Session() as sess:
* sess.run(tf.global_variables_initializer())
  # some training code...
  save_path = saver.save(sess, "./logs/model.ckpt")
```

initializer 를 실행시키면 Variable `a` `b`에 값이 할당됩니다.
---

## Variable Saving, Restoring
그럼 이제, `tf.train.Saver` 객체를 이용해 변수 저장을 해봅시다.
```python
import tensorflow as tf
a = tf.Variable(tf.random_normal(shape=[10])) #a.name="Variable_0:0"
b = tf.Variable(tf.random_normal(shape=[5])) # b.name="Variable_1:0"
saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # some training code...
* save_path = saver.save(sess, "./logs/model.ckpt")
```

현재 Saver 객체가 가지고 있는 dictionary 정보를 디스크의 `"./logs/model.ckpt"` 이름으로 저장(save)합니다. 저장된 파일을 .red[checkpoint] 라고 부릅니다.

---

## Variable Saving, Restoring
그럼 이제, `tf.train.Saver` 객체를 이용해 변수 저장을 해봅시다.
```python
import tensorflow as tf
a = tf.Variable(tf.random_normal(shape=[10])) #a.name="Variable_0:0"
b = tf.Variable(tf.random_normal(shape=[5])) # b.name="Variable_1:0"
saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # some training code...
* save_path = saver.save(sess, "./logs/model.ckpt")
```

다음과 같이 저장되어 있는 것을 확인할 수 있습니다.
```bash
./
├── train.py
└── logs
    ├──checkpoint
    ├──model.ckpt.data-00000-of-00001
    ├──model.ckpt.index
    └──model.ckpt.meta 
```

---

## Variable Saving, Restoring
그럼 이제, `tf.train.Saver` 객체를 이용해 변수 저장을 해봅시다.
```python
import tensorflow as tf
a = tf.Variable(tf.random_normal(shape=[10])) #a.name="Variable_0:0"
b = tf.Variable(tf.random_normal(shape=[5])) # b.name="Variable_1:0"
saver = tf.train.Saver()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # some training code...
* save_path = saver.save(sess, "./logs/model.ckpt", global_step=1000)
```

`global_step` 인자를 통해서 현재 트레이닝 i번째 스텝의 파라미터 값을 가지고 있는 체크포인트임을 명시할 수 있습니다.
```bash
./
├── train.py
└── logs
    ├──checkpoint
    ├──model.ckpt-1000.data-00000-of-00001
    ├──model.ckpt-1000.index
    └──model.ckpt-1000.meta 
```

---
## Variable Saving, Restoring
checkpoint를 저장했으니, 저장한 checkpoint를 불러와 기록되어있는 파라미터 값으로 변수 값을 채워봅시다.

```python
import tensorflow as tf
*a = tf.Variable(tf.random_normal(shape=[10])) #a.name="Variable_0:0"
*b = tf.Variable(tf.random_normal(shape=[5])) # b.name="Variable_1:0"
*saver = tf.train.Saver()
with tf.Session() as sess:
  # some training code...
  saver.restore(sess, "./logs/model.ckpt-1000")
  # sess.run(tf.global_variables_initializer())
```

변수 `a, b`를 생성하고 Saver 객체를 생성합니다.

Saver 객체가 인자 없이 선언되었으니, 생성된 모든 변수들에 대한 dictionary를 가지고 있습니다: `{"Variable_0:0":a, "Variable_1:0":b}`

---

## Variable Saving, Restoring
checkpoint를 저장했으니, 저장한 checkpoint를 불러와 기록되어있는 파라미터 값으로 변수 값을 채워봅시다.

```python
import tensorflow as tf
a = tf.Variable(tf.random_normal(shape=[10])) #a.name="Variable_0:0"
b = tf.Variable(tf.random_normal(shape=[5])) # b.name="Variable_1:0"
saver = tf.train.Saver()
with tf.Session() as sess:
  # some training code...
* saver.restore(sess, "./logs/model.ckpt-1000")
  # sess.run(tf.global_variables_initializer())
```

checkpoint 파일의 이름을 인자로 넣어 저장된 파라미터 값을 불러옵니다.

이 시점에서, saver 객체가 가지고 있는 dictionary 의 key값을 checkpoint파일에서 찾고, 매칭되는 checkpoint 파일의 key값이 존재한다면, 해당 value 텐서의 값을 saver 객체가 가지고 있는 dictionary의 value 에 할당합니다. 

---

## Variable Saving, Restoring
checkpoint를 저장했으니, 저장한 checkpoint를 불러와 기록되어있는 파라미터 값으로 변수 값을 채워봅시다.

```python
import tensorflow as tf
a = tf.Variable(tf.random_normal(shape=[10])) #a.name="Variable_0:0"
b = tf.Variable(tf.random_normal(shape=[5])) # b.name="Variable_1:0"
saver = tf.train.Saver()
with tf.Session() as sess:
  # some training code...
  saver.restore(sess, "./logs/model.ckpt-1000")
* # sess.run(tf.global_variables_initializer())
```

variable initializer를 restoring 이후에 run 하지 않는다는 사실에 주의해야 합니다.

만약 restoring 이후에 initializer run을 하게 되면, 불러온 파라미터 값이 전부 지워지고 원래 변수의 initializer로 초기화됩니다.
---
## Quiz.
1. MNIST에 모델을 트레이닝하고, checkpoint파일을 저장합니다.

2. `eval.py` 파일을 만들고, 모델을 로딩한 후 저장한 checkpoint 파일을 restore합니다.

3. 전체 Validation data에 대해서 불러온 파라미터 값을 가지는 모델을 Fully Evaluation하는(전체 Validation data 대한 Accuracy) 코드를 작성해 봅시다.

Tip. Validation data는 5000개 Image/Label pair이고, `batch_size=100` 으로 50 iteration을 돌려서 Accuracy를 평균내면 됩니다.

---
template: inverse
# DNN using tf.contrib.layers

---
## DNN using tf.contrib.layers
[`tf.contrib.layers.fully_connected`](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected) 모듈을 이용하여 layer수준에서 모델을 디자인할 수 있습니다.
```python
# models.py
class contrib_DNN(object):
  def create_model(self, model_inputs):
    h1 = layers.fully_connected(
      inputs=model_inputs,
      num_outputs=60,
      activation_fn=tf.nn.relu)

    h2 = layers.fully_connected(
      inputs=h1,
      num_outputs=30,
      activation_fn=tf.nn.relu)

    logits = layers.fully_connected(
      inputs=h2,
      num_outputs=10)

    predictions = tf.nn.softmax(logits)

    return predictions

```

---
## DNN using tf.contrib.layers
.center.img-90[![](images/contrib.png)]

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
