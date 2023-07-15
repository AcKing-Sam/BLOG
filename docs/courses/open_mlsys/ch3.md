Application Programming Interface(API)，核心的诉求是：**如何平衡框架性能和易用性？**

为了达到最优的性能，开发者需要利用硬件亲和的编程语言如：C和C++来进行开发。这是因为C和C++可以帮助机器学习框架高效地调用硬件底层API，从而最大限度发挥硬件性能。同时，现代操作系统（如Linux和Windows）提供丰富的基于C和C++的API接口（如文件系统、网络编程、多线程管理等），通过直接调用操作系统API，可以降低框架运行的开销。

从易用性的角度分析，机器学习框架的使用者往往具有丰富的行业背景。他们常用的编程语言是高层次脚本语言：Python、Matlab、R和Julia。相比于C和C++，这些语言在提供编程易用性的同时，丧失了C和C++对底层硬件和操作系统进行深度优化的能力。因此，机器学习框架的核心设计目标是：具有易用的编程接口来支持用户使用高层次语言，如Python实现机器学习算法；同时也要具备以C和C++为核心的低层次编程接口来帮助框架开发者用C和C++实现大量高性能组件，从而在硬件上高效执行。

## evolution of programming models for mlsys

![image-20230715165125538](https://blog-img-acking.oss-cn-beijing.aliyuncs.com/img/202307151651569.png)

在2015年底，谷歌率先推出了TensorFlow。相比于传统的Torch，TensorFlow提出前后端分离相对独立的设计，利用高层次编程语言Python作为面向用户的主要前端语言，而利用C和C++实现高性能后端。大量基于Python的前端API确保了TensorFlow可以被大量的数据科学家和机器学习科学家接受，同时帮助TensorFlow能够快速融入Python为主导的大数据生态（大量的大数据开发库如Numpy、Pandas、SciPy、Matplotlib和PySpark）。同时，Python具有出色的和C/C++语言的互操作性，这种互操作性已经在多个Python库中得到验证。因此，TensorFlow兼有Python的灵活性和生态，同时也通过C/C++后端得以实现高性能。这种设计在日后崛起的PyTorch、MindSpore和PaddlePaddle等机器学习框架得到传承。

随着各国大型企业开源机器学习框架的出现，为了更高效地开发机器学习应用，基于开源机器学习框架为后端的高层次库Keras和TensorLayerX应运而生，它们提供Python API 可以快速导入已有的模型，这些高层次API进一步屏蔽了机器学习框架的实现细节，因此Keras和TensorLayerX可以运行在不同的机器学习框架之上。

随着深度神经网络的进一步发展，对于机器学习框架编程接口的挑战也日益增长。因此在2020年前后，新型的机器学习框架如MindSpore和JAX进一步出现。其中，MindSpore在继承了TensorFlow、PyTorch的Python和C/C++的混合接口的基础上，进一步拓展了机器学习编程模型从而可以高效支持多种AI后端芯片（如华为Ascend、英伟达GPU和ARM芯片），实现了机器学习应用在海量异构设备上的快速部署。

同时，超大型数据集和超大型深度神经网络崛起让分布式执行成为了机器学习编程框架的核心设计需求。为了实现分布式执行，TensorFlow和PyTorch的使用者需要花费大量代码来将数据集和神经网络分配到分布式节点上，而大量的AI开发人员并不具有分布式编程的能力。因此MindSpore进一步完善了机器学习框架的分布式编程模型的能力，从而让单节点的MindSpore程序可以无缝地运行在海量节点上。

在本小节中，我们将以MindSpore作为例子讲解一个现代机器学习框架的Python前端API和C/C++后端API的设计原则。这些设计原则和PyTorch，TensorFlow相似。

## work flow for mlsys

机器学习系统编程模型的首要设计目标是：对开发者的整个工作流进行完整的编程支持。一个常见的机器学习任务一般包含如 [图3.2.1](https://openmlsys.github.io/chapter_programming_interface/ml_workflow.html#img-workflow)所示的工作流。这个工作流完成了训练数据集的读取，模型的训练，测试和调试。通过归纳，我们可以将这一工作流中用户所需要自定义的部分通过定义以下API来支持（我们这里假设用户的高层次API以Python函数的形式提供）：

- **数据处理：** 首先，用户需要数据处理API来支持将数据集从磁盘读入。进一步，用户需要对读取的数据进行预处理，从而可以将数据输入后续的机器学习模型中。
- **模型定义：** 完成数据的预处理后，用户需要模型定义API来定义机器学习模型。这些模型带有模型参数，可以对给定的数据进行推理。
- **优化器定义：** 模型的输出需要和用户的标记进行对比，这个对比差异一般通过损失函数（Loss function）来进行评估。因此，优化器定义API允许用户定义自己的损失函数，并且根据损失来引入（Import）和定义各种优化算法（Optimisation algorithms）来计算梯度（Gradient），完成对模型参数的更新。
- **训练：** 给定一个数据集，模型，损失函数和优化器，用户需要训练API来定义一个循环（Loop）从而将数据集中的数据按照小批量（mini-batch）的方式读取出来，反复计算梯度来更新模型。这个反复的过程称为训练。
- **测试和调试：** 训练过程中，用户需要测试API来对当前模型的精度进行评估。当精度达到目标后，训练结束。这一过程中，用户往往需要调试API来完成对模型的性能和正确性进行验证。

![image-20230715165513430](https://blog-img-acking.oss-cn-beijing.aliyuncs.com/img/202307151655452.png)

> 此处代码验证看不懂，回头再看

## define DNN

随着深度神经网络的飞速发展，各种深度神经网络结构层出不穷，但是不管结构如何复杂，神经网络层数量如何增加，构建深度神经网络结构始终遵循最基本的元素：

- 承载计算的节点
- 可变化的节点权重（节点权重可训练）
- 允许数据流动的节点连接

因此**在机器学习编程库中深度神经网络是以层为核心**，它提供了各类深度神经网络层基本组件；将神经网络层组件按照网络结构进行堆叠、连接就能构造出神经网络模型。

### layer

#### full join

**全连接**是将当前层每个节点都和上一层节点一一连接，本质上是特征空间的线性变换；可以将数据从高维映射到低维，也能从低维映射到高维度。对输入的n个数据变换到大小为m的特征空间，再从大小为m的特征空间变换到大小为p的特征空间；

可见全连接层的参数量巨大，两次变换所需的参数大小为 $m \times n$ 和 $n\times p$ 。

![image-20230715170843474](https://blog-img-acking.oss-cn-beijing.aliyuncs.com/img/202307151708517.png)

#### convolution

**卷积**操作是卷积神经网络中常用的操作之一，卷积相当于对输入进行滑动滤波。根据卷积核（Kernel）、卷积步长（Stride）、填充（Padding）对输入数据从左到右，从上到下进行滑动，每一次滑动操作是矩阵的乘加运算得到的加权值。 如 [图3.3.2](https://openmlsys.github.io/chapter_programming_interface/neural_network_layer.html#conv-comp)卷积操作主要由输入、卷积核、输出组成输出又被称为特征图（Feature Map）。

![image-20230715171051551](https://blog-img-acking.oss-cn-beijing.aliyuncs.com/img/202307151710582.png)

在卷积过程中，如果我们需要对输出矩阵大小进行控制，那么就需要对步长和填充进行设置。还是上面的输入图，如需要得到和输入矩阵大小一样的输出矩阵，步长为1时就需要对上下左右均填充一圈全为0的数。

在上述例子中我们介绍了一个输入一个卷积核的卷积操作。通常情况下我们输入的是彩色图片，有三个输入，这三个输入称为通道（Channel），分别代表红、绿、蓝（RGB）。此时我们执行卷积则为**多通道卷积**，需要三个卷积核，分别对RGB三个通道进行上述卷积过程，之后将结果加起来。

具体如 [图3.3.4](https://openmlsys.github.io/chapter_programming_interface/neural_network_layer.html#channels-conv)描述了一个输入通道为3，输出通道为1，卷积核大小为 $3×3$，卷积步长为1的多通道卷积过程；需要注意的是，每个通道都有各自的卷积核，同一个通道的卷积核参数共享。如果输出通道为$out_c$，输入通道为$in_c$，那么需要 $out_c \times in_c$ 个卷积核。

![image-20230715171414193](https://blog-img-acking.oss-cn-beijing.aliyuncs.com/img/202307151714226.png)

#### pooling

**池化**是常见的降维操作，有最大池化和平均池化。池化操作和卷积的执行类似，通过池化核、步长、填充决定输出；最大池化是在池化核区域范围内取最大值，平均池化则是在池化核范围内做平均。与卷积不同的是池化核没有训练参数；池化层的填充方式也有所不同，平均池化填充的是0，最大池化填充的是$-inf$。对4×4的输入进行2×2区域池化，步长为2，不填充；图左边是最大池化的结果，右边是平均池化的结果。

![image-20230715171620742](https://blog-img-acking.oss-cn-beijing.aliyuncs.com/img/202307151716757.png)

有了卷积、池化、全连接组件就可以构建一个非常简单的卷积神经网络了， [图3.3.6](https://openmlsys.github.io/chapter_programming_interface/neural_network_layer.html#nn-network)展示了一个卷积神经网络的模型结构。 给定输入3×64×64的彩色图片，使用16个3×3×3大小的卷积核做卷积，得到大小为16×64×64的特征图； 再进行池化操作降维，得到大小为16×32×32的特征图； 对特征图再卷积得到大小为32×32×32特征图，再进行池化操作得到32×16×16大小的特征图； 我们需要对特征图做全连接，此时需要把特征图平铺成一维向量这步操作称为Flatten，压平后输入特征大小为32×16×16=8192； 之后做一次全连接对大小为8192特征变换到大小为128的特征，再依次做两次全连接分别得到64，10。 这里最后的输出结果是依据自己的实际问题而定，假设我们的输入是包含0∼9的数字图片，做分类那输出对应是10个概率值，分别对应0∼9的概率大小。

![image-20230715171719620](https://blog-img-acking.oss-cn-beijing.aliyuncs.com/img/202307151717638.png)

```python
# 构建卷积神经网络的组件接口定义：
全连接层接口：fully_connected(input, weights)
卷积层的接口：convolution(input, filters, stride, padding)
最大池化接口：pooling(input, pool_size, stride, padding, mode='max')
平均池化接口：pooling(input, pool_size, stride, padding, mode='mean')

# 构建卷积神经网络描述：
input:(3,64,64)大小的图片
# 创建卷积模型的训练变量,使用随机数初始化变量值
conv1_filters = variable(random(size=(3, 3, 3, 16)))
conv2_filters = variable(random(size=(3, 3, 16, 32)))
fc1_weights = variable(random(size=(8192, 128)))
fc2_weights = variable(random(size=(128, 64)))
fc3_weights = variable(random(size=(64, 10)))
# 将所有需要训练的参数收集起来
all_weights = [conv1_filters, conv2_filters, fc1_weights, fc2_weights, fc3_weights]

# 构建卷积模型的连接过程
output = convolution(input, conv1_filters, stride=1, padding='same')
output = pooling(output, kernel_size=3, stride=2, padding='same', mode='max')
output = convolution(output, conv2_filters, stride=1, padding='same')
output = pooling(output, kernel_size=3, stride=2, padding='same', mode='max')
output = flatten(output)
output = fully_connected(output, fc1_weights)
output = fully_connected(output, fc2_weights)
output = fully_connected(output, fc3_weights)
```

### 神经网络层的实现原理

上面的伪代码定义了一些卷积神经网络接口和模型构建过程，整个构建过程需要创建训练变量和构建连接过程。随着网络层数的增加，手动管理训练变量是一个繁琐的过程，因此上面描述的接口在机器学习库中属于低级API。机器学习编程库大都提供了更高级用户友好的API，它将神经网络层抽象出一个基类，所有的神经网络层都继承基类来实现，如MindSpore提供的mindspore.nn.Cell；PyTorch提供的torch.nn.Module。基于基类他们都提供了高阶API，如MindSpore 提供的mindspore.nn.Conv2d、mindspore.nn.MaxPool2d、mindspore.dataset；PyTorch提供的torch.nn.Conv2d、torch.nn.MaxPool2d、torch.utils.data.Dataset。

![image-20230715172814204](https://blog-img-acking.oss-cn-beijing.aliyuncs.com/img/202307151728253.png)

这张图描述了神经网络构建过程中的基本细节。**基类需要初始化训练参数、管理参数状态以及定义计算过程**；**神经网络模型需要实现对神经网络层和神经网络层参数管理的功能**。

在机器学习编程库中，承担此功能有MindSpore的Cell、PyTorch的Module。Cell和Module是模型抽象方法也是所有网络的基类。现有模型抽象方案有两种，一种是*抽象出两个方法分别为Layer（负责单个神经网络层的参数构建和前向计算），Model（负责对神经网络层进行连接组合和神经网络层参数管理）*；另一种是*将Layer和Model抽象成一个方法，该方法既能表示单层神经网络层也能表示包含多个神经网络层堆叠的模型*，Cell和Module就是这样实现的。

![image-20230715173120946](https://blog-img-acking.oss-cn-beijing.aliyuncs.com/img/202307151731998.png)

这张图展示了设计神经网络层抽象方法的通用表示。通常在init中会选择使用Python中collections模块的OrderedDict来初始化神经网络层和神经网络层参数的存储；它的输出是一个有序的，相比与Dict更适合深度学习这种模型堆叠的模式。参数和神经网络层的管理是在setattr中实现的，当检测到属性是属于神经网络层及神经网络层参数时就记录起来。

神经网络模型比较重要的是计算连接过程，可以在call里重载，实现神经网络层时在这里定义计算过程。训练参数的返回接口给优化器传所有训练参数，这些参数是基类遍历了所有网络层后得到的。这里只列出了一些重要的方法，在自定义方法中，通常需要实现参数插入删除、神经网络层插入删除、神经网络模型信息返回等方法。

神经网络接口层基类实现，仅做了简化的描述，在实际实现时，执行计算的call方法并不会让用户直接重载，它往往在call之外定义一个执行操作的方法（对于神经网络模型该方法是实现网络结构的连接，对于神经网络层则是实现计算过程）后再call调用；如MindSpore的Cell因为动态图和静态图的执行是不一样的，因此在call里定义动态图和计算图的计算执行，在construct方法里定义层或者模型的操作过程。

### 自定义神经网络层

上文使用伪代码定义机器学习库中低级API，有了实现的神经网络基类抽象方法，那么就可以设计更高层次的接口解决手动管理参数的繁琐。假设已经有了神经网络模型抽象方法Cell，**构建Conv2D将继承Cell**，并重构init和call方法，在init里初始化训练参数和输入参数，在call里调用低级API实现计算逻辑。同样使用伪代码描述自定义卷积层的过程。

```python
# 接口定义：
卷积层的接口：convolution(input, filters, stride, padding)
变量：Variable(value, trainable=True)
高斯分布初始化方法：random_normal(shape)
神经网络模型抽象方法：Cell

# 定义卷积层
class Conv2D(Cell):
    def __init__(self, in_channels, out_channels, ksize, stride, padding):
        # 卷积核大小为 ksize x ksize x inchannels x out_channels
        filters_shape = (out_channels, in_channels, ksize, ksize)
        self.stride = stride
        self.padding = padding
        self.filters = Variable(random_normal(filters_shape))

    def __call__(self, inputs):
        outputs = convolution(inputs, self.filters, self.stride, self.padding)
```

有了上述定义在使用卷积层时，就不需要创建训练变量了。 如我们需要对30×30大小10个通道的输入使用3×3的卷积核做卷积，卷积后输出通道为20。 调用方式如下：

```python
conv = Conv2D(in_channel=10, out_channel=20, filter_size=3, stride=2, padding=0)
output = conv(input)
```

在执行过程中，初始化Conv2D时，setattr会判断属性，属于Cell把神经网络层Conv2D记录到self._cells，属于parameter的filters记录到self._params。查看神经网络层参数使用conv.parameters_and_names；查看神经网络层列表使用conv.cells_and_names；执行操作使用conv(input)。

### 自定义神经网络模型

神经网络层是Cell的子类（SubClass）实现，同样的神经网络模型也可以采用SubClass的方法自定义神经网络模型；构建时需要在init里将要使用的神经网络组件实例化，在call里定义神经网络的计算逻辑。

同样的以上述的卷积神经网络模型为例，定义接口和伪代码描述如下：

```python
# 使用Cell子类构建的神经网络层接口定义：
# 构建卷积神经网络的组件接口定义：
全连接层接口：Dense(in_channel, out_channel)
卷积层的接口：Conv2D(in_channel, out_channel, filter_size, stride, padding)
最大池化接口：MaxPool2D(pool_size, stride, padding)
张量平铺：Flatten()

# 使用SubClass方式构建卷积模型
class CNN(Cell):
    def __init__(self):
        self.conv1 = Conv2D(in_channel=3, out_channel=16, filter_size=3, stride=1, padding=0)
        self.maxpool1 = MaxPool2D(pool_size=3, stride=1, padding=0)
        self.conv2 = Conv2D(in_channel=16, out_channel=32, filter_size=3, stride=1, padding=0)
        self.maxpool2 = MaxPool2D(pool_size=3, stride=1, padding=0)
        self.flatten = Flatten()
        self.dense1 = Dense(in_channels=768, out_channel=128)
        self.dense2 = Dense(in_channels=128, out_channel=64)
        self.dense3 = Dense(in_channels=64, out_channel=10)

    def __call__(self, inputs):
        z = self.conv1(inputs)
        z = self.maxpool1(z)
        z = self.conv2(z)
        z = self.maxpool2(z)
        z = self.flatten(z)
        z = self.dense1(z)
        z = self.dense2(z)
        z = self.dense3(z)
        return z
net = CNN()
```

上述卷积模型进行实例化，其执行将   从init开始，第一个是Conv2D，Conv2D也是Cell的子类，会进入到Conv2D的__init__，此时会将第一个Conv2D的卷积参数收集到self.\_params，之后回到Conv2D，将第一个Conv2D收集到self._cells；第二个的组件是MaxPool2D，因为其没有训练参数，因此将MaxPool2D收集到self._cells；依次类推，分别收集第二个卷积层的参数和层信息以及三个全连接层的参数和层信息。实例化之后可以调用net.parameters_and_names来返回训练参数；调用net.cells_and_names查看神经网络层列表。

## c++ interface

在很多时候，开发者也需要添加自定义的算子来帮助实现新的模型，优化器，数据处理函数等。这些自定义算子需要通过C和C++实现，从而获得最优性能。但是为了帮助这些算子被开发者使用，他们也需要暴露为Python函数，从而方便开发者整合入已有的Python为核心编写的工作流和模型。

### how to call c/c++ by python

**由于Python的解释器是由C实现的，因此在Python中可以实现对于C和C++函数的调用。**现代机器学习框架（包括TensorFlow，PyTorch和MindSpore）主要依赖**Pybind11**来将底层的大量C和C++函数**自动生成对应的Python函数**，这一过程一般被称为Python绑定(Binding)。在Pybind11出现以前，将C和C++函数进行Python绑定的手段主要包括：

- Python的C-API。这种方式要求在一个C++程序中包含Python.h，并使用Python的C-API对Python语言进行操作。使用这套API需要对Python的底层实现有一定了解，比如如何管理引用计数等，具有较高的使用门槛。
- 简单包装界面产生器（Simplified Wrapper and Interface Generator，SWIG)。SWIG可以将C和C++代码暴露给Python。SWIG是TensorFlow早期使用的方式。这种方式需要用户编写一个复杂的SWIG接口声明文件，并使用SWIG自动生成使用Python C-API的C代码。自动生成的代码可读性很低，因此具有很大代码维护开销。
- Python的ctypes模块，提供了C语言中的类型，以及直接调用动态链接库的能力。缺点是依赖于C的原生的类型，对自定义类型支持不好。
- Cython是结合了Python和C语言的一种语言，可以简单的认为就是给Python加上了静态类型后的语法，使用者可以维持大部分的Python语法。Cython编写的函数会被自动转译为C和C++代码，因此在Cython中可以插入对于C/C++函数的调用。
- Boost::Python是一个C++库。它可以将C++函数暴露为Python函数。其原理和Python C-API类似，但是使用方法更简单。然而，由于引入了Boost库，因此有沉重的第三方依赖。

相对于上述的提供Python绑定的手段，Pybind11提供了类似于Boost::Python的简洁性和易用性，但是其通过专注支持C++ 11，并且去除Boost依赖，因此成为了轻量级的Python库，从而特别适合在一个复杂的C++项目（例如本书讨论的机器学习系统）中暴露大量的Python函数。

### add self-define calculator

算子是构建神经网络的基础，在前面也称为低级API；通过算子的封装可以实现各类神经网络层，当开发神经网络层遇到内置算子无法满足时，可以通过自定义算子来实现。以MindSpore为例，实现一个GPU算子需要如下步骤：

1. Primitive注册：算子原语是构建网络模型的基础单元，用户可以直接或者间接调用算子原语搭建一个神经网络模型。
2. GPU Kernel实现：GPU Kernel用于调用GPU实现加速计算。
3. GPU Kernel注册：算子注册用于将GPU Kernel及必要信息注册给框架，由框架完成对GPU Kernel的调用。

#### 注册算子原语

**算子原语通常包括算子名、算子输入、算子属性（初始化时需要填的参数，如卷积的stride、padding）、输入数据合法性校验、输出数据类型推导和维度推导**。

假设需要编写加法算子，主要内容如下：

- 算子名：TensorAdd
- 算子属性：构造函数init中初始化属性，因加法没有属性，因此init不需要额外输入
- 算子输入输出及合法性校验：infer_shape方法中约束两个输入维度必须相同，输出的维度和输入维度相同。infer_dtype方法中约束两个输入数据必须是float32类型，输出的数据类型和输入数据类型相同。
- 算子输出

MindSpore中实现注册TensorAdd代码如下：

```python
# mindspore/ops/operations/math_ops.py
class TensorAdd(PrimitiveWithInfer):
    """
    Adds two input tensors element-wise.
    """
    @prim_attr_register
    def __init__(self):
        self.init_prim_io_names(inputs=['x1', 'x2'], outputs=['y'])

    def infer_shape(self, x1_shape, x2_shape):
        validator.check_integer('input dims', len(x1_shape), len(x2_shape), Rel.EQ, self.name)
        for i in range(len(x1_shape)):
            validator.check_integer('input_shape', x1_shape[i], x2_shape[i], Rel.EQ, self.name)
        return x1_shape

    def infer_dtype(self, x1_dtype, x2_type):
        validator.check_tensor_type_same({'x1_dtype': x1_dtype}, [mstype.float32], self.name)
        validator.check_tensor_type_same({'x2_dtype': x2_dtype}, [mstype.float32], self.name)
        return x1_dtype
```

在`mindspore/ops/operations/math_ops.py`文件内**注册加法算子原语后**，需要在`mindspore/ops/operations/__init__`中**导出，方便python导入模块时候调用。**

```python
# mindspore/ops/operations/__init__.py
from .math_ops import (Abs, ACos, ..., TensorAdd)
__all__ = [
  'ReverseSequence',
  'CropAndResize',
  ...,
  'TensorAdd'
]
```

#### GPU算子开发

继承GPU Kernel，实现加法使用类模板定义TensorAddGpuKernel，需要实现以下方法：

- `Init()`: 用于完成GPU Kernel的初始化，通常包括记录算子输入/输出维度，完成Launch前的准备工作；因此在此记录Tensor元素个数。
- `GetInputSizeList()`: 向框架反馈输入Tensor需要占用的显存字节数；返回了输入Tensor需要占用的字节数，TensorAdd有两个Input，每个Input占用字节数为element_num∗sizeof(T)。
- `GetOutputSizeList()`: 向框架反馈输出Tensor需要占用的显存字节数；返回了输出Tensor需要占用的字节数，TensorAdd有一个output，占用element_num∗sizeof(T)字节。
- `GetWorkspaceSizeList()`: 向框架反馈Workspace字节数，Workspace是用于计算过程中存放临时数据的空间；由于TensorAdd不需要Workspace，因此GetWorkspaceSizeList()返回空的std::vector<size_t>。
- `Launch()`: 通常调用CUDA kernel (CUDA kernel是基于Nvidia GPU的并行计算架构开发的核函数)，或者cuDNN接口等方式，完成算子在GPU上加速；Launch()接收input、output在显存的地址，接着调用TensorAdd完成加速。

```c++
// mindspore/ccsrc/backend/kernel_compiler/gpu/math/tensor_add_v2_gpu_kernel.h

template <typename T>
class TensorAddGpuKernel : public GpuKernel {
 public:
  TensorAddGpuKernel() : element_num_(1) {}
  ~TensorAddGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    auto shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    for (size_t i = 0; i < shape.size(); i++) {
      element_num_ *= shape[i];
    }
    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *x1 = GetDeviceAddress<T>(inputs, 0);
    T *x2 = GetDeviceAddress<T>(inputs, 1);
    T *y = GetDeviceAddress<T>(outputs, 0);

    TensorAdd(element_num_, x1, x2, y, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(element_num_ * sizeof(T));
    input_size_list_.push_back(element_num_ * sizeof(T));
    output_size_list_.push_back(element_num_ * sizeof(T));
  }

 private:
  size_t element_num_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
};
```

TensorAdd中调用了CUDA kernelTensorAddKernel来实现element_num个元素的并行相加：

```cpp
// mindspore/ccsrc/backend/kernel_compiler/gpu/math/tensor_add_v2_gpu_kernel.h

 template <typename T>
 __global__ void TensorAddKernel(const size_t element_num, const T* x1, const T* x2, T* y) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < element_num; i += blockDim.x * gridDim.x) {
    y[i] = x1[i] + x2[i];
  }
 }

 template <typename T>
 void TensorAdd(const size_t &element_num, const T* x1, const T* x2, T* y, cudaStream_t stream){
    size_t thread_per_block = 256;
    size_t block_per_grid = (element_num + thread_per_block - 1 ) / thread_per_block;
    TensorAddKernel<<<block_per_grid, thread_per_block, 0, stream>>>(element_num, x1, x2, y);
   return;
 }

 template void TensorAdd(const size_t &element_num, const float* x1, const float* x2, float* y, cudaStream_t stream);
```

#### GPU算子注册

算子信息包含

- Primive
- Input dtype, output dtype
- GPU Kernel class
- CUDA内置数据类型

框架会根据Primive和Input dtype, output dtype，调用以CUDA内置数据类型实例化GPU Kernel class模板类。如下代码中分别注册了支持float和int的TensorAdd算子。

```cpp
// mindspore/ccsrc/backend/kernel_compiler/gpu/math/tensor_add_v2_gpu_kernel.cc

MS_REG_GPU_KERNEL_ONE(TensorAddV2, KernelAttr()
                                    .AddInputAttr(kNumberTypeFloat32)
                                    .AddInputAttr(kNumberTypeFloat32)
                                    .AddOutputAttr(kNumberTypeFloat32),
                      TensorAddV2GpuKernel, float)

MS_REG_GPU_KERNEL_ONE(TensorAddV2, KernelAttr()
                                    .AddInputAttr(kNumberTypeInt32)
                                    .AddInputAttr(kNumberTypeInt32)
                                    .AddOutputAttr(kNumberTypeInt32),
                      TensorAddV2GpuKernel, int)
```

完成上述三步工作后，需要把MindSpore重新编译，在源码的根目录执行bash build.sh -e gpu，最后使用算子进行验证。

## mlsys framework programming formular

### 机器学习框架编程需求

机器学习的训练是其任务中最为关键的一步，训练依赖于优化器算法来描述。目前大部分机器学习任务都使用一阶优化器，因为一阶方法简单易用。随着机器学习的高速发展，软硬件也随之升级，越来越多的研究者开始探索收敛性能更好的高阶优化器。常见的二阶优化器如牛顿法、拟牛顿法、AdaHessians，均需要计算含有二阶导数信息的Hessian矩阵。

Hessian矩阵的计算带来两方面的问题，一方面是计算量巨大如何才能高效计算，另一方面是高阶导数的编程表达。

同时，近年来，工业界发布了非常多的大模型，越来越多的超大规模模型训练需求使得单纯的数据并行难以满足，而模型并行需要靠人工来模型切分耗时耗力，**如何自动并行成为未来机器学习框架所面临的挑战。**最后，构建机器学习模型本质上是数学模型的表示，如何简洁表示机器学习模型也成为机器学习框架编程范式的设计的重点。

为了解决机器学习框架在实际应用中的一些困难，研究人员发现函数式编程能很好地提供解决方案。在计算机科学中，函数式编程是一种编程范式，它将计算视为数学函数的求值，并避免状态变化和数据可变，这是一种更接近于数学思维的编程模式。神经网络由连接的节点组成，每个节点执行简单的数学运算。通过**使用函数式编程语言**，开发人员能够用一种更接近运算本身的语言来描述这些数学运算，使得程序的读取和维护更加容易。同时，函数式语言的函数都是相互隔离的，使得并发性和并行性更容易管理。

因此，机器学习框架使用函数式编程设计具有以下优势：

- 支持高效的科学计算和机器学习场景
- 易于开发并行
- 简洁的代码表示能力

### 机器学习框架编程范式现状

本小节将从目前主流机器学习框架发展历程来看机器学习框架对函数式编程的支持现状。

谷歌在2015年发布了TensorFlow1.0，其代表的编程特点包括计算图(Computational Graphs)、会话（Session）、张量(Tensor)，它是一种声明式编程风格。

2017年Facebook发布了PyTorch，其编程特点为即时执行，它是一种命令式编程风格。

2018年谷歌发布了JAX，它不是存粹为了机器学习而编写的框架，而是针对GPU和TPU做高性能数据并行计算的框架；与传统的机器学习框架相比其核心能力是神经网络计算和数值计算的融合，在接口上兼容了NumPy、Scipy等Python原生的数据科学接口，而且在此基础上扩展分布式、向量化、高阶求导、硬件加速，其编程风格是函数式，主要体现在无副作用、Lambda闭包等。

2020年华为发布了MindSpore，其函数式可微分编程架构可以让用户聚焦机器学习模型数学的原生表达。

2022年PyTorch推出functorch，受到谷歌JAX的极大启发，functorch是一个向PyTorch添加可组合函数转换的库，包括可组合的vmap（向量化）和autodiff转换，可与PyTorch模块和PyTorch autograd一起使用，并具有良好的渴望模式（Eager-Mode）性能，functorch可以说是弥补了PyTorch静态图的分布式并行需求。

从主流的机器学习框架发展历程来看，未来机器学习框架函数式编程风格将会日益得到应用，因为函数式编程能更直观地表达机器学习模型，同时对于自动微分、高阶求导、分布式实现也更加方便。另一方面，未来的机器学习框架在前端接口层次也趋向于分层解耦，其设计不直接为了机器学习场景，而是只提供高性能的科学计算和自动微分算子，更高层次的应用如机器学习模型开发则是通过封装这些高性能算子实现。

### 函数式编程案例

上一小节介绍了机器学习框架编程范式的现状，不管是JAX、MindSpore还是functorch都提到了函数式编程。其在科学计算、分布式方面有着独特的优势。然而在实际应用中纯函数式编程几乎没有能够成为主流开发范式，而现代编程语言几乎不约而同的选择了接纳函数式编程特性。

以MindSpore为例，MindSpore选择将函数式和面向对象编程融合，兼顾用户习惯，提供易用性最好，编程体验最佳的混合编程范式。MindSpore采用混合编程范式道理也很简单，纯函数式会让学习曲线陡增，易用性变差；面向对象构造神经网络的编程范式深入人心。

下面中提供了使用MindSpore编写机器学习模型训练的全流程。其网络构造，满足面向对象编程习惯，函数式编程主要体现在模型训练的反向传播部分；MindSpore使用函数式，将前向计算构造成function，然后通过函数变换，获得grad function，最后通过执行grad function获得权重对应的梯度。

```python
# Class definition
class Net(nn.Cell):
    def __init__(self):
        ......
    def construct(self, inputs):
        ......

# Object instantiation
net = Net() # network
loss_fn = nn.CrossEntropyLoss() # loss function
optimizer = nn.Adam(net.trainable_params(), lr) # optimizer

# define forward function
def forword_fn(inputs, targets):
    logits = net(inputs)
    loss = loss_fn(logits, targets)
    return loss, logits

# get grad function
grad_fn = value_and_grad(forward_fn, None, optim.parameters, has_aux=True)

# define train step function
def train_step(inputs, targets):
    (loss, logits), grads = grad_fn(inputs, targets) # get values and gradients
    optimizer(grads) # update gradient
    return loss, logits

for i in range(epochs):
    for inputs, targets in dataset():
        loss = train_step(inputs, targets)
```

## summary

- 现代机器学习系统需要兼有**易用性和高性能**，因此其一般选择Python作为前端，而C和C++作为后端
- 机器学习框架需要对一个完整的机器学习应用工作流进行编程支持。这些编程支持一般通过提供高层次Python API来实现
- 数据处理编程接口允许用户下载，导入和预处理数据集
- 模型定义编程接口允许用户定义和导入机器学习模型
- 损失函数接口允许用户定义损失函数来评估当前模型性能。同时，优化器接口允许用户定义和导入优化算法来基于损失函数计算梯度
- 机器学习框架同时兼有高层次Python API来对训练过程，模型测试和调试进行支持
- 复杂的深度神经网络可以通过叠加神经网络层来完成
- 用户可以通过Python API定义神经网络层，并指定神经网络层之间的拓扑来定义深度神经网络
- Python和C之间的互操作性一般通过CType等技术实现
- 机器学习框架一般具有多种C和C++接口允许用户定义和注册C++实现的算子。这些算子使得用户可以开发高性能模型，数据处理函数，优化器等一系列框架拓展

## expand

- MindSpore编程指南：[MindSpore](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/index.html)
- Python和C/C++混合编程：[Pybind11](https://pybind11.readthedocs.io/en/latest/basics.html#creating-bindings-for-a-simple-function)