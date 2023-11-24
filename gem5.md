# gem5

## Simple.py

这只是一个普通的 python 文件，将由嵌入式执行 gem5 可执行文件中的 python。因此，您可以使用任何功能和 Python 中可用的库。在这个文件中，我们要做的第一件事是导入 m5 库和所有 我们编译的 SimObjects。

```python
import m5
from m5.objects import *
```

接下来，我们将创建第一个 SimObject：我们将要创建的系统 模拟。**该对象将是所有其他我们模拟系统中的对象的父对象** 。该对象包含大量 功能（非时序级）信息，如物理内存 范围、根时钟域、根电压域、内核（在 全系统仿真）等。要创建系统 SimObject，我们只需 像普通的 Python 类一样实例化它：`SystemSystem`

```python
system = System()
```

 让我们在系统上设置时钟。我们首先必须创建一个时钟 域。然后我们可以在该域上设置时钟频率。设置 SimObject 上的参数与设置 SimObject 的成员完全相同 对象，因此我们可以简单地将时钟设置为 1 GHz。 最后，我们必须为该时钟域指定一个电压域。 由于我们现在不关心系统功耗，因此我们只使用 电压域的默认选项。

```python
# Set the clock fequency of the system (and all of its children)
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '1GHz'
system.clk_domain.voltage_domain = VoltageDomain()
```

有了系统后，让我们来设置如何模拟内存。我们将使用**定时模式**进行内存模拟。除了快进和从检查点恢复等特殊情况外，内存模拟几乎总是使用定时模式。我们还将设置一个 512 MB 大小的单一内存范围，这是一个非常小的系统。请注意，在 python 配置脚本中，只要需要大小，就可以用常用的语言和单位来指定大小。同样，对于时间，也可以使用时间单位（如 ）。这些单位将分别自动转换为通用表示法。

```python
# Set up the system
system.mem_mode = 'timing'               # Use timing accesses
system.mem_ranges = [AddrRange('512MB')] # Create an address range
```

现在，我们可以创建一个 CPU。我们将从 gem5 中最简单的基于 X86 ISA 的定时 CPU X86TimingSimpleCPU 开始。该 CPU 模型在一个时钟周期内执行每条指令，但内存请求除外，内存请求流经内存系统。要创建 CPU，只需实例化该对象即可：

```python
# Create a simple CPU
system.cpu = TimingSimpleCPU()
# system.cpu = X86TimingSimpleCPU()
```

接下来，我们将创建全系统内存总线：

```python
# Create a memory bus, a system crossbar, in this case
system.membus = SystemXBar()
```

既然有了内存总线，我们就将 CPU 上的高速缓存端口连接到内存总线上。在本例中，由于我们要模拟的系统没有任何缓存，我们将把 **iCache和 dCache端口直接连接到内存总线**上。在本示例系统中，我们没有缓存。

```python
# Hook the CPU ports up to the membus
system.cpu.icache_port = system.membus.slave
system.cpu.dcache_port = system.membus.slave
# system.cpu.icache_port = system.membus.cpu_side_ports
# system.cpu.dcache_port = system.membus.cpu_side_ports
```

为了将内存系统组件连接在一起，gem5 使用了**端口抽象**。每个内存对象可以有**两种端口，即请求端口和响应端口**。请求从请求端口发送到响应端口，响应从响应端口发送到请求端口。**连接端口时，必须将请求端口连接到响应端口**。

通过 python 配置文件很容易将端口连接在一起。只需将请求端口设置为响应端口，它们就会连接在一起。例如

```python
system.cpu.icache_port = system.l1_cache.cpu_side
```

在本例中，cpu 的端口是请求端口，缓存的端口是响应端口。请求端口和响应端口可以在任何一边，但都将建立相同的连接。建立连接后，请求者就可以向响应者发送请求。建立连接的幕后工作非常复杂，但对于大多数用户来说，其中的细节并不重要。

在 gem5 Python 配置中，两个端口的另一个值得注意的神奇之处在于，它允许在一侧有一个端口，而在另一侧有一个端口数组。例如

```python
system.cpu.icache_port = system.membus.cpu_side_ports
```

在本例中，cpu 的是一个请求端口，而 membus 的是一个响应端口数组。在这种情况下，一个新的响应端口会在 , 上生成，这个新创建的端口将连接到请求端口。



接下来，我们需要连接其他几个端口，以确保系统正常运行。我们需要**在 CPU 上创建一个 I/O 控制器**，并将其**连接到内存总线**。此外，我们还需要将系统中的一个特殊端口连接到内存总线上。该端口为**功能专用端口，允许系统读写内存**。将 PIO 和中断端口连接到内存总线是 x86 的特定要求。其他 ISA（如 ARM）不需要这 3 条额外的线路。

为 CPU 创建**中断控制器**并连接至内存总线:

```python
# create the interrupt controller for the CPU and connect to the membus
system.cpu.createInterruptController()
```

```python
# For x86 only, make sure the interrupts are connected to the memory
# Note: these are directly connected to the memory bus and are not cached
if m5.defines.buildEnv['TARGET_ISA'] == "x86":
    system.cpu.interrupts[0].pio = system.membus.master
    system.cpu.interrupts[0].int_master = system.membus.slave
    system.cpu.interrupts[0].int_slave = system.membus.master
# system.cpu.interrupts[0].pio = system.membus.mem_side_ports
# system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
# system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports
# system.system_port = system.membus.cpu_side_ports
```

接下来，我们需要创建一个内存控制器，并将其连接到内存总线。在本系统中，我们将使用一个简单的 DDR3 控制器，它将负责整个系统的内存范围。

```python
# Create a DDR3 memory controller and connect it to the membus
system.mem_ctrl = DDR3_1600_8x8()
system.mem_ctrl.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.master


# system.mem_ctrl = MemCtrl()
# system.mem_ctrl.dram = DDR3_1600_8x8()
# system.mem_ctrl.dram.range = system.mem_ranges[0]
# system.mem_ctrl.port = system.membus.mem_side_ports
```





## Cache.py

以前面的配置脚本为起点，本章将介绍更复杂的配置。如下图所示，我们将为系统添加缓存层次结构。此外，本章还将介绍如何理解 gem5 统计输出，以及如何为脚本添加命令行参数。

<img src="D:\桌面\DesktopFile\md笔记\src\advanced_config.png" style="zoom:50%;" />

我们将使用经典缓存，而不是 ruby-intro-chapter，因为我们正在模拟单 CPU 系统，而且我们并不关心缓存一致性建模。我们将扩展缓存 SimObject 并为我们的系统配置它。首先，我们必须了解用于配置 Cache 对象的参数。

gem5 目前有两个完全不同的子系统来模拟系统中的片上缓存，即 "经典缓存 "和 "Ruby"。造成这种情况的历史原因是，gem5 是密歇根州的 m5 和威斯康星州的 GEMS 的组合。GEMS 使用 Ruby 作为其缓存模型，而经典缓存则来自 m5 代码库（因此称为 "经典"）。这两种模型的区别在于，Ruby 的设计旨在详细模拟高速缓存的一致性。SLICC 是 Ruby 的一部分，它是一种用于定义缓存一致性协议的语言。另一方面，经典缓存执行的是简化且缺乏灵活性的 MOESI 一致性协议。

要选择使用哪种模型，您应该问问自己想要模拟什么。如果您要模拟缓存一致性协议的变化，或者一致性协议会对结果产生一阶影响，那么就使用 Ruby。否则，如果一致性协议对您来说并不重要，那就使用经典缓存。

------

缓存 SimObject 声明可在 src/mem/cache/Cache.py 中找到。该 Python 文件定义了可以设置的 SimObject 参数。在引擎盖下，当 SimObject 实例化时，这些参数将传递给对象的 C++ 实现。Cache SimObject 继承自下图所示的 BaseCache 对象。

BaseCache 类中有许多参数。例如，assoc 是一个整数参数。某些参数（如 write_buffers）有一个默认值，本例中为 8。默认参数是 Param.* 的第一个参数，除非第一个参数是字符串。每个参数的字符串参数是对参数内容的描述（例如，tag_latency = Param.Cycles("标签查找延迟")表示 tag_latency 控制 "该缓存的命中延迟"）。

其中许多参数没有默认值，因此我们需要在调用 m5.instantiate() 之前设置这些参数。

现在，为了创建带有特定参数的缓存，我们首先要在 simple.py 的同一目录 configs/tutorial/part1 下创建一个新文件 caches.py。第一步是在该文件中导入我们要扩展的 SimObject。

```python
import m5
from m5.objects import Cache
# Add the common scripts to our path
m5.util.addToPath('../../')
from common import SimpleOpts
```

接下来，我们可以像对待其他 Python 类一样对待 BaseCache 对象，并对其进行扩展。我们可以给新缓存起任何我们想要的名字。让我们从创建 L1 缓存开始。在这里，我们要设置 BaseCache 的一些没有默认值的参数。要查看所有可能的配置选项，并找出哪些是必需的，哪些是可选的，**您必须查看 SimObject 的源代码**。在本例中，我们使用的是 BaseCache。让我们为 L1 缓存添加两个函数：connectCPU 用于将 CPU 连接到缓存，connectBus 用于将缓存连接到总线。我们需要在 L1Cache 类中添加以下代码。

```python
class L1Cache(Cache):
    """Simple L1 Cache with default values"""

    assoc = 2
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 4
    tgts_per_mshr = 20

    def __init__(self, options=None):
        super(L1Cache, self).__init__()
        pass

    def connectBus(self, bus):
        """Connect this cache to a memory-side bus"""
        self.mem_side = bus.slave

    def connectCPU(self, cpu):
        """Connect this cache's port to a CPU-side port
           This must be defined in a subclass"""
        raise NotImplementedError

```

我们扩展了 BaseCache，并在 BaseCache SimObject 中设置了大部分没有默认值的参数。接下来，我们再创建两个 L1Cache 的子类，即 L1DCache 和 L1ICache.现在，我们已经指定了 BaseCache 所需的所有必要参数，接下来要做的就是实例化子类，并将缓存连接到互连网。不过，将大量对象连接到复杂的互连上会使配置文件迅速增大并变得不可读。因此，让我们先为 Cache 的子类添加一些辅助函数。记住，这些只是 Python 类，所以我们可以用它们做任何 Python 类能做的事情。

接下来，我们必须为指令缓存和数据缓存定义单独的 connectCPU 函数，因为 I 缓存和 D 缓存端口的名称不同。我们的 L1ICache 和 L1DCache 类现在变成了

```python
class L1ICache(L1Cache):
    """Simple L1 instruction cache with default values"""

    # Set the default size
    size = '16kB'

    SimpleOpts.add_option('--l1i_size',
                          help="L1 instruction cache size. Default: %s" % size)

    def __init__(self, opts=None):
        super(L1ICache, self).__init__(opts)
        if not opts or not opts.l1i_size:
            return
        self.size = opts.l1i_size

    def connectCPU(self, cpu):
        """Connect this cache's port to a CPU icache port"""
        self.cpu_side = cpu.icache_port

```



```python
class L1DCache(L1Cache):
    """Simple L1 data cache with default values"""

    # Set the default size
    size = '64kB'

    SimpleOpts.add_option('--l1d_size',
                          help="L1 data cache size. Default: %s" % size)

    def __init__(self, opts=None):
        super(L1DCache, self).__init__(opts)
        if not opts or not opts.l1d_size:
            return
        self.size = opts.l1d_size

    def connectCPU(self, cpu):
        """Connect this cache's port to a CPU dcache port"""
        self.cpu_side = cpu.dcache_port

```

最后，让我们为 L2Cache 添加功能，分别连接到内存侧和 CPU 侧总线。

```python
class L2Cache(Cache):
    """Simple L2 Cache with default values"""

    # Default parameters
    size = '256kB'
    assoc = 8
    tag_latency = 20
    data_latency = 20
    response_latency = 20
    mshrs = 20
    tgts_per_mshr = 12

    SimpleOpts.add_option('--l2_size', help="L2 cache size. Default: %s" % size)

    def __init__(self, opts=None):
        super(L2Cache, self).__init__()
        if not opts or not opts.l2_size:
            return
        self.size = opts.l2_size

    def connectCPUSideBus(self, bus):
        self.cpu_side = bus.master

    def connectMemSideBus(self, bus):
        self.mem_side = bus.slave
                                                  
```





现在，让我们将刚刚创建的缓存添加到上一章创建的配置脚本中。

首先，将脚本复制到一个新名称中。

```shell
cp ./configs/tutorial/part1/simple.py ./configs/tutorial/part1/two_level.py
```

首先，我们需要将 caches.py 文件中的名称导入命名空间。我们可以在文件顶部（m5.objects 导入之后）添加以下内容，就像添加任何 Python 源代码一样。

```python
from caches import *
```

创建 CPU 后，我们来创建 L1 缓存：

```python
system.cpu.icache = L1ICache()
system.cpu.dcache = L1DCache()
```

然后使用我们创建的辅助函数将缓存连接到 CPU 端口。

```python
system.cpu.icache.connectCPU(system.cpu)
system.cpu.dcache.connectCPU(system.cpu)
```

您需要移除以下两条将高速缓存端口直接连接到内存总线的线路。

```
system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports
```

我们不能直接将 L1 高速缓存连接到 L2 高速缓存，因为 L2 高速缓存只需要一个端口即可连接。因此，我们需要创建一条 L2 总线，将 L1 缓存连接到 L2 缓存。我们可以使用辅助函数将 L1 缓存连接到 L2 总线。

```python
# Create a memory bus, a coherent crossbar, in this case;创建一条内存总线，在这种情况下是一条连贯的横线
system.l2bus = L2XBar()
# Hook the CPU ports up to the L2bus
system.cpu.icache.connectBus(system.l2bus)
system.cpu.dcache.connectBus(system.l2bus)
```

接下来，我们可以创建二级缓存，并将其连接至二级总线和内存总线。

```python
system.l2cache = L2Cache()
system.l2cache.connectCPUSideBus(system.l2bus)
system.membus = SystemXBar()
system.l2cache.connectMemSideBus(system.membus)
```

请注意，system.membus = SystemXBar() 已在 system.l2cache.connectMemSideBus 之前定义，因此我们可以将其传递给 system.l2cache.connectMemSideBus。文件中的其他内容保持不变！现在，我们有了一个完整的配置，它具有两级缓存层次结构。如果运行当前文件，hello 将在 57467000 ticks 内完成。完整脚本可在 gem5 源代码 configs/learning_gem5/part1/twoo_level.py 中找到。

### Adding parameters to your script

在使用 gem5 进行实验时，你不会希望每次使用不同参数测试系统时都要编辑配置脚本。为了解决这个问题，你可以在 gem5 配置脚本中添加命令行参数。同样，由于配置脚本只是 Python 语言，你可以使用支持参数解析的 Python 库。尽管 pyoptparse 已被正式弃用，但由于 gem5 的 Python 最低版本曾经是 2.5，因此许多随 gem5 一起发布的配置脚本都使用它来代替 pyargparse。现在 Python 的最低版本是 3.6，因此在编写不需要与当前 gem5 脚本交互的新脚本时，Python 的 argparse 是一个更好的选择。要开始使用 :pyoptparse，您可以查阅在线 Python 文档。

为了给我们的二级缓存配置添加选项，在导入缓存后，让我们添加一些选项。

```python
import argparse

parser = argparse.ArgumentParser(description='A simple system with 2-level cache.')
parser.add_argument("binary", default="", nargs="?", type=str,
                    help="Path to the binary to execute.")
parser.add_argument("--l1i_size",
                    help=f"L1 instruction cache size. Default: 16kB.")
parser.add_argument("--l1d_size",
                    help="L1 data cache size. Default: Default: 64kB.")
parser.add_argument("--l2_size",
                    help="L2 cache size. Default: 256kB.")

options = parser.parse_args()
```

需要注意的是，如果你想按上述方式传递二进制文件的路径，并通过选项使用它，则应指定为 options.binary。例如

```python
system.workload = SEWorkload.init_compatible(options.binary)
```

现在，你可以运行 build/X86/gem5.opt configs/tutorial/part1/two_level.py --help 来显示刚刚添加的选项。
