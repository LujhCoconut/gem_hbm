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

在 gem5 Python 配置中，两个端口的另一个值得注意的神奇之处在于，它**允许在一侧有一个端口，而在另一侧有一个端口数组**。例如

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

#	def connectCPU(self, cpu):
    # need to define this in a base class!
#   	raise NotImplementedError

#	def connectBus(self, bus):
#    	self.mem_side = bus.cpu_side_ports       
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



## 了解 gem5 统计和输出

运行 gem5 后，除了模拟脚本打印出的任何信息外，还会在名为 m5out 的目录下生成三个文件：

**config.ini**

包含为模拟创建的每个 SimObject 及其参数值的列表。

**config.json**

与 config.ini 相同，但采用 json 格式

**stats.txt**

模拟登记的所有 gem5 统计数据的文本表示。



### **config.ini**

该文件是模拟结果的最终版本。模拟的每个模拟对象的所有参数，无论是在配置脚本中设置的，还是使用默认值的，都显示在该文件中。

```ini
[root]
type=Root
children=system
eventq_index=0
full_system=false
sim_quantum=0
time_sync_enable=false
time_sync_period=100000000000
time_sync_spin_threshold=100000000

[system]
type=System
children=clk_domain cpu cpu_clk_domain cpu_voltage_domain dvfs_handler l2 mem_ctrls membus tol2bus voltage_domain
boot_osflags=a
cache_line_size=64
clk_domain=system.clk_domain
default_p_state=UNDEFINED
eventq_index=0
exit_on_work_items=false
init_param=0
kernel=
kernel_addr_check=true
kernel_extras=
kvm_vm=Null
load_addr_mask=18446744073709551615
load_offset=0
mem_mode=timing
mem_ranges=0:536870911:0:0:0:0
memories=system.mem_ctrls
mmap_using_noreserve=false
multi_thread=false
num_work_ids=16
p_state_clk_gate_bins=20
p_state_clk_gate_max=1000000000000
p_state_clk_gate_min=1000
power_model=
readfile=
symbolfile=
thermal_components=
thermal_model=Null
work_begin_ckpt_count=0
work_begin_cpu_id_exit=-1
work_begin_exit_count=0
work_cpus_ckpt_count=0
work_end_ckpt_count=0
work_end_exit_count=0
work_item_id=-1
system_port=system.membus.slave[0]
```

在这里我们可以看到，在每个模拟对象的描述开头，首先是其在配置文件中创建的名称，并用方括号包围（例如 [system.membus]）。

接下来，将显示 SimObject 的每个参数及其值，包括配置文件中未明确设置的参数。例如，配置文件将时钟域设置为 1 GHz（本例中为 1000 ticks）。但是，配置文件并未设置缓存行大小（系统中为 64）对象。

配置文件（config.ini）是一个宝贵的工具，可确保你模拟的是你认为你正在模拟的东西。在 gem5 中，有许多可能的方法来设置默认值和覆盖默认值。 **"最佳做法 "是始终检查 config.ini，以确保配置文件中设置的值会传播到实际的 SimObject 实例化中。**



### stats.txt

gem5有一个灵活的统计生成系统。gem5统计在gem5维基站点上有详细介绍。模拟对象的每个实例都有自己的统计数据。在仿真结束时，或在发出特殊的统计转储命令时，**所有模拟对象的当前统计状态都会被转储到一个文件中**。

首先，统计文件包含有关执行情况的一般统计信息：

```txt
---------- Begin Simulation Statistics ----------                                                                          
sim_seconds                                  0.000033                       # Number of seconds simulated                         
sim_ticks                                    33476500                       # Number of ticks simulated                         
final_tick                                   33476500                       # Number of ticks from beginning of simulation (restored from checkpoints and never reset)
sim_freq                                 1000000000000                       # Frequency of simulated ticks                      
host_inst_rate                                  27354                       # Simulator instruction rate (inst/s)                 
host_op_rate                                    49375                       # Simulator op (including micro ops) rate (op/s)    
host_tick_rate                              160246965                       # Simulator tick rate (ticks/s)                    
host_mem_usage                                 663324                       # Number of bytes of host memory used                                         
host_seconds                                     0.21                       # Real time elapsed on the host                     
sim_insts                                        5712                       # Number of instructions simulated                  
sim_ops                                         10313                       # Number of ops (including micro ops) simulated                               
system.voltage_domain.voltage                       1                       # Voltage in Volts                                                                                
```

统计转储以 ---------- Begin Simulation Statistics ---------- 开始。如果 gem5 执行期间有多个统计转储，则单个文件中可能有多个这样的文件。这种情况常见于长时间运行的应用程序，或从检查点恢复时。

每个统计量都有一个名称（第一列）、一个值（第二列）和一个说明（最后一列，前面加 #），然后是统计量的单位。

大多数统计量都可以通过说明自行解释。其中几个重要的统计数据是：sim_seconds（模拟的总模拟时间）、sim_insts（CPU 执行的指令数）和 host_inst_rate（告诉你 gem5 的性能）。

接下来，将打印模拟对象的统计数据。例如，CPU 统计信息包含系统调用次数、缓存系统和翻译缓冲区统计信息等。



文件后面是内存控制器统计数据。其中包含每个组件读取的字节数以及这些组件使用的平均带宽等信息。



## 使用默认配置脚本

在本章中，我们将探讨如何使用 gem5 自带的默认配置脚本。gem5 随附了许多配置脚本，可让你快速使用 gem5。然而，一个常见的误区是，在使用这些脚本时并不完全了解正在模拟的内容。在使用 gem5 进行计算机体系结构研究时，充分了解所模拟的系统非常重要。本章将向你介绍一些重要选项和默认配置脚本的部分内容。

在过去几章中，您已经从头开始创建了自己的配置脚本。这非常强大，因为它允许你指定每一个系统参数。然而，有些系统的设置非常复杂（例如，全系统的 ARM 或 x86 机器）。幸运的是，gem5 开发人员提供了许多脚本来引导构建系统的过程。

### 参观目录结构

gem5 的所有配置文件都可以在 configs/ 目录中找到。目录结构如下所示：

```shell
configs/boot:
bbench-gb.rcS  bbench-ics.rcS  hack_back_ckpt.rcS  halt.sh

configs/common:
Benchmarks.py   Caches.py  cpu2000.py    FileSystemConfig.py  GPUTLBConfig.py   HMC.py       MemConfig.py   Options.py     Simulation.py
CacheConfig.py  cores      CpuConfig.py  FSConfig.py          GPUTLBOptions.py  __init__.py  ObjectList.py  SimpleOpts.py  SysPaths.py

configs/dist:
sw.py

configs/dram:
lat_mem_rd.py  low_power_sweep.py  sweep.py

configs/example:
apu_se.py  etrace_replay.py  garnet_synth_traffic.py  hmctest.py    hsaTopology.py  memtest.py  read_config.py  ruby_direct_test.py      ruby_mem_test.py     sc_main.py
arm        fs.py             hmc_hello.py             hmc_tgen.cfg  memcheck.py     noc_config  riscv           ruby_gpu_random_test.py  ruby_random_test.py  se.py

configs/learning_gem5:
part1  part2  part3  README

configs/network:
__init__.py  Network.py

configs/nvm:
sweep_hybrid.py  sweep.py

configs/ruby:
AMD_Base_Constructor.py  CHI.py        Garnet_standalone.py  __init__.py              MESI_Three_Level.py  MI_example.py      MOESI_CMP_directory.py  MOESI_hammer.py
CHI_config.py            CntrlBase.py  GPU_VIPER.py          MESI_Three_Level_HTM.py  MESI_Two_Level.py    MOESI_AMD_Base.py  MOESI_CMP_token.py      Ruby.py

configs/splash2:
cluster.py  run.py

configs/topologies:
BaseTopology.py  Cluster.py  CrossbarGarnet.py  Crossbar.py  CustomMesh.py  __init__.py  MeshDirCorners_XY.py  Mesh_westfirst.py  Mesh_XY.py  Pt2Pt.py
```

#### **boot/**

这些是在全系统模式下使用的 rcS 文件。这些文件在 Linux 启动后由模拟器加载，并由 shell 执行。其中大部分用于控制在全系统模式下运行的基准。有些是实用功能，如 hack_back_ckpt.rcS。这些文件将在全系统模拟一章中详细介绍。

#### **common/**

该目录包含大量用于创建模拟系统的辅助脚本和函数。例如，Caches.py 与前几章创建的 caches.py 和 caches_opts.py 文件类似。
Options.py 包含多种可在命令行上设置的选项。如 CPU 数量、系统时钟等。你可以在这里查看想要更改的选项是否已经有了命令行参数。

**CacheConfig.py** 包含为经典内存系统设置缓存参数的选项和函数。

**MemConfig.py** 提供了一些用于设置内存系统的辅助函数。

**FSConfig.py** 包含为多种不同系统设置全系统仿真所需的函数。全系统仿真将在单独的一章中进一步讨论。

**Simulation.py** 包含许多用于设置和运行 gem5 的辅助函数。该文件中包含的许多代码都是用来管理保存和恢复检查点的。下文 examples/ 中的示例配置文件使用该文件中的函数来执行 gem5 仿真。该文件相当复杂，但也允许在模拟运行方式上有很大的灵活性。

#### **dram/**

Contains scripts to test DRAM.

#### **example/**

This directory contains some example gem5 configuration scripts that can be used out-of-the-box to run gem5. Specifically, `se.py` and `fs.py` are quite useful. More on these files can be found in the next section. There are also some other utility configuration scripts in this directory.

#### **learning_gem5/**

This directory contains all gem5 configuration scripts found in the learning_gem5 book.

#### **network/**

This directory contains the configurations scripts for a HeteroGarnet network.

#### **nvm/**

This directory contains example scripts using the NVM interface.

#### **ruby/**

This directory contains the configurations scripts for Ruby and its included cache coherence protocols. More details can be found in the chapter on Ruby.

#### **splash2/**

This directory contains scripts to run the splash2 benchmark suite with a few options to configure the simulated system.

#### **topologies/**

This directory contains the implementation of the topologies that can be used when creating the Ruby cache hierarchy. More details can be found in the chapter on Ruby.



### 使用se.py & fs.py

在本节中，我将讨论一些可以通过命令行传递给 se.py 和fs.py 的常用选项。有关如何运行全系统仿真的更多详情，请参阅全系统仿真章节。在此，我将讨论这两个文件共有的选项。

本节讨论的大部分选项都可以在 Options.py 中找到，并在函数 addCommonOptions 中注册。本节不详细介绍所有选项。要查看所有选项，请使用 --help 运行配置脚本，或阅读脚本的源代码。

首先，让我们不带任何参数地运行 hello world 程序：

```shell
build/X86/gem5.opt configs/example/se.py --cmd=tests/test-progs/hello/bin/x86/linux/hello
```

然而，这根本不是一个非常有趣的模拟！默认情况下，gem5 使用原子 CPU 并使用原子内存访问，因此没有真正的时序数据报告！要确认这一点，你可以查看 m5out/config.ini。

要在定时模式下实际运行 gem5，我们需要指定 CPU 类型。同时，我们还可以指定 L1 缓存的大小。

```shell
build/X86/gem5.opt configs/example/se.py --cmd=tests/test-progs/hello/bin/x86/linux/hello --cpu-type=TimingSimpleCPU --l1d_size=64kB --l1i_size=16kB
```

现在，让我们检查一下 config.ini 文件，确保这些选项能正确传播到最终系统中。如果在 m5out/config.ini 文件中搜索 "缓存"，你会发现没有创建缓存！虽然我们指定了缓存的大小，但并没有指定系统应该使用缓存，所以缓存没有创建。**正确的命令行**应该是

```shell
build/X86/gem5.opt configs/example/se.py --cmd=tests/test-progs/hello/bin/x86/linux/hello --cpu-type=TimingSimpleCPU --l1d_size=64kB --l1i_size=16kB --caches
```

在最后一行，我们看到总时间从 454646000 ticks 变为 31680000，快了很多！看来缓存可能已经启用了。不过，**最好还是仔细检查一下 config.ini 文件**。



### 一些se.py & fs.py 的常见选项

运行时会打印所有可能的选项：

```shell
build/X86/gem5.opt configs/example/se.py --help
```

以下是该清单中的几个重要选项：

- `--cpu-type=CPU_TYPE`
  - 运行 CPU 的类型。这是一个必须始终设置的重要参数。默认值为原子，不执行时序模拟。

- `--sys-clock=SYS_CLOCK`
  - 以系统速度运行区块的顶级时钟。

- `--cpu-clock=CPU_CLOCK`
  - 以 CPU 速度运行的程序块的时钟。这与上述系统时钟是分开的。

- `--mem-type=MEM_TYPE`
  - 使用的内存类型。选项包括不同的 DDR 内存和 ruby 内存控制器。

- `--caches`
  - 使用传统缓存进行模拟。

- `--l2cache`
  - 如果使用传统缓存，则使用二级缓存进行模拟。

- `--ruby`
  - Use Ruby instead of the classic caches as the cache system simulation.
- `-m TICKS, --abs-max-tick=TICKS`
  - Run to absolute simulated tick specified including ticks from a restored checkpoint. This is useful if you only want simulate for a certain amount of simulated time.
- `-I MAXINSTS, --maxinsts=MAXINSTS`
  - Total number of instructions to simulate (default: run forever). This is useful if you want to stop simulation after a certain number of instructions has been executed.
- `-c CMD, --cmd=CMD`
  - The binary to run in syscall emulation mode.
- `-o OPTIONS, --options=OPTIONS`
  - The options to pass to the binary, use ” ” around the entire string. This is useful when you are running a command which takes options. You can pass both arguments and options (e.g., –whatever) through this variable.
- `--output=OUTPUT`
  - Redirect stdout to a file. This is useful if you want to redirect the output of the simulated application to a file instead of printing to the screen. Note: to redirect gem5 output, you have to pass a parameter before the configuration script.
- `--errout=ERROUT`
  - Redirect stderr to a file. Similar to above.

[gem5: Using the default configuration scripts](https://www.gem5.org/documentation/learning_gem5/part1/example_configs/)



## 修改与扩展

### 创建一个非常简单的模拟对象

注意：gem5 有一个名为 SimpleObject 的 SimObject。**实现另一个 SimObject 简单对象将导致编译器问题混乱**。

gem5 中的几乎所有对象都继承自基本 SimObject 类型。SimObjects 为 gem5 中的所有对象导出了主要接口。**SimObjects 是经过包装的 C++ 对象，可通过 Python 配置脚本访问。**

SimObjects 可以有很多参数，这些参数通过 Python 配置文件设置。除了整数和浮点数等简单参数外，它们还可以拥有其他模拟对象作为参数。这样就可以创建复杂的系统层次结构，就像真实的机器一样。

在本章中，我们将逐步创建一个简单的 "HelloWorld "模拟对象。目的是向您介绍如何创建 SimObject 以及所有 SimObject 所需的模板代码。我们还将创建一个简单的 Python 配置脚本，用于实例化我们的 SimObject。

在接下来的几章中，我们将使用这个简单的 SimObject 并对其进行扩展，包括调试支持、动态事件和参数。

#### **Using git branches**

在 gem5 中添加每个新功能时，通常都会使用一个新的 git 分支。

在 gem5 中添加新功能或修改某些内容时，第一步是创建一个新分支来存储您的修改。有关 Git 分支的详细信息，请参阅“Git”文档。

```shell
git checkout -b hello-simobject
```



#### **步骤 1：为新的模拟对象创建 Python 类**

**每个模拟对象都有一个与之关联的 Python 类**。**这个 Python 类描述了可以通过 Python 配置文件控制的模拟对象参数**。对于我们这个简单的 SimObject，一开始不需要任何参数。因此，我们只需为我们的 SimObject 声明一个新类，并设置它的名称和定义 SimObject C++ 类的 C++ 头文件。

我们可以在 src/learning_gem5/part2 中创建文件 HelloObject.py。如果你已经克隆了 gem5 软件源，本教程中提到的文件就会在 src/learning_gem5/part2 和 configs/learning_gem5/part2 下完成。你可以删除这些文件或将它们移到其他地方，以便跟上本教程。

```python
from m5.params import *
from m5.SimObject import SimObject

class HelloObject(SimObject):
    type = 'HelloObject'
    cxx_header = "learning_gem5/part2/hello_object.hh"
    cxx_class = "gem5::HelloObject"
```

并不要求类型与类名相同，但这是惯例。类型就是您要用这个 Python SimObject 封装的 C++ 类。只有在特殊情况下，类型和类名才应该不同。

**cxx_header** 是包含作为类型参数的类的声明的**文件**。按照惯例，SimObject 的名称应使用小写字母和下划线，但这只是惯例。您可以在此指定任何头文件。

**cxx_class** 是一个**属性**，指定新创建的模拟对象是在 gem5 命名空间中声明的。gem5 代码库中的大多数 SimObject 都是在 gem5 命名空间中声明的！



#### **步骤2：用 C++ 实现模拟对象**

接下来，我们需要在 src/learning_gem5/part2/ 目录下创建 hello_object.hh 和 hello_object.cc，它们将实现 HelloObject。

我们将从 C++ 对象的头文件开始。按照惯例，gem5 会将所有头文件封装在 #ifndef/#endif 中，并注明文件名及其所在目录，这样就不会出现循环包含的情况。

SimObjects 应在 gem5 命名空间内声明。因此，我们在 gem5 名称空间范围内声明我们的类。

我们在文件中唯一需要做的就是声明我们的类。由于 HelloObject 是一个 SimObject，它必须继承于 C++ SimObject 类。大多数情况下，SimObject 的父类是 SimObject 的子类，而不是 SimObject 本身。

SimObject 类指定了许多虚函数。不过，这些函数都不是纯粹的虚函数，因此在最简单的情况下，除了构造函数外，无需实现任何函数。

**所有 SimObject 的构造函数都假定它将接收一个参数对象**。**这个参数对象是由构建系统自动创建的，它基于 SimObject 的 Python 类，**就像我们上面创建的那个一样。该参数类型的名称由对象名称自动生成。对于我们的 "HelloObject"，参数类型的名称是 "HelloObjectParams"。

下面列出了我们的简单头文件所需的代码。

```c++
#ifndef __LEARNING_GEM5_HELLO_OBJECT_HH__
#define __LEARNING_GEM5_HELLO_OBJECT_HH__

#include "params/HelloObject.hh"
#include "sim/sim_object.hh"

namespace gem5
{

class HelloObject : public SimObject
{
  public:
    HelloObject(const HelloObjectParams &p);
};

} // namespace gem5

#endif // __LEARNING_GEM5_HELLO_OBJECT_HH__
```

接下来，我们需要在 .cc 文件中实现两个函数，而不仅仅是一个。第一个函数是 HelloObject 的构造函数。在这里，我们只需将参数对象传递给 SimObject 父对象，并打印 "Hello world！"

**通常情况下，你绝不会在 gem5 中使用 std::cout，而是应该使用调试标志。**在下一章中，我们将对此进行修改，改用调试标志。不过，现在我们只需使用 std::cout，因为它很简单。

```c++
#include "learning_gem5/part2/hello_object.hh"

#include <iostream>

namespace gem5
{

HelloObject::HelloObject(const HelloObjectParams &params) :
    SimObject(params)
{
    std::cout << "Hello World! From a SimObject!" << std::endl;
}

} // namespace gem5
```

注意：如果模拟对象的构造函数遵循以下签名、

```c++
Foo(const FooParams &)
```

则将自动定义 FooParams::create() 方法。create() 方法的目的是调用 SimObject 构造函数并返回一个 SimObject 实例。大多数 SimObject 都将遵循这种模式；但是，如果您的 SimObject 不遵循这种模式，gem5 SimObject 文档将提供更多有关手动实现 create() 方法的信息。



#### **步骤3：注册模拟对象和 C++ 文件**

为了编译 C++ 文件和解析 Python 文件，我们需要将这些文件告知编译系统。**gem5 使用 SCons 作为编译系统**，**因此您只需在包含 SimObject 代码的目录中创建一个 SConscript 文件。如果该目录下已有 SConscript 文件，只需在该文件中添加以下声明即可。**

该文件只是一个普通的 Python 文件，因此您可以在该文件中编写任何 Python 代码。gem5 可利用这一点来自动创建 SimObjects 代码，并编译 SLICC 和 ISA 语言等特定领域语言。

在 SConscript 文件中，有许多函数是在导入后自动定义的。请参阅相关章节...

要编译新的 SimObject，只需在 src/learning_gem5/part2 目录下新建一个名为 "SConscript "的文件。在该文件中，您必须声明 SimObject 和 .cc 文件。下面是所需代码。

```scons
Import('*')

SimObject('HelloObject.py', sim_objects=['HelloObject'])
Source('hello_object.cc')
```



#### **步骤4：（re-)build gem5**

要编译和链接新文件，你只需重新编译 gem5。下面的示例假定你使用的是 x86 ISA，但我们的对象中没有任何要求 ISA 的内容，因此它可以在 gem5 的任何 ISA 下运行。

```shell
scons build/X86/gem5.opt
```



#### 步骤5：创建配置脚本以使用新的模拟对象

现在您已经实现了一个 SimObject，并已将其编译到 gem5 中，您需要在 configs/learning_gem5/part2 中创建或修改一个 Python 配置文件 run_hello.py，以实例化您的对象。由于您的对象非常简单，因此不需要系统对象！除了 Root 对象外，不需要 CPU、缓存或其他任何东西。**所有 gem5 实例都需要一个 Root 对象。**

**在创建一个非常简单的配置脚本时，首先，导入 m5 和你编译的所有对象。**

```python
import m5
from m5.objects import *
```

接下来，你必须按照所有 gem5 实例的要求实例化 Root 对象。

```python
root = Root(full_system = False)
```

现在，您可以实例化您创建的 HelloObject。您需要做的就是调用 Python "构造函数"。稍后，我们将了解如何通过 Python 构造函数指定参数。除了创建对象的实例外，您还需要确保它是根对象的子对象。在 C++ 中，只有根对象的子对象才能被实例化。

```python
root.hello = HelloObject()
```

最后，需要调用 m5 模块的实例化，并实际运行模拟！

```python
m5.instantiate()

print("Beginning simulation!")
exit_event = m5.simulate()
print('Exiting @ tick {} because {}'
      .format(m5.curTick(), exit_event.getCause()))
```

```shell
[gem5/X86/目录下] $ ./gem5.opt /home/dell/jhlu/gem5_hbm/gem5/configs/learning_gem5/part2/run_simple.py
```



修改 src/ 目录中的文件后，请记得重建 gem5。运行配置文件的命令行位于下面输出中的 "命令行："之后。输出结果应如下所示：

注意：如果未来章节 "为模拟对象添加参数和更多事件"（goodbye_object）的代码位于 src/learning_gem5/part2 目录中，run_hello.py 将导致错误。如果删除这些文件或将它们移到 gem5 目录之外，run_hello.py 就会得到下面的输出结果。



## Debugging gem5

在前几章中，我们介绍了如何创建一个非常简单的 SimObject。在本章中，我们将用 gem5 的调试支持取代简单的打印到 stdout。

gem5通过**调试标记（debug flags）**为你的代码提供printf风格的跟踪/调试支持。这些标志允许每个组件拥有许多调试-打印语句，而无需同时启用所有这些语句。运行 gem5 时，你可以在命令行中指定启用哪些调试标记。

### Using debug flags

例如，在运行 simple-config-chapter 中的第一个 simple.py 脚本时，如果启用 DRAM 调试标志，就会得到以下输出。请注意，这会在控制台中产生大量输出（约 7 MB）。

```shell
 build/X86/gem5.opt --debug-flags=DRAM configs/learning_gem5/part1/simple.py | head -n 50
```

```shell
[build/X86] ./gem5.opt --debug-flags=DRAM ../../configs/learning_gem5/part1/simple.py | head -n 50
```

```shell
gem5 Simulator System.  http://gem5.org
DRAM device capacity (gem5 is copyrighted software; use the --copyright option for details.

gem5 compiled Jan  3 2017 16:03:38
gem5 started Jan  3 2017 16:09:53
gem5 executing on chinook, pid 19223
command line: build/X86/gem5.opt --debug-flags=DRAM configs/learning_gem5/part1/simple.py

Global frequency set at 1000000000000 ticks per second
      0: system.mem_ctrl: Memory capacity 536870912 (536870912) bytes
      0: system.mem_ctrl: Row buffer size 8192 bytes with 128 columns per row buffer
      0: system.remote_gdb.listener: listening for remote gdb #0 on port 7000
Beginning simulation!
info: Entering event queue @ 0.  Starting simulation...
      0: system.mem_ctrl: recvTimingReq: request ReadReq addr 400 size 8
      0: system.mem_ctrl: Read queue limit 32, current size 0, entries needed 1
      0: system.mem_ctrl: Address: 400 Rank 0 Bank 0 Row 0
      0: system.mem_ctrl: Read queue limit 32, current size 0, entries needed 1
      0: system.mem_ctrl: Adding to read queue
      0: system.mem_ctrl: Request scheduled immediately
      0: system.mem_ctrl: Single request, going to a free rank
      0: system.mem_ctrl: Timing access to addr 400, rank/bank/row 0 0 0
      0: system.mem_ctrl: Activate at tick 0
      0: system.mem_ctrl: Activate bank 0, rank 0 at tick 0, now got 1 active
      0: system.mem_ctrl: Access to 400, ready at 46250 bus busy until 46250.
  46250: system.mem_ctrl: processRespondEvent(): Some req has reached its readyTime
  46250: system.mem_ctrl: number of read entries for rank 0 is 0
  46250: system.mem_ctrl: Responding to Address 400..   46250: system.mem_ctrl: Done
  77000: system.mem_ctrl: recvTimingReq: request ReadReq addr 400 size 8
  77000: system.mem_ctrl: Read queue limit 32, current size 0, entries needed 1
  77000: system.mem_ctrl: Address: 400 Rank 0 Bank 0 Row 0
  77000: system.mem_ctrl: Read queue limit 32, current size 0, entries needed 1
  77000: system.mem_ctrl: Adding to read queue
  77000: system.mem_ctrl: Request scheduled immediately
  77000: system.mem_ctrl: Single request, going to a free rank
  77000: system.mem_ctrl: Timing access to addr 400, rank/bank/row 0 0 0
  77000: system.mem_ctrl: Access to 400, ready at 101750 bus busy until 101750.
 101750: system.mem_ctrl: processRespondEvent(): Some req has reached its readyTime
 101750: system.mem_ctrl: number of read entries for rank 0 is 0
 101750: system.mem_ctrl: Responding to Address 400..  101750: system.mem_ctrl: Done
 132000: system.mem_ctrl: recvTimingReq: request ReadReq addr 400 size 8
 132000: system.mem_ctrl: Read queue limit 32, current size 0, entries needed 1
 132000: system.mem_ctrl: Address: 400 Rank 0 Bank 0 Row 0
 132000: system.mem_ctrl: Read queue limit 32, current size 0, entries needed 1
 132000: system.mem_ctrl: Adding to read queue
 132000: system.mem_ctrl: Request scheduled immediately
 132000: system.mem_ctrl: Single request, going to a free rank
 132000: system.mem_ctrl: Timing access to addr 400, rank/bank/row 0 0 0
 132000: system.mem_ctrl: Access to 400, ready at 156750 bus busy until 156750.
 156750: system.mem_ctrl: processRespondEvent(): Some req has reached its readyTime
 156750: system.mem_ctrl: number of read entries for rank 0 is 0
```



或者，您可能希望根据 CPU 正在执行的确切指令进行调试。为此，Exec 调试标志可能很有用。该调试标志会显示模拟 CPU 如何执行每条指令的详细信息。

```shell
build/X86/gem5.opt --debug-flags=Exec configs/learning_gem5/part1/simple.py | head -n 50
```

```shell
[build/X86] ./gem5.opt --debug-flags=Exec ../../configs/learning_gem5/part1/simple.py | head -n 50
```

```shell
gem5 Simulator System.  http://gem5.org
gem5 is copyrighted software; use the --copyright option for details.

gem5 compiled Jan  3 2017 16:03:38
gem5 started Jan  3 2017 16:11:47
gem5 executing on chinook, pid 19234
command line: build/X86/gem5.opt --debug-flags=Exec configs/learning_gem5/part1/simple.py

Global frequency set at 1000000000000 ticks per second
      0: system.remote_gdb.listener: listening for remote gdb #0 on port 7000
warn: ClockedObject: More than one power state change request encountered within the same simulation tick
Beginning simulation!
info: Entering event queue @ 0.  Starting simulation...
  77000: system.cpu T0 : @_start    : xor   rbp, rbp
  77000: system.cpu T0 : @_start.0  :   XOR_R_R : xor   rbp, rbp, rbp : IntAlu :  D=0x0000000000000000
 132000: system.cpu T0 : @_start+3    : mov r9, rdx
 132000: system.cpu T0 : @_start+3.0  :   MOV_R_R : mov   r9, r9, rdx : IntAlu :  D=0x0000000000000000
 187000: system.cpu T0 : @_start+6    : pop rsi
 187000: system.cpu T0 : @_start+6.0  :   POP_R : ld   t1, SS:[rsp] : MemRead :  D=0x0000000000000001 A=0x7fffffffee30
 250000: system.cpu T0 : @_start+6.1  :   POP_R : addi   rsp, rsp, 0x8 : IntAlu :  D=0x00007fffffffee38
 250000: system.cpu T0 : @_start+6.2  :   POP_R : mov   rsi, rsi, t1 : IntAlu :  D=0x0000000000000001
 360000: system.cpu T0 : @_start+7    : mov rdx, rsp
 360000: system.cpu T0 : @_start+7.0  :   MOV_R_R : mov   rdx, rdx, rsp : IntAlu :  D=0x00007fffffffee38
 415000: system.cpu T0 : @_start+10    : and    rax, 0xfffffffffffffff0
 415000: system.cpu T0 : @_start+10.0  :   AND_R_I : limm   t1, 0xfffffffffffffff0 : IntAlu :  D=0xfffffffffffffff0
 415000: system.cpu T0 : @_start+10.1  :   AND_R_I : and   rsp, rsp, t1 : IntAlu :  D=0x0000000000000000
 470000: system.cpu T0 : @_start+14    : push   rax
 470000: system.cpu T0 : @_start+14.0  :   PUSH_R : st   rax, SS:[rsp + 0xfffffffffffffff8] : MemWrite :  D=0x0000000000000000 A=0x7fffffffee28
 491000: system.cpu T0 : @_start+14.1  :   PUSH_R : subi   rsp, rsp, 0x8 : IntAlu :  D=0x00007fffffffee28
 546000: system.cpu T0 : @_start+15    : push   rsp
 546000: system.cpu T0 : @_start+15.0  :   PUSH_R : st   rsp, SS:[rsp + 0xfffffffffffffff8] : MemWrite :  D=0x00007fffffffee28 A=0x7fffffffee20
 567000: system.cpu T0 : @_start+15.1  :   PUSH_R : subi   rsp, rsp, 0x8 : IntAlu :  D=0x00007fffffffee20
 622000: system.cpu T0 : @_start+16    : mov    r15, 0x40a060
 622000: system.cpu T0 : @_start+16.0  :   MOV_R_I : limm   r8, 0x40a060 : IntAlu :  D=0x000000000040a060
 732000: system.cpu T0 : @_start+23    : mov    rdi, 0x409ff0
 732000: system.cpu T0 : @_start+23.0  :   MOV_R_I : limm   rcx, 0x409ff0 : IntAlu :  D=0x0000000000409ff0
 842000: system.cpu T0 : @_start+30    : mov    rdi, 0x400274
 842000: system.cpu T0 : @_start+30.0  :   MOV_R_I : limm   rdi, 0x400274 : IntAlu :  D=0x0000000000400274
 952000: system.cpu T0 : @_start+37    : call   0x9846
 952000: system.cpu T0 : @_start+37.0  :   CALL_NEAR_I : limm   t1, 0x9846 : IntAlu :  D=0x0000000000009846
 952000: system.cpu T0 : @_start+37.1  :   CALL_NEAR_I : rdip   t7, %ctrl153,  : IntAlu :  D=0x00000000004001ba
 952000: system.cpu T0 : @_start+37.2  :   CALL_NEAR_I : st   t7, SS:[rsp + 0xfffffffffffffff8] : MemWrite :  D=0x00000000004001ba A=0x7fffffffee18
 973000: system.cpu T0 : @_start+37.3  :   CALL_NEAR_I : subi   rsp, rsp, 0x8 : IntAlu :  D=0x00007fffffffee18
 973000: system.cpu T0 : @_start+37.4  :   CALL_NEAR_I : wrip   , t7, t1 : IntAlu :
1042000: system.cpu T0 : @__libc_start_main    : push   r15
1042000: system.cpu T0 : @__libc_start_main.0  :   PUSH_R : st   r15, SS:[rsp + 0xfffffffffffffff8] : MemWrite :  D=0x0000000000000000 A=0x7fffffffee10
1063000: system.cpu T0 : @__libc_start_main.1  :   PUSH_R : subi   rsp, rsp, 0x8 : IntAlu :  D=0x00007fffffffee10
1118000: system.cpu T0 : @__libc_start_main+2    : movsxd   rax, rsi
1118000: system.cpu T0 : @__libc_start_main+2.0  :   MOVSXD_R_R : sexti   rax, rsi, 0x1f : IntAlu :  D=0x0000000000000001
1173000: system.cpu T0 : @__libc_start_main+5    : mov  r15, r9
1173000: system.cpu T0 : @__libc_start_main+5.0  :   MOV_R_R : mov   r15, r15, r9 : IntAlu :  D=0x0000000000000000
1228000: system.cpu T0 : @__libc_start_main+8    : push r14
```

事实上，Exec 标志实际上是多个调试标志的集合。通过使用 --debug-help 参数运行 gem5，你可以看到这一点以及所有可用的调试标志。

## Adding a new debug flag

在前几章中，我们使用简单的 std::cout 从模拟对象中打印。虽然可以在 gem5 中使用普通的 C/C++ I/O，但这是极不可取的。因此，我们现在要使用gem5的调试工具来取代它。

创建新的调试标记时，我们首先要在 SConscript 文件中声明它。将以下内容添加到包含 hello 对象代码的目录（src/learning_gem5/SConscript）中的 SConscript 文件中。

```scons
DebugFlag('HelloExample')
```

这样就声明了一个调试标志 "HelloExample"。**现在，我们可以在模拟对象的调试语句中使用该标志。**

**通过在 SConscript 文件中声明调试标志，会自动生成一个调试头文件，允许我们使用调试标志**。头文件位于调试目录中，其名称（和大小写）与我们在 SConscript 文件中声明的名称相同。因此，我们需要在任何计划使用调试标记的文件中包含自动生成的头文件。

在 hello_object.cc 文件中，我们需要包含头文件。

```c++
#include "base/trace.hh"
#include "debug/HelloExample.hh"
```

现在我们已经包含了必要的头文件，让我们用这样的调试语句替换 std::cout 调用。

```C++
DPRINTF(HelloExample, "Created the hello object\n");
```

DPRINTF 是一个 C++ 宏。**第一个参数是 SConscript 文件中声明的调试标志**。我们可以使用 HelloExample 标志，因为我们在 src/learning_gem5/SConscript 文件中声明了它。**其余参数都是变量，可以是任何传递给 printf 语句的参数**。

现在，如果重新编译 gem5 并使用 "HelloExample "调试标记运行，就会得到如下结果。



## 事件驱动编程

gem5 是一个事件驱动的模拟器。在本章中，我们将探讨如何创建和调度事件。我们将从 hello-simobject 章节中的简单 HelloObject 开始构建。

### 创建简单的事件回调

在 gem5 的事件驱动模型中，**每个事件都有一个处理该事件的回调函数**。一般来说，这是一个继承自 :cppEvent 的类。不过，gem5 提供了一个用于创建简单事件的封装函数。

在 HelloObject 的头文件中，我们只需声明一个新函数（processEvent()），每次事件触发时都执行该函数。该函数必须不带参数，也不返回任何内容。

接下来，我们添加一个事件实例。在本例中，我们将使用 EventFunctionWrapper，它允许我们执行任何函数。

我们还添加了一个启动（）函数，下文将对此进行说明。

```c++
class HelloObject : public SimObject
{
  private:
    void processEvent();

    EventFunctionWrapper event;

  public:
    HelloObject(const HelloObjectParams &p);

    void startup() override;
};
```



接下来，我们必须在 HelloObject 的构造函数中构造该事件。EventFuntionWrapper 有两个参数，一个是要执行的函数，另一个是名称。名称通常是拥有该事件的 SimObject 的名称。打印名称时，会在名称末尾自动添加".wrapped_function_event"。

第一个参数是一个不带参数且没有返回值的函数（std::function<void(void)>）。通常，这是一个调用成员函数的简单 lambda 函数。不过，它也可以是任何你想要的函数。下面，我们在 lambda（[this]）中捕获了这个函数，因此我们可以调用类实例的成员函数。

```c++
HelloObject::HelloObject(const HelloObjectParams &params) :
    SimObject(params), event([this]{processEvent();}, name())
{
    DPRINTF(HelloExample, "Created the hello object\n");
}
```

我们还必须定义进程函数的实现。在这种情况下，如果要进行调试，我们只需打印一些内容即可。

```c++
void
HelloObject::processEvent()
{
    DPRINTF(HelloExample, "Hello world! Processing the event!\n");
}
```



### 调度事件

最后，要处理事件，我们首先要对事件进行调度。为此，我们使用 :cppschedule 函数。该函数将某个事件实例调度到未来某个时间（事件驱动模拟不允许事件在过去执行）。

我们最初将在添加到 HelloObject 类的 startup() 函数中调度事件。startup() 函数是允许模拟对象安排内部事件的地方。直到模拟第一次开始（即从 Python 配置文件调用 simulate() 函数），它才会被执行。

```c++
void
HelloObject::startup()
{
    schedule(event, 100);
}
```

在这里，我们只需将事件安排在 tick 100 时执行。通常，你会使用 curTick() 的某个偏移量，但由于我们知道启动()函数是在当前时间为 0 时调用的，所以我们可以使用一个明确的刻度值。

使用 "HelloExample "调试标记运行 gem5 时的输出现在是

```shell
gem5 Simulator System.  http://gem5.org
gem5 is copyrighted software; use the --copyright option for details.

gem5 compiled Jan  4 2017 11:01:46
gem5 started Jan  4 2017 13:41:38
gem5 executing on chinook, pid 1834
command line: build/X86/gem5.opt --debug-flags=Hello configs/learning_gem5/part2/run_hello.py

Global frequency set at 1000000000000 ticks per second
      0: hello: Created the hello object
Beginning simulation!
info: Entering event queue @ 0.  Starting simulation...
    100: hello: Hello world! Processing the event!
Exiting @ tick 18446744073709551615 because simulate() limit reached
```



### 更多的事件调度

我们还可以在事件处理操作中安排新事件。例如，我们将为 HelloObject 添加一个延迟参数，并为触发事件的次数添加一个参数。在下一章中，我们将通过 Python 配置文件访问这些参数。

在 HelloObject 类声明中，为延迟和触发次数添加一个成员变量。

```c++
class HelloObject : public SimObject
{
  private:
    void processEvent();

    EventFunctionWrapper event;

    const Tick latency;

    int timesLeft;

  public:
    HelloObject(const HelloObjectParams &p);

    void startup() override;
};
```

然后，在构造函数中为延迟和剩余时间添加默认值。

```c++
HelloObject::HelloObject(const HelloObjectParams &params) :
    SimObject(params), event([this]{processEvent();}, name()),
    latency(100), timesLeft(10)
{
    DPRINTF(HelloExample, "Created the hello object\n");
}
```

最后，更新 startup() 和 processEvent()。

```c++
void
HelloObject::startup()
{
    schedule(event, latency);
}

void
HelloObject::processEvent()
{
    timesLeft--;
    DPRINTF(HelloExample, "Hello world! Processing the event! %d left\n", timesLeft);

    if (timesLeft <= 0) {
        DPRINTF(HelloExample, "Done firing!\n");
    } else {
        schedule(event, curTick() + latency);
    }
}
```

现在，当我们运行 gem5 时，事件应触发 10 次，模拟将在 1000 ticks 后结束。现在的输出应该如下所示。

```shell
gem5 Simulator System.  http://gem5.org
gem5 is copyrighted software; use the --copyright option for details.

gem5 compiled Jan  4 2017 13:53:35
gem5 started Jan  4 2017 13:54:11
gem5 executing on chinook, pid 2326
command line: build/X86/gem5.opt --debug-flags=Hello configs/learning_gem5/part2/run_hello.py

Global frequency set at 1000000000000 ticks per second
      0: hello: Created the hello object
Beginning simulation!
info: Entering event queue @ 0.  Starting simulation...
    100: hello: Hello world! Processing the event! 9 left
    200: hello: Hello world! Processing the event! 8 left
    300: hello: Hello world! Processing the event! 7 left
    400: hello: Hello world! Processing the event! 6 left
    500: hello: Hello world! Processing the event! 5 left
    600: hello: Hello world! Processing the event! 4 left
    700: hello: Hello world! Processing the event! 3 left
    800: hello: Hello world! Processing the event! 2 left
    900: hello: Hello world! Processing the event! 1 left
   1000: hello: Hello world! Processing the event! 0 left
   1000: hello: Done firing!
Exiting @ tick 18446744073709551615 because simulate() limit reached
```



# 附录：问题合集

1.gem5_hybrid2 使用python3版本，服务器上的gem5使用python2.7版本。最新gem5和gem5_hybrid2版本弃用了python2.7.

两种解决方案：单独在一台服务器配置完全的python3环境；在同一台机器上同时使用Python2.7和Python3来编译不同版本的gem5；

**最好的解决方案 直接使用`/usr/bin/env python3 $(which scons)`  强行python3编译** 所以并未修改默认python版本

但是需要退出`fish shell`（一个我常用的终端），这里有语法冲突。



Hybrid2 环境要求： 

参考 Hybrid^2^             Sconstruct

```python
gem5_Hybrid2/Sconstruct
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
[Line 315] We always use C++14
[Line 340] if compareVersions(main['CXXVERSION'], "5") < 0: (GCC版本>=5)
[Line 370] if compareVersions(main['CXXVERSION'], "3.9") < 0: (Clang 版本>=3.9)
[Line 455] # Based on the availability of the compress stream wrappers, require 2.1.0. (Protobuf版本>=2.1.0)\
[Line 456] min_protoc_version = '2.1.0' (Protobuf版本>=2.1.0)
```



同一台机器不同程序需要不同环境时，只需更改优先级，数字越大，优先级越高；最高优先级为默认版本。

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 50 //更改gcc版本优先级
```

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 60 //将gcc回退到4.8版本，让gem5重新能跑
```

```shell
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 40 //将gcc版本设置为5,让hybrid2能跑
```

```shell
sudo update-alternatives --config gcc //查看版本优先级
```

g++同理

`scons -h` 查看warning部分

* Warning: Can't enable object file debug section compression

* Warning: Can't enable executable debug section compression

* Warning: Couldn't find any HDF5 C++ libraries. Disabling HDF5 support

[看教程，Warning不重要]



重新编译protobuf-2.5.0 [~/gem5/protobuf-2.5.0]

```shell
./autogen.sh    
./configure    
make -j8
make check    
sudo make install    
sudo ldconfig
```

```shell
cd python
python setup.py build
python setup.py test
python setup.py install
```

这一步 python setup.py install 报错，但是使用python import google.protobuf未报错【暂时忽略】

最后`build/X86/gem5.opt`编译成功

**所以之前出现`‘kEmptyString’`的问题是用老版本的gcc/g++编译protobuf ！** **需要用新的gcc....重新编译protobuf !**

...貌似Hybrid2应该`build/ARM/gem5.opt`编译,重新使用`build/ARM/gem5.opt`编译

![](D:\桌面\DesktopFile\md笔记\src\gem5_run_arm.png)

这是个`gem5_hybrid2/run.sh`的最后几行,应该表示的是用的ARM吧



接下来已经可以正常使用了

```shell
bash run.sh <benchmark_name (e.g. mcf_r)>
```

问题就只剩两个目录GEM_DIR / BENCH_DIR

 ```python
gem5_hybrid2/run.sh
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
GEM5_DIR=/home/yifan/gem5_huawei_nomemcpy_timing_swap
BENCH_DIR=/home/yifan/spec2017_build_float_huawei_nomemcpy_timing_swap
RUN_SUFFIX=run_base_refrate_testarm-64.0000
 ```



将`SPEC-CPU-2017.iso.zip`下载到上级目录下

准备SPEC2017的安装和部署

Hybrid^2^采用ARM架构，所以SPEC2017的安装按照ARM指令集

参考链接：

（1）[speccpu2017的安装与运行-CSDN博客](https://blog.csdn.net/weixin_45520085/article/details/131303231)

（2）[Gem5(SE模式)上运行SPEC2017教程 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/222595617)

```shell
cd xxx    #切换到cpu2017.iso所在的目录下
sudo mount -t iso9660 -o ro,exec,loop spec2017.iso /mnt    #挂载cpu2017.iso镜像文件
cd /mnt    #切换目录到挂载目录
./install.sh    #运行spec2017的安装文件，并指定其安装路径（以/root/cpu2017为例）

# 修改配置文件
#在speccpu2017/config目录下，有speccpu2017自带的配置文件，我们可以复制后修改相应的代码进行使用，在这里我使用的是ARM架构，因#此复制Example-gcc-linux-aarch64.cfg文件，并将复制的文件命名为aarch64.cfg。
#并将修改文件中的gcc路径：

# **修改前：**
# %ifndef %{gcc_dir}
# %   define  gcc_dir        /opt/rh/devtoolset-6/root/usr  # EDIT (see above)
# %endif

# **修改后：**
# %ifndef %{gcc_dir}
# %   define  gcc_dir        /usr  # EDIT (see above)
# %endif

cd spec2017/    #切换到安装目录中
source shrc
runcpu --config=aarch64 --action setup --size=test all    #通过size参数可以指定程序输入数据规模大小(test、train、ref)
```

按照（1）  runcpu 时  

`Using 'linux-x86_64' tools` 编译 aarch64 ???

![](D:\桌面\DesktopFile\md笔记\src\spec_bug.png)

应该是`using 'linux-aarch64' tools `才对

但是好像还是能编译？因为用了交叉编译链工具 ??

```shell
// 安装.c转二进制文件的交叉编译链工具
sudo apt-get install gcc-aarch64-linux-gnu
// 安装.cpp转二进制文件的交叉编译链工具
sudo apt-get install g++-aarch64-linux-gnu
// 安装gfortran交叉编译链工具
sudo apt-get install gfortran-aarch64-linux-gnu
```

* `uname -a`得到本机是linux-x86_64，在x86上从源码编译到ARM binary，这是**交叉编译**；如果从x86 binary生成ARM binary,就是**二进制翻译**。 

* hybrid2 的 `run.sh`最后几行

```shell
$GEM5_DIR/build/ARM/gem5.opt  $GEM5_DIR/configs/hybrid/hbm_se.py --num-cpus=1 --cpu-type=TimingSimp ......
```

不确定这里能不能改成 /build/X86/gem5.opt , 如果可以的话 ，事情就变得简单了 。



实测没有必要，但是需要修改：

修改`spec2017/config/aarch64.cfg`的`lable`，将`mytest`更改为`testarm`，然后重新`runcpu`一次

因为`gem5_hybrid2/run.sh`里的各种路径文件名的`lable`是`testarm`

另一种方式，即使修改了整个`run.sh`的`testarm`，最后`bash run.sh <benchmark>`的时候还是会有无法找到的报错提示

这种省时间的方式很可惜不可行，只能重新`runcpu`，耗时较久。

